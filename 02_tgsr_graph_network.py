"""
PROJECT 1B: TEMPORAL GRAPH SECTOR ROTATION (TGSR)
Step 2: Domain-Prior Sector Graph Construction + GATv2 Training

This script:
  1. Builds a static domain-prior sector correlation graph (edges = pairs of
     sectors whose daily-return correlation exceeds a threshold).
  2. Builds monthly node-feature snapshots (21d/63d/126d relative strength,
     21d realized volatility, INDIAVIX regime flag) for each of the 8 NSE
     sectors.
  3. Implements a GATv2-style graph attention layer from scratch in PyTorch
     (no torch_geometric dependency) and trains it to predict each sector's
     forward 21-trading-day relative-strength versus Nifty 50.
  4. Chronologically splits the sample 70% train / 30% out-of-sample (OOS)
     so the OOS predictions used downstream (03_regime_sector_backtest.py)
     are genuinely never seen during training.
  5. Saves the OOS monthly predictions (tgsr_oos_rankings.csv) and the
     latest full-sample prediction (tgsr_predicted_rankings.csv) used in
     the report/plot.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("PROJECT 1B: TEMPORAL GRAPH SECTOR ROTATION (TGSR)")
print("Step 2: Graph Construction + GATv2 Training")
print("=" * 60)

SECTORS = ['BANK', 'IT', 'AUTO', 'PHARMA', 'FMCG', 'METAL', 'REALTY', 'ENERGY']
N = len(SECTORS)

prices = pd.read_csv("nse_sector_prices.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("nse_sector_returns.csv", index_col=0, parse_dates=True)

# ---------------------------------------------------------------------------
# 1. Domain-prior sector correlation graph (static adjacency)
# ---------------------------------------------------------------------------
corr = returns[SECTORS].corr()
EDGE_THRESHOLD = 0.35
adj = (corr.abs() >= EDGE_THRESHOLD).astype(float).copy()
adj_vals = adj.values.copy()
np.fill_diagonal(adj_vals, 1.0)  # self-loops, standard for GAT
adj = pd.DataFrame(adj_vals, index=adj.index, columns=adj.columns)
edge_index = adj.values  # (N, N) binary mask, 1 = edge exists
n_edges = int(edge_index.sum() - N)
print(f"Sector correlation graph: {n_edges} inter-sector edges "
      f"(|corr| >= {EDGE_THRESHOLD}), {N} self-loops")
adj.to_csv("tgsr_sector_graph_adjacency.csv")

# ---------------------------------------------------------------------------
# 2. Monthly node-feature snapshots + forward-return targets
# ---------------------------------------------------------------------------
month_ends = returns.resample('ME').last().index
FWD_HORIZON = 21  # ~1 trading month

rows = []
dates_used = []
for dt in month_ends:
    idx_loc = returns.index.searchsorted(dt, side='right') - 1
    if idx_loc < 130 or idx_loc + FWD_HORIZON >= len(returns):
        continue
    asof = returns.index[idx_loc]

    feat_rows = []
    for s in SECTORS:
        rs21 = returns[s].iloc[idx_loc - 21:idx_loc].mean() - returns['NIFTY50'].iloc[idx_loc - 21:idx_loc].mean()
        rs63 = returns[s].iloc[idx_loc - 63:idx_loc].mean() - returns['NIFTY50'].iloc[idx_loc - 63:idx_loc].mean()
        rs126 = returns[s].iloc[idx_loc - 126:idx_loc].mean() - returns['NIFTY50'].iloc[idx_loc - 126:idx_loc].mean()
        vol21 = returns[s].iloc[idx_loc - 21:idx_loc].std()
        vix_regime = 1.0 if returns['INDIAVIX'].iloc[idx_loc - 21:idx_loc].mean() > returns['INDIAVIX'].mean() else 0.0
        feat_rows.append([rs21, rs63, rs126, vol21, vix_regime])

    fwd_sector = returns[SECTORS].iloc[idx_loc + 1: idx_loc + 1 + FWD_HORIZON].sum()
    fwd_bench = returns['NIFTY50'].iloc[idx_loc + 1: idx_loc + 1 + FWD_HORIZON].sum()
    target = (fwd_sector - fwd_bench).values

    rows.append((np.array(feat_rows, dtype=np.float32), target.astype(np.float32)))
    dates_used.append(asof)

print(f"Built {len(rows)} monthly graph snapshots "
      f"({dates_used[0].date()} to {dates_used[-1].date()})")

X = np.stack([r[0] for r in rows])   # (T, N, F)
Y = np.stack([r[1] for r in rows])   # (T, N)

split = int(len(rows) * 0.70)
feat_mean = X[:split].reshape(-1, X.shape[-1]).mean(axis=0)
feat_std = X[:split].reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
X = (X - feat_mean) / feat_std

X_t = torch.tensor(X)
Y_t = torch.tensor(Y)
A_t = torch.tensor(adj.values, dtype=torch.float32)

print(f"Chronological split: train=1..{split} ({dates_used[split-1].date()}), "
      f"OOS={split}..{len(rows)} ({dates_used[split].date()} to {dates_used[-1].date()})")


class GATv2Layer(nn.Module):
    """GATv2 (Brody et al. 2021): LeakyReLU applied BEFORE the attention
    vector dot-product, fixing the static-attention limitation of GATv1."""
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W_l = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        B, N, _ = x.shape
        h_l = self.W_l(x).view(B, N, self.heads, self.out_dim)
        h_r = self.W_r(x).view(B, N, self.heads, self.out_dim)

        h_sum = h_l.unsqueeze(2) + h_r.unsqueeze(1)          # (B, N_i, N_j, heads, out_dim)
        e = self.leaky(h_sum)
        e = torch.einsum('bijhd,hd->bijh', e, self.attn)     # (B, N_i, N_j, heads)

        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask, float('-1e9'))
        alpha = torch.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        out = torch.einsum('bijh,bjhd->bihd', alpha, h_r)
        out = out.mean(dim=2)
        return out, alpha.detach()


class TGSR_GATv2(nn.Module):
    def __init__(self, in_dim, hidden=16, heads=4):
        super().__init__()
        self.gat1 = GATv2Layer(in_dim, hidden, heads=heads)
        self.gat2 = GATv2Layer(hidden, hidden, heads=heads)
        self.act = nn.ELU()
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, adj):
        h, alpha1 = self.gat1(x, adj)
        h = self.act(h)
        h, alpha2 = self.gat2(h, adj)
        h = self.act(h)
        score = self.head(h).squeeze(-1)
        return score, alpha2


model = TGSR_GATv2(in_dim=X.shape[-1], hidden=16, heads=4)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

X_train, Y_train = X_t[:split], Y_t[:split]
X_oos, Y_oos = X_t[split:], Y_t[split:]

EPOCHS = 300
model.train()
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred, _ = model(X_train, A_t)
    loss = loss_fn(pred, Y_train)
    loss.backward()
    opt.step()
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            oos_pred, _ = model(X_oos, A_t)
            oos_loss = loss_fn(oos_pred, Y_oos).item()
        model.train()
        print(f"Epoch {epoch+1:4d} | train MSE {loss.item():.6f} | OOS MSE {oos_loss:.6f}")

model.eval()
with torch.no_grad():
    oos_pred, oos_alpha = model(X_oos, A_t)
    full_pred, _ = model(X_t, A_t)

oos_df = pd.DataFrame(oos_pred.numpy(), index=dates_used[split:], columns=SECTORS)
oos_df.to_csv("tgsr_oos_rankings.csv")

latest_scores = pd.Series(full_pred[-1].numpy(), index=SECTORS).sort_values(ascending=False)
latest_scores.to_frame(name=str(dates_used[-1].date())).to_csv("tgsr_predicted_rankings.csv")

ic_by_month = []
for i in range(len(oos_df)):
    ic = pd.Series(oos_pred[i].numpy()).corr(pd.Series(Y_oos[i].numpy()), method='spearman')
    ic_by_month.append(ic)
mean_ic = np.nanmean(ic_by_month)
print(f"\nOOS Information Coefficient (Spearman, predicted vs realised fwd RS): "
      f"mean={mean_ic:.3f}, hit-rate(IC>0)={np.mean(np.array(ic_by_month) > 0):.1%}")

baseline_ic = []
for i in range(split, len(rows)):
    naive = pd.Series(X[i][:, 1], index=SECTORS)
    baseline_ic.append(naive.corr(pd.Series(Y[i], index=SECTORS), method='spearman'))
ic_compare = pd.DataFrame({
    'GATv2_Model_IC': ic_by_month,
    'Baseline_63d_Momentum_IC': baseline_ic,
}, index=dates_used[split:])
ic_compare.to_csv("tgsr_ic_comparison.csv")
print(f"Baseline (63d momentum) OOS IC: mean={np.nanmean(baseline_ic):.3f}")

plt.figure(figsize=(10, 6))
latest_scores.sort_values().plot(kind='barh', color='steelblue')
plt.title(f"TGSR GATv2 Predicted Relative-Strength Ranking ({dates_used[-1].date()})")
plt.xlabel("Predicted Forward 21d Relative Strength Score")
plt.tight_layout()
plt.savefig("tgsr_sector_ranking_plot.png")

print("\nSaved: tgsr_sector_graph_adjacency.csv, tgsr_oos_rankings.csv, "
      "tgsr_predicted_rankings.csv, tgsr_ic_comparison.csv, tgsr_sector_ranking_plot.png")
print("Step 2 complete: GATv2 model trained and OOS predictions generated.")
