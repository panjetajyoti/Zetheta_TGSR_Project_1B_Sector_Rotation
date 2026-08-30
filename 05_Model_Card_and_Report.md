# Project 1B: Temporal Graph Sector Rotation Network (TGSR) — Model Card & Report
**Author:** Quantitative Data Analyst Trainee
**Entity:** Zetheta Algorithms Private Limited (CIN: U62012MH2023PTC410415)
**Target:** Long-Only Indian Mutual Fund Sector Allocation

---

## 1. Executive Summary & Directional Thesis
Static, correlation-only sector rotation heuristics struggle to adapt when
inter-sector relationships shift structurally (e.g. a credit event that
decouples Banks from NBFC-adjacent sectors). This project builds a small,
fully-trained **Graph Attention Network (GATv2)** over the 8 major NSE
sectoral indices, learns which sectors should influence each other's
predicted relative strength, and evaluates whether the resulting ranking
adds value over a naive momentum baseline once realistic transaction costs
are applied. The report below states results exactly as produced by
`02_tgsr_graph_network.py` and `03_regime_sector_backtest.py` — including
where the model does **not** yet beat the baseline — rather than only the
favourable numbers.

## 2. Sector Graph Construction & Architecture
- **Nodes (V):** 8 NSE sectoral indices — Bank, IT, Auto, Pharma, FMCG,
  Metal, Realty, Energy.
- **Edges (E):** Static domain-prior graph — an edge exists between two
  sectors if the full-sample |correlation| of their daily returns is
  ≥ 0.35, plus self-loops. This produced **44 inter-sector edges** out of
  56 possible pairs — the graph is fairly dense; IT is the most weakly
  connected node (only linked to Auto and itself), reflecting IT's lower
  correlation with the rest of the cyclical/domestic sectors.
- **Node Features (X):** 5 features per sector per month-end snapshot —
  21d, 63d and 126d relative strength vs Nifty 50, 21d realised volatility,
  and a binary India VIX regime flag (high-vol vs low-vol month).
- **Architecture:** Two stacked GATv2 layers (Brody et al., 2021 attention
  ordering — LeakyReLU applied before the attention vector, which fixes
  GATv1's static-attention limitation), 4 attention heads, 16 hidden
  units, followed by a linear regression head. Trained with pointwise MSE
  against each sector's realised forward 21-trading-day relative strength
  (a listwise ranking loss such as ListMLE was considered but a pointwise
  regression target was used for this iteration, since the resulting score
  is still used purely for ranking).
- **Training/OOS split:** Chronological 70/30 split — trained on
  2012-07 to 2021-10 (112 monthly snapshots), evaluated out-of-sample on
  2021-11 to 2025-11 (49 monthly snapshots) that the model never saw
  during training.

## 3. Predictive Skill: Information Coefficient
Monthly Spearman rank correlation between the model's predicted score and
each sector's realised forward relative strength, on the OOS window only
(see `tgsr_ic_comparison.csv` for the full month-by-month series):

| | Mean IC | Hit rate (IC > 0) |
|---|---|---|
| GATv2 model | **0.030** | 53.1% |
| Baseline (63d momentum, no model) | **0.071** | — |

The IC is noisy and swings between roughly −0.76 and +0.76 month to month
— there is no month where the model reliably dominates. On this sample the
naive 63-day momentum baseline actually shows a higher average IC than the
trained GATv2 model.

## 4. Backtest Performance (OOS, 60bps round-trip cost)
Tactical top-3 equal-weight monthly overlay, Nov 2021 – Nov 2025:

| Series | Total Return | CAGR | Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|---|
| Nifty 50 Benchmark | 48.0% | 10.3% | 13.7% | 0.80 | −17.3% |
| Model-Driven (GATv2) Overlay | 49.1% | 10.5% | 13.5% | 0.82 | −18.7% |
| Baseline Momentum Overlay | 56.0% | 11.8% | 17.2% | 0.75 | −21.8% |

Tracking error vs benchmark: 6.16%, Information Ratio: **0.02** (essentially
flat — not a statistically meaningful edge on this sample size). The model
overlay is roughly in line with the benchmark and modestly lower-vol than
the momentum baseline, but it does **not** demonstrate a clear, robust
outperformance claim.

## 5. Observed Limitation: Over-Smoothing
The latest full-sample prediction (`tgsr_predicted_rankings.csv`) assigns
**near-identical scores to 6 of the 8 sectors** (Bank, Pharma, Metal,
FMCG, Realty, Energy all ≈ 0.00236), with only IT and Auto separated. This
is a classic symptom of **over-smoothing** in stacked graph-attention
layers on a dense graph: with 44/56 possible edges present, two rounds of
neighbourhood averaging pull most node representations toward the graph
mean. Candidate fixes for the next iteration: (a) raise the correlation
threshold to sparsify the graph, (b) drop to a single GAT layer, (c) add
a residual/skip connection from the raw features to the output head, or
(d) add an explicit diversity/regularisation term to the loss.

## 6. SEBI-style Governance Notes
- Full lineage: raw price data → engineered features → static graph →
  model weights → OOS predictions → backtest, each step is a separate,
  re-runnable script with fixed random seeds (`torch.manual_seed(42)`).
- Given the honest performance figures in Sections 3–5, this model should
  currently be positioned as a **research prototype**, not a
  production sector-tilt signal — it has not yet demonstrated a return or
  information-ratio edge over a simple momentum heuristic net of costs.
