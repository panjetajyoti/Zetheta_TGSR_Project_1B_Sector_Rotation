"""
PROJECT 1B: TEMPORAL GRAPH SECTOR ROTATION (TGSR)
Step 3: Regime-Conditioned Sector Overlay Backtest (Out-of-Sample Only)

Uses the OOS monthly predictions produced by 02_tgsr_graph_network.py
(tgsr_oos_rankings.csv) to run a tactical top-3 equal-weight sector overlay,
priced with institutional round-trip transaction costs, and compares it
against:
  (a) the Nifty 50 buy-and-hold benchmark, and
  (b) a naive 63-day trailing-momentum overlay (no model),
over the SAME out-of-sample window only -- this is a genuine OOS test, not
an in-sample backtest.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Step 3: Regime-Conditioned Sector Overlay Backtest (OOS only)")
print("=" * 60)

returns = pd.read_csv("nse_sector_returns.csv", index_col=0, parse_dates=True)
sectors = ['BANK', 'IT', 'AUTO', 'PHARMA', 'FMCG', 'METAL', 'REALTY', 'ENERGY']

# Out-of-sample monthly model rankings, produced by 02_tgsr_graph_network.py
# on the held-out ~30% of the sample -- the model never saw this period
# during training, so this backtest is a genuine OOS overlay test.
oos = pd.read_csv("tgsr_oos_rankings.csv", index_col=0, parse_dates=True)
oos_dates = oos.index

start, end = oos_dates.min(), oos_dates.max()
bt_returns = returns.loc[start:end]
print(f"Backtest window (out-of-sample only): {start.date()} to {end.date()} "
      f"({len(bt_returns)} trading days, {len(oos_dates)} monthly rebalances)")

# Realistic Indian institutional round-trip cost: ~60 bps per Section A9.2
ROUND_TRIP_COST = 0.006


def run_overlay(rank_source_fn, label):
    """rank_source_fn(date) -> ranked sector Series (higher = stronger).
    Rebalances monthly into equal-weighted top-3."""
    cur_weights = pd.Series(0.0, index=sectors)
    equity = [1.0]
    turnover_log = []
    for i, date in enumerate(bt_returns.index):
        if i > 0:
            daily_ret = (cur_weights * bt_returns.loc[date, sectors]).sum()
            equity.append(equity[-1] * (1 + daily_ret))
        if date in oos_dates:
            ranks = rank_source_fn(date)
            top3 = ranks.nlargest(3).index
            target = pd.Series(0.0, index=sectors)
            target[top3] = 1.0 / 3.0
            turnover = (target - cur_weights).abs().sum() / 2
            turnover_log.append(turnover)
            equity[-1] *= (1 - turnover * ROUND_TRIP_COST)
            cur_weights = target
    print(f"{label}: avg monthly turnover = {np.mean(turnover_log):.2%}, "
          f"n_rebalances = {len(turnover_log)}")
    return pd.Series(equity, index=bt_returns.index)


model_curve = run_overlay(lambda d: oos.loc[d], "Model-driven (GATv2) overlay")
baseline_curve = run_overlay(
    lambda d: bt_returns[sectors].loc[:d].tail(63).mean(),
    "Baseline (63d momentum) overlay"
)
bench_curve = (1 + bt_returns['NIFTY50']).cumprod()
bench_curve.iloc[0] = 1.0
bench_curve = bench_curve / bench_curve.iloc[0]

backtest_df = pd.DataFrame({
    'Benchmark_Nifty50': bench_curve.values,
    'Model_Driven_Overlay': model_curve.values,
    'Baseline_Momentum_Overlay': baseline_curve.values,
}, index=bt_returns.index)
backtest_df.to_csv("tgsr_backtest_results.csv")


def perf_stats(series):
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = (daily.mean() * 252) / vol if vol > 0 else np.nan
    dd = (series / series.cummax() - 1).min()
    return dict(total_return=total_ret, cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd)


stats_rows = []
for col in backtest_df.columns:
    s = perf_stats(backtest_df[col])
    s['series'] = col
    stats_rows.append(s)
stats_df = pd.DataFrame(stats_rows).set_index('series')
print("\n--- Out-of-sample performance, 60bps round-trip cost applied ---")
print(stats_df.round(4).to_string())

active_model = backtest_df['Model_Driven_Overlay'].pct_change() - backtest_df['Benchmark_Nifty50'].pct_change()
active_model = active_model.dropna()
te = active_model.std() * np.sqrt(252)
ir = (active_model.mean() * 252) / te if te > 0 else np.nan
print(f"\nModel-driven overlay vs benchmark: Tracking Error={te:.2%}, "
      f"Information Ratio={ir:.2f}, Hit Rate={(active_model > 0).mean():.1%}")

stats_df.to_csv("tgsr_backtest_stats.csv")

plt.figure(figsize=(12, 6))
plt.plot(backtest_df.index, backtest_df['Benchmark_Nifty50'],
         label='Nifty 50 Buy & Hold', color='gray', alpha=0.8)
plt.plot(backtest_df.index, backtest_df['Baseline_Momentum_Overlay'],
         label='Baseline (63d Momentum) Overlay', color='orange', linewidth=1.5, linestyle='--')
plt.plot(backtest_df.index, backtest_df['Model_Driven_Overlay'],
         label='Model-Driven Overlay (GATv2, OOS)', color='darkgreen', linewidth=2)
plt.title("TGSR Sector Overlay vs Benchmark - Out-of-Sample Period Only (ZeTheta Project 1B)")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth (x), rebased to 1.0 at OOS start")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tgsr_backtest_performance.png")
print("\nSaved: tgsr_backtest_results.csv, tgsr_backtest_stats.csv, tgsr_backtest_performance.png")
