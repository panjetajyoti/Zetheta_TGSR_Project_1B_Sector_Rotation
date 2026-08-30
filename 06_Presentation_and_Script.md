# Presentation Deck & Demo Video Script
**Project:** Temporal Graph Sector Rotation Network (TGSR)
**Zetheta Algorithms Private Limited (CIN: U62012MH2023PTC410415)**

---

## SLIDE OUTLINE (14 Slides)
1. Title Slide: Temporal Graph Sector Rotation Network
2. Executive Summary: Can a learned sector-correlation graph beat naive momentum, net of cost?
3. The Indian Sectoral Universe: 8 NSE sectoral indices, 2012–2025 daily data
4. The Cross-Sectional Thesis: Ranking relative strength conditioned on regime
5. Graph Construction: Static domain-prior graph from full-sample return correlation (|corr| ≥ 0.35 → 44/56 edges)
6. Architecture: 2-layer GATv2 (Brody et al. 2021), 4 heads, 16 hidden units, pointwise MSE on forward 21d relative strength
7. Node Feature Engineering: 21d/63d/126d relative strength, 21d volatility, India VIX regime flag
8. Chronological Train/OOS Split: train 2012-07–2021-10, OOS 2021-11–2025-11 (never seen in training)
9. Predictive Skill: Spearman Information Coefficient, model vs 63d-momentum baseline
10. Overlay Backtest Results: active return, Information Ratio, and transaction-cost drag (60 bps round-trip)
11. Honest Finding: model roughly tracks benchmark and trails the momentum baseline on this sample
12. Diagnosed Limitation: over-smoothing — 6/8 sectors converge to near-identical predicted scores
13. Next Iteration Roadmap: sparser graph threshold, single-layer GAT, residual connections, listwise loss
14. Conclusion: research prototype status, not yet a production sector-tilt signal

---

## 10-MINUTE VIDEO DEMO SCRIPT
- **0:00 – 1:30:** Introduction — Indian sector rotation, why static correlation models miss regime shifts.
- **1:30 – 3:30:** Walkthrough of data ingestion (`01`) and the domain-prior correlation graph.
- **3:30 – 5:30:** GATv2 architecture walkthrough (`02`) — attention mechanism, training curve, train/OOS split.
- **5:30 – 7:30:** Information Coefficient and backtest results (`03`), shown against the momentum baseline.
- **7:30 – 9:00:** The over-smoothing finding — what it looks like in the output scores and why it happens.
- **9:00 – 10:00:** Model Card (`05`) summary, governance/lineage notes, and the proposed next-iteration roadmap.
