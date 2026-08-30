# Project 1B: Temporal Graph Sector Rotation Network (TGSR)

**Role:** Quantitative Financial Data Analyst 
**Entity:** ZeTheta Algorithms Private Limited (CIN: U62012MH2023PTC410415) 
**Target:** Long-Only Indian Mutual Fund Schemes / Buy-Side Sector Rotation Desk 

---

## 📌 Executive Summary
Indian sectoral indices partition over Rs 160 lakh crore of market capitalization across 14+ NSE indices, with annual sector dispersion routinely exceeding 40 percentage points. Capturing this dispersion requires more than static correlation-based factor models, which fail during structural shifts (e.g., Banking-NBFC decoupling during IL&FS 2018, COVID tech re-rating 2020).

This project implements a **Temporal Graph Sector Rotation Network (TGSR)** that ranks NSE sectoral indices by relative-strength using a **Graph Attention Network (GATv2)** trained over a domain-prior sector correlation graph. The model, its out-of-sample results, and an honest discussion of its current limitations (including an observed over-smoothing issue) are documented in the Model Card — this is presented as a research prototype, not a production-ready signal.

---

## 🛠️ Repository Architecture & Deliverables Index

### 1. Python Codebase (`Deliverable 2`)
- `01_sector_data_ingestion.py` — Ingests 10+ years of OHLCV data for 8+ NSE sectoral indices, Nifty 50, and India VIX; computes momentum, volatility, and 63-day relative strength features.
- `02_tgsr_graph_network.py` — Constructs the domain-prior sector correlation graph, implements the GATv2 message-passing architecture, computes attention weights, and predicts sector relative-strength rankings.
- `03_regime_sector_backtest.py` — Executes a tactical top-3 sector overlay backtest over the model's out-of-sample window (Nov 2021–Nov 2025) accounting for institutional transaction costs (60 bps round-trip).

### 2. Cross-Verification & Statistical Rigour (`Deliverable 3`)
- `04_r_verification.R` — Independent R implementation that reproduces the inter-sector correlation matrix, cross-checks the Python-generated graph adjacency (|corr| ≥ 0.35 threshold) cell-by-cell, and ranks sectors by full-sample mean return as a sanity check against the Python pipeline.

### 3. Comprehensive Documentation & Reports (`Deliverables 1, 4, 5`)
- `05_Model_Card_and_Report.md` — Technical report covering the graph construction methodology, GATv2 architecture, train/OOS split, Information Coefficient results, backtest performance, the observed over-smoothing limitation, and next-iteration roadmap.

### 4. Presentation & Video Script (`Deliverable 6`)
- `06_Presentation_and_Script.md` — 14-slide executive presentation outline and full script for a 10-minute live demonstration video.

---

## 📊 Visual Outputs & Plots
- `tgsr_sector_ranking_plot.png` — Model-predicted relative strength score ranking across major NSE sectors.
- `tgsr_backtest_performance.png` — Cumulative growth curve comparing the TGSR Sector Rotation Overlay against the Nifty 50 Buy-and-Hold benchmark.

---

## 🔬 Core Methodological Innovations
1. **Direction Over Price:** Prioritizes cross-sectional relative-strength ranking over noisy point price predictions.
2. **Attention-Weighted Graph:** Each GATv2 layer computes learned attention weights over the sector correlation graph rather than treating all neighbours equally — raw attention weights are available for inspection, though translating them into narrative economic rationales is left for a future iteration.
3. **Regime-Aware Features:** Node features include a binary India VIX regime flag (high-vol vs low-vol month) alongside relative-strength and volatility features.
4. **Reproducibility:** Fixed random seeds (`torch.manual_seed(42)`), a chronological (non-shuffled) train/OOS split, and every intermediate output saved to CSV so each step can be independently re-run and checked.

---

## ⚙️ Environment & Setup
To run this repository locally:

```bash
# Clone the repository
git clone https://github.com/panjetajyoti/Zetheta_TGSR_Project_1B.git
cd Zetheta_TGSR_Project_1B_Sector_Rotation

# Install dependencies
pip install numpy pandas matplotlib torch yfinance

# Run the pipeline in order
python 01_sector_data_ingestion.py
python 02_tgsr_graph_network.py
python 03_regime_sector_backtest.py
Rscript 04_r_verification.R
```
