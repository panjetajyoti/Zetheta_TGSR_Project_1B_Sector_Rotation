# ==============================================================================
# PROJECT 1B: R CODEBASE VERIFICATION (TGSR)
# ==============================================================================
cat("Verifying Sector Relative Strength & Cross-Sectional Ranking in R...\n")

if(!require(data.table)) install.packages("data.table")
library(data.table)

# Load Return Data
ret_data <- fread("nse_sector_returns.csv")
sectors <- c("BANK", "IT", "AUTO", "PHARMA", "FMCG", "METAL", "REALTY", "ENERGY")

# Compute Correlation Matrix
corr_mat <- cor(ret_data[, ..sectors], use = "complete.obs")
cat("\n--- Inter-Sector Correlation Matrix ---\n")
print(round(corr_mat, 3))

# Cross-verify the Python-side domain-prior graph: an edge exists where
# |correlation| >= 0.35 (see 02_tgsr_graph_network.py, EDGE_THRESHOLD).
# This independently reproduces the adjacency saved to
# tgsr_sector_graph_adjacency.csv, in a different language/toolchain.
EDGE_THRESHOLD <- 0.35
adj_r <- (abs(corr_mat) >= EDGE_THRESHOLD) * 1
diag(adj_r) <- 1
cat("\n--- R-Verified Sector Graph Adjacency (|corr| >=", EDGE_THRESHOLD, ") ---\n")
print(adj_r)

if (file.exists("tgsr_sector_graph_adjacency.csv")) {
  adj_py <- as.matrix(read.csv("tgsr_sector_graph_adjacency.csv", row.names = 1))
  adj_py <- adj_py[sectors, sectors]
  match_pct <- mean(adj_r == adj_py) * 100
  cat(sprintf("\nAgreement with Python-generated adjacency: %.1f%% of cells match\n", match_pct))
}

# Rank Sectors by Mean Return
mean_rets <- colMeans(ret_data[, ..sectors], na.rm = TRUE)
r_ranks <- rank(-mean_rets)
cat("\nR-Calculated Cross-Sectional Sector Ranks (by full-sample mean daily return):\n")
print(r_ranks)