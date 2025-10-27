#!/usr/bin/env python3
"""
Evaluate retrieval quality on the RAG sample dataset.

Computes Precision@K, Recall@K, and nDCG@K
for each query and averages the results.

Assumes the dataset columns:
  query_id, query_text, candidate_id, candidate_text,
  baseline_rank, baseline_score, gold_label
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("rag_sample_queries_candidates.csv")

# Ensure results are ordered by the baseline rank
df.sort_values(["query_id", "baseline_rank"], inplace=True)

# ---------------------------------------------------------------------
# 2. Metric helpers
# ---------------------------------------------------------------------
def precision_at_k(labels, k):
    """labels: list/array of 0/1 relevance sorted by baseline rank"""
    topk = labels[:k]
    return np.sum(topk) / len(topk)

def recall_at_k(labels, k):
    """Recall = retrieved relevant / total relevant"""
    total_relevant = np.sum(labels)
    if total_relevant == 0:
        return np.nan  # undefined
    topk = labels[:k]
    return np.sum(topk) / total_relevant

def ndcg_at_k(labels, k):
    """Compute nDCG@k with binary relevance (0/1)."""
    labels = np.array(labels)
    k = min(k, len(labels))
    gains = (2 ** labels[:k] - 1)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts)

    # Ideal DCG: sorted by true relevance
    ideal = np.sort(labels)[::-1]
    ideal_gains = (2 ** ideal[:k] - 1)
    idcg = np.sum(ideal_gains * discounts)
    return 0.0 if idcg == 0 else dcg / idcg

# ---------------------------------------------------------------------
# 3. Compute metrics per query
# ---------------------------------------------------------------------
results = []
K = 3

for qid, group in df.groupby("query_id"):
    labels = group["gold_label"].tolist()
    p = precision_at_k(labels, K)
    r = recall_at_k(labels, K)
    n = ndcg_at_k(labels, K)
    results.append({"query_id": qid, f"precision@{K}": p, f"recall@{K}": r, f"nDCG@{K}": n})

metrics = pd.DataFrame(results)

# ---------------------------------------------------------------------
# 4. Display per-query and average metrics
# ---------------------------------------------------------------------
print(metrics.round(3))
print("\nAverage metrics:")
print(metrics[[f"precision@{K}", f"recall@{K}", f"nDCG@{K}"]].mean().round(3))

# ----------------------------
# (Optional) Step 2: LLM scores
# Set env RERANK_COMPUTE_SCORES=1 to compute an llm_score column and save CSV.
# ----------------------------
if __name__ == "__main__":
    import os
    if os.getenv("RERANK_COMPUTE_SCORES") == "1":
        import pandas as pd
        from api_starter import score_pair
        df = pd.read_csv("/mnt/data/rag_sample_queries_candidates.csv")
        # Minimal, no duplicate work beyond exact (query_id, candidate_id) pairs
        seen = {}
        scores = []
        for row in df.itertuples(index=False):
            key = (row.query_id, row.candidate_id)
            if key not in seen:
                seen[key] = score_pair(row.query_text, row.candidate_text)
            scores.append(seen[key])
        df["llm_score"] = scores
        out_path = "/mnt/data/rag_with_llm_scores.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote LLM scores to {out_path}")

# ----------------------------
# (Optional) Step 3: Rerank by llm_score and evaluate
# Trigger with: RERANK_EVAL=1
# Optional: RERANK_FALLBACK_BASELINE=1 (if llm_score missing, derive from baseline_score)
# ----------------------------
