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
        FEWSHOT = os.getenv("RERANK_FEWSHOT") == "1"
        df = pd.read_csv("rag_sample_queries_candidates.csv")
        seen = {}
        scores = []
        for row in df.itertuples(index=False):
            key = (row.query_id, row.candidate_id)
            if key not in seen:
                seen[key] = score_pair(row.query_text, row.candidate_text)
            scores.append(seen[key])
        df["llm_score"] = scores
        out_path = "rag_with_llm_scores.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote LLM scores to {out_path}")

# ----------------------------
# (Optional) Step 3: Rerank by llm_score and evaluate
# Trigger with: RERANK_EVAL=1
# Optional: RERANK_FALLBACK_BASELINE=1 (if llm_score missing, derive from baseline_score)
# ----------------------------
if __name__ == "__main__":
    import os
    from pathlib import Path
    if os.getenv("RERANK_EVAL") == "1":
        import math
        import pandas as pd
        FEWSHOT = os.getenv("RERANK_FEWSHOT") == "1"

        # Load LLM-scored pairs
        in_path = "rag_with_llm_scores.csv" if FEWSHOT else "rag_with_llm_scores.csv"
        if not Path(in_path).exists():
            raise SystemExit(f"Missing {in_path}. Run with RERANK_COMPUTE_SCORES=1 first.")

        df = pd.read_csv(in_path)

        # (Very small) Optional fallback if llm_score is missing or NaN
        if os.getenv("RERANK_FALLBACK_BASELINE") == "1":
            if "llm_score" not in df.columns or df["llm_score"].isna().any():
                b = df["baseline_score"]
                bn = (b - b.min()) / (b.max() - b.min() + 1e-9)
                df["llm_score"] = (bn * 5).round().astype(int).clip(0, 5)

        if "llm_score" not in df.columns:
            raise SystemExit("No llm_score column found and fallback disabled.")

        # Sort within each query by llm_score DESC, then baseline_rank ASC as a stable tie-breaker
        df_sorted = df.sort_values(by=["query_id", "llm_score", "baseline_rank"],
                                   ascending=[True, False, True]).copy()

        # Assign reranked positions per query
        df_sorted["rerank"] = df_sorted.groupby("query_id").cumcount() + 1

        # Save a reranked CSV for inspection
        out_path = "rag_reranked_fewshot.csv" if FEWSHOT else "rag_reranked.csv"
        df_sorted.to_csv(out_path, index=False)
        print(f"Wrote reranked CSV to {out_path}")

        # Evaluate using existing metrics
        try:
            _ = K
        except NameError:
            K = 3

        results = []
        for qid, group in df_sorted.groupby("query_id"):
            labels_in_order = group["gold_label"].tolist()
            p = precision_at_k(labels_in_order, K)
            r = recall_at_k(labels_in_order, K)
            n = ndcg_at_k(labels_in_order, K)
            results.append({"query_id": qid, f"precision@{K}": p, f"recall@{K}": r, f"nDCG@{K}": n})

        metrics = pd.DataFrame(results)
        print("\nReranked metrics (per query):")
        print(metrics.round(3))
        print("\nReranked average metrics:")
        print(metrics[[f"precision@{K}", f"recall@{K}", f"nDCG@{K}"]].mean().round(3))

# ----------------------------
# (Optional) Step 4: Compare baseline vs reranked metrics
# Trigger with: RERANK_COMPARE=1
# ----------------------------
if __name__ == "__main__":
    import os
    if os.getenv("RERANK_COMPARE") == "1":
        import pandas as pd
        from pathlib import Path
        FEWSHOT = os.getenv("RERANK_FEWSHOT") == "1"

        # Baseline metrics (recompute to avoid relying on prior prints)
        base_df = pd.read_csv("rag_sample_queries_candidates.csv").sort_values(["query_id", "baseline_rank"])
        try:
            _ = K
        except NameError:
            K = 3

        base_rows = []
        for qid, group in base_df.groupby("query_id"):
            labels = group["gold_label"].tolist()
            p = precision_at_k(labels, K)
            r = recall_at_k(labels, K)
            n = ndcg_at_k(labels, K)
            base_rows.append({"query_id": qid, f"precision@{K}": p, f"recall@{K}": r, f"nDCG@{K}": n})
        base_metrics = pd.DataFrame(base_rows).set_index("query_id")

        # Reranked metrics
        rerank_path = Path("rag_reranked_fewshot.csv" if FEWSHOT else "rag_reranked.csv")
        if not rerank_path.exists():
            raise SystemExit("Missing rag_reranked.csv. Run Step 3 first (RERANK_EVAL=1).")
        reranked = pd.read_csv(rerank_path)

        rr_rows = []
        for qid, group in reranked.groupby("query_id"):
            labels = group["gold_label"].tolist()
            p = precision_at_k(labels, K)
            r = recall_at_k(labels, K)
            n = ndcg_at_k(labels, K)
            rr_rows.append({"query_id": qid, f"precision@{K}": p, f"recall@{K}": r, f"nDCG@{K}": n})
        rr_metrics = pd.DataFrame(rr_rows).set_index("query_id")

        # Side-by-side comparison (per query)
        compare = base_metrics.add_prefix("baseline_").join(rr_metrics.add_prefix("reranked_"))
        print("\nPer-query comparison:")
        print(compare.round(3))

        # Average comparison
        avg_base = base_metrics.mean().rename(lambda s: "baseline_" + s)
        avg_rr = rr_metrics.mean().rename(lambda s: "reranked_" + s)
        avg = pd.concat([avg_base, avg_rr], axis=0).to_frame("mean")
        print("\nAverage comparison:")
        print(avg.round(3))
