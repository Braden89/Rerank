import pandas as pd

REQUIRED = [
    "query_id","query_text","candidate_id","candidate_text",
    "baseline_rank","baseline_score","gold_label"
]

def load_and_validate(path="queries_candidates.csv"):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # enforce dtypes
    df["query_id"] = df["query_id"].astype(str)
    df["candidate_id"] = df["candidate_id"].astype(str)
    df["baseline_rank"] = pd.to_numeric(df["baseline_rank"], errors="coerce")
    df["baseline_score"] = pd.to_numeric(df["baseline_score"], errors="coerce")
    df["gold_label"] = pd.to_numeric(df["gold_label"], errors="coerce").fillna(0).astype(int)
    return df

def quick_eda(df: pd.DataFrame):
    n_rows = len(df)
    n_queries = df["query_id"].nunique()
    per_q = df.groupby("query_id")["candidate_id"].nunique()
    avg_cands = per_q.mean()
    rel_ratio = df["gold_label"].mean()
    dup_rows = df.duplicated(subset=["query_id","candidate_id"]).sum()
    return {
        "rows": n_rows,
        "queries": n_queries,
        "avg_candidates_per_query": float(avg_cands),
        "relevant_ratio": float(rel_ratio),
        "duplicate_qid_cid_pairs": int(dup_rows),
        "null_counts": df.isna().sum().to_dict()
    }

if __name__ == "__main__":
    df = load_and_validate()
    eda = quick_eda(df)
    print(eda)
