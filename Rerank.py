import argparse, hashlib, json, os, re, sys, time
from dataclasses import dataclass
from typing import Optional
import pandas as pd

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

REQUIRED = [
    "query_id","query_text","candidate_id","candidate_text",
    "baseline_rank","baseline_score","gold_label"
]

@dataclass
class LLMConfig:
    model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", 5))
    initial_backoff: float = float(os.getenv("LLM_BACKOFF", 1.2))

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def build_prompt(query_text: str, candidate_text: str) -> dict:
    system = "You are a careful relevance judge. Return ONLY a single number from 0 to 5 (decimals allowed)."
    user = (
        "Rate how relevant the candidate passage is to the query on a 0–5 scale (decimals allowed).\n"
        "Return ONLY the number, with no words.\n\n"
        f"Query:\n{query_text}\n\nCandidate:\n{candidate_text}\n"
    )
    return {"system": system, "user": user}

def parse_score(text: str) -> float:
    m = NUM_RE.search(text)
    if not m:
        raise ValueError(f"No numeric score found in: {text!r}")
    val = float(m.group(0))
    return max(0.0, min(5.0, val))

# OpenAI-compatible adapter; if not installed or no key, fallback returns 2.5
OPENAI_COMPAT = os.getenv("OPENAI_COMPAT", "1") == "1"
if OPENAI_COMPAT:
    try:
        from openai import OpenAI  # pip install openai
        _client = OpenAI()
    except Exception:
        _client = None
else:
    _client = None

def llm_call(prompt: dict, cfg: LLMConfig) -> str:
    if _client is None:
        return "2.5"  # offline dev fallback
    resp = _client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def score_with_retries(query_text: str, candidate_text: str, cfg: LLMConfig) -> float:
    prompt = build_prompt(query_text, candidate_text)
    backoff = cfg.initial_backoff
    last_err: Optional[Exception] = None
    for _ in range(cfg.max_retries):
        try:
            raw = llm_call(prompt, cfg)
            return parse_score(raw)
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"Failed to score after {cfg.max_retries} retries: {last_err}")

def row_key(qid: str, cid: str) -> str:
    return hashlib.md5(f"{qid}||{cid}".encode()).hexdigest()

def load_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(cache: dict, cache_path: str):
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="queries_candidates.csv")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--cache", default="scores_cache.json")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    args = ap.parse_args()

    cfg = LLMConfig(model=args.model)
    df = load_df(args.input)
    cache = load_cache(args.cache)

    scores = []
    for i, row in df.iterrows():
        qid = str(row["query_id"])
        cid = str(row["candidate_id"])
        k = row_key(qid, cid)
        if k in cache:
            s = cache[k]
        else:
            s = score_with_retries(row["query_text"], row["candidate_text"], cfg)
            cache[k] = s
            if i % 25 == 0:
                save_cache(cache, args.cache)
        scores.append(s)

    df_out = df.copy()
    df_out["llm_score"] = scores
    df_out.sort_values(["query_id","llm_score","baseline_rank"], ascending=[True,False,True], inplace=True)
    df_out["llm_rank"] = df_out.groupby("query_id").cumcount() + 1
    df_out.to_csv(args.out, index=False)
    save_cache(cache, args.cache)
    print(f"Wrote {args.out} with {len(df_out)} rows. Cache at {args.cache}.")

if __name__ == "__main__":
    main()
