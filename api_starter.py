#!/usr/bin/env python3

"""
https://aistudio.google.com/ "Get API key"
https://aistudio.google.com/apikey
Put key in .env file with format:
GEMINI_API_KEY="AIzaSyCaYJAi_J7lLskgI8Bo4mEVCZAnq33sbL8"
Use OpenAIServerModel due to API compatibility
[Select model](https://ai.google.dev/gemini-api/docs/models)
"""

#############################################
# Environment loading
#############################################
import os
import re
import time
from typing import Optional
try:
    import dotenv  # type: ignore
    dotenv.load_dotenv()
except Exception:
    pass

from openai import OpenAI

_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_FEWSHOT = os.getenv("RERANK_FEWSHOT") == "1"

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not _API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY (or OPENAI_API_KEY). Put it in .env or environment.")
        _client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)
    return _client

_SYSTEM = (
    "You are a careful, deterministic relevance judge for retrieval tasks. "
    "Given a user query and a candidate passage, output ONLY a single integer 0,1,2,3,4, or 5. "
    "Meanings: 0=Irrelevant, 5=Highly relevant. Respond with digits only."
)

def score_pair(query: str, candidate: str, retries: int = 2) -> int:
    client = _get_client()

    # Optional few-shot exemplars (enabled with RERANK_FEWSHOT=1)
    examples = ""
    if _FEWSHOT:
        examples = (
            "Examples (format: Query / Candidate -> Score)\n"
            "Q: what is python typing\n"
            "C: Python typing refers to type hints like List[int] and Optional[str]. -> 4\n"
            "Q: who won the 2018 world cup\n"
            "C: France won the FIFA World Cup in 2018. -> 5\n\n"
        )

    prompt = (
        examples +
        "Query:\n" + query.strip() + "\n\n"
        "Candidate:\n" + candidate.strip() + "\n\n"
        "Only output a single integer from 0 to 5, nothing else."
    )

    attempt = 0
    while True:
        attempt += 1
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=4,
            )
            latency = time.time() - start  # optional for logging
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"[0-5]", text)
            if m:
                return int(m.group(0))
        except Exception:
            pass
        if attempt > retries:
            return 0   

__all__ = ["score_pair"]
if __name__ == "__main__":
    import os
    if os.getenv("RERANK_COMPUTE_SCORES") == "1":
        import pandas as pd
        from api_starter import score_pair
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

