#  LLM-Based Semantic Re-Ranking

This project uses a Large Language Model (LLM) to re-score search results and improve retrieval quality over a baseline embedding ranker.  
It implements end-to-end evaluation with Gemini (via the OpenAI-compatible API) and supports few-shot prompting for comparison.

---

## Project Files
| File | Description |
|------|--------------|
| `api_starter.py` | Connects to the Gemini API, sends (query, candidate) pairs, and returns integer relevance scores (0–5). |
| `ranking_starter.py` | Computes baseline and LLM reranked metrics (`Precision@K`, `Recall@K`, `nDCG@K`); supports multiple modes through environment variables. |
| `rag_sample_queries_candidates.csv` | Input dataset of queries, candidate documents, and gold relevance labels. |
| `reranking_report.pdf` | Summary analysis of prompt design, results, and findings. |

---

## Requirements
- **Python 3.10+**
- Required libraries:
  ```bash
  pip install openai pandas python-dotenv
  ```
- (Optional for report generation)  
  ```bash
  pip install reportlab
  ```

---

## Environment Setup
Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

These environment variables allow the code to call Gemini using an OpenAI-style client.

---

## Running the Program (PowerShell)

All steps are run through `ranking_starter.py` using **environment variables** to choose the mode.

## Step 1 — Baseline Evaluation
```powershell
python .\ranking_starter.py
```
Prints baseline `Precision@3`, `Recall@3`, and `nDCG@3` metrics using the embedding-based scores in the CSV.

---

## Step 2 — Compute LLM Scores
```powershell
$env:RERANK_COMPUTE_SCORES = "1"
python .\ranking_starter.py
```
Queries the Gemini model for each `(query, candidate)` pair and writes:
```
rag_with_llm_scores.csv
```

> **Tip:** This step may take several minutes since it calls the API once per candidate.

---

## Step 3 — Re-Rank and Evaluate
```powershell
$env:RERANK_EVAL = "1"
python .\ranking_starter.py
```
Loads the file from Step 2, re-sorts candidates within each query by `llm_score`, evaluates the reranked metrics, and writes:
```
rag_reranked.csv
```

---

## Step 4 — Compare Baseline vs Reranked
```powershell
$env:RERANK_COMPARE = "1"
python .\ranking_starter.py
```
Prints side-by-side metrics for the baseline and LLM reranked results.

---

## Step 5 — Few-Shot Prompting Mode (Optional)
To test improved few-shot prompts:

```powershell
# Step 2: Compute few-shot scores
$env:RERANK_FEWSHOT = "1"
$env:RERANK_COMPUTE_SCORES = "1"
python .\ranking_starter.py

# Step 3: Re-rank few-shot results
$env:RERANK_FEWSHOT = "1"
$env:RERANK_EVAL = "1"
python .\ranking_starter.py

# Step 4: Compare few-shot vs baseline
$env:RERANK_FEWSHOT = "1"
$env:RERANK_COMPARE = "1"
python .\ranking_starter.py
```

Outputs:
- `rag_with_llm_scores_fewshot.csv`
- `rag_reranked_fewshot.csv`

---

## Outputs
| File | Description |
|------|--------------|
| `rag_with_llm_scores.csv` | Original LLM scores per query/candidate |
| `rag_with_llm_scores_fewshot.csv` | Few-shot variant scores |
| `rag_reranked.csv` | Re-ordered candidates by LLM score |
| `rag_reranked_fewshot.csv` | Few-shot re-ordered candidates |
| `reranking_report.pdf` | Final report summarizing findings |

---

## Notes
- If you encounter `FileNotFoundError: /mnt/data/...`, update paths to relative form (`rag_sample_queries_candidates.csv`) as shown.  
- Average latency per call: ~1–2 s. Consider caching responses for larger datasets.
- For Linux/macOS users, replace PowerShell syntax with:
  ```bash
  RERANK_COMPUTE_SCORES=1 python ranking_starter.py
  ```

---

## Author
**Braden Barnes**  
Utah Tech University — CS 4600 Senior Project  
*“LLM-Based Semantic Re-Ranking”*
