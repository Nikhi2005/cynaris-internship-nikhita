# W5D6 – LLM Cost Optimisation

**Cynaris Solutions Private Limited – AI/ML Internship**

## Overview

This practical explores techniques to optimise Large Language Model (LLM) costs through token counting, prompt compression, semantic caching, token usage tracking and model routing.

The implementation uses local Ollama models and a Redis-compatible service to demonstrate these techniques without making paid cloud API calls.

## Objectives

- Measure token counts for different prompt sizes.
- Compress RAG prompts using LLMLingua.
- Implement semantic caching using Redis and embeddings.
- Track token usage and estimate cloud-equivalent costs.
- Develop a local-versus-cloud model routing strategy.

## Technologies Used

- Python
- Ollama
- Llama 3.2 (3B)
- Qwen 2.5 (3B)
- Nomic Embed Text
- tiktoken
- LLMLingua
- Redis / Memurai
- NumPy
- ChromaDB
- Pandas
- OpenPyXL
- Jupyter Notebook

## Project Structure

```text
W5D6_Cost_Optimisation/
│
├── scripts/
│   └── main.py
│
├── notebooks/
│   └── semantic_cache.ipynb
│
├── data/
│
├── outputs/
│   ├── w5d6_results.json
│   ├── cost_log.csv
│   └── W5D6_Token_Cost_Audit.xlsx
│
├── requirements.txt
├── routing_strategy.md
└── README.md
```

## Task 1 – Token Counting

Used tiktoken to measure five different prompt sizes.

| Words | Characters | Tokens |
| ----: | ---------: | -----: |
|    25 |        247 |     38 |
|   100 |        812 |    131 |
|   250 |      1,965 |    317 |
|   500 |      3,876 |    627 |
| 1,000 |      7,699 |  1,248 |

The token counts use a GPT-4o-compatible tokenizer and are estimates for the local Ollama prompts.

## Task 2 – LLMLingua Prompt Compression

Applied LLMLingua-2 to a RAG prompt containing approximately 2,000 tokens.

| Metric            | Result |
| ----------------- | -----: |
| Original tokens   |  2,000 |
| Compressed tokens |    737 |
| Token reduction   | 63.15% |

The original and compressed answers were compared manually.

**Observation:** The compressed answer omitted the deployment or monitoring stage and introduced model selection. Therefore, token reduction was achieved, but factual coverage was not fully preserved.

The execution also displayed a model sequence-length warning (2,382 tokens compared with a stated maximum of 512). This limitation should be considered when interpreting the compression result.

## Task 3 – Redis Semantic Cache

Implemented semantic caching using:

- Ollama `nomic-embed-text` for embeddings.
- Redis for storing question-answer pairs and embeddings.
- NumPy for cosine similarity.
- Similarity threshold: 0.92.
- Cache expiration: 24 hours.

Twenty Q&A pairs were successfully stored.

### Test Results

| Test              | Similarity | Result     |
| ----------------- | ---------: | ---------- |
| Exact-match query |      1.000 | Cache hit  |
| Paraphrase query  |      0.913 | Cache miss |

The exact-match query successfully retrieved its stored answer.

The paraphrase query was below the configured 0.92 threshold and was correctly rejected as a cache hit.

## Task 4 – Token Cost Tracking

The implementation records token usage and generates:

- `cost_log.csv`
- `w5d6_results.json`
- `W5D6_Token_Cost_Audit.xlsx`

The Excel workbook contains the token-cost audit and baseline-versus-optimised estimated spending.

The cost comparison uses lesson pricing assumptions and an illustrative USD-to-INR conversion. These are cloud-equivalent estimates; local Ollama inference does not incur per-token cloud API charges.

## Task 5 – Model Routing

Implemented a basic rule-based routing demonstration.

| Query type                   | Selected model            |
| ---------------------------- | ------------------------- |
| Simple questions             | llama3.2:3b               |
| Medium-complexity questions  | qwen2.5:3b                |
| Complex analytical questions | Cloud API (proposed only) |

No paid cloud API calls were made.

The routing strategy is documented separately in `routing_strategy.md`.

## Installation

Activate or use the existing Cynaris virtual environment.

Install the dependencies:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
```

Ensure the following Ollama models are available:

```powershell
ollama list
```

Required models:

- `llama3.2:3b`
- `qwen2.5:3b`
- `nomic-embed-text:latest`

Ensure the Redis-compatible service is running on port 6379.

## Execution

Run the main practical script:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" ".\scripts\main.py"
```

Open the semantic cache notebook:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" -m jupyter notebook ".\notebooks\semantic_cache.ipynb"
```

Run the notebook cells from top to bottom.

## Key Observations

1. Token counting provides a baseline for estimating prompt usage.
2. LLMLingua reduced prompt tokens by 63.15%, but the answer comparison identified missing information.
3. Redis semantic caching successfully stored and retrieved 20 Q&A pairs.
4. Exact-match retrieval achieved a similarity score of 1.000.
5. The paraphrase query scored 0.913, below the configured threshold.
6. Local model routing can reduce dependence on paid cloud API calls for suitable tasks.

## Conclusion

This practical demonstrated important LLM cost-optimisation techniques, including token measurement, prompt compression, semantic caching, cost tracking and model routing.

The implementation successfully executed the five tasks and generated the required outputs. The observed compression-quality limitation and paraphrase cache miss were documented to ensure transparent evaluation.

---

**Cynaris Solutions Private Limited**  
**AI/ML Internship – Week 5, Day 6**
