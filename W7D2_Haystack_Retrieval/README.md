# W7D2: Haystack Retrieval — BM25 & Dense Retrieval Benchmark

## Task Overview
This project benchmarks two primary document retrieval strategies—**Sparse BM25 (Keyword-based)** and **Dense Vector Retrieval (Embedding-based using `all-MiniLM-L6-v2`)**—across indexed domain documents and custom evaluation questions.

---

## Folder Structure
```text
W7D2_Haystack_Retrieval/
├── data/
│   ├── doc1_crewai.txt
│   ├── doc2_langgraph.txt
│   ├── doc3_mlops.txt
│   ├── doc4_rag.txt
│   └── doc5_haystack.txt
├── outputs/
│   └── retrieval_comparison.csv
├── main.py
├── README.md
└── W7D2_Haystack_Retrieval_Report.md
```

---

## Methodology & Architecture

1. **Document Ingestion:** Reads text chunks across target files into memory indexers.
2. **BM25 Search Node:** Computes sparse term frequency match scores between questions and document text.
3. **Dense Search Node:** Uses `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` to embed queries and document vectors, calculating Cosine Similarity ($\cos(\theta)$).
4. **Benchmarking:** Executes 10 query variations, measuring document rankings and processing latencies.

---

## How to Run

```powershell
Set-Location "D:\Cynaris\Cynaris_Week7"
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week7\W7D2_Haystack_Retrieval\main.py"
```

---

## Benchmark Results Summary

- **BM25 Latency:** $\approx 0.03\text{--}0.05\text{ ms}$ (Extremely fast, but struggles when exact terms are absent).
- **Dense Retrieval Latency:** $\approx 6.6\text{--}12.3\text{ ms}$ (Higher computational cost, but captures semantic intent accurately).
- **Output Metrics File:** `outputs/retrieval_comparison.csv`