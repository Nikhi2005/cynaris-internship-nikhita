# W8D3: Deployment — Haystack API on Railway

## 📌 Project Overview

This project builds and benchmarks an end-to-end Haystack retrieval pipeline comparing **BM25 (Sparse Keyword Search)** against **DenseRetriever (Embedding-Based Vector Search utilizing `sentence-transformers/all-MiniLM-L6-v2`)**. The pipeline evaluates precision, semantic recall, context relevance, and execution latency across 10 technical query benchmarks.

---

## ⚙️ Key Technical Features

1. **Haystack Retrieval Architecture:** Structured node pipeline (`FileConverter` $\rightarrow$ `DocumentStore` $\rightarrow$ `BM25Retriever` / `DenseRetriever` $\rightarrow$ `Reader`).
2. **Comparative Benchmark Suite:** 10 domain questions evaluated across sparse keyword match vs. 384-dimensional dense cosine vector similarity.
3. **MLOps Metrics Export:** Performance results saved to `outputs/haystack_retrieval_comparison.csv`.

---

## 📊 Benchmark Execution Summary

| Q_ID    | BM25 Top Doc         | BM25 Lat (ms) | Dense Top Doc        | Dense Score | Dense Lat (ms) | Precision Match |
| :------ | :------------------- | :------------ | :------------------- | :---------- | :------------- | :-------------- |
| **Q1**  | `doc1_crewai.txt`    | 0.04          | `doc1_crewai.txt`    | 0.4000      | 14.16          | Exact Match     |
| **Q2**  | `doc2_langgraph.txt` | 0.04          | `doc2_langgraph.txt` | 0.5292      | 10.70          | Exact Match     |
| **Q3**  | `doc3_mlops.txt`     | 0.03          | `doc3_mlops.txt`     | 0.6427      | 11.05          | Exact Match     |
| **Q4**  | `doc1_crewai.txt`    | 0.03          | `doc2_langgraph.txt` | 0.0582      | 10.07          | Dense Superior  |
| **Q5**  | `doc3_mlops.txt`     | 0.03          | `doc3_mlops.txt`     | 0.4172      | 7.54           | Exact Match     |
| **Q6**  | `doc4_rag.txt`       | 0.02          | `doc5_haystack.txt`  | 0.2040      | 7.07           | Dense Superior  |
| **Q7**  | `doc5_haystack.txt`  | 0.03          | `doc4_rag.txt`       | 0.1758      | 6.53           | Dense Superior  |
| **Q8**  | `doc5_haystack.txt`  | 0.03          | `doc5_haystack.txt`  | 0.5808      | 6.77           | Exact Match     |
| **Q9**  | `doc1_crewai.txt`    | 0.02          | `doc1_crewai.txt`    | 0.6328      | 6.47           | Exact Match     |
| **Q10** | `doc5_haystack.txt`  | 0.02          | `doc5_haystack.txt`  | 0.5251      | 6.47           | Exact Match     |

---

## 🛠 Execution Instructions

Run the main script using the repository virtual environment:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week8\W8D3_Haystack_Pipeline\main.py"
```
