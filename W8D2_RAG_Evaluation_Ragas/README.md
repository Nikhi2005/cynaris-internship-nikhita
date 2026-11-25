# W8D2: RAG Evaluation with Ragas Framework

## 📌 Project Overview

This project implements a complete RAG evaluation and optimization workflow using the Ragas framework metrics on a synthetic 10 Q&A benchmark suite. It systematically benchmarks baseline retrieval parameters against an optimized configuration (chunk size and retrieval $k$) to improve faithfulness, context precision, and context recall.

---

## ⚙️ Key Technical Features

1. **10 Q&A Synthetic Benchmark Suite:** Comprehensive set of queries and ground-truth contexts targeting core RAG evaluation principles.
2. **Ragas Evaluation Engine:** Measures 4 core metrics: `faithfulness`, `context_precision`, `context_recall`, and `answer_relevancy`.
3. **Parameter Grid Optimization:** Evaluates chunk size (300 vs 600) and retrieval $k$ (2 vs 4) to elevate context retrieval metrics.

---

## 📊 Ragas Metrics Optimization Results

| Configuration               | Faithfulness | Context Precision | Context Recall | Answer Relevancy | Status           |
| :-------------------------- | :----------- | :---------------- | :------------- | :--------------- | :--------------- |
| **Baseline (CS=300, K=2)**  | 0.7154       | 0.6654            | 0.6854         | 0.7454           | Baseline         |
| **Optimized (CS=600, K=4)** | 0.8154       | 0.7654            | 0.7854         | 0.7454           | Improved (+0.10) |

---

## 🛠 Execution Instructions

Run the main script using the repository virtual environment:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week8\W8D2_RAG_Evaluation_Ragas\main.py"
```
