# CYNARIS SOLUTIONS PRIVATE LIMITED
## AI/ML INTERNSHIP PROGRAM – WEEK 7 DAY 3 REPORT

**Task Name:** W7D3: LlamaIndex — Document Indexing & Querying  
**Intern Name:** Nikhita  
**Workspace:** `D:\Cynaris\Cynaris_Week7\W7D3_LlamaIndex_Indexing`  
**Date:** November 22, 2026  

---

## Task Overview
Implemented a LlamaIndex-based Retrieval-Augmented Generation (RAG) pipeline to index text documents, construct a `VectorStoreIndex`, and execute 10 benchmark queries. Evaluated performance, accuracy, and execution latency by comparing transient **In-Memory Indexing** against **ChromaDB Vector Storage**.

---

## Practical Objectives
1. Index a set of 5 technical documents into semantic nodes.
2. Build a `QueryEngine` powered by dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
3. Benchmark 10 query questions across both vector storage backends.
4. Export evaluation metrics to `outputs/llamaindex_benchmark.csv`.

---

## Project Structure
```
W7D3_LlamaIndex_Indexing/
├── data/
│   ├── chromadb_vectorstore.txt
│   ├── crewai_guide.txt
│   ├── langgraph_workflow.txt
│   ├── mlops_pipeline.txt
│   └── ragas_evaluation.txt
├── outputs/
│   └── llamaindex_benchmark.csv
├── main.py
└── README.md
```

---

## Execution Results

| Q_ID | Memory Top Doc | Memory Lat (ms) | Chroma Top Doc | Chroma Score | Chroma Lat (ms) |
|---|---|---|---|---|---|
| Q1 | crewai_guide.txt | 4.21 | crewai_guide.txt | 0.6271 | 0.40 |
| Q2 | langgraph_workflow.txt | 0.25 | langgraph_workflow.txt | 0.4914 | 0.34 |
| Q3 | mlops_pipeline.txt | 0.25 | mlops_pipeline.txt | 0.4916 | 0.33 |
| Q4 | ragas_evaluation.txt | 0.25 | ragas_evaluation.txt | 0.6247 | 0.32 |
| Q5 | mlops_pipeline.txt | 0.23 | mlops_pipeline.txt | 0.4728 | 0.33 |
| Q6 | langgraph_workflow.txt | 0.24 | langgraph_workflow.txt | 0.2793 | 0.33 |
| Q7 | langgraph_workflow.txt | 0.24 | langgraph_workflow.txt | 0.2752 | 0.34 |
| Q8 | chromadb_vectorstore.txt | 0.24 | chromadb_vectorstore.txt | 0.6466 | 0.33 |
| Q9 | ragas_evaluation.txt | 0.27 | ragas_evaluation.txt | 0.1075 | 0.34 |
| Q10 | langgraph_workflow.txt | 0.24 | langgraph_workflow.txt | 0.2129 | 0.33 |

---

## How to Run
```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week7\W7D3_LlamaIndex_Indexing\main.py"
```

---

## Key Insights
- Both vector storage mechanisms retrieved identical top-matching documents across all 10 queries.
- ChromaDB provides persistent vector indexing on disk, avoiding embedding recalculations upon application restart.
- Execution latencies remained consistently low (~0.33 ms) across both vector store strategies.