# CYNARIS SOLUTIONS PRIVATE LIMITED

## AI/ML INTERNSHIP PROGRAM – WEEK 7 CAPSTONE PROJECT REPORT

**Task Name:** W7D5: Week 7 Project: Multi-document RAG System  
**Intern Name:** Nikhita  
**Workspace:** `D:\Cynaris\Cynaris_Week7\W7D5_MultiDoc_RAG_Project`  
**Date:** November 22, 2026  

## Project Overview

Implemented a Multi-Document RAG system integrating the approved **AI/ML 3M Stack** (CrewAI, LangGraph, MLflow, Ragas, and MLOps concepts). The system indexes multiple domain-specific knowledge bases (Legal, Finance, MLOps), performs intelligent agentic query routing, evaluates context precision and faithfulness metrics (Ragas), and logs experiment tracking parameters and metrics (MLflow).

## Practical Objectives

1. Create 3 distinct domain-specific knowledge bases: Legal Compliance, Financial Operations, and MLOps Pipelines.
2. Build an Agentic Router to dynamically route incoming multi-document queries to the correct domain context.
3. Benchmark 10 queries using `sentence-transformers/all-MiniLM-L6-v2` embeddings to measure Context Precision, Faithfulness, and End-to-End Latency.
4. Export evaluation metrics to `outputs/multidoc_rag_metrics.csv` and summarize experiment results for MLOps tracking.

## Project Structure

```
W7D5_MultiDoc_RAG_Project/
├── data/
│   ├── finance/
│   │   └── financial_report.txt
│   ├── legal/
│   │   └── legal_compliance.txt
│   └── mlops/
│       └── mlops_handbook.txt
├── outputs/
│   └── multidoc_rag_metrics.csv
├── main.py
└── README.md
```

## Execution Results & MLOps Summary

| Q_ID | Expected Domain | Routed Domain | Routing Match | Context Precision | Faithfulness | Latency (ms) |
|---|---|---|---|---|---|---|
| Q1 | legal | legal | True | 0.7840 | 0.9598 | 173.31 |
| Q2 | finance | finance | True | 0.5469 | 0.9266 | 44.19 |
| Q3 | mlops | mlops | True | 0.7626 | 0.9568 | 128.53 |
| Q4 | legal | legal | True | 0.6229 | 0.9372 | 40.93 |
| Q5 | finance | finance | True | 0.6577 | 0.9421 | 143.72 |
| Q6 | mlops | mlops | True | 0.6502 | 0.9410 | 46.02 |
| Q7 | finance | finance | True | 0.5950 | 0.9333 | 157.57 |
| Q8 | legal | legal | True | 0.6486 | 0.9408 | 36.56 |
| Q9 | mlops | mlops | True | 0.6510 | 0.9411 | 155.10 |
| Q10 | legal | legal | True | 0.4245 | 0.9094 | 35.98 |

### Aggregated MLOps Experiment Metrics Logged:
- **Agentic Routing Accuracy:** 100.0%
- **Average Context Precision:** 0.6343
- **Average Faithfulness Score:** 0.9388
- **Average End-to-End Latency:** 96.19 ms

## How to Run

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week7\W7D5_MultiDoc_RAG_Project\main.py"
```

## Viva Questions & Answers

1. **What did you build today and why?**
   - Built an Agentic Multi-Document RAG system that dynamically routes user queries across domain indexes (Legal, Finance, MLOps), preventing cross-domain hallucinations and optimizing vector context retrieval.
2. **What was the hardest part and how did you solve it?**
   - Eliminating routing mismatch errors on boundary queries. Solved by calculating semantic similarity vectors between user queries and domain metadata embeddings before retrieving text chunks.
3. **What would you improve with one more day?**
   - Add a hybrid re-ranking node (BM25 + Dense RRF) and integrate live MLflow UI experiment logging for continuous tracking.