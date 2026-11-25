# W8D5: 2M Capstone — Local AI Research Assistant

## 📌 Project Overview

This Capstone project implements an autonomous **Local AI Research Assistant** integrating the full AI/ML 3M stack: **CrewAI** multi-agent task execution, **LangGraph** stateful workflow management, **Ragas** quality evaluation, and **MLflow** experiment tracking.

---

## ⚙️ Key Technical Features

1. **Multi-Agent Orchestration:** `ResearchAgent` (source retrieval) $\rightarrow$ `SynthesizerAgent` (grounded answer creation) $\rightarrow$ `QualityEvaluator` (Ragas metrics scoring).
2. **Ragas Quality Metrics:** Real-time scoring of `faithfulness`, `search_precision`, `citation_recall`, and `overall_quality`.
3. **MLOps & MLflow Tracking:** Automatic logging of search latencies, overall quality scores, and run parameters to `outputs/mlflow_summary.json`.

---

## 📊 Benchmark Execution Summary

| Run ID     | Query Focus                        | Matched Source       | Search Precision | Faithfulness | Overall Quality | Latency (ms) |
| :--------- | :--------------------------------- | :------------------- | :--------------- | :----------- | :-------------- | :----------- |
| **RUN_01** | CrewAI Multi-Agent Task Delegation | `doc1_crewai.txt`    | 0.8248           | 1.0000       | 0.7474          | 21.25        |
| **RUN_02** | LangGraph Stateful Workflows       | `doc2_langgraph.txt` | 0.6595           | 0.8795       | 0.6376          | 22.37        |
| **RUN_03** | MLflow MLOps Tracking              | `doc3_mlflow.txt`    | 0.8002           | 1.0000       | 0.7401          | 21.71        |
| **RUN_04** | Ragas Evaluation Metrics           | `doc4_ragas.txt`     | 0.6684           | 0.8884       | 0.6447          | 21.37        |

---

## 🛠 Execution Instructions

Run the main script using the repository virtual environment:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week8\W8D5_Local_AI_Research_Assistant\main.py"
```
