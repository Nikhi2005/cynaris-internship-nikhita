# W8D1: Production RAG: Evaluation & Optimisation (MLOps Pipeline)

## 📌 Project Overview

This module establishes a production-grade MLOps setup for a FastAPI-based Retrieval-Augmented Generation (RAG) service. It includes Docker multi-stage containerization, a full CI/CD pipeline definition using GitHub Actions, and a comprehensive production monitoring strategy.

---

## ⚙️ Key Technical Features

1. **Multi-Stage Dockerfile:** Implements builder and runtime separation using `python:3.11-slim` with dependency layer caching and non-root user execution.
2. **GitHub Actions CI/CD Pipeline:** Configured in `.github/workflows/ci.yml` for automated linting (`flake8`), code formatting (`black`), unit testing (`pytest`), and Docker image build/push to GitHub Container Registry (`ghcr.io`).
3. **MLOps Monitoring Strategy:** Documented framework (`monitoring_strategy.md`) tracking real-time latency, embedding drift, Ragas faithfulness scores, and automated retraining triggers.

---

## 📊 Benchmark Metrics Summary

| Test ID | Query Target                | Matched Document      | Confidence | Faithfulness | Latency (ms) |
| :------ | :-------------------------- | :-------------------- | :--------- | :----------- | :----------- |
| **T1**  | Docker ML Containerization  | `doc1_docker.txt`     | 0.7615     | 1.0000       | 17.18        |
| **T2**  | GitHub Actions CI Pipeline  | `doc2_cicd.txt`       | 0.6936     | 0.9436       | 16.99        |
| **T3**  | Production MLOps Monitoring | `doc3_monitoring.txt` | 0.7401     | 0.9901       | 16.58        |
| **T4**  | Multi-Stage Docker Build    | `doc1_docker.txt`     | 0.6055     | 0.8555       | 17.52        |

---

## 🛠 Execution Instructions

Run the main script using the repository virtual environment:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week8\W8D1_Production_RAG_MLOps\main.py"
```
