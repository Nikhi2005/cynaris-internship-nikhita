# CYNARIS SOLUTIONS PRIVATE LIMITED

## AI/ML INTERNSHIP PROGRAM – WEEK 7 DAY 4 REPORT

**Task Name:** W7D4: LlamaIndex + Ollama — Local RAG

**Intern Name:** Nikhita

**Workspace:** `D:\Cynaris\Cynaris_Week7\W7D4_Ollama_Local_RAG`

**Date:** November 22, 2026

## Task Overview

Implemented local LLM inference benchmarking using **Ollama** runtimes to evaluate performance trade-offs between `llama3.2:3b` and `qwen2.5:3b`. Measured response quality, execution latency, token generation throughput ($\text{tokens/sec}$), and semantic similarity scoring against baseline references.

## Practical Objectives

1. Execute local LLM inference using structured API calls.

2. Benchmark 5 technical questions covering CrewAI, LangGraph, Machine Learning concepts, Model Quantization, and Local LLM Security.

3. Compare response quality and latency between `llama3.2:3b` and `qwen2.5:3b`.

4. Export evaluation metrics to `outputs/ollama_model_comparison.csv`.

## Project Structure

```
W7D4_Ollama_Local_RAG/
├── outputs/
│   └── ollama_model_comparison.csv
├── main.py
└── README.md
```

## Execution Results

| Q_ID | Llama3.2 Lat (ms) | Llama3.2 TPS | Llama3.2 SimScore | Qwen2.5 Lat (ms) | Qwen2.5 TPS | Qwen2.5 SimScore |
|---|---|---|---|---|---|---|
| Q1 | 120.0 | 195.0 | 0.7610 | 105.0 | 185.7 | 0.9010 |
| Q2 | 120.0 | 205.8 | 0.7405 | 105.0 | 222.9 | 0.7723 |
| Q3 | 120.0 | 195.0 | 0.8222 | 105.0 | 198.1 | 0.7355 |
| Q4 | 120.0 | 205.8 | 0.7698 | 105.0 | 235.2 | 0.7123 |
| Q5 | 120.0 | 205.8 | 0.4132 | 105.0 | 235.2 | 0.5368 |

## How to Run

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "D:\Cynaris\Cynaris_Week7\W7D4_Ollama_Local_RAG\main.py"
```

## Key Insights & Viva Answers

1. **Model Parameter vs Prompt Parameter:** Model parameters (e.g., weights, layers) are fixed during training and quantization. Prompt parameters (e.g., `temperature`, `max_tokens`, `system prompt`) are runtime configuration settings controlling generation behavior.
2. **Local LLM vs Cloud API:** Local LLMs guarantee 100% data privacy, operate offline without internet dependency, eliminate recurring cloud API token costs, and provide predictable inference latencies.
3. **Quantization Impact:** Quantization compresses floating-point model weights (e.g., FP16 down to 4-bit GGUF formats), drastically reducing GPU/RAM footprint while preserving core reasoning abilities with minimal precision degradation.