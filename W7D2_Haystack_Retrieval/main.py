"""
W7D2: Haystack Retrieval — BM25 vs Dense Retrieval Pipeline
Company: Cynaris Solutions Private Limited
Author: AI/ML Intern
Description: Benchmark script comparing BM25 (keyword-based) vs. Dense Retrieval (semantic embeddings)
             across 5 text/PDF documents using 10 standard AI/ML query benchmarks.
"""

import os
import time
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. PREPARE SAMPLE DOCUMENTS
# ---------------------------------------------------------------------
sample_docs = {
    "doc1_crewai.txt": "CrewAI is a framework for orchestrating autonomous AI agents. It enables multi-agent collaboration for complex task automation and reasoning.",
    "doc2_langgraph.txt": "LangGraph is built for constructing stateful multi-agent workflows with loops and conditional logic in LLM applications.",
    "doc3_mlops.txt": "MLOps focuses on continuous integration, deployment, model monitoring, data drift detection, and automated retraining pipelines.",
    "doc4_rag.txt": "Retrieval Augmented Generation combines vector retrieval with generative models to answer user questions grounded in private documents.",
    "doc5_haystack.txt": "Haystack is an open-source NLP framework by Deepset for building search systems, QA pipelines, BM25 indexing, and vector dense retrieval."
}

def create_sample_docs():
    for name, content in sample_docs.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
    print(f"[+] Verified {len(sample_docs)} documents in {DATA_DIR}")

create_sample_docs()

documents = []
doc_names = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt") or file.endswith(".pdf"):
        doc_names.append(file)
        path = os.path.join(DATA_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"name": file, "text": text})

# ---------------------------------------------------------------------
# 2. BENCHMARK QUESTIONS & EMBEDDING MODEL
# ---------------------------------------------------------------------
benchmark_questions = [
    "What is the purpose of the CrewAI library?",
    "Explain the role of LangGraph in an LLM workflow.",
    "How does MLOps differ from traditional ML deployment?",
    "What are the main components of a RAG system?",
    "Give an example of a data drift detection technique.",
    "How to fine-tune a transformer for sentiment analysis?",
    "What is the difference between recall and precision?",
    "Describe how vector search works in Haystack.",
    "What does autonomous AI agent orchestration mean?",
    "List advantages of using dense vector retrieval."
]

print("[+] Initializing SentenceTransformer model ('all-MiniLM-L6-v2')...")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = embedder.encode([d["text"] for d in documents], convert_to_tensor=True)

# ---------------------------------------------------------------------
# 3. BM25 (SPARSE) VS DENSE RETRIEVAL PIPELINE
# ---------------------------------------------------------------------
results = []

print("[+] Running BM25 vs Dense Retrieval Benchmark...")

for idx, q in enumerate(benchmark_questions, 1):
    # BM25 Keyword Search Simulation
    start_bm25 = time.time()
    q_words = set(q.lower().split())
    bm25_scores = []
    for doc in documents:
        doc_words = set(doc["text"].lower().split())
        overlap = len(q_words.intersection(doc_words))
        bm25_scores.append((doc["name"], overlap))
    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    lat_bm25 = (time.time() - start_bm25) * 1000

    # Dense Vector Search (Cosine Similarity)
    start_dense = time.time()
    q_emb = embedder.encode(q, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, doc_embeddings)[0]
    dense_ranked = sorted(zip(doc_names, sims.tolist()), key=lambda x: x[1], reverse=True)
    lat_dense = (time.time() - start_dense) * 1000

    top_bm25_doc, bm25_score = bm25_scores[0]
    top_dense_doc, dense_score = dense_ranked[0]

    results.append({
        "Q_ID": f"Q{idx}",
        "Question": q,
        "BM25_Top_Doc": top_bm25_doc,
        "BM25_Score": round(float(bm25_score), 2),
        "BM25_Latency_ms": round(lat_bm25, 2),
        "Dense_Top_Doc": top_dense_doc,
        "Dense_Score": round(float(dense_score), 4),
        "Dense_Latency_ms": round(lat_dense, 2)
    })

# ---------------------------------------------------------------------
# 4. EXPORT METRICS & DISPLAY SUMMARY
# ---------------------------------------------------------------------
df = pd.DataFrame(results)
csv_out = os.path.join(OUTPUT_DIR, "retrieval_comparison.csv")
df.to_csv(csv_out, index=False)

print("\n" + "="*80)
print("   HAYSTACK RETRIEVAL BENCHMARK SUMMARY (BM25 vs DENSE)")
print("="*80)
print(df[["Q_ID", "BM25_Top_Doc", "BM25_Latency_ms", "Dense_Top_Doc", "Dense_Score", "Dense_Latency_ms"]].to_string(index=False))
print("="*80)
print(f"[+] Benchmark metrics saved to: {csv_out}")