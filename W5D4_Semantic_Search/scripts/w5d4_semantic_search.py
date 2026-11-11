
"""
Cynaris Solutions - Week 5 Day 4
Semantic Search with ChromaDB

Features:
1. Store 20 sample documents using Ollama embeddings.
2. Perform cosine similarity search.
3. Compare filtered and unfiltered retrieval.
4. Extract and chunk a PDF.
5. Retrieve the top 3 PDF chunks.
6. Generate a grounded answer using Ollama.
7. Evaluate known queries and perform sanity checks.
8. Save logs and JSON results.

Run:
python scripts/w5d4_semantic_search.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import requests
from pypdf import PdfReader


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chromadb_store"
LOG_DIR = BASE_DIR / "logs"
RESULTS_FILE = LOG_DIR / "results.json"

PDF_PATH = DATA_DIR / "Machine_Learning_Notes.pdf"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

SAMPLE_COLLECTION_NAME = "w5d4_semantic_documents"
PDF_COLLECTION_NAME = "w5d4_semantic_pdf_chunks"

BATCH_SIZE = 16
TOP_K = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3

LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING
# ============================================================

LOG_FILE = LOG_DIR / (
    f"w5d4_{datetime.now():%Y%m%d_%H%M%S}.log"
)

logger = logging.getLogger("W5D4")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = logging.FileHandler(
    LOG_FILE,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ============================================================
# 3. SAMPLE DOCUMENTS - 20 DOCUMENTS
# ============================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "ml_01",
        "text": "Machine learning is a branch of artificial intelligence "
                "that enables computers to learn patterns from data.",
        "metadata": {"category": "machine_learning", "topic": "definition"}
    },
    {
        "id": "ml_02",
        "text": "Supervised learning trains a model using labelled "
                "training examples containing inputs and expected outputs.",
        "metadata": {"category": "machine_learning", "topic": "supervised"}
    },
    {
        "id": "ml_03",
        "text": "Unsupervised learning discovers hidden patterns and "
                "structures in data without labelled outputs.",
        "metadata": {"category": "machine_learning", "topic": "unsupervised"}
    },
    {
        "id": "ml_04",
        "text": "Reinforcement learning trains an agent through rewards "
                "and penalties received while interacting with an environment.",
        "metadata": {"category": "machine_learning", "topic": "reinforcement"}
    },
    {
        "id": "ml_05",
        "text": "Classification predicts discrete categories such as "
                "spam or not spam using input features.",
        "metadata": {"category": "machine_learning", "topic": "classification"}
    },
    {
        "id": "ml_06",
        "text": "Regression predicts continuous numerical values such as "
                "house prices or temperature.",
        "metadata": {"category": "machine_learning", "topic": "regression"}
    },
    {
        "id": "ml_07",
        "text": "Overfitting occurs when a model learns training data "
                "too closely and performs poorly on unseen data.",
        "metadata": {"category": "machine_learning", "topic": "overfitting"}
    },
    {
        "id": "ml_08",
        "text": "Cross-validation evaluates model performance by "
                "splitting data into multiple training and validation folds.",
        "metadata": {"category": "machine_learning", "topic": "evaluation"}
    },
    {
        "id": "dl_01",
        "text": "Deep learning uses neural networks with multiple layers "
                "to learn complex patterns from large datasets.",
        "metadata": {"category": "deep_learning", "topic": "definition"}
    },
    {
        "id": "dl_02",
        "text": "A neural network contains interconnected artificial "
                "neurons organised into input, hidden and output layers.",
        "metadata": {"category": "deep_learning", "topic": "neural_network"}
    },
    {
        "id": "dl_03",
        "text": "Convolutional neural networks are commonly used for "
                "image classification and computer vision.",
        "metadata": {"category": "deep_learning", "topic": "cnn"}
    },
    {
        "id": "dl_04",
        "text": "Recurrent neural networks process sequential information "
                "such as text, speech and time-series data.",
        "metadata": {"category": "deep_learning", "topic": "rnn"}
    },
    {
        "id": "dl_05",
        "text": "Transformers use attention mechanisms to process "
                "relationships between elements in a sequence.",
        "metadata": {"category": "deep_learning", "topic": "transformer"}
    },
    {
        "id": "dl_06",
        "text": "Backpropagation calculates gradients used to update "
                "neural network weights during training.",
        "metadata": {"category": "deep_learning", "topic": "training"}
    },
    {
        "id": "dl_07",
        "text": "An activation function introduces non-linearity into "
                "a neural network so it can learn complex relationships.",
        "metadata": {"category": "deep_learning", "topic": "activation"}
    },
    {
        "id": "py_01",
        "text": "Python is a high-level programming language known for "
                "readable syntax and a large ecosystem of libraries.",
        "metadata": {"category": "programming", "topic": "python"}
    },
    {
        "id": "py_02",
        "text": "A Python list is an ordered and mutable collection "
                "that can store multiple items.",
        "metadata": {"category": "programming", "topic": "list"}
    },
    {
        "id": "py_03",
        "text": "A Python dictionary stores data as key-value pairs "
                "and supports lookup using keys.",
        "metadata": {"category": "programming", "topic": "dictionary"}
    },
    {
        "id": "py_04",
        "text": "A function is a reusable block of code that performs "
                "a specific task and can accept parameters.",
        "metadata": {"category": "programming", "topic": "function"}
    },
    {
        "id": "py_05",
        "text": "Exception handling in Python uses try, except and "
                "finally blocks to manage runtime errors.",
        "metadata": {"category": "programming", "topic": "exceptions"}
    },
]


# ============================================================
# 4. OLLAMA CONNECTION AND MODEL CHECK
# ============================================================

def check_ollama() -> None:
    """Verify that Ollama and the required models are available."""

    try:
        response = requests.get(
            TAGS_URL,
            timeout=10
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Check that Ollama is running."
        ) from exc

    model_names = {
        item["name"].split(":")[0]
        for item in response.json().get("models", [])
    }

    required_models = {
        EMBED_MODEL.split(":")[0],
        LLM_MODEL.split(":")[0]
    }

    missing = required_models - model_names

    if missing:
        raise RuntimeError(
            f"Required Ollama models are missing: {sorted(missing)}"
        )

    logger.info("Ollama server is running.")
    logger.info("Embedding model verified: %s", EMBED_MODEL)
    logger.info("LLM verified: %s", LLM_MODEL)


# ============================================================
# 5. OLLAMA EMBEDDING FUNCTION WITH RETRIES
# ============================================================

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using Ollama /api/embed.

    Processes inputs in batches and retries temporary failures.
    """

    if not texts:
        return []

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        payload = {
            "model": EMBED_MODEL,
            "input": batch,
            "truncate": True
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    EMBED_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()

                embeddings = response.json().get("embeddings")

                if not embeddings or len(embeddings) != len(batch):
                    raise ValueError(
                        "Ollama returned an unexpected embedding count."
                    )

                for vector in embeddings:
                    if not vector or not all(
                        isinstance(value, (int, float))
                        for value in vector
                    ):
                        raise ValueError(
                            "Invalid embedding vector returned by Ollama."
                        )

                all_embeddings.extend(embeddings)

                logger.info(
                    "Generated embeddings for batch %d (%d texts).",
                    start // BATCH_SIZE + 1,
                    len(batch)
                )
                break

            except (
                requests.RequestException,
                ValueError
            ) as exc:

                logger.warning(
                    "Embedding attempt %d/%d failed: %s",
                    attempt,
                    MAX_RETRIES,
                    exc
                )

                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        "Embedding generation failed after retries."
                    ) from exc

                time.sleep(2 ** (attempt - 1))

    dimensions = {len(vector) for vector in all_embeddings}

    if len(dimensions) != 1:
        raise ValueError(
            "Embedding vectors have inconsistent dimensions."
        )

    return all_embeddings


# ============================================================
# 6. CHROMADB COLLECTIONS
# ============================================================

client = chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(name: str):
    """Create or retrieve a persistent cosine-similarity collection."""

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info(
        "Collection ready: %s | Existing count: %d",
        name,
        collection.count()
    )

    return collection


sample_collection = get_collection(SAMPLE_COLLECTION_NAME)
pdf_collection = get_collection(PDF_COLLECTION_NAME)


# ============================================================
# 7. DOCUMENT INSERTION USING UPSERT
# ============================================================

def upsert_documents(
    collection: Any,
    documents: list[dict[str, Any]]
) -> None:
    """Insert or update documents without creating duplicate IDs."""

    if not documents:
        raise ValueError("No documents supplied for insertion.")

    ids = [item["id"] for item in documents]
    texts = [item["text"] for item in documents]
    metadatas = [item["metadata"] for item in documents]

    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique.")

    embeddings = generate_embeddings(texts)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    logger.info(
        "Upserted %d documents into %s.",
        len(documents),
        collection.name
    )


# ============================================================
# 8. SEMANTIC SEARCH
# ============================================================

def semantic_search(
    collection: Any,
    query: str,
    k: int = TOP_K,
    filter_meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Retrieve the top-k documents using cosine distance."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    available = collection.count()

    if available == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    query_embedding = generate_embeddings([query])[0]

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": min(k, available),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if filter_meta:
        query_args["where"] = filter_meta

    return collection.query(**query_args)


# ============================================================
# 9. PRINT AND ANALYSE SEARCH RESULTS
# ============================================================

def display_results(
    title: str,
    results: dict[str, Any]
) -> list[dict[str, Any]]:

    records: list[dict[str, Any]] = []

    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("No matching documents found.")
        return records

    for rank, (doc_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1
    ):
        similarity = 1 - distance

        print(f"\nRank: {rank}")
        print(f"ID: {doc_id}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Derived similarity: {similarity:.4f}")
        print(f"Metadata: {metadata}")
        print(f"Document: {document}")

        records.append({
            "rank": rank,
            "id": doc_id,
            "distance": float(distance),
            "derived_similarity": float(similarity),
            "metadata": metadata,
            "document": document
        })

    average_distance = sum(
        item["distance"] for item in records
    ) / len(records)

    print(f"\nAverage cosine distance: {average_distance:.4f}")

    logger.info(
        "%s returned %d results; average distance %.4f",
        title,
        len(records),
        average_distance
    )

    return records


# ============================================================
# 10. EVALUATE KNOWN SEMANTIC QUERIES
# ============================================================

def evaluate_sample_search() -> list[dict[str, Any]]:
    """
    Test queries against known sample documents.

    Expected IDs are manually selected relevant examples.
    Search ranking may vary slightly between model versions.
    """

    test_cases = [
        {
            "query": "How do computers learn patterns from examples?",
            "expected_ids": {"ml_01", "ml_02"},
            "filter": None
        },
        {
            "query": "How can a neural network recognise images?",
            "expected_ids": {"dl_03", "dl_02"},
            "filter": None
        },
        {
            "query": "What is a Python structure that stores key value pairs?",
            "expected_ids": {"py_03"},
            "filter": {"category": "programming"}
        }
    ]

    evaluation_results = []

    print("\n\n" + "=" * 65)
    print("SEMANTIC SEARCH EVALUATION")
    print("=" * 65)

    for index, test in enumerate(test_cases, start=1):
        print(f"\nTest case {index}: {test['query']}")

        results = semantic_search(
            sample_collection,
            test["query"],
            k=5,
            filter_meta=test["filter"]
        )

        actual_ids = set(results["ids"][0])
        expected_ids = test["expected_ids"]
        matched_ids = actual_ids.intersection(expected_ids)

        passed = len(matched_ids) > 0

        print(f"Expected relevant IDs: {sorted(expected_ids)}")
        print(f"Retrieved IDs: {results['ids'][0]}")
        print(f"Relevant expected IDs found: {sorted(matched_ids)}")
        print(f"Evaluation: {'PASS' if passed else 'REVIEW'}")

        evaluation_results.append({
            "query": test["query"],
            "expected_ids": sorted(expected_ids),
            "retrieved_ids": results["ids"][0],
            "matched_ids": sorted(matched_ids),
            "passed": passed
        })

    return evaluation_results


# ============================================================
# 11. FILTERED VS UNFILTERED SEARCH
# ============================================================

def compare_metadata_filtering() -> dict[str, Any]:

    query = "How do models learn from data?"

    print("\n\n" + "=" * 65)
    print("METADATA FILTER COMPARISON")
    print("=" * 65)

    unfiltered = semantic_search(
        sample_collection,
        query,
        k=5
    )

    filtered = semantic_search(
        sample_collection,
        query,
        k=5,
        filter_meta={"category": "machine_learning"}
    )

    unfiltered_records = display_results(
        "UNFILTERED SEARCH",
        unfiltered
    )

    filtered_records = display_results(
        "FILTERED SEARCH - machine_learning",
        filtered
    )

    filter_valid = all(
        item["metadata"].get("category") == "machine_learning"
        for item in filtered_records
    )

    print("\nMetadata filter verification:",
          "PASS" if filter_valid else "FAIL")

    if not filter_valid:
        raise AssertionError(
            "Metadata filtering returned an unexpected category."
        )

    return {
        "query": query,
        "unfiltered": unfiltered_records,
        "filtered": filtered_records,
        "filter_validation": filter_valid
    }


# ============================================================
# 12. PDF EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text page by page to preserve page metadata."""

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Copy Machine_Learning_Notes.pdf into the data folder."
        )

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    if not pages:
        raise ValueError(
            "No extractable text found in the PDF. "
            "It may be scanned or image-based."
        )

    logger.info(
        "Extracted text from %d pages of %s.",
        len(pages),
        pdf_path.name
    )

    return pages


# ============================================================
# 13. PAGE-AWARE TEXT CHUNKING
# ============================================================

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into manageable chunks.

    Prefer sentence boundaries where possible.
    Character counts are used as a simple approximation,
    not as exact token counts.
    """

    if chunk_size <= overlap or overlap < 0:
        raise ValueError("Invalid chunk size or overlap.")

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        candidate = (
            f"{current} {sentence}".strip()
            if current else sentence
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(sentence) > chunk_size:
            start = 0

            while start < len(sentence):
                end = start + chunk_size
                piece = sentence[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end >= len(sentence):
                    break

                start = end - overlap

            current = ""
        else:
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def prepare_pdf_chunks(pdf_path: Path) -> list[dict[str, Any]]:
    """Convert PDF pages into ChromaDB-ready chunk records."""

    pages = extract_pdf_pages(pdf_path)
    documents = []
    chunk_number = 0

    for page_data in pages:
        page_chunks = split_into_chunks(page_data["text"])

        for text in page_chunks:
            chunk_number += 1

            documents.append({
                "id": f"pdf_chunk_{chunk_number:04d}",
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_data["page"],
                    "chunk_number": chunk_number,
                    "category": "pdf_content"
                }
            })

    if not documents:
        raise ValueError("PDF chunking produced no chunks.")

    logger.info(
        "Prepared %d chunks from %d PDF pages.",
        len(documents),
        len(pages)
    )

    return documents


# ============================================================
# 14. PDF RETRIEVAL AND RELEVANCE INSPECTION
# ============================================================

def retrieve_pdf_chunks(query: str) -> dict[str, Any]:

    results = semantic_search(
        pdf_collection,
        query,
        k=TOP_K
    )

    records = display_results(
        "TOP-3 PDF CHUNK RETRIEVAL",
        results
    )

    return {
        "query": query,
        "retrieved_chunks": records,
        "count": len(records)
    }


# ============================================================
# 15. RAG PROMPT AND OLLAMA ANSWER GENERATION
# ============================================================

def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]]
) -> str:
    """Build a prompt using only retrieved PDF context."""

    if not retrieved_chunks:
        raise ValueError("Cannot build a RAG prompt without context.")

    context_parts = []

    for item in retrieved_chunks:
        metadata = item["metadata"]

        context_parts.append(
            f"[Page {metadata.get('page', 'Unknown')}]\n"
            f"{item['document']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
You are an assistant answering questions about a supplied document.

Use only the information in the CONTEXT.
Do not invent facts or add information from outside the context.
If the context does not contain the answer, clearly say:
"I don't have enough information in the retrieved document context."

CONTEXT:
{context}

QUESTION:
{query}

Provide a clear, concise answer.
""".strip()


def generate_rag_answer(prompt: str) -> str:
    """Generate an answer using the local Ollama LLM."""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(
            GENERATE_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        answer = response.json().get("response", "").strip()

        if not answer:
            raise ValueError("Ollama returned an empty answer.")

        return answer

    except (
        requests.RequestException,
        ValueError
    ) as exc:
        logger.exception("Ollama answer generation failed.")
        raise RuntimeError(
            "Could not generate the RAG answer."
        ) from exc


# ============================================================
# 16. BASIC SANITY CHECKS
# ============================================================

def run_sanity_checks() -> dict[str, bool]:

    checks = {}

    checks["sample_document_count"] = (
        sample_collection.count() == 20
    )

    checks["pdf_collection_not_empty"] = (
        pdf_collection.count() > 0
    )

    test_results = semantic_search(
        sample_collection,
        "Explain machine learning",
        k=1
    )

    checks["semantic_search_returns_result"] = (
        len(test_results["ids"][0]) == 1
    )

    filtered_results = semantic_search(
        sample_collection,
        "What is a Python dictionary?",
        k=5,
        filter_meta={"category": "programming"}
    )

    checks["metadata_filter_returns_programming"] = (
        len(filtered_results["ids"][0]) > 0
        and all(
            metadata.get("category") == "programming"
            for metadata in filtered_results["metadatas"][0]
        )
    )

    for name, passed in checks.items():
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise AssertionError("One or more sanity checks failed.")

    return checks


# ============================================================
# 17. SAVE RESULTS AS JSON
# ============================================================

def save_results(results: dict[str, Any]) -> None:

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info("Results saved to %s", RESULTS_FILE)


# ============================================================
# 18. MAIN WORKFLOW
# ============================================================

def main() -> None:

    print("\n" + "=" * 65)
    print("CYNARIS SOLUTIONS - W5D4")
    print("SEMANTIC SEARCH WITH CHROMADB")
    print("=" * 65)

    start_time = time.time()

    try:
        # Step 1: Verify Ollama
        check_ollama()

        # Step 2: Insert 20 sample documents.
        # Stable IDs + upsert make reruns idempotent.
        upsert_documents(
            sample_collection,
            SAMPLE_DOCUMENTS
        )

        if sample_collection.count() != 20:
            raise AssertionError(
                "Sample collection should contain exactly 20 documents."
            )

        print(
            f"\nSample documents stored: "
            f"{sample_collection.count()}"
        )

        # Step 3: Evaluate known semantic queries.
        evaluation = evaluate_sample_search()

        # Step 4: Compare filtered and unfiltered results.
        filter_comparison = compare_metadata_filtering()

        # Step 5: Extract and chunk PDF.
        pdf_documents = prepare_pdf_chunks(PDF_PATH)

        # Step 6: Store PDF chunks.
        upsert_documents(
            pdf_collection,
            pdf_documents
        )

        print(
            f"\nPDF chunks stored: {pdf_collection.count()}"
        )

        # Step 7: Retrieve the most relevant PDF chunks.
        pdf_query = (
            "What are the main machine learning concepts "
            "and important points explained in this document?"
        )

        pdf_retrieval = retrieve_pdf_chunks(pdf_query)

        if not pdf_retrieval["retrieved_chunks"]:
            raise RuntimeError(
                "No PDF chunks were retrieved. Cannot continue with RAG."
            )

        # Step 8: Generate an answer using retrieved context.
        prompt = build_rag_prompt(
            pdf_query,
            pdf_retrieval["retrieved_chunks"]
        )

        print("\n" + "=" * 65)
        print("OLLAMA RAG ANSWER")
        print("=" * 65)

        answer = generate_rag_answer(prompt)

        print(answer)

        # Step 9: Manual verification reminder.
        print("\n" + "=" * 65)
        print("MANUAL ANSWER VERIFICATION")
        print("=" * 65)
        print("Check the generated answer against the retrieved PDF chunks.")
        print("Do not mark it correct unless the claims are supported.")
        print("Check for missing facts, unsupported claims and relevance.")

        # Step 10: Sanity checks.
        print("\n" + "=" * 65)
        print("SANITY CHECKS")
        print("=" * 65)

        sanity_checks = run_sanity_checks()

        # Step 11: Save structured results.
        elapsed = round(time.time() - start_time, 2)

        final_results = {
            "task": "W5D4 Semantic Search with ChromaDB",
            "timestamp": datetime.now().isoformat(),
            "embedding_model": EMBED_MODEL,
            "llm_model": LLM_MODEL,
            "sample_document_count": sample_collection.count(),
            "pdf_chunk_count": pdf_collection.count(),
            "evaluation": evaluation,
            "filter_comparison": filter_comparison,
            "pdf_retrieval": pdf_retrieval,
            "rag_query": pdf_query,
            "rag_answer": answer,
            "sanity_checks": sanity_checks,
            "execution_time_seconds": elapsed
        }

        save_results(final_results)

        print("\n" + "=" * 65)
        print("W5D4 PRACTICAL EXECUTION COMPLETED")
        print("=" * 65)
        print(f"Sample documents: {sample_collection.count()}")
        print(f"PDF chunks: {pdf_collection.count()}")
        print(f"Top-k PDF retrieval: {pdf_retrieval['count']}")
        print(f"Execution time: {elapsed} seconds")
        print(f"Log file: {LOG_FILE}")
        print(f"Results file: {RESULTS_FILE}")
        print("\nRemember to manually verify the RAG answer.")

    except Exception as exc:
        logger.exception("W5D4 execution failed: %s", exc)
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
