
"""
Cynaris Solutions - AI/ML Internship
Week 5 Day 3: ChromaDB - Vector Store Setup & Embedding Documents

Requirements:
    pip install chromadb requests pypdf

Ollama models:
    nomic-embed-text
    llama3.2:3b
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import requests
from pypdf import PdfReader


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chromadb_store"
LOG_DIR = BASE_DIR / "logs"

OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

SAMPLE_COLLECTION_NAME = "w5d3_sample_documents"
PDF_COLLECTION_NAME = "w5d3_pdf_chunks"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 16

# Change this question to ask something relevant to your PDF.
USER_QUESTION = "What are the main concepts and important points explained in this document?"

for folder in [DATA_DIR, DB_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOGGING
# ---------------------------------------------------------

LOG_FILE = LOG_DIR / "run.log"
RESULTS_FILE = LOG_DIR / "results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 3. SAMPLE DOCUMENTS - EXACTLY 20
# ---------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_01",
        "text": "Machine learning enables computers to learn patterns from training data and make predictions without being explicitly programmed for every task.",
        "category": "machine_learning",
    },
    {
        "id": "doc_02",
        "text": "Supervised learning trains a model using labelled examples where each input is associated with a known output.",
        "category": "machine_learning",
    },
    {
        "id": "doc_03",
        "text": "Unsupervised learning discovers hidden patterns and structures in data that does not contain labelled outputs.",
        "category": "machine_learning",
    },
    {
        "id": "doc_04",
        "text": "Classification is a supervised learning task that predicts a category or class, such as spam or not spam.",
        "category": "machine_learning",
    },
    {
        "id": "doc_05",
        "text": "Regression predicts continuous numerical values such as house prices, temperatures, or sales figures.",
        "category": "machine_learning",
    },
    {
        "id": "doc_06",
        "text": "Overfitting occurs when a model learns training data too closely and performs poorly on unseen data.",
        "category": "machine_learning",
    },
    {
        "id": "doc_07",
        "text": "Underfitting occurs when a model is too simple to capture important patterns in the training data.",
        "category": "machine_learning",
    },
    {
        "id": "doc_08",
        "text": "Cross-validation evaluates a model by training and testing it across different portions of a dataset.",
        "category": "machine_learning",
    },
    {
        "id": "doc_09",
        "text": "Precision measures how many predicted positive cases are actually positive.",
        "category": "machine_learning",
    },
    {
        "id": "doc_10",
        "text": "Recall measures how many actual positive cases are correctly identified by a model.",
        "category": "machine_learning",
    },
    {
        "id": "doc_11",
        "text": "Python is a popular programming language used for automation, data analysis, artificial intelligence, and web development.",
        "category": "programming",
    },
    {
        "id": "doc_12",
        "text": "NumPy provides multidimensional arrays and mathematical operations for scientific computing in Python.",
        "category": "programming",
    },
    {
        "id": "doc_13",
        "text": "Pandas provides DataFrame and Series structures for cleaning, transforming, and analysing structured data.",
        "category": "programming",
    },
    {
        "id": "doc_14",
        "text": "A database stores and organizes information so that applications can retrieve and manage records efficiently.",
        "category": "programming",
    },
    {
        "id": "doc_15",
        "text": "Git is a version control system that tracks changes in source code and supports collaborative development.",
        "category": "programming",
    },
    {
        "id": "doc_16",
        "text": "A neural network contains interconnected computational units arranged in layers to learn patterns from data.",
        "category": "deep_learning",
    },
    {
        "id": "doc_17",
        "text": "Deep learning uses neural networks with multiple layers to learn complex representations from large datasets.",
        "category": "deep_learning",
    },
    {
        "id": "doc_18",
        "text": "Natural language processing enables computers to understand, analyse, and generate human language.",
        "category": "deep_learning",
    },
    {
        "id": "doc_19",
        "text": "Computer vision enables machines to interpret images and videos for tasks such as object detection and recognition.",
        "category": "deep_learning",
    },
    {
        "id": "doc_20",
        "text": "A large language model is trained on extensive text data to understand and generate natural language.",
        "category": "deep_learning",
    },
]


# ---------------------------------------------------------
# 4. OLLAMA CONNECTION AND MODEL CHECK
# ---------------------------------------------------------

def check_ollama() -> None:
    """Check Ollama availability and required local models."""

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        available_models = [
            model.get("name", "") for model in models
        ]

        logger.info("Ollama server is running.")

        for required_model in [EMBED_MODEL, LLM_MODEL]:
            found = any(
                name == required_model
                or name.startswith(required_model + ":")
                for name in available_models
            )

            if not found:
                raise RuntimeError(
                    f"Model '{required_model}' is not installed. "
                    f"Run: ollama pull {required_model}"
                )

            logger.info("Verified Ollama model: %s", required_model)

    except requests.RequestException as error:
        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama and try again."
        ) from error


# ---------------------------------------------------------
# 5. GENERATE EMBEDDINGS USING OLLAMA
# ---------------------------------------------------------

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings through Ollama's current /api/embed endpoint.
    Accepts a list of text inputs and returns one vector per input.
    """

    if not texts:
        return []

    for attempt in range(1, 4):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={
                    "model": EMBED_MODEL,
                    "input": texts,
                    "truncate": True,
                },
                timeout=180,
            )

            response.raise_for_status()
            data = response.json()

            embeddings = data.get("embeddings")

            if not embeddings or len(embeddings) != len(texts):
                raise ValueError(
                    "Ollama returned an unexpected embedding response."
                )

            return embeddings

        except (requests.RequestException, ValueError) as error:
            logger.warning(
                "Embedding attempt %s/3 failed: %s",
                attempt,
                error,
            )

            if attempt == 3:
                raise RuntimeError(
                    "Could not generate embeddings after 3 attempts."
                ) from error

            time.sleep(1)

    return []


def generate_embeddings_in_batches(
    texts: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> List[List[float]]:
    """Generate embeddings in manageable batches."""

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        logger.info(
            "Generating embeddings: %s to %s of %s",
            start + 1,
            start + len(batch),
            len(texts),
        )

        batch_embeddings = generate_embeddings(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ---------------------------------------------------------
# 6. CREATE PERSISTENT CHROMADB COLLECTIONS
# ---------------------------------------------------------

def create_collection(client, name: str):
    """
    Re-create only this day's collection so repeated runs
    do not leave outdated sample documents or PDF chunks.
    """

    try:
        client.delete_collection(name=name)
        logger.info("Removed previous collection: %s", name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Created cosine collection: %s", name)
    return collection


# ---------------------------------------------------------
# 7. INGEST THE 20 SAMPLE DOCUMENTS
# ---------------------------------------------------------

def ingest_sample_documents(collection) -> None:
    """Embed and store exactly 20 independent sample documents."""

    if len(SAMPLE_DOCUMENTS) != 20:
        raise ValueError("Exactly 20 sample documents are required.")

    texts = [item["text"] for item in SAMPLE_DOCUMENTS]
    embeddings = generate_embeddings_in_batches(texts)

    ids = [item["id"] for item in SAMPLE_DOCUMENTS]

    metadatas = [
        {
            "category": item["category"],
            "doc_type": "sample",
            "source": "sample_documents",
        }
        for item in SAMPLE_DOCUMENTS
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "Successfully stored %s sample documents.",
        collection.count(),
    )


# ---------------------------------------------------------
# 8. SAMPLE DOCUMENT SIMILARITY SEARCH
# ---------------------------------------------------------

def search_sample_documents(collection) -> Dict[str, Any]:
    """Demonstrate cosine similarity search and metadata filtering."""

    query = "How do computers learn patterns from examples?"

    query_embedding = generate_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    logger.info("Sample-document similarity search:")
    print_search_results(results)

    filtered_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"category": "machine_learning"},
        include=["documents", "metadatas", "distances"],
    )

    logger.info("Sample-document search with metadata filter:")
    print_search_results(filtered_results)

    return {
        "query": query,
        "similarity_search": results,
        "metadata_filtered_search": filtered_results,
    }


def print_search_results(results: Dict[str, Any]) -> None:
    """Print search results in a readable format."""

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, document in enumerate(documents):
        logger.info("Result %s: %s", index + 1, document)
        logger.info("Metadata: %s", metadatas[index])
        logger.info("Cosine distance: %.6f", distances[index])


# ---------------------------------------------------------
# 9. EXTRACT TEXT FROM PDF
# ---------------------------------------------------------

def find_pdf() -> Path:
    """Find the first PDF in the data directory."""

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF found in {DATA_DIR}. "
            "Place one text-based PDF in the data folder."
        )

    return pdf_files[0]


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text page by page from a PDF."""

    try:
        reader = PdfReader(str(pdf_path))

        if reader.is_encrypted:
            raise ValueError(
                "The selected PDF is encrypted. Use an accessible PDF."
            )

        page_texts = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                page_texts.append(
                    f"[Page {page_number}]\n{text.strip()}"
                )

        full_text = "\n\n".join(page_texts)

        if not full_text.strip():
            raise ValueError(
                "No extractable text found. "
                "The PDF may be scanned and require OCR."
            )

        logger.info(
            "Extracted text from PDF: %s",
            pdf_path.name,
        )
        logger.info("PDF pages: %s", len(reader.pages))

        return full_text

    except Exception as error:
        raise RuntimeError(
            f"Could not extract PDF text: {error}"
        ) from error


# ---------------------------------------------------------
# 10. SENTENCE-AWARE TEXT CHUNKING WITH OVERLAP
# ---------------------------------------------------------

def split_long_text(text: str, max_size: int) -> List[str]:
    """Split an unusually long sentence at word boundaries."""

    words = text.split()
    pieces = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if len(candidate) > max_size and current:
            pieces.append(current)
            current = word
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Create sentence-aware chunks.
    Consecutive chunks share a small amount of previous context.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(split_long_text(sentence, chunk_size))
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            overlap_text = current[-overlap:] if overlap else ""
            current = f"{overlap_text} {sentence}".strip()

            if len(current) > chunk_size:
                current = sentence

    if current:
        chunks.append(current)

    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    if not chunks:
        raise ValueError("No usable PDF chunks were created.")

    return chunks


# ---------------------------------------------------------
# 11. INGEST PDF CHUNKS INTO A SEPARATE COLLECTION
# ---------------------------------------------------------

def ingest_pdf_chunks(collection, pdf_path: Path) -> List[str]:
    """Extract, chunk, embed and store PDF content."""

    pdf_text = extract_pdf_text(pdf_path)
    chunks = chunk_text(pdf_text)

    logger.info("Created %s PDF chunks.", len(chunks))

    embeddings = generate_embeddings_in_batches(chunks)

    ids = [
        f"pdf_chunk_{index:04d}"
        for index in range(1, len(chunks) + 1)
    ]

    metadatas = [
        {
            "doc_type": "pdf_chunk",
            "source": pdf_path.name,
            "chunk_index": index,
        }
        for index in range(1, len(chunks) + 1)
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "Stored %s PDF chunks in the PDF collection.",
        collection.count(),
    )

    return chunks


# ---------------------------------------------------------
# 12. RETRIEVE TOP 3 PDF CHUNKS
# ---------------------------------------------------------

def retrieve_top_three(
    collection,
    question: str,
) -> Dict[str, Any]:
    """Retrieve exactly the top three matching PDF chunks."""

    query_embedding = generate_embeddings([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0][:3]
    metadatas = results.get("metadatas", [[]])[0][:3]
    distances = results.get("distances", [[]])[0][:3]
    ids = results.get("ids", [[]])[0][:3]

    if not documents:
        raise RuntimeError("No PDF chunks were retrieved.")

    logger.info("Top %s PDF chunks retrieved:", len(documents))

    retrieved = []

    for index, document in enumerate(documents):
        item = {
            "id": ids[index],
            "text": document,
            "metadata": metadatas[index],
            "cosine_distance": distances[index],
        }

        retrieved.append(item)

        logger.info("Chunk %s: %s", index + 1, item["id"])
        logger.info("Distance: %.6f", distances[index])
        logger.info("Text: %s", document)

    return {
        "question": question,
        "retrieved_chunks": retrieved,
    }


# ---------------------------------------------------------
# 13. SEND RETRIEVED CONTEXT TO OLLAMA LLM
# ---------------------------------------------------------

def generate_rag_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """Generate an answer grounded in retrieved PDF context."""

    context = "\n\n".join(
        [
            f"Source chunk {index + 1}:\n{item['text']}"
            for index, item in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are a helpful document-question-answering assistant.

Answer the user's question using only the provided document context.

If the answer is not available in the context, clearly say:
"The provided document context does not contain enough information."

Do not invent facts. Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                },
            },
            timeout=300,
        )

        response.raise_for_status()

        answer = response.json().get("response", "").strip()

        if not answer:
            raise ValueError("Ollama returned an empty answer.")

        logger.info("Generated RAG answer:")
        logger.info("%s", answer)

        return answer

    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            f"Could not generate the RAG answer: {error}"
        ) from error


# ---------------------------------------------------------
# 14. SAVE RESULTS AS JSON
# ---------------------------------------------------------

def save_results(results: Dict[str, Any]) -> None:
    """Save practical results in a structured JSON file."""

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    logger.info("Results saved to: %s", RESULTS_FILE)
    logger.info("Run log saved to: %s", LOG_FILE)


# ---------------------------------------------------------
# 15. MAIN EXECUTION
# ---------------------------------------------------------

def main() -> None:
    logger.info("=" * 65)
    logger.info("CYNARIS - WEEK 5 DAY 3: CHROMADB PRACTICAL")
    logger.info("=" * 65)

    check_ollama()

    # Connect to persistent ChromaDB storage.
    client = chromadb.PersistentClient(path=str(DB_DIR))

    logger.info("Connected to persistent ChromaDB.")
    logger.info("Database path: %s", DB_DIR)

    # Create two separate cosine collections.
    sample_collection = create_collection(
        client,
        SAMPLE_COLLECTION_NAME,
    )

    pdf_collection = create_collection(
        client,
        PDF_COLLECTION_NAME,
    )

    # Part 1: Store and search 20 sample documents.
    ingest_sample_documents(sample_collection)

    sample_results = search_sample_documents(
        sample_collection
    )

    # Part 2: Process the PDF and store its chunks.
    pdf_path = find_pdf()

    pdf_chunks = ingest_pdf_chunks(
        pdf_collection,
        pdf_path,
    )

    # Part 3: Retrieve top 3 chunks and generate an answer.
    rag_results = retrieve_top_three(
        pdf_collection,
        USER_QUESTION,
    )

    answer = generate_rag_answer(
        USER_QUESTION,
        rag_results["retrieved_chunks"],
    )

    # Save all important results.
    final_results = {
        "practical": "W5D3 - ChromaDB",
        "embedding_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "distance_metric": "cosine",
        "sample_collection": {
            "name": SAMPLE_COLLECTION_NAME,
            "document_count": sample_collection.count(),
            "search_results": sample_results,
        },
        "pdf_collection": {
            "name": PDF_COLLECTION_NAME,
            "pdf_file": pdf_path.name,
            "chunk_count": len(pdf_chunks),
            "stored_count": pdf_collection.count(),
        },
        "rag": {
            "question": USER_QUESTION,
            "top_3_retrieved_chunks": rag_results[
                "retrieved_chunks"
            ],
            "generated_answer": answer,
        },
    }

    save_results(final_results)

    logger.info("=" * 65)
    logger.info("W5D3 PRACTICAL COMPLETED SUCCESSFULLY")
    logger.info("Sample documents: %s", sample_collection.count())
    logger.info("PDF chunks: %s", pdf_collection.count())
    logger.info("Top retrieved chunks: %s", len(
        rag_results["retrieved_chunks"]
    ))
    logger.info("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.exception("W5D3 practical failed: %s", error)
        sys.exit(1)
