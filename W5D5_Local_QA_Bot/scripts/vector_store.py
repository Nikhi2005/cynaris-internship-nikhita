import requests
import chromadb

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    TOP_K,
    MAX_WORDS_PER_CHUNK,
    REQUEST_TIMEOUT
)


def generate_query_embedding(question):
    """
    Generate an embedding for the user's question
    using the same embedding model used in W5D4.
    """

    url = f"{OLLAMA_HOST}/api/embed"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": question
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings", [])

        if not embeddings:
            raise RuntimeError(
                "Ollama did not return a query embedding."
            )

        return embeddings[0]

    except requests.RequestException as error:
        raise RuntimeError(
            f"Embedding request failed: {error}"
        ) from error


def load_collection():
    """
    Connect to the existing W5D4 persistent database.
    Do not create a new collection.
    """

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            f"W5D4 ChromaDB database not found: "
            f"{CHROMA_DB_PATH}"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not find collection '{COLLECTION_NAME}'. "
            "Check the W5D4 database path and collection name."
        ) from error

    count = collection.count()

    if count == 0:
        raise RuntimeError(
            "The existing PDF collection is empty."
        )

    print("\nChromaDB connected successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored PDF chunks: {count}")

    return collection


def get_relevant_chunks(
    collection,
    question,
    top_k=TOP_K
):
    """
    Retrieve the most relevant PDF chunks
    using cosine similarity.
    """

    query_embedding = generate_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []

    for index, document in enumerate(documents):

        words = document.split()

        if len(words) > MAX_WORDS_PER_CHUNK:
            document = " ".join(
                words[:MAX_WORDS_PER_CHUNK]
            ) + " ..."

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        chunks.append({
            "text": document,
            "metadata": metadata or {},
            "distance": distance
        })

    return chunks


def format_context(chunks):
    """
    Combine retrieved chunks into one context string.
    """

    if not chunks:
        return "No relevant information was retrieved."

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk["metadata"]
        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"Source {index} (Page: {page}):\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)