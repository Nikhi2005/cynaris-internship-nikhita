
import chromadb
from langchain_ollama import OllamaEmbeddings

# Initialize Ollama embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection using cosine distance
collection = client.get_or_create_collection(
    name="w6d4_documents",
    metadata={"hnsw:space": "cosine"}
)

# Sample documents
documents = [
    "Python is a popular programming language used in AI and machine learning.",
    "Machine learning enables computers to learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural Language Processing helps computers understand human language.",
    "RAG stands for Retrieval Augmented Generation.",
    "RAG combines information retrieval with language model generation.",
    "ChromaDB is a vector database used to store embeddings.",
    "Vector databases support similarity search on numerical vectors.",
    "Embeddings represent text as numerical vectors.",
    "Cosine similarity measures the similarity between two vectors.",
    "Ollama allows users to run large language models locally.",
    "Qwen is a family of large language models.",
    "Document chunking divides large documents into smaller sections.",
    "Metadata provides additional information about stored documents.",
    "A retriever finds relevant documents based on a user query.",
    "FAISS is a library for efficient similarity search.",
    "One-hot encoding represents categories using binary vectors.",
    "Transformers are widely used in modern language models.",
    "PDF loaders extract text from PDF documents.",
    "Prompt templates structure instructions given to language models."
]

# Generate unique IDs and metadata
ids = [f"doc_{i}" for i in range(1, 21)]

metadatas = [
    {
        "category": "AI" if i <= 10 else "Technology",
        "source": "sample_dataset"
    }
    for i in range(1, 21)
]

# Generate embeddings
vectors = embeddings.embed_documents(documents)

# Add documents to ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=vectors,
    metadatas=metadatas
)

print("20 documents added successfully!")
print("Collection name:", collection.name)
print("Total documents:", collection.count())
print("Embedding dimension:", len(vectors[0]))



# -----------------------------------------
# Similarity Search using Cosine Distance
# -----------------------------------------

query = "How does RAG retrieve information for language models?"

query_embedding = embeddings.embed_query(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nTOP 3 SIMILAR DOCUMENTS")
for i, document in enumerate(results["documents"][0]):
    distance = results["distances"][0][i]
    print(f"\n{i + 1}. {document}")
    print(f"Cosine distance: {distance:.4f}")

# -----------------------------------------
# Metadata Filtering
# -----------------------------------------

filtered_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"category": "AI"}
)

print("\nDOCUMENTS FILTERED BY CATEGORY: AI")
for document in filtered_results["documents"][0]:
    print("-", document)

print("\nSimilarity search and metadata filtering completed!")
