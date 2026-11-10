# W5D3 – ChromaDB: Vector Store Setup & Retrieval

## Overview

This practical demonstrates how to use ChromaDB as a persistent vector database for storing document embeddings and performing semantic similarity searches.

It also implements a basic Retrieval-Augmented Generation (RAG) workflow by extracting text from a PDF, splitting it into chunks, retrieving relevant chunks and passing the retrieved context to a locally running Ollama language model.

## Objectives

- Set up ChromaDB with persistent storage.
- Generate embeddings using Ollama.
- Store 20 sample documents in a ChromaDB collection.
- Perform cosine-distance similarity searches.
- Apply metadata filtering during retrieval.
- Extract text from a PDF and split it into chunks.
- Store PDF embeddings in ChromaDB.
- Retrieve the top 3 relevant PDF chunks.
- Pass retrieved context to an Ollama LLM for RAG generation.

## Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Implementation            |
| ChromaDB 1.5.9   | Vector database           |
| Ollama           | Local model execution     |
| nomic-embed-text | Text embedding generation |
| llama3.2:3b      | RAG response generation   |
| Requests         | Ollama API communication  |
| pypdf 7.0.1      | PDF text extraction       |

## Project Structure

```text
W5D3_ChromaDB/
│
├── data/
│   └── Machine_Learning_Notes.pdf
│
├── scripts/
│   └── w5d3_chromadb.py
│
├── chromadb_store/
│
├── logs/
│   ├── results.json
│   └── run.log
│
├── .gitignore
└── README.md
```

## Implementation

### 1. Ollama Verification

The program verifies that the Ollama server is running and checks the availability of:

- nomic-embed-text
- llama3.2:3b

### 2. ChromaDB Setup

A persistent ChromaDB client is initialized.

Two collections are created:

- `w5d3_sample_documents`
- `w5d3_pdf_chunks`

Both collections use cosine distance through the collection metadata configuration.

### 3. Sample Document Embeddings

Twenty sample documents covering machine learning, programming and deep learning are converted into embeddings using `nomic-embed-text`.

The embeddings, document text and metadata are stored in ChromaDB.

### 4. Cosine Similarity Search

A semantic search is performed against the sample-document collection.

The system retrieves the three closest documents and displays their cosine distances.

Lower cosine distance indicates greater similarity.

### 5. Metadata Filtering

A metadata filter is applied:

```python
where={"category": "machine_learning"}
```

This restricts the search results to documents belonging to the machine_learning category.

### 6. PDF Processing

The program reads:

`data/Machine_Learning_Notes.pdf`

The PDF contains four pages.

The extracted text is divided into smaller chunks to support efficient retrieval.

### 7. PDF Embedding and Retrieval

The program generates embeddings for the PDF chunks and stores them in the PDF collection.

A semantic query retrieves the top three relevant chunks.

### 8. Retrieval-Augmented Generation

The retrieved PDF chunks are passed as context to the locally running `llama3.2:3b` model.

The model generates an answer based on the supplied context.

## Execution

Run the script from the W5D3 folder:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" ".\scripts\w5d3_chromadb.py"
```

## Execution Results

| Operation                      | Result     |
| ------------------------------ | ---------- |
| Ollama server verification     | Successful |
| Embedding model verification   | Successful |
| LLM verification               | Successful |
| Persistent ChromaDB connection | Successful |
| Cosine collections created     | 2          |
| Sample documents stored        | 20         |
| Semantic similarity search     | Successful |
| Metadata filtering             | Successful |
| PDF pages extracted            | 4          |
| PDF chunks created             | 13         |
| PDF chunks stored              | 13         |
| Top PDF chunks retrieved       | 3          |
| RAG model invocation           | Completed  |

## Sample Retrieval Results

The sample-document similarity search returned:

1. Machine learning explanation – cosine distance: 0.213009
2. Neural network explanation – cosine distance: 0.275838
3. Supervised learning explanation – cosine distance: 0.292813

The metadata-filtered search returned documents belonging to the machine_learning category.

The PDF retrieval returned three chunks with the following cosine distances:

- Chunk 1: 0.421498
- Chunk 2: 0.488547
- Chunk 3: 0.489260

## Output Files

- `logs/results.json` – structured execution results.
- `logs/run.log` – detailed execution logs.
- `chromadb_store/` – persistent vector database storage.

## Important Observation

The ChromaDB setup, embedding generation, similarity search, metadata filtering, PDF processing and top-three retrieval completed successfully.

The Ollama generation request also completed. However, the generated answer indicated that the retrieved context was insufficient to answer the broad document-summary question.

Therefore, the RAG workflow executed successfully, but the generated answer requires further improvement through better query formulation and prompt refinement.

## Key Learnings

- Understanding vector databases and embeddings.
- Generating text embeddings using Ollama.
- Storing and retrieving vectors with ChromaDB.
- Understanding cosine distance for semantic retrieval.
- Applying metadata filtering.
- Extracting and chunking PDF documents.
- Understanding the RAG pipeline.
- Connecting a vector database with a local LLM.

## Conclusion

The W5D3 practical demonstrated the use of ChromaDB for persistent vector storage, semantic retrieval and metadata filtering.

It also demonstrated how PDF documents can be processed into chunks, embedded and retrieved to provide context to a locally running language model through a basic RAG workflow.
