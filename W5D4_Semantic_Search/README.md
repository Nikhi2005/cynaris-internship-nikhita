# W5D4 – Semantic Search with ChromaDB

**Topic:** Semantic Search with ChromaDB

## 1. Objective

To implement semantic search using ChromaDB, generate text embeddings using Ollama, perform cosine similarity search and metadata filtering, and integrate PDF retrieval with an Ollama language model using Retrieval-Augmented Generation (RAG).

## 2. Technologies Used

- Python
- ChromaDB
- Ollama
- nomic-embed-text
- llama3.2:3b
- PyPDF
- JSON

## 3. Implementation

The practical covered the following tasks:

1. Created ChromaDB collections for sample documents and PDF chunks.
2. Generated embeddings using the Ollama `nomic-embed-text` model.
3. Inserted 20 sample documents into ChromaDB.
4. Configured cosine distance for similarity search.
5. Executed semantic searches using different queries.
6. Applied metadata filtering to restrict search results by category.
7. Extracted text from a four-page PDF.
8. Divided the PDF content into 13 chunks and stored their embeddings.
9. Retrieved the top three relevant PDF chunks.
10. Passed retrieved context to `llama3.2:3b` to generate a RAG response.
11. Executed sanity checks and saved the results.

## 4. ChromaDB Collections

### Sample Documents

`w5d4_semantic_documents`

- Total documents: 20
- Stores sample machine learning, deep learning and Python-related content.
- Supports semantic search and metadata filtering.

### PDF Chunks

`w5d4_semantic_pdf_chunks`

- Source: `Machine_Learning_Notes.pdf`
- Pages extracted: 4
- Chunks stored: 13
- Supports retrieval of relevant PDF passages.

## 5. Execution Results

| Test                         | Result       |
| ---------------------------- | ------------ |
| Ollama server verification   | PASS         |
| Embedding model verification | PASS         |
| LLM verification             | PASS         |
| Sample document insertion    | 20 documents |
| Semantic search – Test 1     | PASS         |
| Semantic search – Test 2     | PASS         |
| Semantic search – Test 3     | PASS         |
| Metadata filter verification | PASS         |
| PDF extraction               | 4 pages      |
| PDF chunk storage            | 13 chunks    |
| Top-3 PDF retrieval          | PASS         |
| Sanity checks                | 4/4 PASS     |

### Semantic Search Evaluation

All three semantic-search test cases successfully retrieved their expected relevant document IDs within the top five results.

### Metadata Filtering

The filtered search successfully restricted results to the `machine_learning` category.

### PDF Retrieval

The system retrieved the top three PDF chunks with an average cosine distance of `0.2528`.

### RAG Response

The retrieved PDF context was passed to the Ollama language model. The generated answer explained:

- Machine learning as a branch of AI.
- Learning patterns from examples.
- The typical machine-learning workflow.
- Supervised learning using labelled examples.

The generated answer was manually checked against the retrieved PDF passages. Its main claims were supported by the retrieved context.

## 6. Sanity Check Results

All four automated sanity checks passed:

- `sample_document_count: PASS`
- `pdf_collection_not_empty: PASS`
- `semantic_search_returns_result: PASS`
- `metadata_filter_returns_programming: PASS`

## 7. Generated Output Files

- `logs/results.json` – Structured execution results.
- `logs/w5d4_20261111_115234.log` – Execution log.
- `chromadb_store/` – Persistent ChromaDB storage.

## 8. Conclusion

The W5D4 practical was completed successfully. ChromaDB was used to store and retrieve document embeddings through semantic search and metadata filtering. PDF content was extracted, divided into chunks, embedded and retrieved. The retrieved context was integrated with Ollama to generate a RAG response.

All four automated sanity checks passed, and the generated answer was manually verified against the retrieved PDF content.

**Execution Status: COMPLETED**
