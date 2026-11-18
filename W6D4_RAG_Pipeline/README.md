# W6D4 - RAG Pipeline using LangChain and ChromaDB

## Objective

To implement a Retrieval Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and Ollama.

## Technologies Used

- Python
- LangChain
- ChromaDB
- Ollama
- nomic-embed-text
- Qwen2.5:3b
- PyPDF
- ReportLab

## Practical 1: ChromaDB Fundamentals

### Tasks

1. Created a persistent ChromaDB collection.
2. Generated embeddings using nomic-embed-text.
3. Added 20 sample documents with metadata.
4. Performed cosine similarity search.
5. Applied metadata filtering.

### Results

- Collection: w6d4_documents
- Documents stored: 20
- Embedding dimension: 768
- Similarity search: Successful
- Metadata filtering: Successful

## Practical 2: RAG Pipeline

### Workflow

1. Created a sample technical PDF.
2. Loaded the PDF using PyPDFLoader.
3. Split the document into smaller chunks.
4. Generated embeddings using nomic-embed-text.
5. Stored the chunks in ChromaDB.
6. Retrieved the top 3 relevant chunks.
7. Passed retrieved context to Qwen2.5:3b.
8. Generated an answer using the retrieved information.

### Results

- PDF pages loaded: 1
- Text chunks generated: 6
- Chunks stored in ChromaDB: 6
- Retrieved documents: 3
- LLM response: Generated successfully

## Project Structure

W6D4_RAG_Pipeline/

- chromadb_demo.py
- rag_pipeline.py
- README.md
- sample_ai_document.pdf
- chroma_db/
- pdf_chroma_db/

## Key Concepts Learned

- Retrieval Augmented Generation
- Text embeddings
- Vector databases
- Cosine distance
- Metadata filtering
- Document chunking
- Similarity search
- Context-based answer generation

## Conclusion

Successfully implemented a RAG pipeline integrating LangChain, ChromaDB, and Ollama. The system retrieves relevant document chunks from a PDF and uses a local language model to generate answers based on the retrieved context.
