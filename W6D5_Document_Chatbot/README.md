# W6D5 – Document Chatbot with LangChain

## Objective

Build a Document Chatbot using LangChain, Ollama, conversation memory, PDF retrieval, ChromaDB and a two-tool agent.

## Technologies

- Python
- LangChain Core
- LangChain Classic
- LangChain Ollama
- ChromaDB
- PyPDF
- Ollama – qwen2.5:3b
- Ollama Embeddings – nomic-embed-text

## Implementation

### 1. LangChain Chain

Implemented a runnable sequence:

PromptTemplate → OllamaLLM → StrOutputParser

Tested using five different questions.

### 2. Conversation Memory

Used ConversationBufferMemory to maintain conversation history across five turns.

The conversation is saved to a JSON file and restored into a new memory object to verify persistence.

### 3. PDF Document Retrieval

- Loaded a PDF document.
- Split the document into smaller text chunks.
- Generated embeddings using nomic-embed-text.
- Stored the chunks in ChromaDB.
- Retrieved relevant chunks for a question.
- Generated an answer using Qwen2.5:3b.

### 4. Two-Tool Agent

Implemented two tools:

- Web Search Stub – searches a small built-in knowledge base.
- Calculator – performs restricted arithmetic using Python AST parsing.

Used ChatOllama tool calling to select and invoke the tools.

## Execution

Run the program using the existing Cynaris virtual environment:

```powershell
& "D:\Cynaris\.venv\Scripts\python.exe" "document_chatbot.py"
```

## Validation Checklist

- [ ] Chain tested with five inputs.
- [ ] Conversation memory tested across five turns.
- [ ] Conversation history saved and restored.
- [ ] PDF loaded and split into chunks.
- [ ] ChromaDB indexing completed.
- [ ] One document-based question answered.
- [ ] Web search stub tested.
- [ ] Calculator tested with two tasks.
- [ ] All outputs reviewed.

## Project Files

- document_chatbot.py
- README.md
- .gitignore

## Notes

The web search tool is a local stub and does not access the internet.

The PDF is read as an input source. Generated ChromaDB data and conversation history are excluded from Git.
