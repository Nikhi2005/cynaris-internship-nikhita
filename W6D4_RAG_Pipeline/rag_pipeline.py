
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# --------------------------------------------------
# 1. CREATE A SAMPLE TECHNICAL PDF
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "sample_ai_document.pdf"
DB_PATH = BASE_DIR / "pdf_chroma_db"

pdf_content = [
    "Artificial Intelligence and Machine Learning",
    "Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence. Examples include problem solving, language understanding, and decision making.",
    "Machine Learning is a branch of Artificial Intelligence. It allows computers to learn patterns from data and improve their performance without being explicitly programmed for every task.",
    "Deep Learning is a subset of Machine Learning. It uses artificial neural networks with multiple layers to learn complex patterns from large datasets.",
    "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. It is used in chatbots, translation systems, and text summarization.",
    "Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with language model generation. It allows a language model to use relevant external documents when answering questions.",
    "A RAG system generally contains a document loading stage, text splitting, embedding generation, vector storage, retrieval, and answer generation.",
    "Embeddings convert text into numerical vectors. These vectors represent semantic meaning and help systems identify related pieces of information.",
    "A vector database stores and searches numerical embeddings. ChromaDB is an example of a vector database that supports similarity search and metadata filtering.",
    "Cosine similarity measures the similarity between two vectors by comparing their directions. In cosine distance, a smaller distance indicates greater similarity.",
    "Document chunking divides large documents into smaller text sections. Chunk size and overlap affect the quality of information retrieval.",
    "A retriever searches a vector database and returns documents that are relevant to a user's query.",
    "Ollama allows developers to run large language models locally. It can also provide embedding models for generating vector representations.",
    "The nomic-embed-text model generates text embeddings and can be used to represent documents and user queries in the same vector space.",
    "A RAG pipeline passes retrieved document content to a language model as context. The model uses this context to generate a grounded response.",
    "RAG can reduce unsupported answers by giving language models access to relevant information. However, retrieval quality and the quality of the source documents remain important."
]

pdf = canvas.Canvas(str(PDF_PATH), pagesize=A4)
width, height = A4

y = height - 60

for paragraph in pdf_content:
    words = paragraph.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (current_line + " " + word).strip()

        if pdf.stringWidth(test_line, "Helvetica", 10) < width - 100:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    for line in lines:
        if y < 60:
            pdf.showPage()
            y = height - 60

        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, line)
        y -= 17

    y -= 12

pdf.save()

print("Sample PDF created:", PDF_PATH.name)


# --------------------------------------------------
# 2. LOAD PDF AND SPLIT INTO CHUNKS
# --------------------------------------------------

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()

print("\nPDF pages loaded:", len(pages))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks = splitter.split_documents(pages)

print("Total text chunks:", len(chunks))


# --------------------------------------------------
# 3. GENERATE EMBEDDINGS AND STORE IN CHROMADB
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    collection_name="w6d4_pdf_rag",
    embedding_function=embeddings,
    persist_directory=str(DB_PATH),
    collection_metadata={"hnsw:space": "cosine"}
)

# Prevent duplicate records when rerunning the program
existing_ids = vectorstore.get()["ids"]

if existing_ids:
    vectorstore.delete(ids=existing_ids)

vectorstore.add_documents(chunks)

print("\nPDF chunks stored in ChromaDB:", len(chunks))


# --------------------------------------------------
# 4. RETRIEVE TOP 3 RELEVANT CHUNKS
# --------------------------------------------------

question = "What is RAG and how does it work?"

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

retrieved_docs = retriever.invoke(question)

print("\nTOP 3 RETRIEVED CHUNKS")

for index, doc in enumerate(retrieved_docs, start=1):
    print(f"\nChunk {index}:")
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content)


# --------------------------------------------------
# 5. PASS RETRIEVED CONTEXT TO OLLAMA
# --------------------------------------------------

context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

prompt = PromptTemplate.from_template(
    """
    Answer the question using only the context provided below.
    If the answer is not available in the context, say that you
    do not have enough information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

rag_chain = prompt | llm

response = rag_chain.invoke({
    "context": context,
    "question": question
})

print("\nRAG GENERATED ANSWER")
print(response.content)

print("\nRAG PIPELINE EXECUTED SUCCESSFULLY!")
