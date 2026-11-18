
from pathlib import Path
import ast
import json
import operator

from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

# Read the existing sample PDF without modifying it.
PDF_PATH = (
    BASE_DIR.parent
    / "W6D4_RAG_Pipeline"
    / "sample_ai_document.pdf"
)

CHROMA_DIR = BASE_DIR / "chroma_db"
HISTORY_FILE = BASE_DIR / "conversation_history.json"

llm = OllamaLLM(model=LLM_MODEL, temperature=0)


# --------------------------------------------------
# PART 1: LANGCHAIN CHAIN
# PromptTemplate -> Ollama LLM -> StrOutputParser
# --------------------------------------------------

def run_chain_demo():
    print("\n" + "=" * 60)
    print("PART 1: LANGCHAIN CHAIN")
    print("=" * 60)

    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant.
        Answer the following question clearly and briefly.

        Question: {question}

        Answer:"""
    )

    chain = prompt | llm | StrOutputParser()

    questions = [
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What is LangChain?",
        "What is Retrieval Augmented Generation?",
        "What is a vector database?"
    ]

    for index, question in enumerate(questions, start=1):
        answer = chain.invoke({"question": question})
        print(f"\nTest {index}")
        print("Question:", question)
        print("Answer:", answer)

    print("\nChain test completed: 5 inputs.")


# --------------------------------------------------
# PART 2: CONVERSATION BUFFER MEMORY
# Verify 5 conversation turns and save/restore history
# --------------------------------------------------

def run_memory_demo():
    print("\n" + "=" * 60)
    print("PART 2: CONVERSATION BUFFER MEMORY")
    print("=" * 60)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=False
    )

    conversation_prompt = PromptTemplate.from_template(
        """You are a helpful assistant.
        Use the conversation history when answering.

        Conversation history:
        {chat_history}

        User: {input}
        Assistant:"""
    )

    conversation_chain = conversation_prompt | llm | StrOutputParser()

    questions = [
        "My name is Alex.",
        "I am learning Python.",
        "I am also learning LangChain.",
        "What programming language am I learning?",
        "What is my name?"
    ]

    memory.chat_memory.clear()

    for turn, question in enumerate(questions, start=1):
        history = memory.load_memory_variables({})["chat_history"]

        answer = conversation_chain.invoke({
            "chat_history": history,
            "input": question
        })

        memory.save_context(
            {"input": question},
            {"output": answer}
        )

        print(f"\nTurn {turn}")
        print("User:", question)
        print("Assistant:", answer)

    # Verify exactly five user-assistant turns.
    messages = memory.chat_memory.messages

    print("\nTotal stored messages:", len(messages))
    print("Expected messages: 10")

    if len(messages) != 10:
        raise RuntimeError("Memory verification failed.")

    # Save the conversation in JSON format.
    history_data = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": message.content
        }
        for index, message in enumerate(messages)
    ]

    HISTORY_FILE.write_text(
        json.dumps(history_data, indent=4),
        encoding="utf-8"
    )

    # Restore the saved conversation into a new memory object.
    restored_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=False
    )

    saved_history = json.loads(
        HISTORY_FILE.read_text(encoding="utf-8")
    )

    for item in saved_history:
        if item["role"] == "user":
            restored_memory.chat_memory.add_user_message(
                item["content"]
            )
        else:
            restored_memory.chat_memory.add_ai_message(
                item["content"]
            )

    restored_messages = restored_memory.chat_memory.messages

    print("Restored messages:", len(restored_messages))

    if len(restored_messages) != 10:
        raise RuntimeError("History restoration failed.")

    print("Memory test completed: 5 turns saved and restored.")


# --------------------------------------------------
# PART 3: PDF DOCUMENT RETRIEVAL
# PDF -> Chunks -> Embeddings -> Chroma -> Retrieval
# --------------------------------------------------

def build_retriever():
    print("\n" + "=" * 60)
    print("PART 3: PDF DOCUMENT RETRIEVAL")
    print("=" * 60)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Check that the sample PDF exists at the configured path."
        )

    print("Loading PDF:", PDF_PATH.name)

    documents = PyPDFLoader(str(PDF_PATH)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted from the PDF.")

    print("PDF pages loaded:", len(documents))
    print("Text chunks created:", len(chunks))

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vector_store = Chroma(
        collection_name="w6d5_document_chatbot",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    # Remove previous records from this W6D5 collection
    # so rerunning the program does not duplicate chunks.
    existing_ids = vector_store.get()["ids"]

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    print("Documents indexed in ChromaDB:", len(chunks))

    return retriever


def run_document_question(retriever):
    print("\n--- Document Chatbot Question ---")

    question = "What is RAG and how does it work?"

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content for document in retrieved_docs
    )

    document_prompt = PromptTemplate.from_template(
        """Answer the question using only the supplied document context.
        If the answer is not present, say that the document does not
        provide enough information.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )

    document_chain = document_prompt | llm | StrOutputParser()

    answer = document_chain.invoke({
        "context": context,
        "question": question
    })

    print("Question:", question)
    print("\nRetrieved chunks:", len(retrieved_docs))
    print("\nAnswer:", answer)


# --------------------------------------------------
# PART 4: TWO-TOOL AGENT
# Web Search Stub + Safe Calculator
# --------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search a small built-in knowledge stub for basic AI and LangChain information."""

    knowledge = {
        "langchain": (
            "LangChain is a framework for building applications "
            "using language models, prompts, chains, memory and tools."
        ),
        "rag": (
            "RAG stands for Retrieval Augmented Generation. "
            "It retrieves relevant external information and provides "
            "it as context to a language model."
        ),
        "ollama": (
            "Ollama allows users to run supported language models "
            "locally on their computers."
        )
    }

    query_lower = query.lower()

    for keyword, result in knowledge.items():
        if keyword in query_lower:
            return result

    return (
        "Web search stub: No matching information found "
        "in the local sample knowledge base."
    )


# Only basic arithmetic operations are allowed.
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}


def safe_calculate(node):
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")

        left = safe_calculate(node.left)
        right = safe_calculate(node.right)

        if operator_type is ast.Pow and abs(right) > 10:
            raise ValueError("Exponent is too large.")

        result = ALLOWED_OPERATORS[operator_type](left, right)

        if abs(result) > 1e12:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in UNARY_OPERATORS:
            raise ValueError("Unary operator not allowed.")

        return UNARY_OPERATORS[operator_type](
            safe_calculate(node.operand)
        )

    raise ValueError("Unsupported mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression such as 18 * 7 or 144 / 12."""

    try:
        parsed_expression = ast.parse(expression, mode="eval")
        result = safe_calculate(parsed_expression.body)
        return str(result)
    except Exception as error:
        return f"Calculation error: {error}"


def run_agent_task(agent, tools_by_name, task):
    print("\nTask:", task)

    messages = [HumanMessage(content=task)]

    # Allow a limited number of tool-calling rounds.
    for _ in range(5):
        response = agent.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("Agent:", response.content)
            return

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name not in tools_by_name:
                raise ValueError(f"Unknown tool: {tool_name}")

            tool_result = tools_by_name[tool_name].invoke(tool_args)

            print(f"Tool used: {tool_name}")
            print("Tool result:", tool_result)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id
                )
            )

    print("Agent stopped after reaching the tool-call limit.")


def run_agent_demo():
    print("\n" + "=" * 60)
    print("PART 4: TWO-TOOL AGENT")
    print("=" * 60)

    tools = [web_search, calculator]
    tools_by_name = {
        selected_tool.name: selected_tool
        for selected_tool in tools
    }

    chat_model = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    agent = chat_model.bind_tools(tools)

    tasks = [
        "Use the web_search tool to find information about LangChain.",
        "Use the calculator tool to calculate 18 * 7.",
        "Use the calculator tool to calculate 144 / 12."
    ]

    for task in tasks:
        run_agent_task(agent, tools_by_name, task)


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():
    print("=" * 60)
    print("W6D5: DOCUMENT CHATBOT WITH LANGCHAIN")
    print("=" * 60)

    run_chain_demo()
    run_memory_demo()

    retriever = build_retriever()
    run_document_question(retriever)

    run_agent_demo()

    print("\n" + "=" * 60)
    print("W6D5 DOCUMENT CHATBOT EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
