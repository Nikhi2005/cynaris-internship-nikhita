from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WEEK_DIR = PROJECT_DIR.parent

W5D4_DIR = WEEK_DIR / "W5D4_Semantic_Search"

CHROMA_DB_PATH = W5D4_DIR / "chromadb_store"

# Existing W5D4 PDF collection
COLLECTION_NAME = "w5d4_semantic_pdf_chunks"


# --------------------------------------------------
# OLLAMA CONFIGURATION
# --------------------------------------------------

OLLAMA_HOST = "http://localhost:11434"

EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_MODEL = "llama3.2:3b"
COMPARISON_MODEL = "qwen2.5:3b"

AVAILABLE_MODELS = [
    DEFAULT_MODEL,
    COMPARISON_MODEL
]


# --------------------------------------------------
# RETRIEVAL CONFIGURATION
# --------------------------------------------------

TOP_K = 3
MAX_WORDS_PER_CHUNK = 250


# --------------------------------------------------
# GENERATION CONFIGURATION
# --------------------------------------------------

TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 500

REQUEST_TIMEOUT = 300


# --------------------------------------------------
# SYSTEM PROMPTS
# --------------------------------------------------

BASE_SYSTEM_PROMPT = """
You are a helpful educational assistant.

Answer the user's question using only the
information provided in the retrieved context.

If the answer is not available in the context,
clearly state that you do not have enough
information.

Do not invent facts.

Keep your answers accurate, clear and relevant.
"""


SYSTEM_PROMPTS = {
    "concise": """
You are an educational assistant.
Answer using only the provided context.
Give a short and direct answer.
If information is missing, say so.
""",

    "step_by_step": """
You are an educational tutor.
Explain the answer using the provided context.
Present the explanation in clear numbered steps.
Do not invent information.
""",

    "bullet_points": """
You are an educational assistant.
Answer using only the provided context.
Organize the answer into easy-to-understand
bullet points.
""",

    "analogy": """
You are a friendly teacher.
Explain the answer using the provided context.
Use a simple everyday analogy when appropriate.
Do not introduce unsupported facts.
""",

    "detailed": """
You are a knowledgeable educational assistant.
Provide a detailed but understandable explanation
using only the provided context.
Clearly state if the context does not contain
the required information.
"""
}


# --------------------------------------------------
# FIVE PROMPT TESTS
# --------------------------------------------------

PROMPT_TESTS = [
    {
        "name": "Concise Explanation",
        "style": "concise",
        "question": "What is machine learning?"
    },
    {
        "name": "Step-by-Step Explanation",
        "style": "step_by_step",
        "question": "Explain the machine learning workflow."
    },
    {
        "name": "Bullet Point Answer",
        "style": "bullet_points",
        "question": "What are the important types of machine learning?"
    },
    {
        "name": "Analogy-Based Explanation",
        "style": "analogy",
        "question": "Explain supervised learning in simple terms."
    },
    {
        "name": "Detailed Explanation",
        "style": "detailed",
        "question": "Explain the difference between supervised and unsupervised learning."
    }
]


# --------------------------------------------------
# MODEL COMPARISON QUESTIONS
# --------------------------------------------------

COMPARISON_QUESTIONS = [
    "What is overfitting?",
    "Explain the bias-variance trade-off.",
    "How does the k-nearest neighbors algorithm work?"
]


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

RESULTS_FILE = PROJECT_DIR / "qa_results.json"