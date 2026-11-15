
from pathlib import Path
from datetime import datetime
import csv
import json
import math
import hashlib

import numpy as np
import requests
import redis
import tiktoken

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
WEEK_DIR = PROJECT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
LLAMA_MODEL = "llama3.2:3b"
QWEN_MODEL = "qwen2.5:3b"

CHROMA_PATH = WEEK_DIR / "W5D4_Semantic_Search" / "chromadb_store"
CHROMA_COLLECTION = "w5d4_semantic_pdf_chunks"

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_PREFIX = "cynaris:w5d6:"

SIMILARITY_THRESHOLD = 0.92

# Lesson-provided GPT-4o pricing assumptions
INPUT_PRICE = 5.00
OUTPUT_PRICE = 15.00

# Illustrative exchange rate. Update for a formal audit.
USD_TO_INR = 82.0

TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4o")

CSV_FILE = OUTPUT_DIR / "cost_log.csv"
JSON_FILE = OUTPUT_DIR / "w5d6_results.json"
EXCEL_FILE = OUTPUT_DIR / "W5D6_Token_Cost_Audit.xlsx"


# ============================================================
# 2. TOKEN COUNTING AND COST CALCULATION
# ============================================================

def count_tokens(text):
    """Count tokens using GPT-4o's tiktoken encoding."""
    return len(TOKEN_ENCODER.encode(text))


def calculate_cost(input_tokens, output_tokens):
    """Calculate estimated GPT-4o-equivalent cost in USD and INR."""

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE

    total_usd = input_cost + output_cost
    total_inr = total_usd * USD_TO_INR

    return total_usd, total_inr


def log_usage(label, model, input_tokens, output_tokens,
              cache_status="MISS"):

    usd, inr = calculate_cost(input_tokens, output_tokens)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_gpt4o_usd": round(usd, 8),
        "estimated_gpt4o_inr": round(inr, 6),
        "cache_status": cache_status
    }

    file_exists = CSV_FILE.exists()

    with CSV_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(record.keys())
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)

    return record


# ============================================================
# 3. OLLAMA API FUNCTIONS
# ============================================================

def get_embeddings(texts):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["embeddings"]


def generate_answer(prompt, model=LLAMA_MODEL):

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 160,
                "temperature": 0.2
            }
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()


def verify_ollama_models():

    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=30
    )

    response.raise_for_status()

    installed_models = [
        model["name"]
        for model in response.json().get("models", [])
    ]

    required_models = [
        EMBED_MODEL,
        LLAMA_MODEL,
        QWEN_MODEL
    ]

    for model in required_models:

        found = any(
            name == model or name.startswith(model + ":")
            for name in installed_models
        )

        if not found:
            raise RuntimeError(
                f"Model not found: {model}. "
                f"Run ollama pull {model}"
            )

    print("All required Ollama models are available.")


# ============================================================
# 4. TASK 1 - TOKEN COUNTING FOR FIVE PROMPT SIZES
# ============================================================

def task1_token_counting():

    print("\n" + "=" * 55)
    print("TASK 1 - TOKEN COUNTING")
    print("=" * 55)

    base_text = (
        "Machine learning enables systems to learn patterns "
        "from data. A typical workflow includes data collection, "
        "preprocessing, feature engineering, model training, "
        "evaluation and deployment. Good evaluation checks "
        "whether a model generalises to unseen data. "
    )

    prompt_sizes = [25, 100, 250, 500, 1000]

    results = []

    words_in_base = base_text.split()

    for target_words in prompt_sizes:

        selected_words = (
            words_in_base *
            math.ceil(target_words / len(words_in_base))
        )[:target_words]

        prompt = (
            "Explain the following machine-learning notes:\n"
            + " ".join(selected_words)
        )

        result = {
            "words": target_words,
            "characters": len(prompt),
            "tokens": count_tokens(prompt)
        }

        results.append(result)

        print(
            f"{target_words:4} words | "
            f"{len(prompt):5} characters | "
            f"{result['tokens']:5} tokens"
        )

    return results


# ============================================================
# 5. TASK 2 - LLMLINGUA PROMPT COMPRESSION
# ============================================================

def retrieve_chroma_context(question):

    import chromadb

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB folder not found: {CHROMA_PATH}"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=CHROMA_COLLECTION
    )

    query_embedding = get_embeddings([question])[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(13, collection.count()),
        include=["documents"]
    )

    documents = result["documents"][0]

    if not documents:
        raise RuntimeError(
            "No documents found in the existing ChromaDB collection."
        )

    context = "\n\n".join(documents)

    prompt = (
        "Use the following retrieved notes to answer the question.\n\n"
        + context
    )

    # Prepare an approximately 2,000-token test prompt.
    # Repeated retrieved passages are used only to reach the
    # required test size when the stored context is shorter.

    while count_tokens(prompt) < 1950:
        prompt += "\n\nRetrieved passage:\n" + context

    encoded = TOKEN_ENCODER.encode(prompt)

    if len(encoded) > 2000:
        prompt = TOKEN_ENCODER.decode(encoded[:2000])

    return prompt


def task2_llmlingua():

    print("\n" + "=" * 55)
    print("TASK 2 - LLMLINGUA COMPRESSION")
    print("=" * 55)

    question = (
        "Explain the main stages of a machine-learning workflow."
    )

    original_prompt = retrieve_chroma_context(question)

    original_tokens = count_tokens(original_prompt)

    print("Original prompt tokens:", original_tokens)

    try:

        from llmlingua import PromptCompressor

        compressor = PromptCompressor(
            model_name=(
                "microsoft/"
                "llmlingua-2-bert-base-multilingual-cased-meetingbank"
            ),
            device_map="cpu",
            use_llmlingua2=True
        )

        compressed_result = compressor.compress_prompt(
            original_prompt,
            rate=0.4,
            force_tokens=["\n", "?"]
        )

        compressed_prompt = compressed_result[
            "compressed_prompt"
        ]

        compressed_tokens = count_tokens(compressed_prompt)

        reduction = (
            1 - compressed_tokens / original_tokens
        ) * 100

        print("Compressed prompt tokens:", compressed_tokens)
        print(f"Token reduction: {reduction:.2f}%")

        system_instruction = (
            "Answer only from the supplied context. "
            "If the context is insufficient, say so."
        )

        original_response = generate_answer(
            f"Context:\n{original_prompt}\n\n"
            f"Question: {question}\n{system_instruction}"
        )

        compressed_response = generate_answer(
            f"Context:\n{compressed_prompt}\n\n"
            f"Question: {question}\n{system_instruction}"
        )

        original_answer = original_response.get("response", "")
        compressed_answer = compressed_response.get("response", "")

        original_output_tokens = original_response.get(
            "eval_count",
            count_tokens(original_answer)
        )

        compressed_output_tokens = compressed_response.get(
            "eval_count",
            count_tokens(compressed_answer)
        )

        log_usage(
            "original_context",
            LLAMA_MODEL,
            original_response.get("prompt_eval_count", original_tokens),
            original_output_tokens
        )

        log_usage(
            "compressed_context",
            LLAMA_MODEL,
            compressed_response.get("prompt_eval_count", compressed_tokens),
            compressed_output_tokens
        )

        print("\nORIGINAL ANSWER:\n")
        print(original_answer)

        print("\nCOMPRESSED ANSWER:\n")
        print(compressed_answer)

        print(
            "\nReview both answers manually for factual coverage "
            "and missing details."
        )

        return {
            "status": "completed",
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "reduction_percent": round(reduction, 2),
            "original_output_tokens": original_output_tokens,
            "compressed_output_tokens": compressed_output_tokens,
            "original_answer": original_answer,
            "compressed_answer": compressed_answer
        }

    except Exception as error:

        print(
            "LLMLingua compression could not complete:",
            type(error).__name__,
            str(error)
        )

        print(
            "No compression result will be claimed. "
            "Other practical tasks will continue."
        )

        return {
            "status": "needs_attention",
            "original_tokens": original_tokens,
            "error": f"{type(error).__name__}: {error}"
        }


# ============================================================
# 6. TASK 3 - SEMANTIC CACHE USING REDIS
# ============================================================

QA_PAIRS = [

    ("What is machine learning?",
     "Machine learning enables systems to learn patterns from data."),

    ("Define supervised learning.",
     "Supervised learning trains on labelled examples."),

    ("What is unsupervised learning?",
     "Unsupervised learning discovers patterns in unlabelled data."),

    ("Explain reinforcement learning.",
     "Reinforcement learning learns through actions and rewards."),

    ("What is overfitting?",
     "Overfitting occurs when a model memorises training data and generalises poorly."),

    ("Define underfitting.",
     "Underfitting occurs when a model is too simple to learn data patterns."),

    ("What is a training dataset?",
     "A training dataset is used to fit a machine-learning model."),

    ("What is a test dataset?",
     "A test dataset evaluates a model on unseen examples."),

    ("Explain feature engineering.",
     "Feature engineering creates or transforms useful input variables."),

    ("What is classification?",
     "Classification predicts a category or class label."),

    ("What is regression?",
     "Regression predicts a continuous numerical value."),

    ("What is clustering?",
     "Clustering groups similar data points together."),

    ("What is an embedding?",
     "An embedding represents text as a numerical vector."),

    ("What is cosine similarity?",
     "Cosine similarity compares the direction of two vectors."),

    ("What is a vector database?",
     "A vector database stores and searches vector representations."),

    ("What is RAG?",
     "RAG retrieves relevant information and supplies it to a generative model."),

    ("What is a prompt?",
     "A prompt is the input or instruction supplied to a language model."),

    ("What is tokenisation?",
     "Tokenisation splits text into units that a model processes."),

    ("What is model evaluation?",
     "Model evaluation measures performance using suitable metrics."),

    ("What is data preprocessing?",
     "Data preprocessing cleans and transforms raw data before modelling.")
]


class SemanticCache:

    def __init__(self):

        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5
        )

        self.client.ping()

    @staticmethod
    def cosine_similarity(vector_a, vector_b):

        a = np.asarray(vector_a, dtype=np.float32)
        b = np.asarray(vector_b, dtype=np.float32)

        denominator = (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b) / denominator
        )

    def store(self, question, answer, embedding):

        key_hash = hashlib.sha256(
            question.encode("utf-8")
        ).hexdigest()

        key = REDIS_PREFIX + key_hash

        payload = {
            "question": question,
            "answer": answer,
            "embedding": embedding
        }

        self.client.set(
            key,
            json.dumps(payload),
            ex=86400
        )

    def search(self, query, embedding):

        best_item = None
        best_score = -1.0

        for key in self.client.scan_iter(
            match=REDIS_PREFIX + "*"
        ):

            raw_value = self.client.get(key)

            if not raw_value:
                continue

            item = json.loads(raw_value)

            score = self.cosine_similarity(
                embedding,
                item["embedding"]
            )

            if score > best_score:
                best_score = score
                best_item = item

        if (
            best_item is not None
            and best_score >= SIMILARITY_THRESHOLD
        ):

            return {
                "cache_hit": True,
                "similarity": round(best_score, 4),
                "matched_question": best_item["question"],
                "answer": best_item["answer"]
            }

        return {
            "cache_hit": False,
            "best_similarity": round(best_score, 4)
        }

    def clear_demo_entries(self):

        keys = list(
            self.client.scan_iter(
                match=REDIS_PREFIX + "*"
            )
        )

        if keys:
            self.client.delete(*keys)


def task3_semantic_cache():

    print("\n" + "=" * 55)
    print("TASK 3 - REDIS SEMANTIC CACHE")
    print("=" * 55)

    cache = SemanticCache()

    # Clears only this W5D6 namespace.
    # Other Redis data remains untouched.

    cache.clear_demo_entries()

    questions = [
        question for question, answer in QA_PAIRS
    ]

    embeddings = get_embeddings(questions)

    for (question, answer), embedding in zip(
        QA_PAIRS,
        embeddings
    ):

        cache.store(
            question,
            answer,
            embedding
        )

    print(
        f"Successfully stored {len(QA_PAIRS)} Q&A pairs."
    )

    test_query = QA_PAIRS[0][0]

    test_embedding = get_embeddings(
        [test_query]
    )[0]

    result = cache.search(
        test_query,
        test_embedding
    )

    print("Test query:", test_query)
    print("Cache result:", result)

    return {
        "pairs_stored": len(QA_PAIRS),
        "threshold": SIMILARITY_THRESHOLD,
        "test_query": test_query,
        "lookup_result": result
    }


# ============================================================
# 7. TASK 4 - EXCEL COST AUDIT
# ============================================================

def create_cost_audit(
    token_results,
    compression_results,
    cache_results,
    routing_results
):

    workbook = Workbook()

    token_sheet = workbook.active
    token_sheet.title = "Token Audit"

    token_sheet.append([
        "Prompt Size (Words)",
        "Characters",
        "Token Count"
    ])

    for item in token_results:

        token_sheet.append([
            item["words"],
            item["characters"],
            item["tokens"]
        ])

    cost_sheet = workbook.create_sheet("Cost Summary")

    cost_sheet.append([
        "Metric",
        "Value",
        "Notes"
    ])

    cost_sheet.append([
        "GPT-4o Input Rate USD / 1M",
        INPUT_PRICE,
        "Lesson-provided rate"
    ])

    cost_sheet.append([
        "GPT-4o Output Rate USD / 1M",
        OUTPUT_PRICE,
        "Lesson-provided rate"
    ])

    cost_sheet.append([
        "USD to INR",
        USD_TO_INR,
        "Illustrative exchange-rate assumption"
    ])

    for key in [
        "status",
        "original_tokens",
        "compressed_tokens",
        "reduction_percent"
    ]:

        cost_sheet.append([
            key,
            compression_results.get(key, "Not available"),
            "Measured LLMLingua result"
        ])

    if compression_results.get("status") == "completed":

        baseline_usd, baseline_inr = calculate_cost(
            compression_results["original_tokens"],
            compression_results["original_output_tokens"]
        )

        optimised_usd, optimised_inr = calculate_cost(
            compression_results["compressed_tokens"],
            compression_results["compressed_output_tokens"]
        )

        cost_sheet.append([
            "Baseline estimated cost USD",
            round(baseline_usd, 8),
            "Original prompt and response"
        ])

        cost_sheet.append([
            "Optimised estimated cost USD",
            round(optimised_usd, 8),
            "Compressed prompt and response"
        ])

        cost_sheet.append([
            "Baseline estimated cost INR",
            round(baseline_inr, 6),
            "Illustrative cloud equivalent"
        ])

        cost_sheet.append([
            "Optimised estimated cost INR",
            round(optimised_inr, 6),
            "Illustrative cloud equivalent"
        ])

        cost_sheet.append([
            "Estimated cost reduction %",
            round(
                (1 - optimised_inr / max(baseline_inr, 1e-12)) * 100,
                2
            ),
            "Based on measured token counts"
        ])

    cost_sheet.append([
        "Redis Q&A Pairs",
        cache_results["pairs_stored"],
        "Stored locally"
    ])

    cost_sheet.append([
        "Cosine Similarity Threshold",
        SIMILARITY_THRESHOLD,
        "Required practical threshold"
    ])

    routing_sheet = workbook.create_sheet("Model Routing")

    routing_sheet.append([
        "Query",
        "Selected Route",
        "Reason"
    ])

    for item in routing_results:

        routing_sheet.append([
            item["query"],
            item["route"],
            item["reason"]
        ])

    # Basic spreadsheet formatting

    for sheet in workbook.worksheets:

        sheet.freeze_panes = "A2"

        for cell in sheet[1]:

            cell.font = Font(
                bold=True,
                color="FFFFFF"
            )

            cell.fill = PatternFill(
                "solid",
                fgColor="244062"
            )

        for column in sheet.columns:

            maximum_length = max(
                len(str(cell.value or ""))
                for cell in column
            )

            sheet.column_dimensions[
                get_column_letter(column[0].column)
            ].width = min(maximum_length + 4, 60)

    workbook.save(EXCEL_FILE)

    print("\nExcel cost audit saved:")
    print(EXCEL_FILE)


# ============================================================
# 8. TASK 5 - MODEL ROUTING
# ============================================================

def choose_model_route(query):

    query_lower = query.lower()

    complex_terms = [
        "critically analyse",
        "design an architecture",
        "multiple constraints",
        "prove",
        "research limitations"
    ]

    medium_terms = [
        "explain",
        "compare",
        "differentiate",
        "workflow",
        "summarise"
    ]

    if any(term in query_lower for term in complex_terms):

        return {
            "query": query,
            "route": "Cloud API (proposed only)",
            "reason": (
                "Complex reasoning may benefit from a stronger model. "
                "No cloud API call is made in this practical."
            )
        }

    if any(term in query_lower for term in medium_terms):

        return {
            "query": query,
            "route": QWEN_MODEL,
            "reason": (
                "Use the local Qwen model for a structured explanation."
            )
        }

    return {
        "query": query,
        "route": LLAMA_MODEL,
        "reason": (
            "Use the local Llama model for a straightforward question."
        )
    }


def create_routing_document():

    routing_examples = [
        choose_model_route("What is a Python list?"),
        choose_model_route(
            "Explain the machine-learning workflow."
        ),
        choose_model_route(
            "Critically analyse the limitations of this research method."
        )
    ]

    lines = [
        "# W5D6 Model Routing Strategy",
        "",
        "| Query | Route | Reason |",
        "|---|---|---|"
    ]

    for item in routing_examples:

        lines.append(
            f"| {item['query']} | "
            f"{item['route']} | "
            f"{item['reason']} |"
        )

    lines.extend([
        "",
        "## Routing Rules",
        "",
        "- Repeated similar queries: check Redis first.",
        "- Cache hit: reuse answer only when similarity is >= 0.92.",
        f"- Simple questions: {LLAMA_MODEL}.",
        f"- Medium-complexity questions: {QWEN_MODEL}.",
        "- Complex questions: cloud model may be considered after budget and privacy checks.",
        "- No paid cloud API calls are made by this practical.",
        "",
        "## Limitations",
        "",
        "- This is a beginner-friendly keyword-based router.",
        "- It is not a trained complexity classifier.",
        "- Local inference still consumes hardware resources and electricity."
    ])

    document_path = PROJECT_DIR / "routing_strategy.md"

    document_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nRouting strategy saved:")
    print(document_path)

    return routing_examples


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("CYNARIS SOLUTIONS - W5D6")
    print("LLM COST OPTIMISATION")
    print("=" * 60)

    verify_ollama_models()

    # Task 1
    token_results = task1_token_counting()

    # Task 2
    compression_results = task2_llmlingua()

    # Task 3
    cache_results = task3_semantic_cache()

    # Task 5
    print("\n" + "=" * 55)
    print("TASK 5 - MODEL ROUTING")
    print("=" * 55)

    routing_results = create_routing_document()

    for item in routing_results:

        print(
            item["query"],
            "->",
            item["route"]
        )

    # Task 4
    create_cost_audit(
        token_results,
        compression_results,
        cache_results,
        routing_results
    )

    # Save complete results

    final_results = {
        "token_counting": token_results,
        "llmlingua_compression": compression_results,
        "semantic_cache": cache_results,
        "model_routing": routing_results,
        "pricing_assumptions": {
            "input_usd_per_million": INPUT_PRICE,
            "output_usd_per_million": OUTPUT_PRICE,
            "usd_to_inr": USD_TO_INR,
            "note": (
                "Estimated GPT-4o-equivalent costs only. "
                "Local Ollama inference is not billed per token."
            )
        }
    }

    JSON_FILE.write_text(
        json.dumps(
            final_results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("W5D6 EXECUTION FINISHED")
    print("=" * 60)

    print("\nGenerated files:")
    print("1.", JSON_FILE)
    print("2.", CSV_FILE)
    print("3.", EXCEL_FILE)

    print(
        "\nReview the original and compressed answers manually "
        "before claiming quality was preserved."
    )


if __name__ == "__main__":
    main()
