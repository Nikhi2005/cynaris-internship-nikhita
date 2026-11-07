
# W5D2: Prompt Engineering & System Prompts with Ollama
# Demonstrates role-based prompting, reusable prompt templates,
# and few-shot prompting using two locally running Ollama models.

import json
import requests
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

OUTPUT_FILE = Path(__file__).resolve().parent / "outputs_w5d2.json"

# --------------------------------------------------
# 2. System Prompt (Role-Based Prompting)
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a senior AI/ML mentor teaching undergraduate
Computer Science engineering students.

Explain technical concepts in simple and clear language.
Use relevant examples and keep answers well structured.
"""

# --------------------------------------------------
# 3. Reusable Prompt Template
# --------------------------------------------------

PROMPT_TEMPLATE = """
Learn the expected answering style from these examples.

Example 1:
Question: What is a transformer in NLP?
Answer: A transformer is a neural network architecture
that uses self-attention to process sequence data.

Example 2:
Question: What is few-shot prompting?
Answer: Few-shot prompting provides a few examples
inside a prompt to guide the model's response.

Now answer the following question using a similar
clear and informative style.

Question: {question}

Answer:
"""

# --------------------------------------------------
# 4. Test Questions
# --------------------------------------------------

QUESTIONS = [
    "How does back-propagation work in a neural network?",
    "What are the advantages of quantisation for LLMs?",
    "What is the difference between supervised learning and unsupervised learning?"
]

# --------------------------------------------------
# 5. Function to Call Ollama API
# --------------------------------------------------

def query_ollama(model, question):

    user_prompt = PROMPT_TEMPLATE.format(question=question)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return {
        "response": result["message"]["content"],
        "model": result.get("model", model)
    }

# --------------------------------------------------
# 6. Main Execution
# --------------------------------------------------

def main():

    print("=" * 65)
    print("W5D2 - PROMPT ENGINEERING WITH OLLAMA")
    print("=" * 65)

    print("\nPrompt Engineering Techniques:")
    print("1. Role-Based Prompting")
    print("2. Reusable Prompt Template")
    print("3. Few-Shot Prompting")

    results = {}

    for model in MODELS:

        print("\n" + "=" * 65)
        print(f"MODEL: {model}")
        print("=" * 65)

        results[model] = []

        for index, question in enumerate(QUESTIONS, start=1):

            print(f"\nQuestion {index}: {question}")

            try:
                answer = query_ollama(model, question)

                results[model].append({
                    "question": question,
                    "response": answer["response"],
                    "status": "success"
                })

                print("\nResponse:")
                print(answer["response"])

            except requests.exceptions.RequestException as error:

                error_message = str(error)

                results[model].append({
                    "question": question,
                    "response": None,
                    "status": "failed",
                    "error": error_message
                })

                print(f"\nAPI Error: {error_message}")

            except (KeyError, ValueError) as error:

                error_message = str(error)

                results[model].append({
                    "question": question,
                    "response": None,
                    "status": "failed",
                    "error": error_message
                })

                print(f"\nResponse Processing Error: {error_message}")

    # --------------------------------------------------
    # 7. Save Results to JSON
    # --------------------------------------------------

    output_data = {
        "task": "W5D2 - Prompt Engineering & System Prompts",
        "techniques": [
            "Role-Based Prompting",
            "Reusable Prompt Template",
            "Few-Shot Prompting"
        ],
        "system_prompt": SYSTEM_PROMPT.strip(),
        "questions": QUESTIONS,
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("EXECUTION COMPLETED")
    print("=" * 65)

    print(f"\nJSON results saved to:\n{OUTPUT_FILE}")

    print("\nCheck the model responses and compare their quality.")
    print("Verify that both models completed the questions successfully.")


# --------------------------------------------------
# 8. Program Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()
