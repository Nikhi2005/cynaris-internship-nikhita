
import requests
import json
from datetime import datetime

# Ollama API configuration
API_URL = "http://localhost:11434/api/chat"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

# Custom system prompt
SYSTEM_PROMPT = """
You are a helpful AI assistant for computer science students.
Give accurate, clear, and easy-to-understand answers.
Use simple language and practical examples.
Keep your answers well-structured and concise.
"""

# Five prompts for testing both models
PROMPTS = [
    "Explain artificial intelligence with a real-life example.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how a neural network works in simple words.",
    "What are the main challenges of AI adoption in Indian MSMEs?",
    "Suggest a computer vision project idea for a BTech final-year student."
]


def get_response(model, prompt):
    """Send a prompt to a local Ollama model and return its response."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    result = response.json()

    return result["message"]["content"]


def main():
    results = []

    print("=" * 65)
    print("LOCAL LLM INFERENCE AND MODEL COMPARISON")
    print("=" * 65)

    for model in MODELS:
        print(f"\nTesting model: {model}")

        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"\nPrompt {index}: {prompt}")

            try:
                answer = get_response(model, prompt)

                print("\nResponse:")
                print(answer)

                results.append({
                    "model": model,
                    "prompt_number": index,
                    "prompt": prompt,
                    "response": answer
                })

            except requests.exceptions.RequestException as error:
                print(f"API error for {model}: {error}")

                results.append({
                    "model": model,
                    "prompt_number": index,
                    "prompt": prompt,
                    "error": str(error)
                })

    # Save actual responses
    output_file = "ollama_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("TESTING COMPLETED")
    print(f"Results saved to: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
