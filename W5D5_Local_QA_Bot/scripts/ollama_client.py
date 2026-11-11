import time
import requests

from config import (
    OLLAMA_HOST,
    REQUEST_TIMEOUT,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS
)


def check_ollama_connection():
    """
    Verify that the Ollama server is running.
    """

    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        ) from error


def check_models(required_models):
    """
    Verify that the required models are installed.
    """

    data = check_ollama_connection()

    installed_models = {
        model["name"]
        for model in data.get("models", [])
    }

    missing_models = [
        model
        for model in required_models
        if model not in installed_models
    ]

    if missing_models:
        raise RuntimeError(
            "The following Ollama models are missing: "
            + ", ".join(missing_models)
            + ". Check the output of 'ollama list'."
        )

    print("\nOllama connection successful.")

    for model in required_models:
        print(f"Model available: {model}")


def generate_answer(
    question,
    context,
    system_prompt,
    model
):
    """
    Generate an answer using Ollama's local API.

    Returns:
        answer
        response_time
    """

    url = f"{OLLAMA_HOST}/api/generate"

    full_prompt = f"""
### Retrieved Context

{context}

### User Question

{question}

### Instructions

Answer the question using the retrieved context.
If the context does not contain the answer,
clearly state that the information is unavailable.
"""

    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_OUTPUT_TOKENS
        }
    }

    start_time = time.perf_counter()

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        response_time = (
            time.perf_counter() - start_time
        )

        return answer, round(response_time, 2)

    except requests.RequestException as error:
        raise RuntimeError(
            f"Ollama generation failed for {model}: {error}"
        ) from error