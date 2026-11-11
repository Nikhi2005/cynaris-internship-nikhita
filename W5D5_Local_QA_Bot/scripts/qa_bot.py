import json
from datetime import datetime

from config import (
    DEFAULT_MODEL,
    COMPARISON_MODEL,
    AVAILABLE_MODELS,
    BASE_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    PROMPT_TESTS,
    COMPARISON_QUESTIONS,
    RESULTS_FILE
)

from vector_store import (
    load_collection,
    get_relevant_chunks,
    format_context
)

from ollama_client import (
    check_models,
    generate_answer
)


# --------------------------------------------------
# RESULTS STORAGE
# --------------------------------------------------

results = {
    "project": "W5D5 Local Q&A Bot",
    "created_at": datetime.now().isoformat(),
    "prompt_tests": [],
    "model_comparison": [],
    "interactive_questions": []
}


def save_results():
    """
    Save all practical results into a JSON file.
    """

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results["last_updated"] = datetime.now().isoformat()

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nResults saved to: {RESULTS_FILE}")


# --------------------------------------------------
# DISPLAY RETRIEVED CHUNKS
# --------------------------------------------------

def display_retrieved_chunks(chunks):

    print("\nRetrieved PDF chunks:")

    if not chunks:
        print("No relevant chunks found.")
        return

    for index, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {index}")

        print(
            f"Distance: {chunk['distance']}"
        )

        print(
            f"Text: {chunk['text'][:180]}..."
        )


# --------------------------------------------------
# FIVE PROMPT TESTS
# --------------------------------------------------

def run_prompt_tests(collection):

    print("\n" + "=" * 65)
    print("W5D5 - FIVE PROMPT TESTS")
    print("=" * 65)

    for index, test in enumerate(PROMPT_TESTS, start=1):

        question = test["question"]
        style = test["style"]
        system_prompt = SYSTEM_PROMPTS[style]

        print(f"\nTest {index}: {test['name']}")
        print(f"Question: {question}")

        try:
            chunks = get_relevant_chunks(
                collection,
                question
            )

            context = format_context(chunks)

            display_retrieved_chunks(chunks)

            answer, response_time = generate_answer(
                question=question,
                context=context,
                system_prompt=system_prompt,
                model=DEFAULT_MODEL
            )

            print("\nGenerated Answer:")
            print(answer)

            print(
                f"\nResponse Time: {response_time} seconds"
            )

            results["prompt_tests"].append({
                "test_number": index,
                "test_name": test["name"],
                "prompt_style": style,
                "question": question,
                "model": DEFAULT_MODEL,
                "retrieved_chunks": len(chunks),
                "answer": answer,
                "response_time_seconds": response_time,
                "manual_review": {
                    "accuracy": None,
                    "relevance": None,
                    "clarity": None,
                    "hallucination": None,
                    "observations": ""
                }
            })

            save_results()

        except Exception as error:
            print(f"\nTest failed: {error}")


# --------------------------------------------------
# COMPARE LLAMA AND QWEN
# --------------------------------------------------

def run_model_comparison(collection):

    print("\n" + "=" * 65)
    print("W5D5 - MODEL COMPARISON")
    print("=" * 65)

    for index, question in enumerate(
        COMPARISON_QUESTIONS,
        start=1
    ):

        print(f"\nComparison Question {index}")
        print(f"Question: {question}")

        try:
            # Retrieve context only once.
            # Both models receive identical context.

            chunks = get_relevant_chunks(
                collection,
                question
            )

            context = format_context(chunks)

            display_retrieved_chunks(chunks)

            for model in AVAILABLE_MODELS:

                print("\n" + "-" * 50)
                print(f"Model: {model}")

                answer, response_time = generate_answer(
                    question=question,
                    context=context,
                    system_prompt=BASE_SYSTEM_PROMPT,
                    model=model
                )

                print("\nGenerated Answer:")
                print(answer)

                print(
                    f"\nResponse Time: {response_time} seconds"
                )

                results["model_comparison"].append({
                    "question_number": index,
                    "question": question,
                    "model": model,
                    "retrieved_chunks": len(chunks),
                    "answer": answer,
                    "response_time_seconds": response_time,
                    "manual_review": {
                        "accuracy": None,
                        "relevance": None,
                        "clarity": None,
                        "hallucination": None,
                        "observations": ""
                    }
                })

                save_results()

        except Exception as error:
            print(f"\nComparison failed: {error}")


# --------------------------------------------------
# INTERACTIVE Q&A BOT
# --------------------------------------------------

def interactive_qa(collection):

    print("\n" + "=" * 65)
    print("LOCAL Q&A BOT")
    print("=" * 65)

    print("Ask questions about the indexed PDF.")
    print("Type 'exit', 'quit' or 'q' to stop.")

    while True:

        try:
            question = input(
                "\nEnter your question: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
                "q"
            }:
                print("\nExiting Local Q&A Bot.")
                break

            if not question:
                print("Please enter a valid question.")
                continue

            try:
                chunks = get_relevant_chunks(
                    collection,
                    question
                )

                context = format_context(chunks)

                print("\nRetrieving information...")
                display_retrieved_chunks(chunks)

                answer, response_time = generate_answer(
                    question=question,
                    context=context,
                    system_prompt=BASE_SYSTEM_PROMPT,
                    model=DEFAULT_MODEL
                )

                print("\nBot Answer:")
                print(answer)

                print(
                    f"\nResponse Time: {response_time} seconds"
                )

                results["interactive_questions"].append({
                    "question": question,
                    "model": DEFAULT_MODEL,
                    "retrieved_chunks": len(chunks),
                    "answer": answer,
                    "response_time_seconds": response_time
                })

                save_results()

            except Exception as error:
                print(f"\nCould not answer the question: {error}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nQ&A Bot stopped.")
            break


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("CYNARIS SOLUTIONS - WEEK 5 DAY 5")
    print("LOCAL Q&A BOT USING OLLAMA + CHROMADB")
    print("=" * 65)

    try:
        # Step 1: Verify Ollama models

        print("\nStep 1: Checking Ollama models...")

        check_models(AVAILABLE_MODELS)

        # Step 2: Connect to existing ChromaDB

        print("\nStep 2: Connecting to ChromaDB...")

        collection = load_collection()

        # Step 3: Run five prompt tests

        run_prompt_tests(collection)

        # Step 4: Compare both models

        run_model_comparison(collection)

        # Step 5: Save results

        save_results()

        # Step 6: Start interactive bot

        interactive_qa(collection)

        # Final save

        save_results()

        print("\n" + "=" * 65)
        print("W5D5 EXECUTION COMPLETED")
        print("=" * 65)

        print("\nCompleted:")
        print("1. Ollama model verification")
        print("2. Existing ChromaDB connection")
        print("3. Five prompt tests")
        print("4. Llama vs Qwen comparison")
        print("5. Interactive Local Q&A Bot")
        print("6. JSON results saved")

    except Exception as error:

        print("\nW5D5 execution failed.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()