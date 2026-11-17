
# W6D1: Build and test a basic LangChain chain.
# Flow: PromptTemplate -> Ollama LLM -> StrOutputParser

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


MODEL_NAME = "qwen2.5:3b"


def main():
    # 1. Create a reusable prompt template.
    prompt = PromptTemplate.from_template(
        """You are a helpful AI tutor.
Explain the following topic in simple words,
using a short and clear answer.

Topic: {question}
Answer:"""
    )

    # 2. Connect to the locally available Ollama model.
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2
    )

    # 3. Create an output parser.
    parser = StrOutputParser()

    # 4. Connect all three components into one chain.
    chain = prompt | llm | parser

    # 5. Test the chain with five inputs.
    test_questions = [
        "What is Python?",
        "Explain Machine Learning.",
        "What is a neural network?",
        "Explain cloud computing.",
        "What is Natural Language Processing?"
    ]

    print("\n========== LANGCHAIN CHAIN TEST ==========")

    for index, question in enumerate(test_questions, start=1):
        print(f"\nTest {index}")
        print(f"Question: {question}")

        answer = chain.invoke({"question": question})

        print(f"Answer: {answer}")
        print("-" * 60)

    print("\nChain testing completed: 5 inputs processed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nChain execution failed: {error}")
        print("Check whether Ollama is running and the model is available.")
