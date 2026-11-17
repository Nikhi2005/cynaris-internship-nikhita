
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


def main():
    llm = OllamaLLM(
        model="qwen2.5:3b",
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are a helpful AI assistant. "
            "Answer the following question clearly and briefly:\n"
            "{question}"
        )
    )

    chain = prompt | llm | StrOutputParser()

    test_inputs = [
        "What is LangChain?",
        "What is a language model?",
        "Explain Python in simple words.",
        "What is machine learning?",
        "What is the purpose of an AI agent?"
    ]

    print("=" * 60)
    print("W6D3 - LANGCHAIN CHAIN DEMO")
    print("=" * 60)

    for index, question in enumerate(test_inputs, start=1):
        print(f"\nTest {index}")
        print(f"Question: {question}")

        response = chain.invoke({"question": question})

        print(f"Answer: {response}")

    print("\nAll 5 chain tests completed.")


if __name__ == "__main__":
    main()
