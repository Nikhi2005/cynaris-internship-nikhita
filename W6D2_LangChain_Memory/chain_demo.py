
# W6D2: LangChain chain using PromptTemplate, OllamaLLM and OutputParser

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

MODEL_NAME = "qwen2.5:3b"


def main():
    prompt = PromptTemplate.from_template(
        """You are a helpful AI tutor.
Explain the following concept in simple and clear language.

Question: {question}
Answer:"""
    )

    llm = OllamaLLM(model=MODEL_NAME, temperature=0.2)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    questions = [
        "What is conversation history in a chatbot?",
        "Why is context important in AI conversations?",
        "What is the purpose of LangChain?",
        "What is the difference between memory and a prompt?",
        "How does an AI chatbot remember previous messages?"
    ]

    print("=" * 55)
    print("W6D2 - LANGCHAIN CHAIN")
    print("=" * 55)

    for number, question in enumerate(questions, start=1):
        print(f"\nQuestion {number}: {question}")

        answer = chain.invoke({"question": question})

        print("Answer:", answer)

    print("\nSUCCESS: 5 inputs processed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Chain execution failed:", error)
