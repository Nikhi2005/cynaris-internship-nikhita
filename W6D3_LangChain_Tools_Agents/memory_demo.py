
from langchain_ollama import OllamaLLM
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


def main():
    llm = OllamaLLM(
        model="qwen2.5:3b",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question"
    )

    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "You are a helpful AI assistant.\n\n"
            "Previous conversation:\n"
            "{chat_history}\n\n"
            "User: {question}\n"
            "Assistant:"
        )
    )

    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    questions = [
        "My name is Nikhita.",
        "I am learning Python and artificial intelligence.",
        "What programming language am I learning?",
        "I am also exploring LangChain.",
        "What are the three things I mentioned learning or exploring?"
    ]

    print("=" * 60)
    print("W6D3 - CONVERSATION MEMORY DEMO")
    print("=" * 60)

    for index, question in enumerate(questions, start=1):
        print(f"\nTurn {index}")
        print(f"User: {question}")

        response = conversation.invoke({"question": question})

        print(f"Assistant: {response['text']}")

    print("\n" + "=" * 60)
    print("MEMORY VERIFICATION")
    print("=" * 60)

    messages = memory.chat_memory.messages

    print(f"Total stored messages: {len(messages)}")
    print("\nStored conversation:")

    for message in messages:
        print(f"{message.type}: {message.content}")

    print("\nConversation memory test completed.")


if __name__ == "__main__":
    main()
