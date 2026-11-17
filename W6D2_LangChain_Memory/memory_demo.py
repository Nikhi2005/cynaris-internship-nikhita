
# W6D2: ConversationBufferMemory and conversation history

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory

MODEL_NAME = "qwen2.5:3b"


def main():
    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant.
Use the conversation history to answer the user's questions.
Remember details shared in earlier messages.

Conversation History:
{history}

User: {input}
Assistant:"""
    )

    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        output_key="text",
        return_messages=False
    )

    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        output_key="text",
        verbose=False
    )

    conversations = [
        "I am developing a chatbot for my AI project.",
        "I am using Python and LangChain to build it.",
        "What application am I developing?",
        "Which programming language and framework am I using?",
        "Summarize the important details I shared about my project."
    ]

    print("=" * 55)
    print("W6D2 - CONVERSATION MEMORY")
    print("=" * 55)

    for turn, message in enumerate(conversations, start=1):
        print(f"\n--- Conversation Turn {turn} ---")
        print("User:", message)

        response = conversation_chain.invoke({
            "input": message
        })

        print("Assistant:", response["text"])

    # Retrieve the complete stored conversation
    history = memory.load_memory_variables({})["history"]
    stored_messages = memory.chat_memory.messages

    print("\n" + "=" * 55)
    print("CONVERSATION HISTORY VERIFICATION")
    print("=" * 55)

    print("\nStored Conversation:")
    print(history)

    user_messages = len([
        message for message in stored_messages
        if message.type == "human"
    ])

    assistant_messages = len([
        message for message in stored_messages
        if message.type == "ai"
    ])

    print("\nVerification Results:")
    print("User messages:", user_messages)
    print("Assistant messages:", assistant_messages)
    print("Total stored messages:", len(stored_messages))

    if user_messages == 5 and assistant_messages == 5:
        print("\nSUCCESS: All five conversation turns were stored.")
    else:
        print("\nWARNING: Conversation history verification failed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Memory execution failed:", error)
