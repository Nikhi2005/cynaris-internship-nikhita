
# W6D1: Demonstrate ConversationBufferMemory.
# The program stores and recalls conversation history across five turns.

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory


MODEL_NAME = "qwen2.5:3b"


def main():
    # 1. Create the Ollama language model.
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2
    )

    # 2. Create a prompt containing conversation history.
    prompt = PromptTemplate.from_template(
        """You are a helpful conversational assistant.
Use the conversation history to answer questions.
Do not guess personal information that is not provided.

Conversation History:
{history}

Human: {input}
Assistant:"""
    )

    # 3. Create conversation memory.
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        output_key="text",
        return_messages=False
    )

    # 4. Connect the LLM, prompt and memory.
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    # 5. Five conversation turns.
    conversation_inputs = [
        "My name is Nikhita.",
        "I am learning Python for AI and Machine Learning.",
        "Which programming language am I learning?",
        "What is my name?",
        "Summarize my name and what I am learning."
    ]

    print("\n========== CONVERSATION MEMORY TEST ==========")

    for turn, message in enumerate(conversation_inputs, start=1):
        print(f"\nTurn {turn}")
        print(f"Human: {message}")

        response = conversation_chain.invoke(
            {"input": message}
        )

        print(f"Assistant: {response['text']}")
        print("-" * 60)

    # 6. Inspect the stored conversation history.
    history = memory.load_memory_variables({})["history"]

    print("\n========== STORED CONVERSATION HISTORY ==========")
    print(history)

    # 7. Verify that five exchanges were stored.
    stored_messages = memory.chat_memory.messages

    print("\n========== MEMORY VERIFICATION ==========")
    print(f"Expected conversation turns: 5")
    print(f"Stored user messages: {len(stored_messages) // 2}")

    if len(stored_messages) == 10:
        print("SUCCESS: All five conversation exchanges were stored.")
    else:
        print("WARNING: Conversation history count does not match.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nMemory execution failed: {error}")
        print("Check the LangChain packages and Ollama connection.")
