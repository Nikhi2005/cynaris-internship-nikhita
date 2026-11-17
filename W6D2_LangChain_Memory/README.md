# W6D2 - LangChain Memory & Conversation History

## 1. Objective

To understand LangChain chains, conversation memory, and tool-based agents using a locally running Ollama language model.

This practical demonstrates how to:

- Build a LangChain chain using PromptTemplate, OllamaLLM, and StrOutputParser.
- Maintain conversation history using ConversationBufferMemory.
- Create a simple agent workflow using WebSearch and Calculator tools.

## 2. Environment and Tools

- Language: Python
- Framework: LangChain
- Local LLM: Ollama
- Model: qwen2.5:3b
- IDE: Visual Studio Code

## 3. Project Structure

```text
W6D2_LangChain_Memory/
│
├── chain_demo.py
├── memory_demo.py
├── react_agent.py
└── README.md
```

## 4. Program 1 - LangChain Chain

### Description

The program creates a processing pipeline using:

PromptTemplate → OllamaLLM → StrOutputParser

It accepts five different questions related to LangChain and conversation history and generates answers using the local language model.

### Execution

```powershell
python chain_demo.py
```

### Result

- All five questions were processed successfully.
- The language model generated relevant responses.
- The output parser returned the responses as strings.

**Status:** Successfully executed.

## 5. Program 2 - Conversation Memory

### Description

This program uses ConversationBufferMemory to maintain conversation history across multiple interactions.

The conversation covers a chatbot project, its programming language, and its framework.

The program:

- Processes five conversation turns.
- Stores user and assistant messages.
- Uses previous conversation context to answer follow-up questions.
- Displays the complete stored conversation history.
- Verifies the number of stored messages.

### Execution

```powershell
python memory_demo.py
```

### Result

| Verification       | Actual Result |
| ------------------ | ------------: |
| User messages      |             5 |
| Assistant messages |             5 |
| Total messages     |            10 |

The model successfully recalled the chatbot project, Python, and LangChain from previous messages.

**Status:** Successfully executed.

### Technical Observation

The installed LangChain version displayed deprecation warnings for:

- ConversationBufferMemory
- InMemoryChatMessageHistory
- LLMChain

The program completed successfully. These APIs were retained because ConversationBufferMemory was explicitly required by the practical.

## 6. Program 3 - Two-Tool Agent

### Description

This program demonstrates a simple tool-based agent workflow with two tools.

**Tool 1: WebSearch**

- Uses predefined information.
- Demonstrates a simulated search operation.
- Does not access the live internet.

**Tool 2: Calculator**

- Performs arithmetic operations.
- Uses AST-based expression parsing.
- Restricts supported operators and validates calculation results.

The workflow executes the selected tools and passes their observations to the language model to generate a final response.

### Execution

```powershell
python react_agent.py
```

### Result

| Task                                               | Actual Result                     |
| -------------------------------------------------- | --------------------------------- |
| Search for LangChain information                   | Successful                        |
| Calculate 18 × 7                                   | 126                               |
| Explain conversation memory and calculate 144 ÷ 12 | Information returned; result 12.0 |

Both tools executed successfully, and all three tasks were completed.

**Status:** Successfully executed.

### Implementation Note

The implementation uses deterministic tool routing followed by an LLM-generated final response. It is a lightweight ReAct-style workflow, not a fully autonomous LLM-driven ReAct planner.

## 7. Key Concepts Learned

- PromptTemplate: Structures input prompts for a language model.
- OllamaLLM: Runs a language model locally.
- StrOutputParser: Converts model output into a string.
- ConversationBufferMemory: Stores previous conversation messages.
- Tool: Allows an agent workflow to execute specific functions.
- Tool observation: The result returned after a tool is executed.
- ReAct: A reasoning and acting pattern involving thoughts, actions, and observations.

## 8. Self-Review Checklist

- [x] Created a LangChain chain.
- [x] Tested the chain with five inputs.
- [x] Implemented ConversationBufferMemory.
- [x] Verified five conversation turns.
- [x] Created two tools.
- [x] Executed three agent tasks.
- [x] Verified program outputs.
- [x] Documented observations and limitations.

## 9. Viva Questions and Answers

**1. What is the difference between a LangChain chain and a single LLM call?**

A single LLM call sends a prompt directly to a language model. A LangChain chain connects multiple components into an organized workflow.

**2. What problem does LangChain Memory solve?**

It stores previous conversation information so that the application can use earlier context when responding to new messages.

**3. What is the ReAct pattern?**

ReAct stands for Reasoning and Acting. It combines reasoning with actions such as calling tools and observing their results to complete a task.

## 10. Conclusion

The practical demonstrated LangChain chains, conversation history, and tool execution using the local qwen2.5:3b model. All three programs executed successfully, and the required outputs were verified.
