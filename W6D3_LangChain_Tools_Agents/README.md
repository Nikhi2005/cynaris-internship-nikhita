# W6D3 - LangChain Tools & Agents

## 1. Overview

This practical demonstrates LangChain tools and agents using the local Ollama model `qwen2.5:3b`.

The implementation covers:

- Building an LLM chain using PromptTemplate and StrOutputParser.
- Maintaining conversation history using ConversationBufferMemory.
- Defining and invoking custom tools.
- Using an LLM to select tools and process their observations.

## 2. Technologies Used

- Python
- LangChain
- LangChain Core
- LangChain Classic
- LangChain Ollama
- Ollama
- Qwen 2.5 (3B)

## 3. Project Structure

```text
W6D3_LangChain_Tools_Agents/
│
├── chain_demo.py
├── memory_demo.py
├── react_agent.py
└── README.md
```

## 4. Task 1 - LangChain Chain

### Objective

Build a chain using:

PromptTemplate → Ollama LLM → StrOutputParser

### Implementation

The program creates a prompt template, sends five questions to the local Ollama model, and converts the model responses into plain text using StrOutputParser.

### Result

All five test inputs executed successfully and generated relevant responses.

## 5. Task 2 - Conversation Memory

### Objective

Implement ConversationBufferMemory and verify conversation history across five turns.

### Implementation

The program uses:

- ConversationBufferMemory to retain conversation history.
- LLMChain to connect the prompt, language model, and memory.
- Five conversation turns to test information recall.

### Result

All five turns executed successfully.

The final question correctly recalled:

1. Python
2. Artificial Intelligence
3. LangChain

Total stored messages: 10 (5 user messages and 5 assistant messages).

### Note

LangChain displayed deprecation warnings for ConversationBufferMemory, InMemoryChatMessageHistory, and LLMChain. These warnings did not affect program execution.

## 6. Task 3 - LangChain Tools & Agent

### Objective

Create an agent with two tools:

1. Web Search
2. Calculator

### Tool Descriptions

**Web Search**

- Implemented as a simulated search tool.
- Searches a small predefined knowledge base.
- Returns information about matching topics.

**Calculator**

- Evaluates basic mathematical expressions.
- Uses Python AST parsing and allowed arithmetic operators.
- Avoids unsafe use of eval().

### Agent Workflow

1. The user provides a task.
2. The LLM selects an appropriate tool.
3. The selected tool executes the request.
4. The tool returns an observation.
5. The LLM generates a final response.

The implementation uses the native tool-calling interface provided by ChatOllama.

### Test Results

| Task                             | Selected Tool | Result     |
| -------------------------------- | ------------- | ---------- |
| Search for LangChain information | Web Search    | Successful |
| Calculate 18 × 7                 | Calculator    | 126        |
| Calculate 144 ÷ 12               | Calculator    | 12.0       |

All three agent tasks executed successfully.

**Note:** Web Search is a simulated tool and does not access the live internet.

## 7. Execution Commands

Run the programs using the existing virtual environment:

```powershell
python chain_demo.py
python memory_demo.py
python react_agent.py
```

## 8. Final Results

| Practical           | Status    |
| ------------------- | --------- |
| LangChain Chain     | Completed |
| Conversation Memory | Completed |
| Tools and Agent     | Completed |

All three practical demonstrations executed successfully.

## 9. Learning Outcomes

- Understood how LangChain chains connect prompts, language models, and output parsers.
- Implemented conversation history using memory.
- Created custom LangChain tools.
- Understood native LLM tool selection and tool invocation.
- Observed how tool results are passed back to the LLM to generate final responses.

## 10. Conclusion

The W6D3 practical successfully demonstrated LangChain chains, conversation memory, and tool-based agent execution using Ollama and the Qwen 2.5 model.
