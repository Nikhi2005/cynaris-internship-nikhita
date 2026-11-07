# W5D1 – Running LLMs Locally with Ollama

## 1. Overview

This task demonstrates how to install and use Ollama to run Large Language Models (LLMs) locally. Two models, Llama 3.2 and Qwen 2.5, are tested using Python and the Ollama REST API.

## 2. Objectives

- Install and verify Ollama.
- Run Llama 3.2 locally.
- Download and run Qwen 2.5.
- Connect Python to the Ollama REST API.
- Apply a custom system prompt.
- Test five prompts on both models.
- Compare model responses.
- Save results in JSON format.

## 3. Tools and Technologies

- Python
- Ollama 0.40.15
- Llama 3.2 (llama3.2:3b)
- Qwen 2.5 (qwen2.5:3b)
- Requests 2.34.2
- JSON
- VS Code

## 4. Environment Setup

Verify Ollama:

```powershell
ollama --version
```

Check available models:

```powershell
ollama list
```

Download Qwen:

```powershell
ollama pull qwen2.5:3b
```

Run the first Llama inference:

```powershell
ollama run llama3.2:3b "Explain artificial intelligence in simple words with one real-life example."
```

The existing Python virtual environment was reused:

`D:\Cynaris\.venv`

## 5. Python Implementation

**File:** `ollama_chat.py`

The Python script connects to the local Ollama API:

`http://localhost:11434/api/chat`

It:

- Sends system and user messages.
- Uses a custom system prompt.
- Tests five identical prompts on both models.
- Uses non-streaming responses.
- Handles HTTP request errors.
- Saves generated responses in JSON format.

## 6. Prompts Used

1. Explain artificial intelligence with a real-life example.
2. What is the difference between supervised and unsupervised learning?
3. Explain how a neural network works in simple words.
4. What are the main challenges of AI adoption in Indian MSMEs?
5. Suggest a computer vision project idea for a BTech final-year student.

## 7. Model Comparison

| Evaluation         | Llama 3.2                                    | Qwen 2.5                                          |
| ------------------ | -------------------------------------------- | ------------------------------------------------- |
| AI explanation     | Structured explanation                       | Conversational example                            |
| Learning types     | Image classification and clustering examples | Dog-training and toy-sorting analogies            |
| Neural networks    | Layers, activation and training              | Layers, weights, forward and backward propagation |
| Indian MSMEs       | Listed 10 challenges                         | Listed 8 challenges                               |
| Project suggestion | Fruit ripeness detection                     | Pedestrian crossing detection                     |

### Observation

Llama 3.2 generally provided structured explanations, while Qwen 2.5 often used conversational analogies and more implementation-oriented details.

These observations are based on the five prompts tested and are not formal benchmark scores.

## 8. Execution

Run the script from the project folder:

```powershell
python ollama_chat.py
```

Both models successfully answered all five prompts in the recorded execution.

## 9. Output

The script generates:

`ollama_results.json`

The JSON file stores:

- Model name
- Prompt number
- Prompt text
- Generated response

## 10. Conclusion

Successfully implemented local LLM inference using Ollama and Python. Tested Llama 3.2 and Qwen 2.5 with a common system prompt and five identical prompts. Saved the generated responses in JSON format and compared the models qualitatively.

## 11. Viva Questions

**1. Difference between model parameters and prompt parameters?**

Model parameters are learned numerical values such as weights and biases. Prompt parameters are instructions or settings supplied during inference to guide the model.

**2. Why use a local LLM instead of a cloud API?**

Local LLMs can keep data on the local machine, reduce dependence on internet connectivity and avoid per-request cloud API charges. They require suitable local hardware.

**3. What is quantisation?**

Quantisation reduces the numerical precision used to represent model weights. It reduces memory requirements and can improve inference efficiency, but aggressive quantisation may affect response quality.
