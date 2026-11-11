# W5D5 – Local Q&A Bot using Ollama + ChromaDB

## 1. Objective

The objective of this practical was to develop a Local Question-Answering (Q&A) Bot by integrating ChromaDB semantic retrieval with locally hosted Large Language Models using Ollama.

The application retrieves relevant information from an existing PDF collection and uses an LLM to generate context-based answers.

## 2. Technologies Used

- Python
- Ollama
- ChromaDB
- Requests
- nomic-embed-text
- llama3.2:3b
- qwen2.5:3b
- JSON

## 3. Project Structure

```text
W5D5_Local_QA_Bot/
│
├── scripts/
│   ├── config.py
│   ├── vector_store.py
│   ├── ollama_client.py
│   └── qa_bot.py
│
└── qa_results.json
```

The application reuses the persistent ChromaDB database and PDF collection created during W5D4.

## 4. Existing ChromaDB Collection

- Database: W5D4 persistent ChromaDB database
- Collection: `w5d4_semantic_pdf_chunks`
- Stored PDF chunks: 13
- Embedding model: `nomic-embed-text`

The existing PDF embeddings were reused. Only new user questions were converted into embeddings for similarity search.

## 5. Implementation

### A. Configuration – config.py

Contains:

- Project and database paths
- Ollama configuration
- Model names
- Retrieval settings
- Generation settings
- Custom system prompts
- Five prompt-test questions
- Three model-comparison questions
- Results output path

### B. Vector Store – vector_store.py

Responsibilities:

- Connects to the existing W5D4 ChromaDB database.
- Loads the existing PDF collection.
- Generates query embeddings using `nomic-embed-text`.
- Retrieves the top three relevant PDF chunks.
- Formats retrieved chunks into context for the LLM.

### C. Ollama Client – ollama_client.py

Responsibilities:

- Checks the Ollama server connection.
- Verifies the required models.
- Sends generation requests to Ollama's local API.
- Applies system prompts and generation settings.
- Measures response time.

API endpoints:

- `/api/tags` – checks installed models.
- `/api/embed` – generates query embeddings.
- `/api/generate` – generates answers.

### D. Main Application – qa_bot.py

The main program performs:

1. Ollama connection and model verification.
2. ChromaDB collection connection.
3. Five prompt-style tests.
4. Llama 3.2 and Qwen 2.5 comparison.
5. Interactive command-line question answering.
6. JSON result generation.

## 6. Five Prompt Tests

Five different system-prompt styles were tested using `llama3.2:3b`.

| Test | Prompt Style              | Response Time |
| ---- | ------------------------- | ------------: |
| 1    | Concise explanation       | 23.27 seconds |
| 2    | Step-by-step explanation  | 35.86 seconds |
| 3    | Bullet-point answer       | 24.51 seconds |
| 4    | Analogy-based explanation | 32.65 seconds |
| 5    | Detailed explanation      | 32.03 seconds |

Average response time: **29.66 seconds**

All five tests completed successfully.

## 7. Model Comparison

The two models were evaluated using the same three questions, retrieved context and base system prompt.

| Question                             |   llama3.2:3b |    qwen2.5:3b |
| ------------------------------------ | ------------: | ------------: |
| What is overfitting?                 |     17.22 sec |     23.79 sec |
| Explain the bias-variance trade-off. |     12.16 sec |     10.37 sec |
| How does k-nearest neighbors work?   |     16.98 sec |     14.40 sec |
| **Average**                          | **15.45 sec** | **16.19 sec** |

### Quality Observations

- Both models generated relevant explanations for overfitting.
- Both models identified that the retrieved context did not contain enough information about the bias-variance trade-off.
- Both models identified that the retrieved context did not explain the k-nearest neighbors algorithm.
- Llama was faster on the overfitting question.
- Qwen was faster on the other two questions.
- Llama had a slightly lower average response time in this three-question test.

The results do not establish that one model is universally better.

## 8. Interactive Q&A Testing

The interactive bot was tested with three questions.

| Question                                                | Result           | Response Time |
| ------------------------------------------------------- | ---------------- | ------------: |
| Difference between supervised and unsupervised learning | Answer generated |     25.88 sec |
| Explain the machine learning workflow                   | Answer generated |     19.78 sec |
| Explain supervised learning                             | Answer generated |     20.15 sec |

The bot accepted questions through the command line and stopped normally when `exit` was entered.

## 9. Validation Results

| Validation                         | Result |
| ---------------------------------- | ------ |
| Ollama connection                  | PASS   |
| Llama 3.2 availability             | PASS   |
| Qwen 2.5 availability              | PASS   |
| ChromaDB connection                | PASS   |
| Existing collection with 13 chunks | PASS   |
| Five prompt tests                  | PASS   |
| Three-question model comparison    | PASS   |
| Interactive Q&A                    | PASS   |
| JSON results saved                 | PASS   |
| Normal program termination         | PASS   |

## 10. Generated Results

The application generated:

`qa_results.json`

The JSON file records:

- Five prompt-test questions and answers.
- Prompt styles and model names.
- Three model-comparison questions and both model responses.
- Response times.
- Interactive questions and answers.
- Fields for manual quality observations.

## 11. Conclusion

The W5D5 Local Q&A Bot practical was completed successfully.

The application integrated ChromaDB semantic retrieval with locally hosted Ollama language models. It demonstrated prompt engineering, context-based answer generation, model comparison and interactive question answering.

The practical also demonstrated the importance of grounding LLM responses in retrieved information. When the PDF context did not contain sufficient information, both models appropriately indicated that they could not answer the question from the available context.

The execution completed successfully, and the results were saved in JSON format.
