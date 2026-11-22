# W7D1 – Haystack Pipeline Architecture

## Overview

This project implements a document retrieval system using Haystack. It indexes five PDF documents and compares two retrieval approaches:

- BM25 Retrieval – keyword-based document retrieval.
- Dense Retrieval – semantic document retrieval using sentence embeddings.

An extractive Reader is connected to both pipelines to generate answers from retrieved documents.

## Objectives

- Understand Haystack pipeline architecture.
- Convert PDF files into Haystack documents.
- Store documents using InMemoryDocumentStore.
- Implement BM25 retrieval.
- Implement dense retrieval using sentence-transformer embeddings.
- Connect an extractive Reader.
- Evaluate both retrieval methods using Precision@3.
- Compare retrieval performance using ten questions.

## Project Structure

```text
W7D1_Haystack_Pipeline_Architecture/
│
├── data/
│   └── pdfs/
│       ├── 01_policy_overview.pdf
│       ├── 02_eligibility_and_underwriting.pdf
│       ├── 03_premium_and_payment.pdf
│       ├── 04_claims_and_settlement.pdf
│       └── 05_exclusions_renewal_and_cancellation.pdf
│
├── outputs/
│   └── retrieval_comparison.csv
│
├── main.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python 3.14.3
- Haystack
- PyPDF
- Sentence Transformers
- Transformers
- InMemoryDocumentStore

## Models Used

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Extractive Reader:** `deepset/roberta-base-squad2-distilled`

## Pipeline Architecture

### 1. PDF Conversion

`PyPDFToDocument` converts the five PDF files into Haystack documents.

### 2. Document Embedding

`SentenceTransformersDocumentEmbedder` generates vector embeddings for the documents.

### 3. Document Storage

`DocumentWriter` stores the processed documents in `InMemoryDocumentStore`.

### 4. BM25 Retrieval

`InMemoryBM25Retriever` retrieves documents based on keyword relevance.

### 5. Dense Retrieval

`SentenceTransformersTextEmbedder` converts the user query into an embedding, and `InMemoryEmbeddingRetriever` retrieves semantically relevant documents.

### 6. Answer Extraction

`TransformersExtractiveReader` extracts answers from the retrieved documents.

### 7. Evaluation

Both retrieval methods are evaluated using the same ten questions. Precision@3 is calculated using manually defined relevant-document labels.

## Evaluation Results

| Metric              |   BM25 | Dense Retrieval |
| ------------------- | -----: | --------------: |
| PDFs Indexed        |      5 |               5 |
| Questions Evaluated |     10 |              10 |
| Average Precision@3 | 0.4000 |          0.4000 |

**Precision difference:** 0.0000

Both methods achieved the same average Precision@3 in this experiment.

## Key Observations

- Both retrieval pipelines executed successfully.
- Five PDF documents were indexed.
- Both methods completed all ten queries.
- Q7 and Q9 achieved Precision@3 of 0.67.
- Some Reader answers were incomplete, particularly for Q1, Q4 and Q6.
- Retrieval relevance and answer extraction quality are separate aspects of the pipeline.

## Output

The evaluation results are saved in:

`outputs/retrieval_comparison.csv`

The CSV contains the retrieval method, question, retrieved documents, relevant documents, Precision@3 and Reader answer.

## Execution

Activate the existing virtual environment and navigate to this project directory.

Run:

```powershell
python main.py
```

## Limitations

- The experiment uses five synthetic insurance reference PDFs.
- Evaluation is based on ten manually labelled questions.
- Some extracted answers require improvement.
- The results represent this small experiment and should not be generalized to all document retrieval tasks.

## Conclusion

The Haystack pipeline was successfully implemented and executed. Both BM25 and dense retrieval completed the evaluation with an average Precision@3 of 0.4000.

The experiment demonstrates PDF indexing, keyword retrieval, semantic retrieval, answer extraction and retrieval evaluation using Haystack.
