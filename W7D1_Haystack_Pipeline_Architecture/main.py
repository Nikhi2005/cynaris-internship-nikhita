
# W7D1 - Haystack Pipeline Architecture
# Purpose:
# Convert five insurance reference PDFs into Haystack documents,
# index them, compare BM25 and Dense retrieval, and evaluate
# retrieval quality using Precision@3 for ten questions.

import csv
import gc
import sys
from pathlib import Path

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from haystack_integrations.components.readers.transformers import (
    TransformersExtractiveReader,
)


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_DIR = BASE_DIR / "outputs"
CSV_PATH = OUTPUT_DIR / "retrieval_comparison.csv"

TOP_K = 3

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# 2. TEN QUESTIONS AND MANUAL RELEVANCE JUDGEMENTS
# ---------------------------------------------------------

QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is the minimum and maximum applicant age?",
        "relevant_docs": ["02_eligibility_and_underwriting"],
    },
    {
        "id": "Q2",
        "question": "Which documents are required before an insurance application can be accepted?",
        "relevant_docs": ["02_eligibility_and_underwriting"],
    },
    {
        "id": "Q3",
        "question": "What details should a premium payment receipt contain?",
        "relevant_docs": ["03_premium_and_payment"],
    },
    {
        "id": "Q4",
        "question": "What grace period applies to monthly and annual premium payments?",
        "relevant_docs": ["03_premium_and_payment"],
    },
    {
        "id": "Q5",
        "question": "How should a nominee notify the insurer about a claim?",
        "relevant_docs": ["04_claims_and_settlement"],
    },
    {
        "id": "Q6",
        "question": "Which documents may be requested during claim settlement?",
        "relevant_docs": ["04_claims_and_settlement"],
    },
    {
        "id": "Q7",
        "question": "Who receives policy proceeds when the insured event occurs?",
        "relevant_docs": [
            "01_policy_overview",
            "04_claims_and_settlement",
        ],
    },
    {
        "id": "Q8",
        "question": "How long is the free-look cancellation period?",
        "relevant_docs": ["01_policy_overview"],
    },
    {
        "id": "Q9",
        "question": "What can happen if premiums are unpaid and how can a lapsed policy be revived?",
        "relevant_docs": [
            "03_premium_and_payment",
            "05_exclusions_renewal_and_cancellation",
        ],
    },
    {
        "id": "Q10",
        "question": "Which application records should have matching name and date of birth?",
        "relevant_docs": ["02_eligibility_and_underwriting"],
    },
]


# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------

def get_source_id(document):
    """Get the original PDF identifier from document metadata."""

    metadata = document.meta or {}

    source_id = metadata.get("source_id")

    if source_id:
        return source_id

    file_path = metadata.get("file_path", "")

    if file_path:
        return Path(str(file_path)).stem

    return "unknown"


def get_reader_answer(result):
    """Extract the best answer returned by the Reader."""

    answers = result.get("reader", {}).get("answers", [])

    if not answers:
        return "No answer extracted"

    answer = answers[0]

    answer_text = getattr(answer, "data", None)

    if not answer_text:
        answer_text = getattr(answer, "answer", None)

    return str(answer_text) if answer_text else "No answer extracted"


# ---------------------------------------------------------
# 4. CREATE THE DOCUMENT STORE
# ---------------------------------------------------------

def create_document_store():

    print("\nCreating Haystack DocumentStore...")

    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )

    print("DocumentStore created successfully.")

    return document_store


# ---------------------------------------------------------
# 5. BUILD AND RUN THE PDF INDEXING PIPELINE
# ---------------------------------------------------------

def index_pdf_documents(document_store):

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if len(pdf_files) != 5:
        raise ValueError(
            f"Expected exactly 5 PDF documents, found {len(pdf_files)}."
        )

    print(f"\nFound {len(pdf_files)} PDF documents.")

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(
        "converter",
        PyPDFToDocument()
    )

    indexing_pipeline.add_component(
        "document_embedder",
        SentenceTransformersDocumentEmbedder(
            model=EMBEDDING_MODEL
        )
    )

    indexing_pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store
        )
    )

    indexing_pipeline.connect(
        "converter.documents",
        "document_embedder.documents"
    )

    indexing_pipeline.connect(
        "document_embedder.documents",
        "writer.documents"
    )

    print("\nIndexing PDF documents...")

    for pdf_file in pdf_files:

        print(f"Processing: {pdf_file.name}")

        indexing_pipeline.run(
            {
                "converter": {
                    "sources": [pdf_file],
                    "meta": {
                        "source_id": pdf_file.stem,
                        "file_name": pdf_file.name,
                    },
                }
            }
        )

    indexed_documents = document_store.filter_documents()

    print("\nPDF indexing completed.")
    print(f"Number of PDFs processed: {len(pdf_files)}")
    print(f"Number of documents indexed: {len(indexed_documents)}")

    if not indexed_documents:
        raise RuntimeError("No documents were indexed.")

    return len(pdf_files)


# ---------------------------------------------------------
# 6. BUILD BM25 RETRIEVAL PIPELINE
# ---------------------------------------------------------

def build_bm25_pipeline(document_store):

    pipeline = Pipeline()

    retriever = InMemoryBM25Retriever(
        document_store=document_store
    )

    reader = TransformersExtractiveReader()

    pipeline.add_component(
        "retriever",
        retriever
    )

    pipeline.add_component(
        "reader",
        reader
    )

    pipeline.connect(
        "retriever.documents",
        "reader.documents"
    )

    return pipeline


# ---------------------------------------------------------
# 7. BUILD DENSE RETRIEVAL PIPELINE
# ---------------------------------------------------------

def build_dense_pipeline(document_store):

    pipeline = Pipeline()

    text_embedder = SentenceTransformersTextEmbedder(
        model=EMBEDDING_MODEL
    )

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store
    )

    reader = TransformersExtractiveReader()

    pipeline.add_component(
        "text_embedder",
        text_embedder
    )

    pipeline.add_component(
        "retriever",
        retriever
    )

    pipeline.add_component(
        "reader",
        reader
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    pipeline.connect(
        "retriever.documents",
        "reader.documents"
    )

    return pipeline


# ---------------------------------------------------------
# 8. RUN QUESTIONS AND EVALUATE RETRIEVAL
# ---------------------------------------------------------

def evaluate_pipeline(pipeline, method):

    evaluation_results = []

    print(f"\n{'=' * 65}")
    print(f"{method} RETRIEVAL EVALUATION")
    print(f"{'=' * 65}")

    for item in QUESTIONS:

        query_id = item["id"]
        question = item["question"]

        relevant_docs = set(item["relevant_docs"])

        if method == "BM25":

            pipeline_input = {
                "retriever": {
                    "query": question,
                    "top_k": TOP_K,
                },
                "reader": {
                    "query": question,
                    "top_k": 1,
                },
            }

        else:

            pipeline_input = {
                "text_embedder": {
                    "text": question,
                },
                "retriever": {
                    "top_k": TOP_K,
                },
                "reader": {
                    "query": question,
                    "top_k": 1,
                },
            }

        result = pipeline.run(
            pipeline_input,
            include_outputs_from={"retriever"}
        )

        retrieved_documents = result["retriever"]["documents"]

        retrieved_ids = [
            get_source_id(document)
            for document in retrieved_documents
        ]

        relevant_count = sum(
            1 for doc_id in retrieved_ids
            if doc_id in relevant_docs
        )

        precision = relevant_count / TOP_K

        answer = get_reader_answer(result)

        print(f"\n{query_id}: {question}")

        print("Retrieved documents:")

        for index, doc_id in enumerate(retrieved_ids, start=1):
            print(f"  {index}. {doc_id}")

        print(f"Relevant documents retrieved: {relevant_count}")
        print(f"Precision@{TOP_K}: {precision:.2f}")
        print(f"Reader answer: {answer}")

        evaluation_results.append({
            "method": method,
            "query_id": query_id,
            "question": question,
            "retrieved_documents": " | ".join(retrieved_ids),
            "relevant_documents": " | ".join(sorted(relevant_docs)),
            "relevant_count": relevant_count,
            "precision_at_3": round(precision, 4),
            "reader_answer": answer,
        })

    average_precision = sum(
        row["precision_at_3"]
        for row in evaluation_results
    ) / len(evaluation_results)

    print(f"\nAverage {method} Precision@{TOP_K}: "
          f"{average_precision:.4f}")

    return evaluation_results, average_precision


# ---------------------------------------------------------
# 9. SAVE COMPARISON RESULTS
# ---------------------------------------------------------

def save_results(results):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "query_id",
        "question",
        "retrieved_documents",
        "relevant_documents",
        "relevant_count",
        "precision_at_3",
        "reader_answer",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nComparison results saved to: {CSV_PATH}")


# ---------------------------------------------------------
# 10. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("W7D1 - HAYSTACK PIPELINE ARCHITECTURE")
    print("=" * 65)

    print(f"Python version: {sys.version.split()[0]}")

    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_DIR}"
        )

    document_store = create_document_store()

    # Stage 1: Index all five PDFs.
    pdf_count = index_pdf_documents(document_store)

    # Stage 2: Run BM25 evaluation.
    bm25_pipeline = build_bm25_pipeline(document_store)

    bm25_results, bm25_average = evaluate_pipeline(
        bm25_pipeline,
        "BM25"
    )

    # Release the BM25 Reader before loading another Reader.
    del bm25_pipeline
    gc.collect()

    # Stage 3: Run Dense retrieval evaluation.
    dense_pipeline = build_dense_pipeline(document_store)

    dense_results, dense_average = evaluate_pipeline(
        dense_pipeline,
        "DENSE"
    )

    # Stage 4: Save both sets of results.
    all_results = bm25_results + dense_results

    save_results(all_results)

    # Stage 5: Final comparison.
    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 65)

    print(f"PDFs indexed: {pdf_count}")
    print(f"Questions evaluated: {len(QUESTIONS)}")

    print(f"BM25 Average Precision@{TOP_K}: "
          f"{bm25_average:.4f}")

    print(f"Dense Average Precision@{TOP_K}: "
          f"{dense_average:.4f}")

    difference = dense_average - bm25_average

    print(f"Precision difference (Dense - BM25): "
          f"{difference:+.4f}")

    print("\nExecution completed successfully.")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERROR:", error)
        raise
