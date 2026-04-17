# RAG with Mistral 7B and FAISS

A compact Retrieval-Augmented Generation (RAG) project built with **Mistral-7B-Instruct-v0.3**, **multilingual-e5-base** embeddings, and **FAISS** for document-grounded question answering.

## Overview

This project implements a simple RAG pipeline:

**document -> chunking -> embeddings -> FAISS -> retrieval -> prompt -> generation**

The system answers questions **only from the provided source document**. If the required information is not present in the retrieved context, it returns a refusal instead of generating unsupported content.

## Tech Stack

- Python
- Google Colab
- Hugging Face Transformers
- Mistral-7B-Instruct-v0.3
- Sentence Transformers
- FAISS
- PyTorch
- bitsandbytes
- accelerate

## Key Features

- Document-grounded question answering
- Semantic retrieval with FAISS
- 4-bit quantized Mistral model for more efficient Colab usage
- Configurable retrieval depth with `k`
- Optional similarity filtering with `score_threshold`
- Temperature-based generation control
- Explicit refusal when no relevant context is found

## How It Works

1. The source document is split into chunks.
2. Each chunk is embedded with `multilingual-e5-base`.
3. Embeddings are indexed in FAISS.
4. A user question is embedded and matched against the document chunks.
5. The retrieved context is inserted into the prompt.
6. Mistral generates a short answer constrained to the retrieved content.

## Core Functions

- `chunk_text(text)` – prepares document chunks
- `retrieve_context(query, k=3, score_threshold=None, return_meta=False)` – retrieves relevant chunks
- `ask_bot(question, k=3, score_threshold=None, temperature=0.01, show_meta=False)` – runs the full RAG flow

## Example Test Scenarios

- Answering questions covered by the document
- Refusing out-of-scope questions
- Comparing different `k` values
- Testing the effect of `score_threshold`
- Observing answer style changes with different `temperature` values

## Running the Project

Install dependencies:

```bash
pip install -U bitsandbytes sentence-transformers faiss-cpu transformers accelerate
```

Then run the notebook or script in **Google Colab with GPU enabled**.

## Notes

This project uses a single manually defined source document and a simple chunking strategy. It is designed as a focused demonstration of the RAG workflow rather than a production-ready knowledge system.
