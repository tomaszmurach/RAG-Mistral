# RAG with Mistral 7B and FAISS

A compact Retrieval-Augmented Generation (RAG) project built with **Mistral-7B-Instruct-v0.3**, **multilingual-e5-base** embeddings, and **FAISS** for document-grounded question answering.

## Overview

This project implements a simple RAG pipeline:

**document -> chunking -> embeddings -> FAISS -> retrieval -> prompt -> generation**

The model is instructed to answer only from retrieved document passages. If no
passages survive similarity filtering, the pipeline returns the fixed refusal
`Brak informacji w dokumencie.` without calling the language model. When passages
exist, grounding and refusal depend on model instructions: answers are not
independently verified and unsupported content remains possible.

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
- Greedy generation at `temperature=0`; sampling at positive temperatures
- Deterministic refusal when no passages survive filtering
- Structured query results, with separate console presentation

## How It Works

1. Each numbered section becomes one chunk, keeping its heading and all its
   content together. The title/preamble stays with the first section.
2. Chunks are embedded as `passage: <text>` with `multilingual-e5-base`.
3. Normalized embeddings are indexed in FAISS `IndexFlatIP`, giving cosine
   similarity scores.
4. The question is embedded as `query: <text>`, also normalized, and matched
   against the document chunks. E5 prefixes affect embedding inputs only;
   stored passages, displayed text, and prompt context remain unprefixed.
5. Kept passages enter a single user message formatted with the tokenizer's chat
   template. Special tokens are not added a second time during tokenization.
6. Mistral is instructed to produce a short, context-grounded answer. The returned
   completion is retained, apart from surrounding whitespace, so formatting
   deviations are not hidden by truncating it to one line.

The E5 formatting follows the [model's retrieval conventions](https://huggingface.co/intfloat/multilingual-e5-base#faq).
Chunking targets this short, numbered document; it does not split arbitrarily
long sections to fit model token limits.

## Core Functions

- `chunk_text(text)` prepares complete section chunks.
- `Retriever.from_text(text, embedder)` builds retrieval state once: original
  chunks, the embedding model, and its FAISS index.
- `retriever.retrieve_context(query, k=3, score_threshold=None)` returns ranked
  passages, kept passages, scores, context, and requested/effective retrieval settings.
- `AnswerGenerator(tokenizer, generator)` holds the separate generation state.
- `ask_bot(question, retriever, answer_generator, *, k=3, score_threshold=None,
  temperature=0.0, max_new_tokens=70)` returns a dictionary containing the question,
  retrieval details, answer, generation settings, and `no_context_refusal` flag.
  Settings are recorded even if generation is skipped. A model-generated refusal
  does not set this flag.
- `print_result(result)` handles console output; `load_pipeline()` explicitly
  initializes the two models and retrieval index.

Blank questions/documents, nonpositive or noninteger `k`, and invalid token
limits raise `ValueError`. Oversized `k` is capped at the number of chunks, with
both requested and effective values recorded. A threshold must be `None` (no
filter) or a finite number in `[-1, 1]`; temperature must be finite and nonnegative.
Boolean values are not accepted as numeric parameters. Questions are trimmed.
Zero temperature uses greedy decoding without passing sampling-only parameters.
It does not imply bit-for-bit reproducibility across different hardware/software.

## Example Test Scenarios

- Answering questions covered by the document
- Observing refusal behavior on out-of-scope questions
- Comparing different `k` values
- Testing the effect of `score_threshold`
- Observing answer style changes with different `temperature` values

These are demonstrations, not measured evaluation results. Thresholds `0.20`,
`0.35`, and `0.89` remain **provisional**: E5 prefixes and section-level chunking
change scores, so Phase 3 GPU evaluation must reassess them. Cosine scores are not
confidence probabilities. With filtering disabled, nearest passages are returned
even for unrelated questions.

## Running the Project

The standalone script is `rag_demo.py`; no notebook is tracked. It loads the
models, builds the FAISS index, and runs the ten demonstration questions when
executed directly. Importing the module uses only the Python standard library;
ML dependencies and models are loaded explicitly when needed. The source uses
Python 3.10+ syntax; a complete supported GPU environment is not yet verified.

The current implementation requires a compatible NVIDIA GPU/PyTorch environment
and sufficient GPU memory: Mistral is loaded in 4-bit mode entirely on GPU 0,
and the embedding model also uses GPU 0. Startup checks CUDA availability and
quantization dependencies. There is no CPU fallback. First execution downloads the models from Hugging Face
and requires internet access and local cache space.

From the repository root, using your chosen Python environment:

```bash
python -m pip install -r requirements.txt
python rag_demo.py
```

For Google Colab, select a GPU runtime, clone the repository, and run those
commands from its directory in a shell cell.

Dependencies are currently unpinned. An exact compatible combination of Python,
PyTorch/CUDA, and the remaining libraries has not yet been verified, and a full
GPU run is still pending. The dependency list is not a tested environment lock.

Lightweight regression checks use `unittest` and NumPy, with embedding, FAISS,
tokenizer, and generation doubles; they do not download models or require CUDA:

```bash
python -m unittest discover -s tests -v
```

## Notes

This project uses a single manually defined source document and a simple chunking strategy. It is designed as a focused demonstration of the RAG workflow rather than a production-ready knowledge system.
