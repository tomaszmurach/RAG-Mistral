# RAG with Mistral 7B and FAISS

A focused document-grounded question-answering demonstration using
**Mistral-7B-Instruct-v0.3**, **multilingual-e5-base**, and **FAISS** on a short
Polish IT incident-response procedure.

**document → section chunks → E5 embeddings → FAISS retrieval → chat prompt → answer**

The final implementation completed a real **Google Colab / Tesla T4 GPU smoke
run**, exercising both greedy and sampling generation. All **22/22 lightweight
tests passed** in the verified environment.

## How it works

- One chunk per numbered section preserves headings and all section content;
  the document title stays with the first section (15 chunks total).
- E5 embeds `passage: <text>` and `query: <text>`, with normalized vectors and
  FAISS `IndexFlatIP` cosine-similarity search. Prefixes affect embeddings only.
- Defaults: **k = 3**, **similarity threshold = 0.82**, **temperature = 0.0**,
  with up to 70 new tokens. Zero temperature uses greedy decoding; positive
  temperatures enable sampling.
- Mistral uses 4-bit NF4 quantization and its tokenizer's chat template. Both
  models run on GPU 0; CPU inference and multi-GPU execution are not implemented.

## Grounding and evaluation

Refusal has two distinct paths:

1. **Pipeline refusal:** no passages survive filtering → return
   `Brak informacji w dokumencie.` without calling Mistral.
2. **Model-instruction refusal:** passages exist, but the requested fact is not
   explicitly stated → Mistral is instructed to return the same wording. This
   behavior is not independently verified or guaranteed.

The **0.82 threshold was selected empirically for this corpus and small evaluation
set**. It is a coarse relevance filter, not an answerability test, confidence
probability, or universal E5 threshold. On-topic unanswerable questions can score
highly and still produce unsupported answers.

The original 10 answerable questions retrieved the expected section at top-1.
All 16 answerable paraphrases scored at least 0.82; 15 retrieved the expected
section first. These historical measurements motivated the current threshold
and stricter grounding instructions.

In the final smoke run, the known hard negative, “Kto jest właścicielem tej
procedury?”, was rejected by retrieval at **0.82** and correctly refused by
Mistral at an explicit **0.80** override. The strengthened prompt corrected the
earlier unsupported answer, “Organizacja”, in this tested case. The out-of-domain
control was also rejected. See [recorded results](examples/results.md) for the
separate historical evaluation and final verification; this is a small evaluation
set, not a general benchmark or statistical claim.

## Verified environment

One successful environment is recorded here; it does not establish compatibility
elsewhere or imply that this is the only environment that could work.

| Component | Observed version / hardware |
|---|---|
| Platform / GPU | Google Colab / Tesla T4, 15360 MiB VRAM |
| Python | 3.13.15 |
| PyTorch / CUDA reported by PyTorch | 2.11.0+cu128 / 12.8 |
| Transformers | 5.16.1 |
| Sentence Transformers | 5.7.0 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.14.0 |
| NumPy | 2.1.3 |
| FAISS (`faiss-cpu`) | 1.15.1 |

Observed GPU memory after final pipeline loading/use: approximately **5500 MiB
(~5.4 GiB)**. This is an observed loaded-runtime value, not peak generation memory
or a minimum VRAM guarantee.

## Run

In Colab, select a **T4 GPU runtime**. Run the following from a `%%bash` cell
(or a shell in an equivalent Linux environment). Model downloads need internet
access and local cache space.

```bash
git clone https://github.com/tomaszmurach/RAG-Mistral.git
cd RAG-Mistral
# Reuse this build if already installed; otherwise install the verified CUDA wheel.
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python rag_demo.py
```

The explicit CUDA-wheel step follows [PyTorch's versioned installation guidance](https://pytorch.org/get-started/previous-versions/).
`requirements.txt` pins only direct project dependencies. Its `torch==2.11.0`
pin accepts the verified `+cu128` build but does not select a CUDA wheel by itself.
No CUDA/NVIDIA transitive packages or unrelated Colab packages are pinned.
These are observed working versions, not a complete environment lock.

Unauthenticated Hugging Face Hub requests may produce a warning when `HF_TOKEN`
is unset. This is expected, not a project error; a token is not required for this
demo, and the warning is not suppressed.

The demo covers a direct question, a paraphrase, an on-topic missing fact, an
out-of-domain question, and a question asking for two responsibilities.

## Code and lightweight checks

`load_pipeline()` initializes a `Retriever` and an `AnswerGenerator` once.
`ask_bot(question, retriever, answer_generator, ...)` returns a dictionary with
ranked/kept passages, scores, context, parameters, answer, and `no_context_refusal`.
`print_result()` provides console output. Importing the module loads no models.

Override `k`, `score_threshold`, `temperature`, or `max_new_tokens` via `ask_bot()`;
`score_threshold=None` disables filtering. Oversized `k` is capped and recorded.
Blank questions/documents, invalid integer limits, nonfinite numbers, thresholds
outside `[-1, 1]`, and negative temperatures raise `ValueError`.

```bash
python -m unittest discover -s tests -v
```

The **22 lightweight checks** use NumPy and model/FAISS doubles without downloads
or CUDA. They passed alongside the separate final GPU smoke run. In that GPU run,
both greedy and sampling (`temperature=0.5`) produced correct grounded answers
for the answerable control. The previous Transformers warnings about mixed
generation configuration/kwargs, conflicting length settings, and BPE tokenization
cleanup were no longer observed. Lightweight tests alone do not validate model
quality or real-library integration.

This remains a small demonstration: the section chunker targets this short
corpus, answers are not independently verified, and longer documents would need
explicit token-budget handling.
