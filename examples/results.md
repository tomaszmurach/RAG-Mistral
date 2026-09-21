# Recorded evaluation results

These are supplied observations from the successful **pre-update** Colab run.
This is a small evaluation set for the demonstration corpus, not a benchmark;
no statistical significance or generalization to other corpora is claimed.

## Environment

| Component | Observed version / hardware |
|---|---|
| Platform / GPU | Google Colab / Tesla T4, 15360 MiB VRAM |
| Python | 3.13.15 |
| PyTorch / CUDA reported by PyTorch | 2.11.0+cu128 / 12.8 |
| Transformers / Sentence Transformers | 5.16.1 / 5.7.0 |
| bitsandbytes / accelerate | 0.50.2 / 1.14.0 |
| NumPy / FAISS | 2.1.3 / 1.15.1 |

Approximately 5317 MiB (~5.2 GiB) GPU memory was observed after loading both
models; peak generation memory was not reported. The complete Mistral + E5 +
FAISS pipeline succeeded, and the then-current lightweight suite passed 20/20.

## Retrieval evaluation

Scores are top-1 cosine similarities from normalized E5 embeddings.

| Question group | n | Expected section at top-1 | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|---:|
| Answerable | 10 | 10/10 | 0.8447 | 0.9024 | 0.8692 |
| Out-of-domain | 6 | Not applicable | 0.7020 | 0.7771 | 0.7412 |
| Unanswerable, on-topic | 6 | Not applicable | 0.7858 | 0.8653 | 0.8300 |

On-topic unanswerable scores overlap answerable scores. Similarity can filter
coarse irrelevance, but cannot establish that the requested fact exists.

## Paraphrase robustness

- 16/16 answerable paraphrases scored at least 0.82; minimum score: **0.8337**.
- 15/16 retrieved the expected section at top-1.
- The mismatch was semantically reasonable: a question about restoring systems
  ranked **9. Działania naprawcze** above **10. Przywracanie systemów**.

These observations support 0.82 as a default coarse filter for this small corpus
and evaluation set. They do not establish a universal E5 cutoff or guarantee
recall on new questions.

## Grounding and refusal at threshold 0.80

| Group | Observed outcome |
|---|---|
| Answerable controls | 2/2 answered |
| Out-of-domain | 3/3 rejected by the pipeline |
| On-topic, unanswerable | 5/6 correctly refused, by retrieval or the model |

The one unsupported inference was:

- Question: **Kto jest właścicielem tej procedury?**
- Top retrieval score: **0.8191**.
- At threshold **0.80**, passages reached Mistral, which answered **Organizacja**.
- At threshold **0.82**, that recorded score would fail retrieval filtering.
  This is a consequence of the recorded score, not a new measured model run.

The failure motivated two changes: a corpus-specific default threshold of 0.82
and a stricter Polish prompt requiring explicit evidence and forbidding inference
from organization names, roles, context, or general knowledge. Some other
on-topic unanswerable scores exceed 0.82, so model refusal remains necessary
and fallible. The stricter prompt's effectiveness has not yet been measured.

## Verification boundary

The observations above precede the new prompt and Transformers warning cleanup.
The expanded 22-test lightweight suite exercises filtering, generation settings,
and refusal control flow; model and FAISS doubles do not reproduce GPU inference.
A follow-up Colab smoke run must check the current demo, greedy and sampling
calls, the known hard negative, and warning output before the modified version
can be described as GPU-verified. For the hard negative, check both the default
0.82 filter and an explicit 0.80 override: rejecting it before generation does
not evaluate the stricter prompt's model-refusal behavior.

The supplied evidence contains aggregate results and the hard-negative example,
not the full question list, raw outputs, or model revision hashes. This summary
therefore records the available evidence rather than claiming a fully
reproducible evaluation dataset.
