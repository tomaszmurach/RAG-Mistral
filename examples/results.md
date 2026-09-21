# Recorded evaluation results

This records the maintainer's Colab/T4 observations, separating **historical
pre-update evaluation** from **final post-update verification**. This is a small
evaluation set for the demonstration corpus, not a benchmark; no statistical
significance or generalization to other corpora is claimed.

## Verified environment (both runs)

| Component | Observed version / hardware |
|---|---|
| Platform / GPU | Google Colab / Tesla T4, 15360 MiB VRAM |
| Python | 3.13.15 |
| PyTorch / CUDA reported by PyTorch | 2.11.0+cu128 / 12.8 |
| Transformers / Sentence Transformers | 5.16.1 / 5.7.0 |
| bitsandbytes / accelerate | 0.50.2 / 1.14.0 |
| NumPy / FAISS | 2.1.3 / 1.15.1 |

## Historical evaluation (pre-update)

Before the prompt and generation-configuration updates, approximately 5317 MiB
(~5.2 GiB) GPU memory was observed after loading both models. The complete
Mistral + E5 + FAISS pipeline succeeded, and the then-current suite passed 20/20.
The measurements in this section were not collected from the final code.

### Retrieval evaluation

Scores are top-1 cosine similarities from normalized E5 embeddings.

| Question group | n | Expected section at top-1 | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|---:|
| Answerable | 10 | 10/10 | 0.8447 | 0.9024 | 0.8692 |
| Out-of-domain | 6 | Not applicable | 0.7020 | 0.7771 | 0.7412 |
| Unanswerable, on-topic | 6 | Not applicable | 0.7858 | 0.8653 | 0.8300 |

On-topic unanswerable scores overlap answerable scores. Similarity can filter
coarse irrelevance, but cannot establish that the requested fact exists.

### Paraphrase robustness

- 16/16 answerable paraphrases scored at least 0.82; minimum score: **0.8337**.
- 15/16 retrieved the expected section at top-1.
- The mismatch was semantically reasonable: a question about restoring systems
  ranked **9. Działania naprawcze** above **10. Przywracanie systemów**.

These observations support 0.82 as a default coarse filter for this small corpus
and evaluation set. They do not establish a universal E5 cutoff or guarantee
recall on new questions.

### Grounding and refusal at threshold 0.80

| Group | Observed outcome |
|---|---|
| Answerable controls | 2/2 answered |
| Out-of-domain | 3/3 rejected by the pipeline |
| On-topic, unanswerable | 5/6 correctly refused, by retrieval or the model |

The one unsupported inference was:

- Question: **Kto jest właścicielem tej procedury?**
- Top retrieval score: **0.8191**.
- At threshold **0.80**, passages reached Mistral, which answered **Organizacja**.
- That score predicted rejection at threshold **0.82**. This was initially an
  inference from the recorded score; the final run below subsequently confirmed it.

The failure motivated two changes: a corpus-specific default threshold of 0.82
and a stricter Polish prompt requiring explicit evidence and forbidding inference
from organization names, roles, context, or general knowledge. Some other
on-topic unanswerable scores exceed 0.82, so model refusal remains necessary
and fallible. These historical measurements did not evaluate the strengthened
prompt; that check is recorded in the final verification below.

## Final verification (post-update)

The final modified implementation completed its real-GPU smoke run in the
environment above, and **22/22 lightweight tests passed**. Defaults were
`k=3`, `score_threshold=0.82`, and `temperature=0.0`.

The smoke controls were:

- **Answerable:** “Jak należy zgłosić incydent?”
- **Known hard negative:** “Kto jest właścicielem tej procedury?”
- **Out-of-domain:** “W jakiej temperaturze wrze woda?”

| Case | Threshold | Temperature | Observed result | `no_context_refusal` |
|---|---:|---:|---|---|
| Answerable, greedy | 0.82 | 0.0 | Relevant passages kept; correct grounded answer | `False` |
| Answerable, sampling | 0.82 | 0.5 | Sampling succeeded; grounded answer remained correct | `False` |
| Hard negative, default filter | 0.82 | 0.0 | Top score ~0.8191; all passages rejected; fixed refusal | `True` |
| Hard negative, lower filter | 0.80 | 0.0 | Passages reached Mistral; exact model refusal | `False` |
| Out-of-domain control | 0.82 | 0.0 | Passages rejected; fixed refusal | `True` |

The fixed refusal and the exact model refusal were both
`Brak informacji w dokumencie.` At threshold 0.80, the strengthened prompt
corrected the previously observed unsupported answer, `Organizacja`, for this
tested question. The `False` flag distinguishes model refusal from the
deterministic no-context pipeline path. This does not guarantee refusal on all
unanswerable questions.

The previous project-owned Transformers warnings were **no longer observed**:

- `generation_config` passed together with generation-related kwargs;
- conflicting `max_length` and `max_new_tokens` settings;
- `clean_up_tokenization_spaces` for a BPE tokenizer.

The unauthenticated Hugging Face Hub request warning still appeared with
`HF_TOKEN` unset. This is expected, not a project error; no token requirement
or warning suppression was introduced.

GPU memory after final pipeline loading/use was approximately **5500 MiB
(~5.4 GiB)**. Both this and the earlier 5317 MiB reading are observed loaded-runtime
values, not peak-memory measurements or minimum-VRAM guarantees.

## Evidence and limitations

Historical retrieval aggregates and final smoke observations are separate; the
historical evaluation was not claimed to have been rerun in full on the final
code. The supplied evidence does not include the complete historical question
list, raw outputs, or model revision hashes, so this is a record of the available
evidence rather than a fully reproducible evaluation dataset.

Threshold 0.82 is specific to this corpus/evaluation set. Grounding remains
instruction-based and is not independently verified. The successful smoke cases
do not establish statistical significance, general model accuracy, or compatibility
outside the verified environment.
