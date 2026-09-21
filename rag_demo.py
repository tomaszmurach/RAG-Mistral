"""A small GPU RAG demo; importing this module never loads models."""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Any

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"
REFUSAL = "Brak informacji w dokumencie."
DEFAULT_K = 3
# Selected from a small evaluation of THIS corpus; not an answerability cutoff.
DEFAULT_SCORE_THRESHOLD = 0.82
DEFAULT_TEMPERATURE = 0.0

DOCUMENT = """
Procedura reagowania na incydenty bezpieczeństwa IT w organizacji

1. Cel procedury
Celem procedury jest zapewnienie spójnego i skutecznego reagowania na incydenty bezpieczeństwa IT w organizacji oraz minimalizacja ich skutków.

2. Definicja incydentu
Incydentem bezpieczeństwa IT jest każde zdarzenie, które może prowadzić do naruszenia poufności, integralności lub dostępności systemów informatycznych lub danych.

3. Przykłady incydentów
Do incydentów zalicza się między innymi: nieautoryzowany dostęp do systemu, wyciek danych, infekcję złośliwym oprogramowaniem, ataki phishingowe, ataki typu DDoS oraz utratę urządzenia służbowego.

4. Role i odpowiedzialności
Użytkownik końcowy jest zobowiązany do niezwłocznego zgłoszenia podejrzanego zdarzenia.
Administrator IT odpowiada za analizę techniczną incydentu.
Kierownik zespołu IT odpowiada za decyzję o eskalacji incydentu.
Zespół bezpieczeństwa IT odpowiada za koordynację działań naprawczych.

5. Zgłaszanie incydentu
Incydent należy zgłosić poprzez dedykowany system zgłoszeń IT lub drogą mailową.
Zgłoszenie powinno zawierać opis zdarzenia, czas wystąpienia oraz systemy, których dotyczy.

6. Klasyfikacja incydentu
Po otrzymaniu zgłoszenia administrator IT klasyfikuje incydent jako niski, średni lub krytyczny.
Klasyfikacja zależy od skali wpływu na systemy oraz potencjalnych skutków dla organizacji.

7. Reakcja na incydent
W przypadku incydentu niskiego poziomu administrator IT podejmuje działania naprawcze samodzielnie.
W przypadku incydentu średniego lub krytycznego kierownik zespołu IT podejmuje decyzję o eskalacji.

8. Eskalacja
Incydenty krytyczne są eskalowane do zespołu bezpieczeństwa IT.
W przypadku incydentów o charakterze prawnym informowany jest dział prawny organizacji.

9. Działania naprawcze
Działania naprawcze obejmują izolację zagrożonych systemów, usunięcie przyczyny incydentu oraz przywrócenie poprawnego działania usług.

10. Przywracanie systemów
Systemy przywracane są na podstawie aktualnych kopii zapasowych.
Po przywróceniu systemów wykonywane są testy poprawności działania.

11. Dokumentacja incydentu
Każdy incydent musi zostać udokumentowany w systemie zgłoszeń IT.
Dokumentacja zawiera opis incydentu, podjęte działania oraz wnioski.

12. Zamknięcie incydentu
Incydent uznaje się za zamknięty po usunięciu skutków oraz zatwierdzeniu przez kierownika zespołu IT.

13. Przegląd po incydencie
Po incydencie krytycznym przeprowadzany jest przegląd w celu zapobiegania podobnym zdarzeniom w przyszłości.

14. Zakres procedury
Procedura dotyczy wyłącznie systemów informatycznych organizacji.
Procedura nie obejmuje incydentów niezwiązanych z infrastrukturą IT ani zdarzeń losowych niezależnych od systemów informatycznych.

15. Postanowienia końcowe
Nieprzestrzeganie procedury może skutkować konsekwencjami służbowymi zgodnie z obowiązującymi zasadami organizacji.

"""


def chunk_text(text: str) -> list[str]:
    """Keep each numbered section intact; attach any preamble to the first one.

    This deliberately targets the short, numbered demo document, not arbitrary
    long documents. Empty input is invalid. Even heading-only sections are kept.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The source document must be a non-empty string.")

    chunks = []
    current = []
    seen_heading = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_heading = bool(re.match(r"^\d+\.\s+\S", line))
        if is_heading and seen_heading:
            chunks.append("\n".join(current))
            current = []
        current.append(line)
        seen_heading = seen_heading or is_heading
    if current:
        chunks.append("\n".join(current))
    return chunks


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


@dataclass
class Retriever:
    """One source's original chunks, embedding model, and normalized FAISS index."""

    chunks: tuple[str, ...]
    embedder: Any
    index: Any

    @classmethod
    def from_text(cls, text: str, embedder: Any) -> "Retriever":
        chunks = tuple(chunk_text(text))
        import faiss
        import numpy as np

        # E5 prefixes belong only to embedding inputs, never the stored text.
        embeddings = embedder.encode(
            [f"passage: {chunk}" for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(chunks, embedder, index)

    def retrieve_context(self, query: str, k: int = DEFAULT_K,
                         score_threshold: float | None = DEFAULT_SCORE_THRESHOLD) -> dict:
        """Return ranked/kept passages and context. Clamp k to the corpus size.

        Thresholds are finite cosine similarities in [-1, 1], not confidence
        probabilities. None disables filtering. Invalid inputs raise ValueError.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The question must be a non-empty string.")
        _positive_integer(k, "k")
        if score_threshold is not None:
            _finite_number(score_threshold, "score_threshold")
            if not -1 <= score_threshold <= 1:
                raise ValueError("score_threshold must be between -1 and 1.")
        if not self.chunks:
            raise ValueError("The retriever must contain at least one chunk.")

        import numpy as np

        question = query.strip()
        effective_k = min(k, len(self.chunks))
        q_emb = self.embedder.encode(
            [f"query: {question}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Inner product of normalized vectors is cosine similarity.
        scores, indices = self.index.search(
            np.ascontiguousarray(q_emb, dtype=np.float32), effective_k
        )
        retrieved = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:  # FAISS can use -1 for an unfilled result slot.
                continue
            retrieved.append({
                "rank": rank,
                "idx": int(idx),
                "score": float(score),
                "text": self.chunks[int(idx)],
            })
        kept = [item for item in retrieved
                if score_threshold is None or item["score"] >= score_threshold]
        return {
            "question": question,
            "context": "\n\n".join(item["text"] for item in kept),
            "retrieved_passages": retrieved,
            "kept_passages": kept,
            "retrieval_parameters": {
                "k": k,
                "effective_k": effective_k,
                "score_threshold": score_threshold,
            },
        }


@dataclass
class AnswerGenerator:
    """Generation state, separate from retrieval; both objects are loaded once."""

    tokenizer: Any
    generator: Any

    def generate(self, question: str, context: str, *,
                 temperature: float, max_new_tokens: int) -> str:
        # These are instructions, not independent verification of the answer.
        prompt_content = f"""
Odpowiadasz wyłącznie na podstawie sekcji KONTEKST.
Żądany fakt musi być podany wprost i jednoznacznie. Samo podobieństwo tematyczne nie wystarcza.
Nie wyciągaj wniosków o brakujących faktach z nazw organizacji, ról, kontekstu ani wiedzy ogólnej.
Jeśli kontekst jest związany z pytaniem, ale nie zawiera wprost żądanego faktu,
lub nie ma w nim potrzebnej informacji, zwróć DOKŁADNIE:
{REFUSAL}
Odpowiedz krótko: maksymalnie jedno zdanie, jedna linia, bez wypunktowań.
Jeśli pytanie prosi o listę, wypisz tylko elementy podane w kontekście, oddzielone przecinkami.

KONTEKST:
{context}

PYTANIE:
{question}

ODPOWIEDŹ: (krotko, jedna linia)"""
        # A single user message is sufficient for this Mistral instruction task.
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Keep token IDs/model defaults, but never mutate shared config per query.
        # Transformers 5.16.1 expects a config OR generation kwargs, not both.
        generation_config = deepcopy(self.generator.generation_config)
        generation_config.max_length = None
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = temperature > 0
        generation_config.num_beams = 1
        # Neutral when greedy, avoiding an inherited sampling temperature warning.
        generation_config.temperature = temperature if temperature > 0 else 1.0
        output = self.generator(
            prompt,
            # The chat template already supplies the model's special tokens.
            add_special_tokens=False,
            return_full_text=False,
            clean_up_tokenization_spaces=False,
            generation_config=generation_config,
        )[0]["generated_text"]
        # Preserve the completion rather than silently discarding later lines.
        return output.strip()


def ask_bot(question: str, retriever: Retriever, answer_generator: AnswerGenerator,
            *, k: int = DEFAULT_K, score_threshold: float | None = DEFAULT_SCORE_THRESHOLD,
            temperature: float = DEFAULT_TEMPERATURE, max_new_tokens: int = 70) -> dict:
    """Return an inspectable result without printing; reject invalid parameters.

    Generation parameters record the requested settings even when generation is
    skipped. no_context_refusal identifies only the deterministic pipeline path,
    not a refusal produced by the language model itself.
    """
    _finite_number(temperature, "temperature")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    _positive_integer(max_new_tokens, "max_new_tokens")
    retrieval = retriever.retrieve_context(question, k, score_threshold)
    no_context_refusal = not retrieval["kept_passages"]
    if no_context_refusal:
        answer = REFUSAL
    else:
        answer = answer_generator.generate(
            retrieval["question"], retrieval["context"],
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
    return {
        **retrieval,
        "answer": answer,
        "generation_parameters": {
            "temperature": temperature,
            "do_sample": temperature > 0,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
        },
        "no_context_refusal": no_context_refusal,
    }


def print_result(result: dict) -> None:
    """Console presentation only; query execution returns the full metadata."""
    print("-" * 80)
    print("PYTANIE:", result["question"], sep="\n")
    print("RETRIEVAL:", result["retrieval_parameters"])
    print("GENERATION:", result["generation_parameters"])
    kept_ids = {item["idx"] for item in result["kept_passages"]}
    for item in result["retrieved_passages"]:
        status = "KEPT" if item["idx"] in kept_ids else "DROP"
        print(f"  #{item['rank']} score={item['score']:.4f} {status}")
        print(item["text"])
    print("KONTEKST:", result["context"] or "Brak pasujacych fragmentow.", sep="\n")
    print("ODPOWIEDŹ:", result["answer"], sep="\n")
    print("NO-CONTEXT REFUSAL:", result["no_context_refusal"])


def load_pipeline(text: str = DOCUMENT) -> tuple[Retriever, AnswerGenerator]:
    """Initialize this single-GPU demo explicitly; never called during import."""
    chunk_text(text)  # Reject empty input before importing ML packages/downloading.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements.txt with a CUDA-enabled PyTorch build to run the demo."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This demo requires a CUDA-capable NVIDIA GPU and CUDA-enabled PyTorch. "
            "Both models use GPU 0; CPU inference is not supported."
        )

    try:
        # Check quantization dependencies before attempting model downloads.
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements.txt in a compatible CUDA environment; this demo "
            "requires bitsandbytes, accelerate, transformers, and sentence-transformers."
        ) from exc

    embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device="cuda:0")
    retriever = Retriever.from_text(text, embedder)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        quantization_config=bnb_config,
    )
    # In 5.16.1, the length-warning check consults BOTH the model config and
    # the per-call config before filling defaults. Clear the legacy limit here
    # and on each query so max_new_tokens is the only explicit length control.
    model.generation_config.max_length = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
    )
    return retriever, AnswerGenerator(tokenizer, generator)


def main() -> None:
    retriever, answer_generator = load_pipeline()
    # Representative cases, all using the corpus-specific defaults.
    demonstrations = [
        "Jak należy zgłosić incydent według dokumentu?",
        "Co powinien zrobić pracownik, gdy zauważy podejrzane zdarzenie?",
        "Kto jest właścicielem tej procedury?",
        "W jakiej temperaturze wrze woda?",
        "Kto odpowiada za analizę techniczną i kto za eskalację incydentu?",
    ]
    for question in demonstrations:
        result = ask_bot(question, retriever, answer_generator)
        print_result(result)


if __name__ == "__main__":
    main()
