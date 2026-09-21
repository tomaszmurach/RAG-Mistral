"""Small CPU checks with model/FAISS doubles; no downloads or GPU inference."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import rag_demo as rag


class FakeIndex:
    """Exact NumPy inner products for exercising code around the FAISS API."""

    def __init__(self, dim):
        self.dim = dim

    def add(self, vectors):
        self.vectors = vectors

    def search(self, query, k):
        self.last_k = k
        self.last_query = query
        scores = query @ self.vectors.T
        indices = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, indices, axis=1), indices


class ChunkingTests(unittest.TestCase):
    def test_embedded_document_preserves_every_nonempty_line(self):
        chunks = rag.chunk_text(rag.DOCUMENT)
        self.assertEqual(len(chunks), 15)
        self.assertEqual(
            "\n".join(chunks).splitlines(),
            [line.strip() for line in rag.DOCUMENT.splitlines() if line.strip()],
        )
        for number, chunk in enumerate(chunks, start=1):
            heading = next(line for line in chunk.splitlines() if line.startswith(f"{number}. "))
            self.assertIn(heading, chunk)
        self.assertIn("Administrator IT odpowiada za analizę techniczną", chunks[3])
        self.assertIn("Zespół bezpieczeństwa IT odpowiada za koordynację", chunks[3])

    def test_preamble_and_consecutive_or_trailing_headings_are_preserved(self):
        self.assertEqual(
            rag.chunk_text("Title\n1. First\n2. Second\nBody\nMore\n3. Last"),
            ["Title\n1. First", "2. Second\nBody\nMore", "3. Last"],
        )

    def test_long_heading_and_unnumbered_text(self):
        text = "1. A section heading longer than the previous six word limit\nBody\n2. Next\nEnd"
        self.assertEqual(len(rag.chunk_text(text)), 2)
        self.assertEqual(rag.chunk_text(" First \n\n Second "), ["First\nSecond"])

    def test_empty_document_is_rejected_before_embedding(self):
        embedder = Mock()
        for text in ("", " \n\t", None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                rag.Retriever.from_text(text, embedder)
        embedder.encode.assert_not_called()


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.text = "Title\n1. Alpha\nFirst fact\n2. Beta\nSecond fact\n3. Gamma\nThird fact"
        self.embedder = Mock()
        self.embedder.encode.side_effect = lambda texts, **kwargs: (
            np.array([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float64)
            if len(texts) == 3 else np.array([[1, 0]], dtype=np.float64)
        )
        with patch.dict(sys.modules, {"faiss": SimpleNamespace(IndexFlatIP=FakeIndex)}):
            self.retriever = rag.Retriever.from_text(self.text, self.embedder)
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.return_value = "template-formatted prompt"
        self.generator = Mock(return_value=[{"generated_text": "  First line.\nSecond line.  "}])
        self.generator.generation_config = SimpleNamespace(
            max_length=4096, max_new_tokens=256, do_sample=True,
            temperature=0.7, num_beams=1, eos_token_id=2, pad_token_id=2,
        )
        self.answer_generator = rag.AnswerGenerator(self.tokenizer, self.generator)

    def ask(self, **kwargs):
        return rag.ask_bot("  A question?  ", self.retriever, self.answer_generator, **kwargs)

    def test_prefixes_only_reach_embeddings_and_normalized_float32_reaches_index(self):
        expected_chunks = rag.chunk_text(self.text)
        self.embedder.encode.assert_called_once_with(
            [f"passage: {chunk}" for chunk in expected_chunks],
            normalize_embeddings=True, convert_to_numpy=True,
        )
        result = self.ask()
        self.embedder.encode.assert_called_with(
            ["query: A question?"], normalize_embeddings=True, convert_to_numpy=True,
        )
        self.assertEqual(list(self.retriever.chunks), expected_chunks)
        self.assertEqual([item["text"] for item in result["retrieved_passages"]], expected_chunks)
        message = self.tokenizer.apply_chat_template.call_args.args[0][0]
        self.assertIn(result["context"], message["content"])
        self.assertNotIn("passage: ", message["content"])
        self.assertNotIn("query: ", message["content"])
        for vectors in (self.retriever.index.vectors, self.retriever.index.last_query):
            self.assertEqual(vectors.dtype, np.float32)
            self.assertTrue(vectors.flags.c_contiguous)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rag.print_result(result)
        self.assertNotIn("passage: ", output.getvalue())

    def test_filtering_preserves_ranked_metadata_and_only_prompts_with_kept_text(self):
        result = self.ask(score_threshold=0.5)
        self.assertEqual(len(result["retrieved_passages"]), 3)
        self.assertEqual([item["idx"] for item in result["kept_passages"]], [0, 1])
        self.assertEqual([item["rank"] for item in result["retrieved_passages"]], [1, 2, 3])
        self.assertNotIn("Third fact", result["context"])
        self.assertNotIn("Third fact", self.tokenizer.apply_chat_template.call_args.args[0][0]["content"])
        self.assertFalse(result["no_context_refusal"])
        self.assertEqual(result["question"], "A question?")
        self.assertEqual(result["generation_parameters"]["max_new_tokens"], 70)
        json.dumps(result)  # Results can be recorded without custom serialization.

    def test_threshold_boundary_is_inclusive_and_none_disables_filtering(self):
        self.assertEqual(len(self.ask(score_threshold=1.0)["kept_passages"]), 1)
        self.assertEqual(len(self.ask(score_threshold=None)["kept_passages"]), 3)
        self.assertEqual(len(self.ask(score_threshold=-1.0)["kept_passages"]), 3)

    def test_no_context_refusal_never_calls_tokenizer_or_generator(self):
        self.retriever.index.search = Mock(return_value=(
            np.array([[0.3, 0.2, 0.1]]), np.array([[0, 1, 2]]),
        ))
        result = self.ask(score_threshold=0.9)
        self.assertEqual(result["answer"], "Brak informacji w dokumencie.")
        self.assertTrue(result["no_context_refusal"])
        self.assertEqual(result["context"], "")
        self.assertEqual(len(result["retrieved_passages"]), 3)
        self.assertEqual(result["kept_passages"], [])
        self.tokenizer.apply_chat_template.assert_not_called()
        self.generator.assert_not_called()

    def test_model_refusal_is_not_reported_as_pipeline_refusal(self):
        self.generator.return_value = [{"generated_text": rag.REFUSAL}]
        result = self.ask()
        self.assertEqual(result["answer"], rag.REFUSAL)
        self.assertFalse(result["no_context_refusal"])

    def test_corpus_defaults_keep_control_but_reject_observed_hard_negative_score(self):
        # Supplied evaluation scores exercise filtering, not simulated LLM quality.
        self.retriever.index.search = Mock(return_value=(
            np.array([[0.8337, 0.8191, 0.7771]]), np.array([[0, 1, 2]]),
        ))
        result = self.ask()
        self.assertEqual(result["retrieval_parameters"], {
            "k": 3, "effective_k": 3, "score_threshold": 0.82,
        })
        self.assertEqual(result["generation_parameters"]["temperature"], 0.0)
        self.assertEqual([item["idx"] for item in result["kept_passages"]], [0])
        self.retriever.index.search.return_value = (
            np.array([[0.8191, 0.7771, 0.7020]]), np.array([[0, 1, 2]]),
        )
        self.generator.reset_mock()
        self.assertEqual(self.retriever.retrieve_context("Question?")["kept_passages"], [])
        self.assertTrue(self.ask()["no_context_refusal"])
        self.generator.assert_not_called()
        self.assertFalse(self.ask(score_threshold=0.80)["no_context_refusal"])

    def test_chat_template_and_greedy_generation_preserve_completion(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.ask(temperature=0, max_new_tokens=90)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(result["answer"], "First line.\nSecond line.")
        args, kwargs = self.tokenizer.apply_chat_template.call_args
        self.assertEqual(len(args[0]), 1)
        self.assertEqual(args[0][0]["role"], "user")
        self.assertIn(rag.REFUSAL, args[0][0]["content"])
        self.assertIn("podany wprost i jednoznacznie", args[0][0]["content"])
        self.assertIn("Samo podobieństwo tematyczne nie wystarcza", args[0][0]["content"])
        self.assertIn(
            "Nie wyciągaj wniosków o brakujących faktach z nazw organizacji, ról, kontekstu ani wiedzy ogólnej",
            args[0][0]["content"],
        )
        self.assertNotIn("[INST]", args[0][0]["content"])
        self.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": True})
        self.generator.assert_called_once()
        generation_args, generation_kwargs = self.generator.call_args
        self.assertEqual(generation_args, ("template-formatted prompt",))
        generation_kwargs = dict(generation_kwargs)
        config = generation_kwargs.pop("generation_config")
        self.assertEqual(generation_kwargs, {
            "add_special_tokens": False, "return_full_text": False,
            "clean_up_tokenization_spaces": False,
        })  # No generation-related kwargs accompany the config.
        self.assertIsNone(config.max_length)
        self.assertEqual(config.max_new_tokens, 90)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.num_beams, 1)
        self.assertEqual(config.temperature, 1.0)  # Neutral, ignored in greedy mode.
        self.assertEqual((config.eos_token_id, config.pad_token_id), (2, 2))
        self.assertIsNot(config, self.generator.generation_config)
        self.assertEqual(self.generator.generation_config.max_length, 4096)
        self.assertEqual(self.generator.generation_config.temperature, 0.7)

    def test_positive_temperature_enables_sampling(self):
        result = self.ask(temperature=0.5)
        config = self.generator.call_args.kwargs["generation_config"]
        self.assertTrue(config.do_sample)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(result["generation_parameters"]["temperature"], 0.5)
        self.ask(temperature=0)
        self.assertFalse(self.generator.call_args.kwargs["generation_config"].do_sample)
        self.assertTrue(config.do_sample)  # Later calls do not mutate earlier configs.
        self.assertEqual(config.temperature, 0.5)

    def test_oversized_k_is_capped_and_recorded(self):
        result = self.ask(k=100)
        self.assertEqual(self.retriever.index.last_k, 3)
        self.assertEqual(result["retrieval_parameters"], {
            "k": 100, "effective_k": 3, "score_threshold": 0.82,
        })

    def test_invalid_questions_do_not_encode_or_generate(self):
        self.embedder.reset_mock()
        for question in ("", " \n", None, 42):
            with self.subTest(question=question), self.assertRaises(ValueError):
                rag.ask_bot(question, self.retriever, self.answer_generator)
        self.embedder.encode.assert_not_called()
        self.generator.assert_not_called()

    def test_invalid_parameters_fail_before_query_embedding(self):
        invalid = {
            "k": [0, -1, 1.5, True, "3"],
            "score_threshold": [-1.1, 1.1, float("nan"), float("inf"), True, "0.5"],
            "temperature": [-0.1, float("nan"), float("inf"), True, None],
            "max_new_tokens": [0, -1, 1.5, True],
        }
        self.embedder.reset_mock()
        for parameter, values in invalid.items():
            for value in values:
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    self.ask(**{parameter: value})
        self.embedder.encode.assert_not_called()
        self.generator.assert_not_called()

    def test_faiss_missing_slots_do_not_select_last_chunk(self):
        self.retriever.index.search = Mock(return_value=(
            np.array([[0.9, -float("inf")]]), np.array([[0, -1]]),
        ))
        result = self.ask(k=2)
        self.assertEqual([item["idx"] for item in result["retrieved_passages"]], [0])


class InitializationTests(unittest.TestCase):
    def test_model_length_and_padding_config_are_set_before_pipeline_construction(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), float16="float16")
        model = SimpleNamespace(generation_config=SimpleNamespace(max_length=4096, pad_token_id=None))
        tokenizer = SimpleNamespace(eos_token="</s>", pad_token_id=2)

        def create_pipeline(*args, **kwargs):
            self.assertIsNone(model.generation_config.max_length)
            self.assertEqual(model.generation_config.pad_token_id, 2)
            self.assertEqual(args, ("text-generation",))
            self.assertEqual(kwargs, {"model": model, "tokenizer": tokenizer})
            return Mock()

        modules = {
            "torch": torch, "accelerate": SimpleNamespace(), "bitsandbytes": SimpleNamespace(),
            "sentence_transformers": SimpleNamespace(SentenceTransformer=Mock()),
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                BitsAndBytesConfig=Mock(), pipeline=create_pipeline,
            ),
        }
        with patch.dict(sys.modules, modules), patch.object(rag.Retriever, "from_text"):
            rag.load_pipeline()

    def test_import_needs_no_ml_packages_and_has_no_side_effect_output(self):
        script = """
import sys
for name in ('torch', 'numpy', 'faiss', 'transformers', 'sentence_transformers', 'bitsandbytes', 'accelerate'):
    sys.modules[name] = None
import rag_demo
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(rag.__file__).resolve().parent,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_no_cuda_fails_before_model_imports(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": torch, "transformers": None, "sentence_transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "CUDA-capable NVIDIA GPU"):
                rag.load_pipeline()

    def test_missing_torch_reports_setup_requirement(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements.txt"):
                rag.load_pipeline()

    def test_missing_quantization_dependency_fails_before_model_imports(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {"torch": torch, "accelerate": None, "transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "bitsandbytes"):
                rag.load_pipeline()

    def test_empty_document_fails_before_torch_import(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "source document"):
                rag.load_pipeline("  ")


if __name__ == "__main__":
    unittest.main()
