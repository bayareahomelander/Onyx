import io
import json
import re
from dataclasses import dataclass

import pytest
import probe_cuda_tokenizer

from onyx_cuda.tokenizer_probe import (
    _load_tokenizer_metadata,
    build_grammar_vocabulary,
    byte_level_token_for_bytes,
    inspect_loaded_tokenizer,
)


@dataclass(frozen=True)
class FakeAddedToken:
    content: str
    special: bool


class FakeBackendTokenizer:
    def __init__(self, *, model_type="BPE", decoder_type="ByteLevel"):
        self.model_type = model_type
        self.decoder_type = decoder_type

    def to_str(self):
        return json.dumps(
            {
                "model": {"type": self.model_type},
                "decoder": {"type": self.decoder_type},
            }
        )


class FakeTokenizer:
    is_fast = True

    def __init__(self, text_to_id, *, config_width=8):
        self.text_to_id = dict(text_to_id)
        self.backend_tokenizer = FakeBackendTokenizer()
        self.all_special_ids = [len(self.text_to_id)]
        self.added_tokens_decoder = {
            self.all_special_ids[0]: FakeAddedToken("<|endoftext|>", special=True)
        }
        self._vocab = {
            byte_level_token_for_bytes(text.encode("utf-8")): token_id
            for text, token_id in self.text_to_id.items()
        }
        self._vocab["<|endoftext|>"] = self.all_special_ids[0]
        self.vocab_size = len(self._vocab)
        self.config_width = config_width

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [self.text_to_id[text]]


@dataclass
class FakeConfig:
    vocab_size: int = 8
    model_type: str = "qwen2"
    architectures: tuple = ("Qwen2ForCausalLM",)


class RecordingConfigLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        config = FakeConfig()
        config._commit_hash = "resolved-commit-sha"
        return config


class RecordingTokenizerLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        tokenizer, _ = make_compatible_tokenizer()
        return tokenizer


class LiteralGrammar:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pattern = None
        self.states = {}
        self.next_state = 1

    def compile_regex(self, pattern):
        self.pattern = re.compile(pattern)

    def init_state(self):
        return self._insert(b"")

    def _insert(self, value):
        state = self.next_state
        self.next_state += 1
        self.states[state] = value
        return state

    def get_valid_token_ids(self, state):
        prefix = self.states[state]
        valid = []
        for token_id, token_bytes in enumerate(self.vocabulary):
            try:
                candidate = (prefix + token_bytes).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if token_bytes and self.pattern.fullmatch(candidate):
                valid.append(token_id)
        return valid

    def advance_state(self, state, token_id):
        return self._insert(self.states[state] + self.vocabulary[token_id])

    def release_state(self, state):
        self.states.pop(state, None)

    def is_match_state(self, state):
        return self.pattern.fullmatch(self.states[state].decode("utf-8")) is not None


def make_compatible_tokenizer():
    samples = {
        "ONY-2026": 0,
        "café": 1,
        '{"name":"café"}': 2,
        "你好，Onyx": 3,
        "🙂": 4,
    }
    return FakeTokenizer(samples, config_width=8), FakeConfig(vocab_size=8)


def test_compatible_byte_level_tokenizer_aligns_with_padded_logits_width():
    tokenizer, config = make_compatible_tokenizer()

    report, vocabulary = inspect_loaded_tokenizer(
        tokenizer,
        config,
        model_id="fake/qwen",
        grammar_factory=LiteralGrammar,
    )

    assert report.compatible is True
    assert report.maximum_token_id == 5
    assert report.expected_logits_width == 8
    assert report.padded_or_missing_ids == 2
    assert report.special_token_ids == [5]
    assert vocabulary[5] == b""
    assert vocabulary[6:] == [b"", b""]
    assert all(check.matches for check in report.text_byte_checks)
    assert all(check.every_token_allowed for check in report.grammar_alignment_checks)
    assert all(check.grammar_matched for check in report.grammar_alignment_checks)


def test_metadata_loader_pins_tokenizer_to_resolved_config_revision():
    RecordingConfigLoader.calls.clear()
    RecordingTokenizerLoader.calls.clear()

    _, _, resolved_revision = _load_tokenizer_metadata(
        "fake/qwen",
        revision="requested-branch",
        local_files_only=True,
        config_loader=RecordingConfigLoader,
        tokenizer_loader=RecordingTokenizerLoader,
    )

    assert resolved_revision == "resolved-commit-sha"
    assert RecordingConfigLoader.calls == [
        (
            "fake/qwen",
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "revision": "requested-branch",
            },
        )
    ]
    assert RecordingTokenizerLoader.calls == [
        (
            "fake/qwen",
            {
                "use_fast": True,
                "local_files_only": True,
                "trust_remote_code": False,
                "revision": "resolved-commit-sha",
            },
        )
    ]


def test_build_vocabulary_rejects_token_id_outside_logits_width():
    tokenizer, _ = make_compatible_tokenizer()
    token = byte_level_token_for_bytes(b"outside")
    tokenizer._vocab[token] = 8

    _, _, errors = build_grammar_vocabulary(tokenizer, logits_width=8)

    assert any("outside logits range" in error for error in errors)


def test_probe_rejects_non_byte_level_base_token():
    tokenizer, config = make_compatible_tokenizer()
    tokenizer._vocab["∯"] = 6

    report, _ = inspect_loaded_tokenizer(
        tokenizer,
        config,
        model_id="fake/qwen",
        grammar_factory=LiteralGrammar,
    )

    assert report.compatible is False
    assert any("non-byte-level character" in error for error in report.errors)


def test_probe_detects_sequence_byte_reconstruction_mismatch():
    tokenizer, config = make_compatible_tokenizer()
    cafe_id = tokenizer.text_to_id["café"]
    original_token = next(
        token for token, token_id in tokenizer._vocab.items() if token_id == cafe_id
    )
    del tokenizer._vocab[original_token]
    tokenizer._vocab[byte_level_token_for_bytes(b"cafe")] = cafe_id

    report, _ = inspect_loaded_tokenizer(
        tokenizer,
        config,
        model_id="fake/qwen",
        grammar_factory=LiteralGrammar,
        text_samples=("café",),
        grammar_samples=(),
    )

    assert report.compatible is False
    assert any("do not reconstruct UTF-8" in error for error in report.errors)


def test_probe_rejects_slow_or_uninspectable_tokenizer():
    tokenizer, config = make_compatible_tokenizer()
    tokenizer.is_fast = False
    tokenizer.backend_tokenizer = None

    report, _ = inspect_loaded_tokenizer(
        tokenizer,
        config,
        model_id="fake/qwen",
        grammar_factory=LiteralGrammar,
    )

    assert report.compatible is False
    assert any("fast tokenizer" in error for error in report.errors)
    assert any("does not expose" in error for error in report.errors)


def test_constructed_vocabulary_aligns_with_real_rust_grammar_when_available():
    try:
        from onyx._rust import GrammarConstraint
    except ImportError:
        pytest.skip("onyx Rust extension is not available")

    tokenizer, config = make_compatible_tokenizer()
    report, _ = inspect_loaded_tokenizer(
        tokenizer,
        config,
        model_id="fake/qwen",
        grammar_factory=GrammarConstraint,
    )

    assert report.compatible is True
    assert all(check.grammar_matched for check in report.grammar_alignment_checks)


def test_report_dictionary_includes_derived_compatibility():
    tokenizer, config = make_compatible_tokenizer()
    report, _ = inspect_loaded_tokenizer(
        tokenizer,
        config,
        model_id="fake/qwen",
        grammar_factory=LiteralGrammar,
        requested_revision="main",
        resolved_revision="abc123",
    )

    serialized = report.to_dict()

    assert serialized["compatible"] is True
    assert serialized["requested_revision"] == "main"
    assert serialized["resolved_revision"] == "abc123"
    assert serialized["text_byte_checks"][0]["token_ids"] == (0,)


def test_console_output_escapes_unicode_for_cp1252(monkeypatch):
    output = io.BytesIO()
    console = io.TextIOWrapper(output, encoding="cp1252", errors="strict")
    monkeypatch.setattr(probe_cuda_tokenizer.sys, "stdout", console)

    probe_cuda_tokenizer.print_console_safe("你好🙂")
    console.flush()

    rendered = output.getvalue().decode("cp1252")
    assert rendered.splitlines() == [r"\u4f60\u597d\U0001f642"]
