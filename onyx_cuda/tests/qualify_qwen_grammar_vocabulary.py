"""Pinned offline qualification for the installed Qwen grammar-vocabulary builder.

This file is intentionally not named ``test_*.py``. The normal source suite remains independent
of a local Hugging Face cache; distribution qualification invokes this script from a clean
environment containing the built wheel, Hugging Face Hub, and Tokenizers only.
"""

from __future__ import annotations

import hashlib
import sys
import unicodedata
from pathlib import Path

import onyx_cuda
from onyx_cuda import build_qwen_grammar_vocabulary, load_qwen_tokenizer


EXPECTED_BASE_VOCAB_SIZE = 151_643
EXPECTED_VOCAB_SIZE = 151_665
EXPECTED_FINGERPRINT = "63ae520f9b74ae136cae96ce06470a10edfd3d5a3ae857d90b64ba8f870345f8"
SPECIAL_EMPTY_IDS = tuple(range(151_643, 151_657))
NON_SPECIAL_ADDED_TOKENS = (
    (151_657, b"<tool_call>"),
    (151_658, b"</tool_call>"),
    (151_659, b"<|fim_prefix|>"),
    (151_660, b"<|fim_middle|>"),
    (151_661, b"<|fim_suffix|>"),
    (151_662, b"<|fim_pad|>"),
    (151_663, b"<|repo_name|>"),
    (151_664, b"<|file_sep|>"),
)
CORPUS_CASES = (
    ("multilingual", "Hello, 你好 — café 🙂"),
    ("normalization", "Cafe\u0301 and A\u030a"),
    ("whitespace", "  leading\tspaces\r\ntrailing  "),
    ("control", "prefix\x00\x01\x02\n\t\r suffix"),
    (
        "added_tokens",
        "<tool_call></tool_call><|fim_prefix|><|fim_middle|><|fim_suffix|>"
        "<|fim_pad|><|repo_name|><|file_sep|>",
    ),
)
FORBIDDEN_RUNTIME_PREFIXES = (
    "onyx",
    "mlx",
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "onnxruntime",
    "psutil",
)


def main() -> None:
    _require_installed_package()
    if "huggingface_hub" in sys.modules or "tokenizers" in sys.modules:
        raise AssertionError("normal onyx_cuda import loaded tokenizer dependencies eagerly")
    _require_forbidden_runtimes_absent()

    loaded = load_qwen_tokenizer(local_files_only=True)
    tokenizer = loaded.tokenizer
    if tokenizer.base_vocab_size != EXPECTED_BASE_VOCAB_SIZE:
        raise AssertionError(f"unexpected pinned base vocabulary size: {tokenizer.base_vocab_size}")
    if tokenizer.vocab_size != EXPECTED_VOCAB_SIZE:
        raise AssertionError(f"unexpected pinned vocabulary size: {tokenizer.vocab_size}")

    first = build_qwen_grammar_vocabulary(tokenizer)
    second = build_qwen_grammar_vocabulary(tokenizer)
    if first != second or first is second:
        raise AssertionError(
            "independent grammar-vocabulary builds are not deterministic and uncached"
        )
    if len(first) != EXPECTED_VOCAB_SIZE:
        raise AssertionError(f"grammar vocabulary has {len(first)} entries")
    if any(not isinstance(token_bytes, bytes) for token_bytes in first):
        raise AssertionError("grammar vocabulary contains a non-bytes entry")

    empty_ids = tuple(token_id for token_id, token_bytes in enumerate(first) if not token_bytes)
    if empty_ids != SPECIAL_EMPTY_IDS:
        raise AssertionError(f"unexpected empty grammar token IDs: {empty_ids!r}")
    for token_id in SPECIAL_EMPTY_IDS:
        if first[token_id] != b"":
            raise AssertionError(f"special token {token_id} is not grammar-non-emitting")
    for token_id, expected in NON_SPECIAL_ADDED_TOKENS:
        if first[token_id] != expected:
            raise AssertionError(
                f"added token {token_id} bytes differ: {first[token_id]!r} != {expected!r}"
            )

    expected_partial = {160: "e4", 121: "bd", 254: "a0"}
    for token_id, expected_hex in expected_partial.items():
        if first[token_id].hex() != expected_hex:
            raise AssertionError(
                f"partial UTF-8 token {token_id} is {first[token_id].hex()}; "
                f"expected {expected_hex}"
            )
    if b"".join(first[token_id] for token_id in (160, 121, 254)) != "你".encode():
        raise AssertionError("characterized partial UTF-8 tokens do not reconstruct '你'")
    if first[18_947] != "个".encode():
        raise AssertionError("complete multibyte token 18947 does not reconstruct '个'")

    for label, text in CORPUS_CASES:
        token_ids = tokenizer.encode(text)
        reconstructed = b"".join(first[token_id] for token_id in token_ids)
        expected = unicodedata.normalize("NFC", text).encode("utf-8")
        if reconstructed != expected:
            raise AssertionError(
                f"{label} corpus reconstruction differs: "
                f"actual={reconstructed!r} expected={expected!r} ids={token_ids!r}"
            )

    fingerprint = _fingerprint(first)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise AssertionError(
            f"grammar-vocabulary fingerprint is {fingerprint}; expected {EXPECTED_FINGERPRINT}"
        )
    if "onyx_cuda._grammar_native" in sys.modules:
        raise AssertionError("grammar-vocabulary qualification loaded the native extension")
    _require_forbidden_runtimes_absent()

    print(
        "installed Qwen grammar-vocabulary qualification passed: "
        f"base_tokens={EXPECTED_BASE_VOCAB_SIZE} added_tokens=22 "
        f"empty_special_tokens={len(SPECIAL_EMPTY_IDS)} corpus_cases={len(CORPUS_CASES)} "
        f"fingerprint={fingerprint}"
    )


def _fingerprint(vocabulary: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    for token_bytes in vocabulary:
        digest.update(len(token_bytes).to_bytes(4, "big"))
        digest.update(token_bytes)
    return digest.hexdigest()


def _require_installed_package() -> None:
    package_path = Path(onyx_cuda.__file__).resolve()
    environment = Path(sys.prefix).resolve()
    if not package_path.is_relative_to(environment):
        raise AssertionError(
            f"qualification imported onyx_cuda outside the clean environment: {package_path}"
        )


def _require_forbidden_runtimes_absent() -> None:
    loaded = tuple(sys.modules)
    for prefix in FORBIDDEN_RUNTIME_PREFIXES:
        if any(name == prefix or name.startswith(f"{prefix}.") for name in loaded):
            raise AssertionError(f"qualification imported forbidden runtime {prefix!r}")


if __name__ == "__main__":
    main()
