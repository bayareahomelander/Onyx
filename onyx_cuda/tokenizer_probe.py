"""Tokenizer/vocabulary compatibility checks for the Windows CUDA path.

This module deliberately stops before model-weight loading or inference.  It
verifies the token-ID boundary shared by a Hugging Face tokenizer, the Rust
``GrammarConstraint`` vocabulary, and a future CUDA logits tensor.
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TEXT_SAMPLES = (
    "ONY-2026",
    '{"name":"café"}',
    "你好，Onyx",
    "🙂",
)
DEFAULT_GRAMMAR_SAMPLES = ("ONY-2026", "café")


@dataclass(frozen=True)
class TextByteCheck:
    text: str
    token_ids: Tuple[int, ...]
    expected_hex: str
    actual_hex: str
    matches: bool


@dataclass(frozen=True)
class GrammarAlignmentCheck:
    text: str
    token_ids: Tuple[int, ...]
    every_token_allowed: bool
    grammar_matched: bool
    error: Optional[str] = None


@dataclass
class TokenizerProbeReport:
    model_id: str
    requested_revision: Optional[str]
    resolved_revision: Optional[str]
    tokenizer_class: str
    tokenizer_is_fast: bool
    backend_model_type: Optional[str]
    backend_decoder_type: Optional[str]
    config_model_type: Optional[str]
    config_architectures: List[str]
    tokenizer_vocab_size: int
    tokenizer_length: int
    maximum_token_id: int
    config_vocab_size: int
    expected_logits_width: int
    populated_token_ids: int
    padded_or_missing_ids: int
    special_token_ids: List[int]
    regular_added_token_ids: List[int]
    duplicate_byte_sequences: int
    text_byte_checks: List[TextByteCheck] = field(default_factory=list)
    grammar_alignment_checks: List[GrammarAlignmentCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rss_before_bytes: Optional[int] = None
    rss_loaded_bytes: Optional[int] = None
    rss_after_cleanup_bytes: Optional[int] = None
    cuda_allocated_before_bytes: Optional[int] = None
    cuda_allocated_loaded_bytes: Optional[int] = None
    cuda_allocated_after_cleanup_bytes: Optional[int] = None

    @property
    def compatible(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["compatible"] = self.compatible
        return result


def _gpt2_byte_encoder() -> Dict[int, str]:
    """Return the reversible byte-to-Unicode map used by byte-level BPE."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)

    extra_index = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + extra_index)
            extra_index += 1

    return dict(zip(byte_values, (chr(value) for value in unicode_values)))


_BYTE_ENCODER = _gpt2_byte_encoder()
_BYTE_DECODER = {value: key for key, value in _BYTE_ENCODER.items()}


def byte_level_token_for_bytes(value: bytes) -> str:
    """Encode bytes into the token-string alphabet used by byte-level BPE."""
    return "".join(_BYTE_ENCODER[byte] for byte in value)


def _decode_byte_level_token(token: str) -> bytes:
    try:
        return bytes(_BYTE_DECODER[character] for character in token)
    except KeyError as exc:
        codepoint = f"U+{ord(exc.args[0]):04X}"
        raise ValueError(f"token contains non-byte-level character {codepoint}") from exc


def _backend_metadata(tokenizer: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        return None, None, "tokenizer does not expose a fast-tokenizer backend"

    try:
        backend_config = json.loads(backend.to_str())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, None, f"could not inspect tokenizer backend: {exc}"

    model_type = backend_config.get("model", {}).get("type")
    decoder = backend_config.get("decoder") or {}
    decoder_type = decoder.get("type")
    if decoder_type == "Sequence":
        child_types = [item.get("type") for item in decoder.get("decoders", [])]
        decoder_type = "+".join(item for item in child_types if item) or "Sequence"

    error = None
    if model_type != "BPE":
        error = f"unsupported tokenizer backend model {model_type!r}; expected byte-level BPE"
    elif decoder_type is None or "ByteLevel" not in decoder_type:
        error = f"unsupported tokenizer decoder {decoder_type!r}; expected ByteLevel"

    return model_type, decoder_type, error


def _added_token_metadata(tokenizer: Any) -> Dict[int, Any]:
    decoder = getattr(tokenizer, "added_tokens_decoder", {})
    return {int(token_id): token for token_id, token in dict(decoder).items()}


def build_grammar_vocabulary(
    tokenizer: Any,
    logits_width: int,
) -> Tuple[List[bytes], Dict[str, Any], List[str]]:
    """Build an ID-aligned byte vocabulary for ``GrammarConstraint``.

    Empty byte strings intentionally reserve special, missing, and padded IDs so
    the list index always remains identical to the future logits index.
    """
    if isinstance(logits_width, bool) or not isinstance(logits_width, int) or logits_width < 1:
        raise ValueError("logits_width must be a positive integer")

    raw_vocab = tokenizer.get_vocab()
    if not isinstance(raw_vocab, dict) or not raw_vocab:
        raise ValueError("tokenizer.get_vocab() must return a non-empty mapping")

    vocabulary = [b""] * logits_width
    errors: List[str] = []
    seen_ids: Dict[int, str] = {}
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    added_tokens = _added_token_metadata(tokenizer)
    regular_added_ids = {
        token_id
        for token_id, token in added_tokens.items()
        if token_id not in special_ids and not bool(getattr(token, "special", False))
    }

    mapped_ids = set()
    for token, raw_token_id in raw_vocab.items():
        if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
            errors.append(f"token {token!r} has non-integer ID {raw_token_id!r}")
            continue

        token_id = int(raw_token_id)
        if token_id < 0 or token_id >= logits_width:
            errors.append(
                f"token {token!r} has ID {token_id}, outside logits range [0, {logits_width})"
            )
            continue
        if token_id in seen_ids and seen_ids[token_id] != token:
            errors.append(
                f"token ID {token_id} is assigned to both {seen_ids[token_id]!r} and {token!r}"
            )
            continue

        seen_ids[token_id] = token
        mapped_ids.add(token_id)

        if token_id in special_ids or bool(getattr(added_tokens.get(token_id), "special", False)):
            continue

        if token_id in regular_added_ids:
            content = str(getattr(added_tokens[token_id], "content", token))
            vocabulary[token_id] = content.encode("utf-8")
            continue

        try:
            vocabulary[token_id] = _decode_byte_level_token(str(token))
        except ValueError as exc:
            errors.append(f"token ID {token_id} ({token!r}) cannot be converted to bytes: {exc}")

    byte_to_ids: Dict[bytes, List[int]] = {}
    for token_id, token_bytes in enumerate(vocabulary):
        if token_bytes:
            byte_to_ids.setdefault(token_bytes, []).append(token_id)
    duplicate_byte_sequences = sum(1 for ids in byte_to_ids.values() if len(ids) > 1)

    stats = {
        "populated_token_ids": len(mapped_ids),
        "padded_or_missing_ids": logits_width - len(mapped_ids),
        "special_token_ids": sorted(
            token_id for token_id in special_ids if token_id < logits_width
        ),
        "regular_added_token_ids": sorted(
            token_id for token_id in regular_added_ids if token_id < logits_width
        ),
        "duplicate_byte_sequences": duplicate_byte_sequences,
    }
    return vocabulary, stats, errors


def _encode_without_special_tokens(tokenizer: Any, text: str) -> Tuple[int, ...]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return tuple(int(token_id) for token_id in token_ids)


def check_text_byte_reconstruction(
    tokenizer: Any,
    vocabulary: Sequence[bytes],
    samples: Iterable[str],
) -> List[TextByteCheck]:
    checks = []
    for text in samples:
        token_ids = _encode_without_special_tokens(tokenizer, text)
        expected = text.encode("utf-8")
        if any(token_id < 0 or token_id >= len(vocabulary) for token_id in token_ids):
            actual = b""
        else:
            actual = b"".join(vocabulary[token_id] for token_id in token_ids)
        checks.append(
            TextByteCheck(
                text=text,
                token_ids=token_ids,
                expected_hex=expected.hex(),
                actual_hex=actual.hex(),
                matches=actual == expected,
            )
        )
    return checks


def check_grammar_alignment(
    tokenizer: Any,
    vocabulary: Sequence[bytes],
    grammar_factory: Callable[[Sequence[bytes]], Any],
    samples: Iterable[str],
) -> List[GrammarAlignmentCheck]:
    checks = []
    for text in samples:
        token_ids = _encode_without_special_tokens(tokenizer, text)
        every_token_allowed = True
        matched = False
        error = None
        constraint = None
        state = None

        try:
            constraint = grammar_factory(list(vocabulary))
            constraint.compile_regex(re.escape(text))
            state = constraint.init_state()
            for token_id in token_ids:
                valid_ids = constraint.get_valid_token_ids(state)
                if token_id not in valid_ids:
                    every_token_allowed = False
                    break
                next_state = constraint.advance_state(state, token_id)
                constraint.release_state(state)
                state = next_state
            if every_token_allowed:
                matched = bool(constraint.is_match_state(state))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if constraint is not None and state is not None:
                try:
                    constraint.release_state(state)
                except Exception:
                    pass

        checks.append(
            GrammarAlignmentCheck(
                text=text,
                token_ids=token_ids,
                every_token_allowed=every_token_allowed,
                grammar_matched=matched,
                error=error,
            )
        )
    return checks


def inspect_loaded_tokenizer(
    tokenizer: Any,
    config: Any,
    *,
    model_id: str,
    grammar_factory: Callable[[Sequence[bytes]], Any],
    requested_revision: Optional[str] = None,
    resolved_revision: Optional[str] = None,
    text_samples: Iterable[str] = DEFAULT_TEXT_SAMPLES,
    grammar_samples: Iterable[str] = DEFAULT_GRAMMAR_SAMPLES,
) -> Tuple[TokenizerProbeReport, List[bytes]]:
    """Inspect already-loaded metadata without loading model weights."""
    errors: List[str] = []
    warnings: List[str] = []

    is_fast = bool(getattr(tokenizer, "is_fast", False))
    if not is_fast:
        errors.append("tokenizer must be a fast tokenizer so its backend can be inspected")

    backend_model_type, backend_decoder_type, backend_error = _backend_metadata(tokenizer)
    if backend_error:
        errors.append(backend_error)

    config_vocab_size = getattr(config, "vocab_size", None)
    if (
        isinstance(config_vocab_size, bool)
        or not isinstance(config_vocab_size, int)
        or config_vocab_size < 1
    ):
        raise ValueError("model config must expose a positive integer vocab_size")

    raw_vocab = tokenizer.get_vocab()
    integer_ids = [
        int(token_id)
        for token_id in raw_vocab.values()
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    ]
    maximum_token_id = max(integer_ids, default=-1)

    vocabulary, stats, mapping_errors = build_grammar_vocabulary(tokenizer, config_vocab_size)
    errors.extend(mapping_errors)

    text_checks = check_text_byte_reconstruction(tokenizer, vocabulary, text_samples)
    for check in text_checks:
        if not check.matches:
            errors.append(
                f"token bytes do not reconstruct UTF-8 input for {check.text!a}: "
                f"expected {check.expected_hex}, got {check.actual_hex}"
            )

    grammar_checks = check_grammar_alignment(
        tokenizer,
        vocabulary,
        grammar_factory,
        grammar_samples,
    )
    for check in grammar_checks:
        if check.error:
            errors.append(f"grammar alignment check failed for {check.text!a}: {check.error}")
        elif not check.every_token_allowed or not check.grammar_matched:
            errors.append(
                f"grammar rejected tokenizer IDs for {check.text!a} "
                f"(allowed={check.every_token_allowed}, matched={check.grammar_matched})"
            )

    tokenizer_length = len(tokenizer)
    tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size", tokenizer_length))
    if tokenizer_length > config_vocab_size:
        errors.append(
            f"tokenizer length {tokenizer_length} exceeds config vocab size {config_vocab_size}"
        )
    if stats["padded_or_missing_ids"]:
        warnings.append(
            f"reserved {stats['padded_or_missing_ids']} empty vocabulary entries so grammar IDs "
            "remain aligned with the configured logits width"
        )
    if stats["duplicate_byte_sequences"]:
        warnings.append(
            f"found {stats['duplicate_byte_sequences']} byte sequences represented by multiple IDs"
        )
    if resolved_revision is None:
        warnings.append(
            "model source did not expose a resolved revision; preserve the local artifact "
            "separately if this report must be reproduced"
        )

    architectures = getattr(config, "architectures", None) or []
    report = TokenizerProbeReport(
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_is_fast=is_fast,
        backend_model_type=backend_model_type,
        backend_decoder_type=backend_decoder_type,
        config_model_type=getattr(config, "model_type", None),
        config_architectures=[str(item) for item in architectures],
        tokenizer_vocab_size=tokenizer_vocab_size,
        tokenizer_length=tokenizer_length,
        maximum_token_id=maximum_token_id,
        config_vocab_size=config_vocab_size,
        expected_logits_width=config_vocab_size,
        populated_token_ids=stats["populated_token_ids"],
        padded_or_missing_ids=stats["padded_or_missing_ids"],
        special_token_ids=stats["special_token_ids"],
        regular_added_token_ids=stats["regular_added_token_ids"],
        duplicate_byte_sequences=stats["duplicate_byte_sequences"],
        text_byte_checks=text_checks,
        grammar_alignment_checks=grammar_checks,
        errors=errors,
        warnings=warnings,
    )
    return report, vocabulary


def _process_rss_bytes() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _cuda_allocated_bytes() -> Optional[int]:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_allocated())


def _load_tokenizer_metadata(
    model_id: str,
    *,
    revision: Optional[str],
    local_files_only: bool,
    config_loader: Any = None,
    tokenizer_loader: Any = None,
) -> Tuple[Any, Any, Optional[str]]:
    """Load config first and pin tokenizer files to its resolved snapshot."""
    if config_loader is None or tokenizer_loader is None:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "tokenizer probe requires Transformers; install with "
                "`python -m pip install -e .[cuda]`"
            ) from exc
        config_loader = AutoConfig
        tokenizer_loader = AutoTokenizer

    common_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    config_kwargs = dict(common_kwargs)
    if revision is not None:
        config_kwargs["revision"] = revision

    config = config_loader.from_pretrained(model_id, **config_kwargs)
    resolved_revision = getattr(config, "_commit_hash", None)

    tokenizer_kwargs = dict(common_kwargs)
    tokenizer_revision = resolved_revision or revision
    if tokenizer_revision is not None:
        tokenizer_kwargs["revision"] = tokenizer_revision
    tokenizer = tokenizer_loader.from_pretrained(
        model_id,
        use_fast=True,
        **tokenizer_kwargs,
    )
    return config, tokenizer, resolved_revision


def run_tokenizer_probe(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> TokenizerProbeReport:
    """Load tokenizer/config metadata and run the compatibility probe."""
    try:
        from onyx._rust import GrammarConstraint
    except ImportError as exc:
        raise RuntimeError(
            "tokenizer probe requires the Rust extension; run `python -m maturin develop --release`"
        ) from exc

    rss_before = _process_rss_bytes()
    cuda_before = _cuda_allocated_bytes()
    config, tokenizer, resolved_revision = _load_tokenizer_metadata(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    rss_loaded = _process_rss_bytes()
    cuda_loaded = _cuda_allocated_bytes()
    vocabulary = None

    try:
        report, vocabulary = inspect_loaded_tokenizer(
            tokenizer,
            config,
            model_id=model_id,
            grammar_factory=GrammarConstraint,
            requested_revision=revision,
            resolved_revision=resolved_revision,
        )
        report.rss_before_bytes = rss_before
        report.rss_loaded_bytes = rss_loaded
        report.cuda_allocated_before_bytes = cuda_before
        report.cuda_allocated_loaded_bytes = cuda_loaded
    finally:
        del vocabulary
        del tokenizer
        del config
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    report.rss_after_cleanup_bytes = _process_rss_bytes()
    report.cuda_allocated_after_cleanup_bytes = _cuda_allocated_bytes()
    return report


def format_probe_report(report: TokenizerProbeReport) -> str:
    """Format a compact human-readable compatibility report."""
    status = "COMPATIBLE" if report.compatible else "INCOMPATIBLE"
    lines = [
        f"Tokenizer compatibility probe: {status}",
        f"  Model candidate: {report.model_id}",
        f"  Requested revision: {report.requested_revision or 'default'}",
        f"  Resolved revision: {report.resolved_revision or 'unavailable'}",
        (
            f"  Backend: {report.tokenizer_class} "
            f"({report.backend_model_type}/{report.backend_decoder_type})"
        ),
        (
            f"  ID space: tokenizer_len={report.tokenizer_length}, "
            f"max_id={report.maximum_token_id}, logits_width={report.expected_logits_width}"
        ),
        (
            f"  Reserved IDs: {report.padded_or_missing_ids}; "
            f"special IDs: {len(report.special_token_ids)}"
        ),
        "  Text byte checks: "
        + ", ".join(
            f"{check.text!a}={'ok' if check.matches else 'FAIL'}"
            for check in report.text_byte_checks
        ),
        "  Grammar checks: "
        + ", ".join(
            f"{check.text!a}={'ok' if check.every_token_allowed and check.grammar_matched and not check.error else 'FAIL'}"
            for check in report.grammar_alignment_checks
        ),
    ]
    if report.rss_before_bytes is not None and report.rss_loaded_bytes is not None:
        delta_mib = (report.rss_loaded_bytes - report.rss_before_bytes) / (1024 * 1024)
        lines.append(f"  Host RSS increase while loaded: {delta_mib:.1f} MiB")
    if report.cuda_allocated_loaded_bytes is None:
        lines.append("  CUDA allocation: unavailable or CUDA not active")
    else:
        delta_mib = (
            report.cuda_allocated_loaded_bytes - (report.cuda_allocated_before_bytes or 0)
        ) / (1024 * 1024)
        lines.append(f"  CUDA allocation increase while loaded: {delta_mib:.1f} MiB")

    lines.extend(f"  Warning: {warning}" for warning in report.warnings)
    lines.extend(f"  Error: {error}" for error in report.errors)
    return "\n".join(lines)


__all__ = [
    "DEFAULT_GRAMMAR_SAMPLES",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TEXT_SAMPLES",
    "GrammarAlignmentCheck",
    "TextByteCheck",
    "TokenizerProbeReport",
    "build_grammar_vocabulary",
    "byte_level_token_for_bytes",
    "check_grammar_alignment",
    "check_text_byte_reconstruction",
    "format_probe_report",
    "inspect_loaded_tokenizer",
    "run_tokenizer_probe",
]
