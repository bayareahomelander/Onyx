"""Tokenizer compatibility and grammar-vocabulary helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_RECONSTRUCTION_SAMPLES = (
    "ONY-2026",
    '{"name":"caf\u00e9"}',
    "\u4f60\u597d\uff0cOnyx",
    "\U0001f642",
)


def _gpt2_byte_encoder() -> Dict[int, str]:
    """Return the reversible byte-to-Unicode mapping used by byte-level BPE."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("\u00a1"), ord("\u00ac") + 1))
    visible += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    byte_values = list(visible)
    codepoints = list(visible)
    extra_index = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra_index)
            extra_index += 1
    return dict(zip(byte_values, (chr(codepoint) for codepoint in codepoints)))


_BYTE_ENCODER = _gpt2_byte_encoder()
_BYTE_DECODER = {character: byte for byte, character in _BYTE_ENCODER.items()}


def byte_level_token_for_bytes(value: bytes) -> str:
    """Encode raw bytes into a synthetic byte-level BPE token string."""
    return "".join(_BYTE_ENCODER[byte] for byte in value)


def decode_byte_level_token(token: str) -> bytes:
    """Decode one byte-level BPE vocabulary token into its original bytes."""
    try:
        return bytes(_BYTE_DECODER[character] for character in token)
    except KeyError as exc:
        codepoint = f"U+{ord(exc.args[0]):04X}"
        raise ValueError(f"token contains non-byte-level character {codepoint}") from exc


def _raw_vocabulary(tokenizer: Any) -> Dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("tokenizer must expose get_vocab() for exact ID compatibility checks")

    raw_vocab = get_vocab()
    if not isinstance(raw_vocab, dict) or not raw_vocab:
        raise ValueError("tokenizer.get_vocab() must return a non-empty mapping")
    return raw_vocab


def tokenizer_id_map(tokenizer: Any) -> Dict[int, str]:
    """Return an ID-to-token map, rejecting malformed or duplicate IDs."""
    id_map: Dict[int, str] = {}
    for token, raw_token_id in _raw_vocabulary(tokenizer).items():
        if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
            raise ValueError(f"token {token!r} has non-integer ID {raw_token_id!r}")
        token_id = int(raw_token_id)
        if token_id < 0:
            raise ValueError(f"token {token!r} has negative ID {token_id}")
        if token_id in id_map and id_map[token_id] != str(token):
            raise ValueError(
                f"token ID {token_id} is assigned to both {id_map[token_id]!r} and {token!r}"
            )
        id_map[token_id] = str(token)
    return id_map


def validate_tokenizer_pair(
    draft_tokenizer: Any,
    target_tokenizer: Any,
    *,
    draft_logits_width: int,
    target_logits_width: int,
) -> Tuple[int, ...]:
    """Validate shared token meanings and return IDs safe for paired sampling."""
    if draft_logits_width != target_logits_width:
        raise ValueError(
            "Tokenizer mismatch: model vocabulary widths differ "
            f"(draft={draft_logits_width}, target={target_logits_width})"
        )

    draft_map = tokenizer_id_map(draft_tokenizer)
    target_map = tokenizer_id_map(target_tokenizer)
    incompatible_ids = [
        token_id for token_id, token in draft_map.items() if target_map.get(token_id) != token
    ]
    if incompatible_ids:
        differing_id = min(incompatible_ids)
        raise ValueError(
            "Tokenizer mismatch at token ID "
            f"{differing_id}: draft={draft_map.get(differing_id)!r}, "
            f"target={target_map.get(differing_id)!r}"
        )

    draft_special_ids = tuple(sorted(int(value) for value in draft_tokenizer.all_special_ids))
    target_special_ids = tuple(sorted(int(value) for value in target_tokenizer.all_special_ids))
    if draft_special_ids != target_special_ids:
        raise ValueError(
            "Tokenizer mismatch: draft and target special token IDs differ "
            f"(draft={draft_special_ids}, target={target_special_ids})"
        )

    return tuple(sorted(draft_map))


def _added_token_metadata(tokenizer: Any) -> Dict[int, Any]:
    decoder = getattr(tokenizer, "added_tokens_decoder", {})
    return {int(token_id): token for token_id, token in dict(decoder).items()}


def build_grammar_vocabulary(
    tokenizer: Any,
    logits_width: int,
) -> Tuple[List[bytes], Dict[str, Any], List[str]]:
    """Build an ID-aligned byte vocabulary for the Rust grammar engine."""
    if isinstance(logits_width, bool) or not isinstance(logits_width, int) or logits_width < 1:
        raise ValueError("logits_width must be a positive integer")

    raw_vocab = _raw_vocabulary(tokenizer)
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
            vocabulary[token_id] = decode_byte_level_token(str(token))
        except ValueError as exc:
            errors.append(f"token ID {token_id} ({token!r}) cannot be converted to bytes: {exc}")

    byte_to_ids: Dict[bytes, List[int]] = {}
    for token_id, token_bytes in enumerate(vocabulary):
        if token_bytes:
            byte_to_ids.setdefault(token_bytes, []).append(token_id)

    stats = {
        "populated_token_ids": len(mapped_ids),
        "padded_or_missing_ids": logits_width - len(mapped_ids),
        "special_token_ids": sorted(
            token_id for token_id in special_ids if token_id < logits_width
        ),
        "regular_added_token_ids": sorted(
            token_id for token_id in regular_added_ids if token_id < logits_width
        ),
        "duplicate_byte_sequences": sum(1 for ids in byte_to_ids.values() if len(ids) > 1),
    }
    return vocabulary, stats, errors


def validate_byte_reconstruction(
    tokenizer: Any,
    vocabulary: Sequence[bytes],
    samples: Iterable[str] = DEFAULT_RECONSTRUCTION_SAMPLES,
) -> None:
    """Verify tokenizer IDs reconstruct representative source UTF-8 bytes."""
    for text in samples:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
        token_ids = tuple(int(token_id) for token_id in encoded)
        if any(token_id < 0 or token_id >= len(vocabulary) for token_id in token_ids):
            raise ValueError(
                f"tokenizer produced an ID outside the model logits width for {text!r}"
            )
        actual = b"".join(vocabulary[token_id] for token_id in token_ids)
        expected = text.encode("utf-8")
        if actual != expected:
            raise ValueError(
                f"tokenizer byte reconstruction failed for {text!r}: "
                f"expected {expected.hex()}, got {actual.hex()}"
            )


__all__ = [
    "DEFAULT_RECONSTRUCTION_SAMPLES",
    "build_grammar_vocabulary",
    "byte_level_token_for_bytes",
    "decode_byte_level_token",
    "tokenizer_id_map",
    "validate_byte_reconstruction",
    "validate_tokenizer_pair",
]
