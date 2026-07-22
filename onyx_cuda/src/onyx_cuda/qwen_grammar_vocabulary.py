"""Strict, framework-neutral construction of Qwen grammar token bytes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class _QwenGrammarVocabularyError(ValueError):
    """Raised when a tokenizer asset cannot produce an exact grammar vocabulary."""


@dataclass(frozen=True, slots=True)
class _QwenGrammarVocabularyRuntimeMetadata:
    """Raw Tokenizers metadata captured lazily at grammar-vocabulary construction time."""

    base_vocab_size: Any
    vocab_size: Any
    vocabulary: Any
    added_tokens: Any


@dataclass(frozen=True, slots=True)
class _AddedTokenRecord:
    token_id: int
    content: str
    single_word: bool
    lstrip: bool
    rstrip: bool
    normalized: bool
    special: bool


_ADDED_TOKEN_BOOLEAN_FIELDS = (
    "single_word",
    "lstrip",
    "rstrip",
    "normalized",
    "special",
)


def _byte_level_alphabet() -> tuple[str, ...]:
    direct_bytes = [
        *range(ord("!"), ord("~") + 1),
        *range(0xA1, 0xAC + 1),
        *range(0xAE, 0xFF + 1),
    ]
    direct_set = frozenset(direct_bytes)
    mapped_code_points = list(direct_bytes)
    next_code_point = 0x100
    for byte in range(256):
        if byte not in direct_set:
            direct_bytes.append(byte)
            mapped_code_points.append(next_code_point)
            next_code_point += 1

    characters_by_byte: list[str | None] = [None] * 256
    for byte, code_point in zip(direct_bytes, mapped_code_points, strict=True):
        characters_by_byte[byte] = chr(code_point)
    if any(character is None for character in characters_by_byte):
        raise RuntimeError("the canonical ByteLevel alphabet is incomplete")
    return tuple(character for character in characters_by_byte if character is not None)


_BYTE_TO_CHARACTER = _byte_level_alphabet()
_CHARACTER_TO_BYTE = {character: byte for byte, character in enumerate(_BYTE_TO_CHARACTER)}


def _build_qwen_grammar_vocabulary_from_source(
    tokenizer_json_path: Any,
    runtime_metadata: _QwenGrammarVocabularyRuntimeMetadata,
    *,
    expected_base_vocab_size: int,
    expected_vocab_size: int,
) -> tuple[bytes, ...]:
    """Build one exact byte string per token ID from a tokenizer asset and runtime metadata."""

    base_vocab_size = _positive_size(
        expected_base_vocab_size,
        label="adapter base vocabulary size",
    )
    vocab_size = _positive_size(expected_vocab_size, label="adapter vocabulary size")
    if base_vocab_size > vocab_size:
        raise _QwenGrammarVocabularyError(
            "adapter base vocabulary size cannot exceed adapter vocabulary size"
        )
    if not isinstance(runtime_metadata, _QwenGrammarVocabularyRuntimeMetadata):
        raise _QwenGrammarVocabularyError(
            "runtime grammar-vocabulary metadata has an unsupported type"
        )

    asset = _read_tokenizer_asset(tokenizer_json_path)
    base_pieces, added_tokens = _normalize_asset(
        asset,
        expected_base_vocab_size=base_vocab_size,
        expected_vocab_size=vocab_size,
    )
    _validate_runtime_agreement(
        runtime_metadata,
        base_pieces=base_pieces,
        added_tokens=added_tokens,
        expected_base_vocab_size=base_vocab_size,
        expected_vocab_size=vocab_size,
    )

    result: list[bytes | None] = [None] * vocab_size
    for token_id, piece in enumerate(base_pieces):
        result[token_id] = _decode_byte_level_piece(
            piece,
            label=f"base token {token_id}",
        )
    for token in added_tokens:
        result[token.token_id] = b"" if token.special else _decode_added_token_content(token)

    if any(token_bytes is None for token_bytes in result):
        raise _QwenGrammarVocabularyError(
            "constructed grammar vocabulary contains an unassigned token ID"
        )
    vocabulary = tuple(token_bytes for token_bytes in result if token_bytes is not None)
    if len(vocabulary) != vocab_size:
        raise _QwenGrammarVocabularyError(
            f"constructed grammar vocabulary contains {len(vocabulary)} entries; "
            f"expected {vocab_size}"
        )
    for token_id, token_bytes in enumerate(vocabulary):
        if not isinstance(token_bytes, bytes):
            raise _QwenGrammarVocabularyError(
                f"constructed grammar vocabulary entry {token_id} is not bytes"
            )
    return vocabulary


def _read_tokenizer_asset(path: Any) -> Mapping[str, Any]:
    try:
        source_path = Path(path)
        payload = source_path.read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        raise _QwenGrammarVocabularyError(f"failed to read pinned tokenizer.json: {exc}") from exc
    try:
        asset = json.loads(
            payload,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _QwenGrammarVocabularyError(f"failed to parse pinned tokenizer.json: {exc}") from exc
    if not isinstance(asset, Mapping):
        raise _QwenGrammarVocabularyError("pinned tokenizer.json root must be an object")
    return asset


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is unsupported")


def _normalize_asset(
    asset: Mapping[str, Any],
    *,
    expected_base_vocab_size: int,
    expected_vocab_size: int,
) -> tuple[tuple[str, ...], tuple[_AddedTokenRecord, ...]]:
    model = _required_mapping(asset, "model", label="tokenizer root")
    model_type = model.get("type")
    if model_type != "BPE":
        raise _QwenGrammarVocabularyError(
            f"tokenizer model type must be 'BPE'; found {model_type!r}"
        )
    byte_fallback = model.get("byte_fallback", False)
    if byte_fallback is not False:
        raise _QwenGrammarVocabularyError("tokenizer BPE byte_fallback must be absent or false")

    decoder = _required_mapping(asset, "decoder", label="tokenizer root")
    decoder_type = decoder.get("type")
    if decoder_type != "ByteLevel":
        raise _QwenGrammarVocabularyError(
            f"tokenizer decoder type must be 'ByteLevel'; found {decoder_type!r}"
        )

    raw_vocabulary = _required_mapping(model, "vocab", label="tokenizer model")
    base_pieces = _normalize_base_vocabulary(
        raw_vocabulary,
        expected_base_vocab_size=expected_base_vocab_size,
    )
    raw_added_tokens = asset.get("added_tokens")
    if not isinstance(raw_added_tokens, list):
        raise _QwenGrammarVocabularyError("tokenizer added_tokens must be an array")
    added_tokens = _normalize_added_tokens(
        raw_added_tokens,
        expected_base_vocab_size=expected_base_vocab_size,
        expected_vocab_size=expected_vocab_size,
        base_pieces=base_pieces,
    )
    return base_pieces, added_tokens


def _normalize_base_vocabulary(
    raw_vocabulary: Mapping[str, Any],
    *,
    expected_base_vocab_size: int,
) -> tuple[str, ...]:
    if len(raw_vocabulary) != expected_base_vocab_size:
        raise _QwenGrammarVocabularyError(
            f"tokenizer model vocabulary contains {len(raw_vocabulary)} entries; "
            f"expected {expected_base_vocab_size}"
        )

    pieces: list[str | None] = [None] * expected_base_vocab_size
    for piece, raw_token_id in raw_vocabulary.items():
        if not isinstance(piece, str):
            raise _QwenGrammarVocabularyError("tokenizer model vocabulary keys must be strings")
        if not piece:
            raise _QwenGrammarVocabularyError(
                "tokenizer model vocabulary contains an empty base piece"
            )
        token_id = _strict_token_id(raw_token_id, label=f"base token {piece!r}")
        if token_id < 0 or token_id >= expected_base_vocab_size:
            raise _QwenGrammarVocabularyError(
                f"base token {piece!r} ID {token_id} is outside base range "
                f"[0, {expected_base_vocab_size})"
            )
        if pieces[token_id] is not None:
            raise _QwenGrammarVocabularyError(
                f"tokenizer model vocabulary contains duplicate base token ID {token_id}"
            )
        invalid_character = next(
            (character for character in piece if character not in _CHARACTER_TO_BYTE),
            None,
        )
        if invalid_character is not None:
            raise _QwenGrammarVocabularyError(
                f"base token {token_id} contains character {invalid_character!r} outside "
                "the canonical ByteLevel alphabet"
            )
        pieces[token_id] = piece

    missing = tuple(token_id for token_id, piece in enumerate(pieces) if piece is None)
    if missing:
        raise _QwenGrammarVocabularyError(
            f"tokenizer model vocabulary is missing base token IDs {missing!r}"
        )
    return tuple(piece for piece in pieces if piece is not None)


def _normalize_added_tokens(
    raw_added_tokens: list[Any],
    *,
    expected_base_vocab_size: int,
    expected_vocab_size: int,
    base_pieces: tuple[str, ...],
) -> tuple[_AddedTokenRecord, ...]:
    expected_added_count = expected_vocab_size - expected_base_vocab_size
    if len(raw_added_tokens) != expected_added_count:
        raise _QwenGrammarVocabularyError(
            f"tokenizer added_tokens contains {len(raw_added_tokens)} entries; "
            f"expected {expected_added_count}"
        )

    records: list[_AddedTokenRecord | None] = [None] * expected_added_count
    seen_contents = set(base_pieces)
    for position, raw_token in enumerate(raw_added_tokens):
        if not isinstance(raw_token, Mapping):
            raise _QwenGrammarVocabularyError(
                f"tokenizer added token at position {position} must be an object"
            )
        token_id = _strict_token_id(
            raw_token.get("id"),
            label=f"added token at position {position}",
        )
        if token_id < expected_base_vocab_size or token_id >= expected_vocab_size:
            raise _QwenGrammarVocabularyError(
                f"added token ID {token_id} is outside added-token range "
                f"[{expected_base_vocab_size}, {expected_vocab_size})"
            )
        content = raw_token.get("content")
        if not isinstance(content, str):
            raise _QwenGrammarVocabularyError(f"added token {token_id} content must be a string")
        if content in seen_contents:
            raise _QwenGrammarVocabularyError(
                f"tokenizer contains duplicate token content {content!r}"
            )
        boolean_values = {
            field: _required_boolean(raw_token, field, label=f"added token {token_id}")
            for field in _ADDED_TOKEN_BOOLEAN_FIELDS
        }
        offset = token_id - expected_base_vocab_size
        if records[offset] is not None:
            raise _QwenGrammarVocabularyError(
                f"tokenizer added_tokens contains duplicate token ID {token_id}"
            )
        records[offset] = _AddedTokenRecord(
            token_id=token_id,
            content=content,
            **boolean_values,
        )
        seen_contents.add(content)

    missing = tuple(
        expected_base_vocab_size + offset for offset, record in enumerate(records) if record is None
    )
    if missing:
        raise _QwenGrammarVocabularyError(
            f"tokenizer added_tokens is missing token IDs {missing!r}"
        )
    return tuple(record for record in records if record is not None)


def _validate_runtime_agreement(
    runtime: _QwenGrammarVocabularyRuntimeMetadata,
    *,
    base_pieces: tuple[str, ...],
    added_tokens: tuple[_AddedTokenRecord, ...],
    expected_base_vocab_size: int,
    expected_vocab_size: int,
) -> None:
    runtime_base_vocab_size = _positive_size(
        runtime.base_vocab_size,
        label="runtime base vocabulary size",
    )
    runtime_vocab_size = _positive_size(
        runtime.vocab_size,
        label="runtime vocabulary size",
    )
    if runtime_base_vocab_size != expected_base_vocab_size:
        raise _QwenGrammarVocabularyError(
            f"runtime base vocabulary size is {runtime_base_vocab_size}; "
            f"adapter reports {expected_base_vocab_size}"
        )
    if runtime_vocab_size != expected_vocab_size:
        raise _QwenGrammarVocabularyError(
            f"runtime vocabulary size is {runtime_vocab_size}; "
            f"adapter reports {expected_vocab_size}"
        )

    runtime_vocabulary = _normalize_runtime_vocabulary(
        runtime.vocabulary,
        expected_vocab_size=expected_vocab_size,
    )
    expected_tokens = (*base_pieces, *(token.content for token in added_tokens))
    for token_id, (actual, expected) in enumerate(
        zip(runtime_vocabulary, expected_tokens, strict=True)
    ):
        if actual != expected:
            raise _QwenGrammarVocabularyError(
                f"runtime token ID {token_id} contains {actual!r}; asset contains {expected!r}"
            )

    runtime_added_tokens = _normalize_runtime_added_tokens(
        runtime.added_tokens,
        expected_base_vocab_size=expected_base_vocab_size,
        expected_vocab_size=expected_vocab_size,
    )
    for actual, expected in zip(runtime_added_tokens, added_tokens, strict=True):
        if actual != expected:
            raise _QwenGrammarVocabularyError(
                f"runtime added-token metadata for ID {expected.token_id} disagrees "
                "with tokenizer.json"
            )


def _normalize_runtime_vocabulary(
    raw_vocabulary: Any,
    *,
    expected_vocab_size: int,
) -> tuple[str, ...]:
    if not isinstance(raw_vocabulary, Mapping):
        raise _QwenGrammarVocabularyError("runtime vocabulary must be a mapping")
    if len(raw_vocabulary) != expected_vocab_size:
        raise _QwenGrammarVocabularyError(
            f"runtime vocabulary contains {len(raw_vocabulary)} entries; "
            f"expected {expected_vocab_size}"
        )

    tokens: list[str | None] = [None] * expected_vocab_size
    for token, raw_token_id in raw_vocabulary.items():
        if not isinstance(token, str):
            raise _QwenGrammarVocabularyError("runtime vocabulary keys must be strings")
        token_id = _strict_token_id(raw_token_id, label=f"runtime token {token!r}")
        if token_id < 0 or token_id >= expected_vocab_size:
            raise _QwenGrammarVocabularyError(
                f"runtime token {token!r} ID {token_id} is outside tokenizer range "
                f"[0, {expected_vocab_size})"
            )
        if tokens[token_id] is not None:
            raise _QwenGrammarVocabularyError(
                f"runtime vocabulary contains duplicate token ID {token_id}"
            )
        tokens[token_id] = token
    missing = tuple(token_id for token_id, token in enumerate(tokens) if token is None)
    if missing:
        raise _QwenGrammarVocabularyError(f"runtime vocabulary is missing token IDs {missing!r}")
    return tuple(token for token in tokens if token is not None)


def _normalize_runtime_added_tokens(
    raw_added_tokens: Any,
    *,
    expected_base_vocab_size: int,
    expected_vocab_size: int,
) -> tuple[_AddedTokenRecord, ...]:
    if not isinstance(raw_added_tokens, Mapping):
        raise _QwenGrammarVocabularyError("runtime added-token decoder must be a mapping")
    expected_count = expected_vocab_size - expected_base_vocab_size
    if len(raw_added_tokens) != expected_count:
        raise _QwenGrammarVocabularyError(
            f"runtime added-token decoder contains {len(raw_added_tokens)} entries; "
            f"expected {expected_count}"
        )

    records: list[_AddedTokenRecord | None] = [None] * expected_count
    for raw_token_id, metadata in raw_added_tokens.items():
        token_id = _strict_token_id(raw_token_id, label="runtime added token")
        if token_id < expected_base_vocab_size or token_id >= expected_vocab_size:
            raise _QwenGrammarVocabularyError(
                f"runtime added token ID {token_id} is outside added-token range "
                f"[{expected_base_vocab_size}, {expected_vocab_size})"
            )
        content = _runtime_metadata_field(metadata, "content", token_id=token_id)
        if not isinstance(content, str):
            raise _QwenGrammarVocabularyError(
                f"runtime added token {token_id} content must be a string"
            )
        boolean_values = {}
        for field in _ADDED_TOKEN_BOOLEAN_FIELDS:
            value = _runtime_metadata_field(metadata, field, token_id=token_id)
            if not isinstance(value, bool):
                raise _QwenGrammarVocabularyError(
                    f"runtime added token {token_id} {field} must be a boolean"
                )
            boolean_values[field] = value
        offset = token_id - expected_base_vocab_size
        if records[offset] is not None:
            raise _QwenGrammarVocabularyError(
                f"runtime added-token decoder contains duplicate token ID {token_id}"
            )
        records[offset] = _AddedTokenRecord(
            token_id=token_id,
            content=content,
            **boolean_values,
        )

    missing = tuple(
        expected_base_vocab_size + offset for offset, record in enumerate(records) if record is None
    )
    if missing:
        raise _QwenGrammarVocabularyError(
            f"runtime added-token decoder is missing token IDs {missing!r}"
        )
    return tuple(record for record in records if record is not None)


def _runtime_metadata_field(metadata: Any, field: str, *, token_id: int) -> Any:
    try:
        if isinstance(metadata, Mapping):
            return metadata[field]
        return getattr(metadata, field)
    except (AttributeError, KeyError, TypeError) as exc:
        raise _QwenGrammarVocabularyError(
            f"runtime added token {token_id} does not expose {field} metadata"
        ) from exc


def _decode_byte_level_piece(piece: str, *, label: str) -> bytes:
    try:
        return bytes(_CHARACTER_TO_BYTE[character] for character in piece)
    except KeyError as exc:
        raise _QwenGrammarVocabularyError(
            f"{label} contains character {exc.args[0]!r} outside the canonical ByteLevel alphabet"
        ) from exc


def _decode_added_token_content(token: _AddedTokenRecord) -> bytes:
    if not token.content:
        return b""
    if all(character in _CHARACTER_TO_BYTE for character in token.content):
        return _decode_byte_level_piece(
            token.content,
            label=f"added token {token.token_id}",
        )
    try:
        return token.content.encode("utf-8")
    except UnicodeError as exc:
        raise _QwenGrammarVocabularyError(
            f"added token {token.token_id} content cannot be encoded as UTF-8"
        ) from exc


def _required_mapping(
    value: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    result = value.get(field)
    if not isinstance(result, Mapping):
        raise _QwenGrammarVocabularyError(f"{label} {field} must be an object")
    return result


def _required_boolean(value: Mapping[str, Any], field: str, *, label: str) -> bool:
    result = value.get(field)
    if not isinstance(result, bool):
        raise _QwenGrammarVocabularyError(f"{label} {field} must be a boolean")
    return result


def _strict_token_id(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _QwenGrammarVocabularyError(f"{label} ID must be an integer")
    return value


def _positive_size(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _QwenGrammarVocabularyError(f"{label} must be an integer")
    if value <= 0:
        raise _QwenGrammarVocabularyError(f"{label} must be greater than zero")
    return value
