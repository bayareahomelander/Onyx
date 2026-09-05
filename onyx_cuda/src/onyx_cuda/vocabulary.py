"""Tokenizer ID to UTF-8 byte vocabulary mapping."""

import json
from typing import NamedTuple

from transformers import PreTrainedTokenizerBase
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class TokenByteVocabulary(NamedTuple):
    token_bytes: list[bytes]
    special_token_count: int
    empty_token_count: int


def build_token_byte_vocabulary(
    tokenizer: PreTrainedTokenizerBase, logits_vocab_size: int
) -> TokenByteVocabulary:
    """Preserve raw ByteLevel bytes, including tokens splitting a UTF-8 character."""
    if (
        isinstance(logits_vocab_size, bool)
        or not isinstance(logits_vocab_size, int)
        or logits_vocab_size <= 0
    ):
        raise ValueError("Model logits vocabulary size must be a positive integer")

    vocabulary = tokenizer.get_vocab()
    tokenizer_ids = set(vocabulary.values())
    special_ids = set(tokenizer.all_special_ids)
    known_ids = tokenizer_ids | special_ids
    if not known_ids:
        raise ValueError("Tokenizer vocabulary cannot be empty")
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in known_ids):
        raise ValueError("Tokenizer token IDs must be non-negative integers")

    max_token_id = max(known_ids)
    if max_token_id >= logits_vocab_size:
        raise ValueError(
            f"Tokenizer token ID {max_token_id} exceeds model logits vocabulary "
            f"size {logits_vocab_size}"
        )

    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if decoder is None or json.loads(decoder.__getstate__()).get("type") != "ByteLevel":
        raise ValueError("Grammar constraints currently require a ByteLevel tokenizer")
    byte_decoder = {character: byte for byte, character in bytes_to_unicode().items()}
    token_bytes = [b""] * logits_vocab_size
    for token, token_id in vocabulary.items():
        if token_id in special_ids:
            continue  # These disappear with skip_special_tokens=True and cannot satisfy a grammar.
        try:
            token_bytes[token_id] = bytes(byte_decoder[character] for character in token)
        except KeyError as error:
            raise ValueError(f"Token {token_id} is not a ByteLevel byte sequence") from error

    return TokenByteVocabulary(
        token_bytes=token_bytes,
        special_token_count=sum(
            0 <= token_id < logits_vocab_size for token_id in special_ids
        ),
        empty_token_count=sum(not value for value in token_bytes),
    )
