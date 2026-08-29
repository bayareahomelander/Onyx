"""Tokenizer ID to UTF-8 byte vocabulary mapping."""

from typing import NamedTuple

from transformers import PreTrainedTokenizerBase


class TokenByteVocabulary(NamedTuple):
    token_bytes: list[bytes]
    special_token_count: int
    empty_token_count: int


def build_token_byte_vocabulary(
    tokenizer: PreTrainedTokenizerBase, logits_vocab_size: int
) -> TokenByteVocabulary:
    """Map every model logit ID to its standalone decoded UTF-8 bytes."""
    if (
        isinstance(logits_vocab_size, bool)
        or not isinstance(logits_vocab_size, int)
        or logits_vocab_size <= 0
    ):
        raise ValueError("Model logits vocabulary size must be a positive integer")

    tokenizer_ids = set(tokenizer.get_vocab().values())
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

    token_bytes: list[bytes] = []
    for token_id in range(logits_vocab_size):
        try:
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            value = b"" if "\ufffd" in text else text.encode("utf-8")
        except Exception:
            value = b""
        token_bytes.append(value)

    return TokenByteVocabulary(
        token_bytes=token_bytes,
        special_token_count=sum(
            0 <= token_id < logits_vocab_size for token_id in special_ids
        ),
        empty_token_count=sum(not value for value in token_bytes),
    )
