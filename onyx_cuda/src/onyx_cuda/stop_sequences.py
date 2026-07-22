"""Internal ordered token-sequence validation and matching helpers."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def normalize_stop_token_sequences(
    stop_token_sequences: Sequence[Sequence[int]],
    vocab_size: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(stop_token_sequences, (str, bytes, bytearray)) or not isinstance(
        stop_token_sequences, Sequence
    ):
        raise TypeError("stop_token_sequences must be an ordered sequence")

    normalized = []
    for position, stop_sequence in enumerate(stop_token_sequences):
        if isinstance(stop_sequence, (str, bytes, bytearray)) or not isinstance(
            stop_sequence, Sequence
        ):
            raise TypeError(
                f"stop token sequence at position {position} must be an ordered sequence"
            )
        token_ids = tuple(stop_sequence)
        if not token_ids:
            raise ValueError(f"stop token sequence at position {position} cannot be empty")
        validate_token_ids(
            token_ids,
            vocab_size,
            label=f"stop sequence {position}",
            allow_empty=False,
        )
        normalized.append(token_ids)
    return tuple(normalized)


def match_stop_token_sequence(
    generated_token_ids: Sequence[int],
    stop_token_sequences: Sequence[tuple[int, ...]],
) -> tuple[int, ...] | None:
    for stop_sequence in stop_token_sequences:
        if len(generated_token_ids) < len(stop_sequence):
            continue
        if tuple(generated_token_ids[-len(stop_sequence) :]) == stop_sequence:
            return stop_sequence
    return None


def pending_stop_prefix_length(
    pending_token_ids: Sequence[int],
    stop_token_sequences: Sequence[tuple[int, ...]],
) -> int:
    """Return the longest pending suffix that is a prefix of any configured stop."""

    longest = 0
    for stop_sequence in stop_token_sequences:
        maximum = min(len(pending_token_ids), len(stop_sequence))
        for length in range(maximum, longest, -1):
            if tuple(pending_token_ids[-length:]) == stop_sequence[:length]:
                longest = length
                break
    return longest


def validate_token_ids(
    token_ids: Collection[int],
    vocab_size: int,
    *,
    label: str,
    allow_empty: bool,
) -> None:
    if not token_ids and not allow_empty:
        raise ValueError(f"{label} token IDs cannot be empty")
    for token_id in token_ids:
        validate_token_id(token_id, vocab_size, label=f"{label} token")


def validate_token_id(token_id: int, vocab_size: int, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} ID must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{label} ID {token_id} is outside vocabulary range [0, {vocab_size})")
