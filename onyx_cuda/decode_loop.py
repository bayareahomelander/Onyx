"""Model-free multi-token CUDA decoding with Rust grammar constraints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from .grammar_handoff import CudaValidIdCache
from .masked_argmax import masked_argmax_tensor


@dataclass(frozen=True)
class CudaGrammarDecodeTimings:
    """Host-observed stage timings before final state/cache cleanup."""

    valid_id_lookup_us: float
    selection_call_us: float
    result_sync_us: float
    grammar_advance_us: float
    pre_cleanup_us: float


@dataclass(frozen=True)
class CudaGrammarDecodeResult:
    """Result of a model-free grammar-constrained CUDA decode."""

    token_ids: Tuple[int, ...]
    final_state: int
    matched: bool
    termination_reason: str
    timings: CudaGrammarDecodeTimings

    @property
    def steps(self) -> int:
        return len(self.token_ids)


def _microseconds(nanoseconds: int) -> float:
    return nanoseconds / 1_000.0


def decode_greedy_from_logits(
    logits_steps: Iterable[Any],
    grammar_constraint: Any,
    initial_state: int,
    *,
    max_steps: int = 256,
    check_inputs: bool = True,
) -> CudaGrammarDecodeResult:
    """Greedily decode one constrained sequence from precomputed CUDA logits.

    Each item in ``logits_steps`` must be a CUDA tensor shaped ``[vocab]`` or
    ``[1, vocab]``. The function obtains grammar-valid token IDs for the current
    Rust state, selects the highest-logit valid token on CUDA, copies that one
    token ID to the host, advances the grammar state, and repeats.

    This helper intentionally performs no model inference. It isolates the
    grammar-to-CUDA selection and state-advancement path needed by a future
    Windows inference loop.

    State ownership is transactional. On success, ``initial_state`` and every
    intermediate state are released, while the returned ``final_state`` remains
    live and is owned by the caller. If decoding raises, every state created by
    this function is released and the caller's ``initial_state`` remains live.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    cache = CudaValidIdCache(grammar_constraint)
    state = int(initial_state)
    token_ids = []
    created_states = []
    lookup_ns = 0
    selection_ns = 0
    sync_ns = 0
    advance_ns = 0
    total_start = time.perf_counter_ns()

    try:
        matched = bool(grammar_constraint.is_match_state(state))
        termination_reason = "grammar_complete" if matched else "logits_exhausted"

        if not matched:
            logits_iterator = iter(logits_steps)
            for _ in range(max_steps):
                try:
                    logits = next(logits_iterator)
                except StopIteration:
                    termination_reason = "logits_exhausted"
                    break

                if getattr(logits, "ndim", None) == 2 and logits.shape[0] != 1:
                    raise ValueError("decode loop requires one logits row per step")

                stage_start = time.perf_counter_ns()
                valid_ids = cache.get(state, logits.device)
                lookup_ns += time.perf_counter_ns() - stage_start

                stage_start = time.perf_counter_ns()
                selected = masked_argmax_tensor(logits, valid_ids, check_inputs=check_inputs)
                selection_ns += time.perf_counter_ns() - stage_start

                if selected.numel() != 1:
                    raise ValueError("decode loop requires one selected token per step")

                stage_start = time.perf_counter_ns()
                token_id = int(selected.item())
                sync_ns += time.perf_counter_ns() - stage_start
                cache.discard(state)

                stage_start = time.perf_counter_ns()
                next_state = int(grammar_constraint.advance_state(state, token_id))
                created_states.append(next_state)
                state = next_state
                matched = bool(grammar_constraint.is_match_state(state))
                advance_ns += time.perf_counter_ns() - stage_start

                token_ids.append(token_id)
                if matched:
                    termination_reason = "grammar_complete"
                    break
            else:
                termination_reason = "max_steps"

        pre_cleanup_ns = time.perf_counter_ns() - total_start
        timings = CudaGrammarDecodeTimings(
            valid_id_lookup_us=_microseconds(lookup_ns),
            selection_call_us=_microseconds(selection_ns),
            result_sync_us=_microseconds(sync_ns),
            grammar_advance_us=_microseconds(advance_ns),
            pre_cleanup_us=_microseconds(pre_cleanup_ns),
        )

        result = CudaGrammarDecodeResult(
            token_ids=tuple(token_ids),
            final_state=state,
            matched=matched,
            termination_reason=termination_reason,
            timings=timings,
        )

        if created_states:
            consumed_states = [
                consumed_state
                for consumed_state in [int(initial_state), *created_states[:-1]]
                if consumed_state != state
            ]
            grammar_constraint.release_states(consumed_states)

        return result
    except BaseException:
        if created_states:
            try:
                grammar_constraint.release_states(created_states)
            except Exception:
                pass
        raise
    finally:
        cache.clear()


__all__ = [
    "CudaGrammarDecodeResult",
    "CudaGrammarDecodeTimings",
    "decode_greedy_from_logits",
]
