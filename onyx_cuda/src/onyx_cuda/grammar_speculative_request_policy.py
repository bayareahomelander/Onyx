"""Framework-neutral request policy over one completed D51 result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast

from .grammar_speculative_final_outcome import (
    GrammarMaskedSpeculativeFinalOutcomeError,
    GrammarMaskedSpeculativeFinalOutcomeResult,
)
from .grammar_speculative_iteration import GrammarMaskedSpeculativeIterationResult
from .grammar_speculative_outcome import GrammarMaskedSpeculativeOutcomeResult

__all__ = [
    "GrammarMaskedSpeculativeRequestPolicyError",
    "GrammarMaskedSpeculativeRequestPolicyInvariantError",
    "GrammarMaskedSpeculativeRequestPolicyResult",
    "decide_grammar_masked_speculative_request_policy",
]


class GrammarMaskedSpeculativeRequestPolicyError(
    GrammarMaskedSpeculativeFinalOutcomeError
):
    """Base error raised while deciding request-level speculative policy."""


class GrammarMaskedSpeculativeRequestPolicyInvariantError(
    GrammarMaskedSpeculativeRequestPolicyError
):
    """Raised when completed D51 or composed request evidence is inconsistent."""


StateT = TypeVar("StateT")
FinalDisposition = Literal[
    "grammar_complete",
    "grammar_no_continuation",
    "iteration_bound_exhausted",
]
RequestDisposition = Literal[
    "stop",
    "grammar_complete",
    "output_budget_exhausted",
    "grammar_no_continuation",
    "continuation_permitted",
]


@dataclass(frozen=True, slots=True)
class _FinalOutcomeSnapshot:
    output_token_ids: tuple[int, ...]
    final_iteration: GrammarMaskedSpeculativeIterationResult[object]
    final_outcome: GrammarMaskedSpeculativeOutcomeResult
    disposition: FinalDisposition
    grammar_completion_token_id: int | None
    final_iteration_output_token_ids: tuple[int, ...]
    uncached_next_token_id: int | None
    committed_state: object
    committed_state_is_match: bool
    final_outcome_kind: str
    sampled_token_ids: tuple[int, ...]
    visible_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GrammarMaskedSpeculativeRequestPolicyResult(Generic[StateT]):
    """One retained D51 result plus its request-level policy disposition."""

    final_outcome_result: GrammarMaskedSpeculativeFinalOutcomeResult[StateT]
    request_disposition: RequestDisposition
    matched_stop_token_ids: tuple[int, ...] | None
    remaining_output_token_budget: int

    def __post_init__(self) -> None:
        final_outcome_result = _require_final_outcome_result(
            self.final_outcome_result,
            error_type=TypeError,
        )
        snapshot = _snapshot_final_outcome_result(
            final_outcome_result,
            vocab_size=None,
        )
        request_disposition = _validate_request_disposition(
            self.request_disposition
        )
        matched_stop_token_ids = _validate_result_stop_token_ids(
            self.matched_stop_token_ids
        )
        remaining_output_token_budget = _validate_result_remaining_budget(
            self.remaining_output_token_budget
        )
        _validate_result_relationship(
            snapshot=snapshot,
            request_disposition=request_disposition,
            matched_stop_token_ids=matched_stop_token_ids,
            remaining_output_token_budget=remaining_output_token_budget,
        )

    @property
    def sampled_token_ids(self) -> tuple[int, ...]:
        """Delegate to D51 sampled metadata without changing any occurrence."""

        return self.final_outcome_result.sampled_token_ids

    @property
    def visible_token_ids(self) -> tuple[int, ...]:
        """Delegate to the exact unchanged D51 visible token view."""

        return self.final_outcome_result.visible_token_ids

    @property
    def request_is_terminal(self) -> bool:
        """Whether this policy result prohibits further generation work."""

        return self.request_disposition != "continuation_permitted"

    @property
    def further_generation_permitted(self) -> bool:
        """Whether a later owner may perform more generation work."""

        return self.request_disposition == "continuation_permitted"


_FINAL_OUTCOME_RESULT_TYPE = GrammarMaskedSpeculativeFinalOutcomeResult
_REQUEST_POLICY_RESULT_TYPE = GrammarMaskedSpeculativeRequestPolicyResult


def decide_grammar_masked_speculative_request_policy(
    final_outcome_result: GrammarMaskedSpeculativeFinalOutcomeResult[StateT],
    *,
    vocab_size: int,
    matched_stop_token_ids: tuple[int, ...] | None,
    matched_stop_is_eligible: bool,
    available_output_token_budget: int,
) -> GrammarMaskedSpeculativeRequestPolicyResult[StateT]:
    """Validate request evidence and apply terminal/continuation precedence."""

    vocab_size = _validate_vocab_size(vocab_size)
    available_output_token_budget = _validate_available_output_token_budget(
        available_output_token_budget
    )
    matched_stop_is_eligible = _validate_matched_stop_is_eligible(
        matched_stop_is_eligible
    )
    matched_stop_token_ids = _validate_operation_stop_token_ids(
        matched_stop_token_ids,
        vocab_size=vocab_size,
        matched_stop_is_eligible=matched_stop_is_eligible,
    )
    final_outcome_result = cast(
        GrammarMaskedSpeculativeFinalOutcomeResult[StateT],
        _require_final_outcome_result(
            final_outcome_result,
            error_type=TypeError,
        ),
    )

    before = _snapshot_final_outcome_result(
        final_outcome_result,
        vocab_size=vocab_size,
    )
    consumed_output_token_budget = len(before.sampled_token_ids)
    if consumed_output_token_budget > available_output_token_budget:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "completed D51 sampled output exceeds the available output token budget"
        )
    remaining_output_token_budget = (
        available_output_token_budget - consumed_output_token_budget
    )
    effective_stop_token_ids = (
        matched_stop_token_ids if matched_stop_is_eligible else None
    )

    if effective_stop_token_ids is not None:
        request_disposition: RequestDisposition = "stop"
    elif before.disposition == "grammar_complete":
        request_disposition = "grammar_complete"
    elif remaining_output_token_budget == 0:
        request_disposition = "output_budget_exhausted"
    elif before.disposition == "grammar_no_continuation":
        request_disposition = "grammar_no_continuation"
    else:
        request_disposition = "continuation_permitted"

    result = GrammarMaskedSpeculativeRequestPolicyResult(
        final_outcome_result=final_outcome_result,
        request_disposition=request_disposition,
        matched_stop_token_ids=effective_stop_token_ids,
        remaining_output_token_budget=remaining_output_token_budget,
    )
    if type(result) is not _REQUEST_POLICY_RESULT_TYPE:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy construction must return a "
            "GrammarMaskedSpeculativeRequestPolicyResult"
        )

    result_final_outcome = _read_attribute(
        result,
        "final_outcome_result",
        label="request-policy result",
    )
    result_disposition = _read_attribute(
        result,
        "request_disposition",
        label="request-policy result",
    )
    result_stop = _read_attribute(
        result,
        "matched_stop_token_ids",
        label="request-policy result",
    )
    result_remaining_budget = _read_attribute(
        result,
        "remaining_output_token_budget",
        label="request-policy result",
    )
    if result_final_outcome is not final_outcome_result:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy result must retain the exact D51 result"
        )
    if type(result_disposition) is not str or result_disposition != request_disposition:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy result must retain the requested disposition"
        )
    if result_stop is not effective_stop_token_ids:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy result must retain the exact effective stop evidence"
        )
    if (
        type(result_remaining_budget) is not int
        or result_remaining_budget != remaining_output_token_budget
    ):
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy result must retain the exact remaining output budget"
        )

    after = _snapshot_final_outcome_result(
        cast(GrammarMaskedSpeculativeFinalOutcomeResult[object], result_final_outcome),
        vocab_size=vocab_size,
    )
    _require_unchanged_final_outcome(before, after)

    result_sampled = _read_attribute(
        result,
        "sampled_token_ids",
        label="request-policy result",
    )
    result_visible = _read_attribute(
        result,
        "visible_token_ids",
        label="request-policy result",
    )
    result_is_terminal = _read_attribute(
        result,
        "request_is_terminal",
        label="request-policy result",
    )
    result_permits_generation = _read_attribute(
        result,
        "further_generation_permitted",
        label="request-policy result",
    )
    if result_sampled != before.sampled_token_ids:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy sampled_token_ids must delegate to D51 unchanged"
        )
    if result_visible is not before.output_token_ids:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy visible_token_ids must retain the exact D51 output tuple"
        )
    expected_terminal = request_disposition != "continuation_permitted"
    if type(result_is_terminal) is not bool or result_is_terminal is not expected_terminal:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy result reports inconsistent terminal status"
        )
    if (
        type(result_permits_generation) is not bool
        or result_permits_generation is expected_terminal
    ):
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "request-policy result reports inconsistent continuation permission"
        )
    return cast(GrammarMaskedSpeculativeRequestPolicyResult[StateT], result)


def _validate_vocab_size(value: object) -> int:
    if type(value) is not int:
        raise TypeError("vocab_size must be an exact integer")
    if value <= 0:
        raise ValueError("vocab_size must be greater than zero")
    return value


def _validate_available_output_token_budget(value: object) -> int:
    if type(value) is not int:
        raise TypeError("available_output_token_budget must be an exact integer")
    if value <= 0:
        raise ValueError("available_output_token_budget must be greater than zero")
    return value


def _validate_matched_stop_is_eligible(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError("matched_stop_is_eligible must be an exact boolean")
    return value


def _validate_operation_stop_token_ids(
    value: object,
    *,
    vocab_size: int,
    matched_stop_is_eligible: bool,
) -> tuple[int, ...] | None:
    if value is None:
        if matched_stop_is_eligible:
            raise ValueError(
                "matched_stop_is_eligible cannot be true without stop evidence"
            )
        return None
    if type(value) is not tuple:
        raise TypeError("matched_stop_token_ids must be an exact tuple or None")
    if not value:
        raise ValueError("matched_stop_token_ids cannot be empty")
    for position, token_id in enumerate(value):
        if type(token_id) is not int:
            raise TypeError(
                f"matched stop token at position {position} must be an integer"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"matched stop token at position {position} {token_id} is outside "
                f"vocabulary range [0, {vocab_size})"
            )
    return cast(tuple[int, ...], value)


def _validate_result_stop_token_ids(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if type(value) is not tuple:
        raise TypeError("matched_stop_token_ids must be an exact tuple or None")
    if not value:
        raise ValueError("matched_stop_token_ids cannot be empty")
    for position, token_id in enumerate(value):
        if type(token_id) is not int:
            raise TypeError(
                f"matched stop token at position {position} must be an integer"
            )
        if token_id < 0:
            raise ValueError(
                f"matched stop token at position {position} cannot be negative"
            )
    return cast(tuple[int, ...], value)


def _validate_result_remaining_budget(value: object) -> int:
    if type(value) is not int:
        raise TypeError("remaining_output_token_budget must be an exact integer")
    if value < 0:
        raise ValueError("remaining_output_token_budget cannot be negative")
    return value


def _validate_request_disposition(value: object) -> RequestDisposition:
    if type(value) is not str:
        raise TypeError("request_disposition must be an exact string")
    if value not in {
        "stop",
        "grammar_complete",
        "output_budget_exhausted",
        "grammar_no_continuation",
        "continuation_permitted",
    }:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            f"unsupported grammar-masked speculative request disposition: {value!r}"
        )
    return cast(RequestDisposition, value)


def _require_final_outcome_result(
    value: object,
    *,
    error_type: type[Exception],
) -> GrammarMaskedSpeculativeFinalOutcomeResult[object]:
    try:
        is_result = isinstance(value, _FINAL_OUTCOME_RESULT_TYPE)
    except Exception as exc:
        raise error_type("final_outcome_result type could not be determined") from exc
    if not is_result:
        raise error_type(
            "final_outcome_result must be a "
            "GrammarMaskedSpeculativeFinalOutcomeResult"
        )
    return cast(GrammarMaskedSpeculativeFinalOutcomeResult[object], value)


def _require_iteration_result(
    value: object,
) -> GrammarMaskedSpeculativeIterationResult[object]:
    try:
        is_result = isinstance(value, GrammarMaskedSpeculativeIterationResult)
    except Exception as exc:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_iteration type could not be determined"
        ) from exc
    if not is_result:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_iteration must be a GrammarMaskedSpeculativeIterationResult"
        )
    return cast(GrammarMaskedSpeculativeIterationResult[object], value)


def _require_outcome_result(value: object) -> GrammarMaskedSpeculativeOutcomeResult:
    try:
        is_result = isinstance(value, GrammarMaskedSpeculativeOutcomeResult)
    except Exception as exc:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_outcome type could not be determined"
        ) from exc
    if not is_result:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_outcome must be a GrammarMaskedSpeculativeOutcomeResult"
        )
    return cast(GrammarMaskedSpeculativeOutcomeResult, value)


def _snapshot_final_outcome_result(
    final_outcome_result: GrammarMaskedSpeculativeFinalOutcomeResult[object],
    *,
    vocab_size: int | None,
) -> _FinalOutcomeSnapshot:
    output_token_ids = _read_attribute(
        final_outcome_result,
        "output_token_ids",
        label="final_outcome_result",
    )
    final_iteration = _read_attribute(
        final_outcome_result,
        "final_iteration",
        label="final_outcome_result",
    )
    final_outcome = _read_attribute(
        final_outcome_result,
        "final_outcome",
        label="final_outcome_result",
    )
    disposition = _read_attribute(
        final_outcome_result,
        "disposition",
        label="final_outcome_result",
    )
    grammar_completion_token_id = _read_attribute(
        final_outcome_result,
        "grammar_completion_token_id",
        label="final_outcome_result",
    )

    output_token_ids = _validate_snapshot_token_ids(
        output_token_ids,
        label="output_token_ids",
        vocab_size=vocab_size,
    )
    final_iteration = _require_iteration_result(final_iteration)
    final_outcome = _require_outcome_result(final_outcome)
    disposition = _validate_snapshot_disposition(disposition)

    final_iteration_output_token_ids = _read_attribute(
        final_iteration,
        "output_token_ids",
        label="final_iteration",
    )
    uncached_next_token_id = _read_attribute(
        final_iteration,
        "uncached_next_token_id",
        label="final_iteration",
    )
    committed_state = _read_attribute(
        final_iteration,
        "committed_state",
        label="final_iteration",
    )
    committed_state_is_match = _read_attribute(
        final_iteration,
        "committed_state_is_match",
        label="final_iteration",
    )
    final_outcome_kind = _read_attribute(
        final_outcome,
        "kind",
        label="final_outcome",
    )
    visible_token_ids = _read_attribute(
        final_outcome_result,
        "visible_token_ids",
        label="final_outcome_result",
    )
    sampled_token_ids = _read_attribute(
        final_outcome_result,
        "sampled_token_ids",
        label="final_outcome_result",
    )

    final_iteration_output_token_ids = _validate_snapshot_token_ids(
        final_iteration_output_token_ids,
        label="final_iteration output_token_ids",
        vocab_size=vocab_size,
    )
    if len(final_iteration_output_token_ids) > len(output_token_ids) or (
        final_iteration_output_token_ids
        and output_token_ids[-len(final_iteration_output_token_ids) :]
        != final_iteration_output_token_ids
    ):
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_iteration output must be an exact suffix of output_token_ids"
        )
    uncached_next_token_id = _validate_snapshot_optional_token(
        uncached_next_token_id,
        label="final_iteration uncached_next_token_id",
        vocab_size=vocab_size,
    )
    if type(committed_state_is_match) is not bool:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_iteration committed_state_is_match must be a boolean"
        )
    final_outcome_kind = _validate_snapshot_outcome_kind(final_outcome_kind)

    if disposition == "grammar_complete":
        grammar_completion_token_id = _validate_snapshot_required_token(
            grammar_completion_token_id,
            label="grammar_completion_token_id",
            vocab_size=vocab_size,
        )
        if uncached_next_token_id is not None or committed_state_is_match is not True:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "grammar completion requires matching terminal evidence without a handoff"
            )
    elif disposition == "grammar_no_continuation":
        if grammar_completion_token_id is not None:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "grammar no-continuation cannot contain a completion token"
            )
        if uncached_next_token_id is not None or committed_state_is_match is not False:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "grammar no-continuation requires nonmatching terminal evidence "
                "without a handoff"
            )
    else:
        if grammar_completion_token_id is not None:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "iteration-bound exhaustion cannot contain a completion token"
            )
        if uncached_next_token_id is None:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "iteration-bound exhaustion requires one uncached handoff token"
            )
        if not output_token_ids or output_token_ids[-1] != uncached_next_token_id:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "final handoff token must be the final accumulated output token"
            )

    expected_kind = {
        "grammar_complete": "grammar_complete",
        "grammar_no_continuation": "grammar_no_continuation",
        "iteration_bound_exhausted": "handoff_available",
    }[disposition]
    if final_outcome_kind != expected_kind:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "D51 disposition disagrees with final_outcome kind"
        )

    if visible_token_ids is not output_token_ids:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "D51 visible_token_ids must be the exact output_token_ids tuple"
        )
    sampled_token_ids = _validate_snapshot_token_ids(
        sampled_token_ids,
        label="sampled_token_ids",
        vocab_size=vocab_size,
    )
    expected_sampled = (
        output_token_ids + (cast(int, grammar_completion_token_id),)
        if disposition == "grammar_complete"
        else output_token_ids
    )
    if sampled_token_ids != expected_sampled:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "D51 sampled_token_ids disagree with its disposition"
        )
    if disposition != "grammar_complete" and sampled_token_ids is not output_token_ids:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "non-completion D51 sampled_token_ids must retain the exact output tuple"
        )

    return _FinalOutcomeSnapshot(
        output_token_ids=output_token_ids,
        final_iteration=final_iteration,
        final_outcome=final_outcome,
        disposition=disposition,
        grammar_completion_token_id=cast(int | None, grammar_completion_token_id),
        final_iteration_output_token_ids=final_iteration_output_token_ids,
        uncached_next_token_id=uncached_next_token_id,
        committed_state=committed_state,
        committed_state_is_match=committed_state_is_match,
        final_outcome_kind=final_outcome_kind,
        sampled_token_ids=sampled_token_ids,
        visible_token_ids=cast(tuple[int, ...], visible_token_ids),
    )


def _validate_snapshot_token_ids(
    value: object,
    *,
    label: str,
    vocab_size: int | None,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            f"{label} must be an exact tuple"
        )
    for position, token_id in enumerate(value):
        if type(token_id) is not int:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                f"{label} token at position {position} must be an integer"
            )
        if token_id < 0 or (vocab_size is not None and token_id >= vocab_size):
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                f"{label} token at position {position} is outside the numeric domain"
            )
    return cast(tuple[int, ...], value)


def _validate_snapshot_optional_token(
    value: object,
    *,
    label: str,
    vocab_size: int | None,
) -> int | None:
    if value is None:
        return None
    return _validate_snapshot_required_token(
        value,
        label=label,
        vocab_size=vocab_size,
    )


def _validate_snapshot_required_token(
    value: object,
    *,
    label: str,
    vocab_size: int | None,
) -> int:
    if type(value) is not int:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            f"{label} must be an integer"
        )
    if value < 0 or (vocab_size is not None and value >= vocab_size):
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            f"{label} is outside the numeric domain"
        )
    return value


def _validate_snapshot_disposition(value: object) -> FinalDisposition:
    if type(value) is not str or value not in {
        "grammar_complete",
        "grammar_no_continuation",
        "iteration_bound_exhausted",
    }:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_outcome_result contains an unsupported disposition"
        )
    return cast(FinalDisposition, value)


def _validate_snapshot_outcome_kind(value: object) -> str:
    if type(value) is not str or value not in {
        "handoff_available",
        "grammar_complete",
        "grammar_no_continuation",
    }:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "final_outcome contains an unsupported kind"
        )
    return value


def _validate_result_relationship(
    *,
    snapshot: _FinalOutcomeSnapshot,
    request_disposition: RequestDisposition,
    matched_stop_token_ids: tuple[int, ...] | None,
    remaining_output_token_budget: int,
) -> None:
    if request_disposition == "stop":
        if matched_stop_token_ids is None:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "stop disposition requires effective stop evidence"
            )
        return
    if matched_stop_token_ids is not None:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "effective stop evidence is valid only for stop disposition"
        )
    if request_disposition == "grammar_complete":
        if snapshot.disposition != "grammar_complete":
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "grammar_complete request disposition requires D51 grammar completion"
            )
        return
    if request_disposition == "output_budget_exhausted":
        if remaining_output_token_budget != 0:
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "output_budget_exhausted requires zero remaining budget"
            )
        if snapshot.disposition == "grammar_complete":
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "D51 grammar completion takes precedence over output budget exhaustion"
            )
        return
    if remaining_output_token_budget == 0:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            f"{request_disposition} requires positive remaining budget"
        )
    if request_disposition == "grammar_no_continuation":
        if snapshot.disposition != "grammar_no_continuation":
            raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
                "grammar_no_continuation requires the matching D51 disposition"
            )
        return
    if snapshot.disposition != "iteration_bound_exhausted":
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "continuation_permitted requires D51 iteration-bound exhaustion"
        )


def _require_unchanged_final_outcome(
    before: _FinalOutcomeSnapshot,
    after: _FinalOutcomeSnapshot,
) -> None:
    if (
        after.output_token_ids is not before.output_token_ids
        or after.final_iteration is not before.final_iteration
        or after.final_outcome is not before.final_outcome
    ):
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "D51 retained identity changed during request-policy construction"
        )
    if (
        after.disposition != before.disposition
        or after.grammar_completion_token_id != before.grammar_completion_token_id
        or after.final_iteration_output_token_ids
        != before.final_iteration_output_token_ids
        or after.uncached_next_token_id != before.uncached_next_token_id
        or after.committed_state_is_match is not before.committed_state_is_match
        or after.final_outcome_kind != before.final_outcome_kind
        or after.sampled_token_ids != before.sampled_token_ids
    ):
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "D51 policy evidence changed during request-policy construction"
        )
    if after.committed_state is not before.committed_state:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "D51 committed-state identity changed during request-policy construction"
        )
    if after.visible_token_ids is not before.visible_token_ids:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            "D51 visible-token identity changed during request-policy construction"
        )


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedSpeculativeRequestPolicyInvariantError(
            f"{label} {name} could not be read"
        ) from exc
