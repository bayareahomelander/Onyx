"""Framework-neutral final policy over completed grammar-masked routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast

from .grammar_speculative_handoff import (
    GrammarMaskedSpeculativeHandoffError,
    GrammarMaskedSpeculativeHandoffResult,
)
from .grammar_speculative_iteration import GrammarMaskedSpeculativeIterationResult
from .grammar_speculative_outcome import (
    GrammarMaskedSpeculativeOutcomeResult,
    classify_grammar_masked_speculative_outcome,
)

__all__ = [
    "GrammarMaskedSpeculativeFinalOutcomeError",
    "GrammarMaskedSpeculativeFinalOutcomeInvariantError",
    "GrammarMaskedSpeculativeFinalOutcomeResult",
    "decide_grammar_masked_speculative_final_outcome",
]


class GrammarMaskedSpeculativeFinalOutcomeError(
    GrammarMaskedSpeculativeHandoffError
):
    """Base error raised while deciding final grammar-masked routing policy."""


class GrammarMaskedSpeculativeFinalOutcomeInvariantError(
    GrammarMaskedSpeculativeFinalOutcomeError
):
    """Raised when completed routing or final-policy evidence is inconsistent."""


StateT = TypeVar("StateT")
FinalDisposition = Literal[
    "grammar_complete",
    "grammar_no_continuation",
    "iteration_bound_exhausted",
]


@dataclass(frozen=True, slots=True)
class GrammarMaskedSpeculativeFinalOutcomeResult(Generic[StateT]):
    """Exact D50 evidence plus one explicit final grammar disposition."""

    output_token_ids: tuple[int, ...]
    final_iteration: GrammarMaskedSpeculativeIterationResult[StateT]
    final_outcome: GrammarMaskedSpeculativeOutcomeResult
    disposition: FinalDisposition
    grammar_completion_token_id: int | None

    def __post_init__(self) -> None:
        output_token_ids = _validate_result_output(self.output_token_ids)
        final_iteration = _require_iteration_result(
            self.final_iteration,
            label="final_iteration",
            error_type=TypeError,
        )
        final_outcome = _require_outcome_result(
            self.final_outcome,
            label="final_outcome",
            error_type=TypeError,
        )
        disposition = _validate_disposition(self.disposition)
        completion_token_id = _validate_result_completion_token(
            self.grammar_completion_token_id
        )
        _validate_final_relationship(
            output_token_ids=output_token_ids,
            final_iteration=final_iteration,
            final_outcome=final_outcome,
            disposition=disposition,
            grammar_completion_token_id=completion_token_id,
        )

    @property
    def sampled_token_ids(self) -> tuple[int, ...]:
        """Return sampled metadata, including a hidden grammar-completion EOS."""

        if self.disposition == "grammar_complete":
            return self.output_token_ids + (
                cast(int, self.grammar_completion_token_id),
            )
        return self.output_token_ids

    @property
    def visible_token_ids(self) -> tuple[int, ...]:
        """Return the exact accumulated visible D50 output."""

        return self.output_token_ids


_FINAL_OUTCOME_RESULT_TYPE = GrammarMaskedSpeculativeFinalOutcomeResult


def decide_grammar_masked_speculative_final_outcome(
    handoff_result: GrammarMaskedSpeculativeHandoffResult[StateT],
    *,
    vocab_size: int,
    eos_token_id: int,
) -> GrammarMaskedSpeculativeFinalOutcomeResult[StateT]:
    """Validate completed D50 evidence and map it to one final disposition."""

    vocab_size = _validate_vocab_size(vocab_size)
    eos_token_id = _validate_eos_token_id(eos_token_id, vocab_size=vocab_size)
    _require_handoff_result(handoff_result)

    output_token_ids = _read_attribute(
        handoff_result,
        "output_token_ids",
        label="handoff_result",
    )
    final_iteration = _read_attribute(
        handoff_result,
        "final_iteration",
        label="handoff_result",
    )
    final_outcome = _read_attribute(
        handoff_result,
        "final_outcome",
        label="handoff_result",
    )

    output_token_ids = _validate_operation_output(
        output_token_ids,
        vocab_size=vocab_size,
    )
    final_iteration = _require_iteration_result(
        final_iteration,
        label="final_iteration",
        error_type=GrammarMaskedSpeculativeFinalOutcomeInvariantError,
    )
    final_outcome = _require_outcome_result(
        final_outcome,
        label="final_outcome",
        error_type=GrammarMaskedSpeculativeFinalOutcomeInvariantError,
    )

    try:
        recomputed_outcome = classify_grammar_masked_speculative_outcome(
            final_iteration
        )
    except Exception as exc:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final_iteration could not be reclassified"
        ) from exc
    recomputed_outcome = _require_outcome_result(
        recomputed_outcome,
        label="recomputed final_outcome",
        error_type=GrammarMaskedSpeculativeFinalOutcomeInvariantError,
    )
    stored_kind = _read_outcome_kind(final_outcome, label="final_outcome")
    recomputed_kind = _read_outcome_kind(
        recomputed_outcome,
        label="recomputed final_outcome",
    )
    if recomputed_kind != stored_kind:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "stored final_outcome kind disagrees with reclassified final_iteration"
        )

    if stored_kind == "grammar_complete":
        disposition: FinalDisposition = "grammar_complete"
        completion_token_id: int | None = eos_token_id
    elif stored_kind == "grammar_no_continuation":
        disposition = "grammar_no_continuation"
        completion_token_id = None
    else:
        disposition = "iteration_bound_exhausted"
        completion_token_id = None

    before = _validate_final_relationship(
        output_token_ids=output_token_ids,
        final_iteration=final_iteration,
        final_outcome=final_outcome,
        disposition=disposition,
        grammar_completion_token_id=completion_token_id,
        vocab_size=vocab_size,
    )

    result = GrammarMaskedSpeculativeFinalOutcomeResult(
        output_token_ids=output_token_ids,
        final_iteration=cast(
            GrammarMaskedSpeculativeIterationResult[StateT],
            final_iteration,
        ),
        final_outcome=final_outcome,
        disposition=disposition,
        grammar_completion_token_id=completion_token_id,
    )
    if type(result) is not _FINAL_OUTCOME_RESULT_TYPE:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final-outcome construction must return a "
            "GrammarMaskedSpeculativeFinalOutcomeResult"
        )

    result_output = _read_attribute(
        result,
        "output_token_ids",
        label="final-outcome result",
    )
    result_iteration = _read_attribute(
        result,
        "final_iteration",
        label="final-outcome result",
    )
    result_outcome = _read_attribute(
        result,
        "final_outcome",
        label="final-outcome result",
    )
    result_disposition = _read_attribute(
        result,
        "disposition",
        label="final-outcome result",
    )
    result_completion_token = _read_attribute(
        result,
        "grammar_completion_token_id",
        label="final-outcome result",
    )
    if result_output is not output_token_ids:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final-outcome result must retain the exact output_token_ids tuple"
        )
    if result_iteration is not final_iteration:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final-outcome result must retain the exact final D47 result"
        )
    if result_outcome is not final_outcome:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final-outcome result must retain the exact final D48 result"
        )
    if result_disposition != disposition:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final-outcome result must retain the requested disposition"
        )
    if result_completion_token != completion_token_id:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final-outcome result must retain the requested completion token"
        )

    after = _validate_final_relationship(
        output_token_ids=cast(tuple[int, ...], result_output),
        final_iteration=cast(
            GrammarMaskedSpeculativeIterationResult[object],
            result_iteration,
        ),
        final_outcome=cast(GrammarMaskedSpeculativeOutcomeResult, result_outcome),
        disposition=cast(FinalDisposition, result_disposition),
        grammar_completion_token_id=cast(int | None, result_completion_token),
        vocab_size=vocab_size,
    )
    if after[0] != before[0] or after[1] != before[1]:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final D47 token evidence changed during final-outcome construction"
        )
    if after[2] is not before[2] or after[3:] != before[3:]:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final D47 state or outcome evidence changed during construction"
        )

    visible_token_ids = _read_attribute(
        result,
        "visible_token_ids",
        label="final-outcome result",
    )
    sampled_token_ids = _read_attribute(
        result,
        "sampled_token_ids",
        label="final-outcome result",
    )
    if visible_token_ids is not output_token_ids:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "visible_token_ids must be the exact accumulated output tuple"
        )
    expected_sampled = (
        output_token_ids + (eos_token_id,)
        if disposition == "grammar_complete"
        else output_token_ids
    )
    if sampled_token_ids != expected_sampled:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "sampled_token_ids disagree with the final disposition"
        )
    if disposition != "grammar_complete" and sampled_token_ids is not output_token_ids:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "non-completion sampled_token_ids must retain the exact output tuple"
        )
    return cast(GrammarMaskedSpeculativeFinalOutcomeResult[StateT], result)


def _validate_vocab_size(value: object) -> int:
    if type(value) is not int:
        raise TypeError("vocab_size must be an exact integer")
    if value <= 0:
        raise ValueError("vocab_size must be greater than zero")
    return value


def _validate_eos_token_id(value: object, *, vocab_size: int) -> int:
    if type(value) is not int:
        raise TypeError("eos_token_id must be an exact integer")
    if value < 0 or value >= vocab_size:
        raise ValueError(
            f"eos_token_id {value} is outside vocabulary range [0, {vocab_size})"
        )
    return value


def _validate_result_output(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("output_token_ids must be an exact tuple")
    for position, token_id in enumerate(value):
        if type(token_id) is not int:
            raise TypeError(
                f"output token at position {position} must be an integer"
            )
        if token_id < 0:
            raise ValueError(
                f"output token at position {position} cannot be negative"
            )
    return cast(tuple[int, ...], value)


def _validate_operation_output(
    value: object,
    *,
    vocab_size: int,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "output_token_ids must be an exact tuple"
        )
    for position, token_id in enumerate(value):
        if type(token_id) is not int:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                f"output token at position {position} must be an integer"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                f"output token at position {position} {token_id} is outside "
                f"vocabulary range [0, {vocab_size})"
            )
    return cast(tuple[int, ...], value)


def _validate_result_completion_token(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("grammar_completion_token_id must be an integer or None")
    if value < 0:
        raise ValueError("grammar_completion_token_id cannot be negative")
    return value


def _validate_disposition(value: object) -> FinalDisposition:
    if type(value) is not str:
        raise TypeError("disposition must be an exact string")
    if value not in {
        "grammar_complete",
        "grammar_no_continuation",
        "iteration_bound_exhausted",
    }:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            f"unsupported grammar-masked speculative disposition: {value!r}"
        )
    return cast(FinalDisposition, value)


def _require_handoff_result(value: object) -> GrammarMaskedSpeculativeHandoffResult[object]:
    try:
        is_result = isinstance(value, GrammarMaskedSpeculativeHandoffResult)
    except Exception as exc:
        raise TypeError("handoff_result type could not be determined") from exc
    if not is_result:
        raise TypeError(
            "handoff_result must be a GrammarMaskedSpeculativeHandoffResult"
        )
    return cast(GrammarMaskedSpeculativeHandoffResult[object], value)


def _require_iteration_result(
    value: object,
    *,
    label: str,
    error_type: type[Exception],
) -> GrammarMaskedSpeculativeIterationResult[object]:
    try:
        is_result = isinstance(value, GrammarMaskedSpeculativeIterationResult)
    except Exception as exc:
        raise error_type(f"{label} type could not be determined") from exc
    if not is_result:
        raise error_type(
            f"{label} must be a GrammarMaskedSpeculativeIterationResult"
        )
    return cast(GrammarMaskedSpeculativeIterationResult[object], value)


def _require_outcome_result(
    value: object,
    *,
    label: str,
    error_type: type[Exception],
) -> GrammarMaskedSpeculativeOutcomeResult:
    try:
        is_result = isinstance(value, GrammarMaskedSpeculativeOutcomeResult)
    except Exception as exc:
        raise error_type(f"{label} type could not be determined") from exc
    if not is_result:
        raise error_type(
            f"{label} must be a GrammarMaskedSpeculativeOutcomeResult"
        )
    return cast(GrammarMaskedSpeculativeOutcomeResult, value)


def _read_outcome_kind(
    result: GrammarMaskedSpeculativeOutcomeResult,
    *,
    label: str,
) -> str:
    kind = _read_attribute(result, "kind", label=label)
    if type(kind) is not str or kind not in {
        "handoff_available",
        "grammar_complete",
        "grammar_no_continuation",
    }:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            f"{label} contains an unsupported kind"
        )
    return kind


def _validate_final_relationship(
    *,
    output_token_ids: tuple[int, ...],
    final_iteration: GrammarMaskedSpeculativeIterationResult[object],
    final_outcome: GrammarMaskedSpeculativeOutcomeResult,
    disposition: FinalDisposition,
    grammar_completion_token_id: int | None,
    vocab_size: int | None = None,
) -> tuple[tuple[int, ...], int | None, object, bool, str]:
    final_output = _read_attribute(
        final_iteration,
        "output_token_ids",
        label="final_iteration",
    )
    if vocab_size is None:
        final_output = _validate_direct_final_output(final_output)
    else:
        final_output = _validate_operation_output(
            final_output,
            vocab_size=vocab_size,
        )
    if len(final_output) > len(output_token_ids) or (
        final_output and output_token_ids[-len(final_output) :] != final_output
    ):
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final_iteration output must be an exact suffix of output_token_ids"
        )

    handoff = _read_attribute(
        final_iteration,
        "uncached_next_token_id",
        label="final_iteration",
    )
    if handoff is not None:
        if type(handoff) is not int:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "final_iteration uncached_next_token_id must be an integer or None"
            )
        if handoff < 0 or (vocab_size is not None and handoff >= vocab_size):
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "final_iteration uncached_next_token_id is outside the numeric domain"
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
    if type(committed_state_is_match) is not bool:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final_iteration committed_state_is_match must be a boolean"
        )
    kind = _read_outcome_kind(final_outcome, label="final_outcome")

    expected_kind = {
        "grammar_complete": "grammar_complete",
        "grammar_no_continuation": "grammar_no_continuation",
        "iteration_bound_exhausted": "handoff_available",
    }[disposition]
    if kind != expected_kind:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "disposition disagrees with final_outcome kind"
        )

    if disposition == "grammar_complete":
        if handoff is not None or committed_state_is_match is not True:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "grammar completion requires matching terminal evidence without a handoff"
            )
        if grammar_completion_token_id is None:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "grammar completion requires a completion token"
            )
    elif disposition == "grammar_no_continuation":
        if handoff is not None or committed_state_is_match is not False:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "grammar no-continuation requires nonmatching terminal evidence "
                "without a handoff"
            )
        if grammar_completion_token_id is not None:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "grammar no-continuation cannot contain a completion token"
            )
    else:
        if handoff is None:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "iteration-bound exhaustion requires one uncached handoff token"
            )
        if not output_token_ids or output_token_ids[-1] != handoff:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "final handoff token must be the final accumulated output token"
            )
        if grammar_completion_token_id is not None:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                "iteration-bound exhaustion cannot contain a completion token"
            )

    return (
        final_output,
        cast(int | None, handoff),
        committed_state,
        committed_state_is_match,
        kind,
    )


def _validate_direct_final_output(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            "final_iteration output_token_ids must be an exact tuple"
        )
    for position, token_id in enumerate(value):
        if type(token_id) is not int:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                f"final_iteration output token at position {position} must be an integer"
            )
        if token_id < 0:
            raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
                f"final_iteration output token at position {position} cannot be negative"
            )
    return cast(tuple[int, ...], value)


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedSpeculativeFinalOutcomeInvariantError(
            f"{label} {name} could not be read"
        ) from exc
