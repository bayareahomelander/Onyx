"""Framework-neutral target-only generation confined by one grammar state."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from .backend import AutoregressiveBackend
from .generation import (
    FinishReason,
    GenerationResult,
    TargetGenerationStep,
    _validate_backend_cache_length,
    _validate_backend_step,
    _validate_generation_inputs,
)
from .grammar import (
    GrammarConstraint,
    GrammarType,
    _validate_json_schema,
    _validate_regex_pattern,
)
from .metrics import (
    GrammarTimingSession,
    TargetMetricsSession,
    create_grammar_timing_session,
    create_target_metrics_session,
)


class ConstrainedGenerationError(RuntimeError):
    """Base error raised by constrained-generation orchestration."""


class ConstrainedGenerationInvariantError(ConstrainedGenerationError):
    """Raised when constrained components violate their shared contract."""


class GrammarNoContinuationError(ConstrainedGenerationError):
    """Raised when a live, incomplete grammar state has no valid continuation."""


class ConstrainedGenerationCleanupError(ConstrainedGenerationError):
    """Raised when a failed generation also cannot be cleaned up completely."""

    def __init__(
        self,
        original_failure: BaseException,
        cleanup_failures: Sequence[tuple[str, Exception]],
    ) -> None:
        failures = tuple(cleanup_failures)
        if not failures:
            raise ValueError("cleanup_failures cannot be empty")
        self.original_failure = original_failure
        self.cleanup_failures = failures
        details = "; ".join(
            f"{operation} also failed: {failure}" for operation, failure in failures
        )
        super().__init__(f"constrained generation failed: {original_failure}; {details}")


@dataclass(frozen=True, slots=True)
class RegexGrammar:
    """One immutable native-regex generation specification."""

    pattern: str

    def __post_init__(self) -> None:
        _validate_regex_pattern(self.pattern)


@dataclass(frozen=True, slots=True)
class JsonSchemaGrammar:
    """One immutable native-JSON-Schema generation specification."""

    schema: str

    def __post_init__(self) -> None:
        _validate_json_schema(self.schema)


GrammarSpecification: TypeAlias = RegexGrammar | JsonSchemaGrammar
LogitsT = TypeVar("LogitsT")
StateT = TypeVar("StateT")


@runtime_checkable
class GrammarLogitMask(Protocol[LogitsT]):
    """Stateless masking boundary for one immutable valid-token tuple."""

    @property
    def vocab_size(self) -> int: ...

    def apply(
        self,
        logits: LogitsT,
        valid_token_ids: tuple[int, ...],
        /,
    ) -> LogitsT: ...


@runtime_checkable
class _TimedGrammarLogitMask(Protocol[LogitsT]):
    """Optional internal mask capability for separate completed stage timings."""

    def apply_with_timing(
        self,
        logits: LogitsT,
        valid_token_ids: tuple[int, ...],
        timing_session: GrammarTimingSession,
        /,
    ) -> LogitsT: ...


@dataclass(frozen=True, slots=True)
class GrammarGenerationContext(Generic[LogitsT, StateT]):
    """Fresh owned constraint/timing plus borrowed mask and terminal EOS configuration."""

    constraint: GrammarConstraint[StateT]
    logit_mask: GrammarLogitMask[LogitsT]
    eos_token_id: int
    timing_session: GrammarTimingSession = field(default_factory=create_grammar_timing_session)


def generate_constrained_target(
    backend: AutoregressiveBackend[LogitsT],
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    grammar_context: GrammarGenerationContext[LogitsT, StateT],
    stop_token_sequences: Sequence[Sequence[int]] = (),
    metrics_session: TargetMetricsSession | None = None,
) -> GenerationResult:
    """Generate one non-streaming target sequence confined by a fresh grammar context."""

    return _generate_constrained_target(
        backend,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        select_token=select_token,
        grammar_context=grammar_context,
        stop_token_sequences=stop_token_sequences,
        metrics_session=metrics_session,
        expected_grammar_type=None,
        create_metrics_session=None,
    )


def _generate_constrained_target(
    backend: AutoregressiveBackend[LogitsT],
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    grammar_context: GrammarGenerationContext[LogitsT, StateT],
    stop_token_sequences: Sequence[Sequence[int]],
    metrics_session: TargetMetricsSession | None,
    expected_grammar_type: GrammarType | None,
    create_metrics_session: Callable[[], TargetMetricsSession] | None,
) -> GenerationResult:
    for step in _iterate_constrained_target(
        backend,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        select_token=select_token,
        grammar_context=grammar_context,
        stop_token_sequences=stop_token_sequences,
        metrics_session=metrics_session,
        expected_grammar_type=expected_grammar_type,
        create_metrics_session=create_metrics_session,
    ):
        if step.result is not None:
            return step.result
    raise ConstrainedGenerationInvariantError(
        "constrained generation ended without a terminal result"
    )


def _iterate_constrained_target(
    backend: AutoregressiveBackend[LogitsT],
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    grammar_context: GrammarGenerationContext[LogitsT, StateT],
    stop_token_sequences: Sequence[Sequence[int]],
    metrics_session: TargetMetricsSession | None,
    expected_grammar_type: GrammarType | None,
    create_metrics_session: Callable[[], TargetMetricsSession] | None,
) -> Generator[TargetGenerationStep, None, None]:
    """Iterate one validated constrained generation with deterministic cleanup."""

    constraint = (
        grammar_context.constraint
        if isinstance(grammar_context, GrammarGenerationContext)
        else None
    )
    owned_states: list[StateT] = []
    constraint_reset = False
    backend_started = False
    metrics_begin_attempted = False
    metrics_finished = False
    grammar_timing_session = (
        grammar_context.timing_session
        if isinstance(grammar_context, GrammarGenerationContext)
        and isinstance(grammar_context.timing_session, GrammarTimingSession)
        else None
    )
    grammar_timing_finished = False
    completed = False

    try:
        prompt, stops, vocab_size = _validate_generation_inputs(
            backend,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            select_token=select_token,
            stop_token_sequences=stop_token_sequences,
        )
        constraint, logit_mask, eos_token_id, grammar_timing_session = _validate_grammar_context(
            grammar_context,
            vocab_size=vocab_size,
            expected_grammar_type=expected_grammar_type,
        )

        current_state = constraint.init_state()
        owned_states.append(current_state)

        if metrics_session is None and create_metrics_session is not None:
            metrics_session = create_metrics_session()
        elif metrics_session is None:
            metrics_session = create_target_metrics_session(cache_mode="unknown")
        if not isinstance(metrics_session, TargetMetricsSession):
            raise TypeError("metrics_session must be a TargetMetricsSession")
        metrics_begin_attempted = True
        metrics_session.begin()

        generated: list[int] = []
        prefix_matches: list[bool] = []
        expected_cache_length = len(prompt)
        finish_reason: FinishReason | None = None
        matched_stop_token_ids: tuple[int, ...] | None = None
        grammar_completion_token_id: int | None = None

        while len(generated) < max_new_tokens:
            with metrics_session.active():
                current_is_dead = _require_state_boolean(
                    constraint.is_dead_state(current_state),
                    operation="is_dead_state",
                )
                if current_is_dead:
                    raise ConstrainedGenerationInvariantError(
                        "constrained generation retained a dead grammar state"
                    )
                current_is_match = _require_state_boolean(
                    constraint.is_match_state(current_state),
                    operation="is_match_state",
                )
                if not prefix_matches:
                    prefix_matches.append(current_is_match)
                elif prefix_matches[-1] != current_is_match:
                    raise ConstrainedGenerationInvariantError(
                        "retained grammar state changed its recorded match status"
                    )

                with grammar_timing_session.state_scan():
                    scanned_valid_token_ids = constraint.get_valid_token_ids(current_state)
                native_valid_token_ids = _validate_native_valid_token_ids(
                    scanned_valid_token_ids,
                    vocab_size=vocab_size,
                )
                effective_valid_token_ids = _effective_valid_token_ids(
                    native_valid_token_ids,
                    is_match=current_is_match,
                    eos_token_id=eos_token_id,
                )
                if not effective_valid_token_ids:
                    raise GrammarNoContinuationError(
                        "live nonmatching grammar state has no valid token continuation"
                    )

                if not generated:
                    backend_started = True
                    backend_step = backend.prefill(prompt)
                else:
                    expected_cache_length += 1
                    backend_started = True
                    backend_step = backend.decode(generated[-1])
                _validate_backend_step(backend, backend_step, expected_cache_length)

                if isinstance(logit_mask, _TimedGrammarLogitMask):
                    masked_logits = logit_mask.apply_with_timing(
                        backend_step.logits,
                        effective_valid_token_ids,
                        grammar_timing_session,
                    )
                else:
                    if (
                        getattr(logit_mask, "transport_name", None)
                        == "sparse_valid_indices"
                    ):
                        raise ConstrainedGenerationInvariantError(
                            "production sparse-index grammar masks must provide completed "
                            "transfer/application timing"
                        )
                    with grammar_timing_session.mask_application():
                        masked_logits = logit_mask.apply(
                            backend_step.logits,
                            effective_valid_token_ids,
                        )
                token_id = _validate_selected_token(
                    select_token(masked_logits),
                    vocab_size=vocab_size,
                    effective_valid_token_ids=effective_valid_token_ids,
                )

                parent_state = current_state
                child_state = constraint.advance_state(parent_state, token_id)
                if child_state is parent_state:
                    raise ConstrainedGenerationInvariantError(
                        "grammar advancement must return an independent child state"
                    )
                owned_states.append(child_state)

                child_is_dead = _require_state_boolean(
                    constraint.is_dead_state(child_state),
                    operation="is_dead_state",
                )
                child_is_match = _require_state_boolean(
                    constraint.is_match_state(child_state),
                    operation="is_match_state",
                )
                if child_is_dead:
                    raise ConstrainedGenerationInvariantError(
                        f"grammar-valid token ID {token_id} advanced to a dead child state"
                    )
                selected_eos = token_id == eos_token_id
                if selected_eos and not child_is_match:
                    raise ConstrainedGenerationInvariantError(
                        "injected EOS must preserve a matching grammar state"
                    )

                constraint.release_state(parent_state)
                owned_states[:] = [child_state]
                current_state = child_state

                generated.append(token_id)
                prefix_matches.append(child_is_match)
                if len(generated) == 1:
                    metrics_session.mark_first_token()

                matched_stop_token_ids = _match_eligible_stop_token_sequence(
                    generated,
                    stops,
                    prefix_matches,
                )
                if matched_stop_token_ids is not None:
                    finish_reason = "stop"
                elif selected_eos:
                    finish_reason = "grammar_complete"
                    grammar_completion_token_id = token_id
                elif len(generated) >= max_new_tokens:
                    finish_reason = "length"

            if finish_reason is not None:
                break
            yield TargetGenerationStep(token_id=token_id)

        if finish_reason is None:
            raise ConstrainedGenerationInvariantError(
                "constrained generation ended without a terminal reason"
            )

        constraint.release_state(current_state)
        owned_states.clear()
        constraint.reset()
        constraint_reset = True
        final_cache_length = _validate_backend_cache_length(
            backend,
            expected_cache_length,
        )
        grammar_timing = grammar_timing_session.finish(len(generated))
        grammar_timing_finished = True
        metrics = metrics_session.finish(
            len(generated),
            grammar_timing=grammar_timing,
        )
        metrics_finished = True
        result = GenerationResult(
            model_id=backend.model_id,
            token_ids=tuple(generated),
            finish_reason=finish_reason,
            prompt_tokens=len(prompt),
            final_cache_length=final_cache_length,
            metrics=metrics,
            matched_stop_token_ids=matched_stop_token_ids,
            grammar_completion_token_id=grammar_completion_token_id,
        )
        completed = True
        yield TargetGenerationStep(token_id=generated[-1], result=result)
    except BaseException as failure:
        if completed:
            raise

        cleanup_failures: list[tuple[str, Exception]] = []

        if constraint is not None and not constraint_reset:
            if owned_states:
                try:
                    constraint.release_states(tuple(owned_states))
                except Exception as cleanup_failure:
                    cleanup_failures.append(("grammar state release", cleanup_failure))
                else:
                    owned_states.clear()
            try:
                constraint.reset()
            except Exception as cleanup_failure:
                cleanup_failures.append(("grammar constraint reset", cleanup_failure))
            else:
                constraint_reset = True

        if backend_started:
            try:
                backend.reset()
            except Exception as cleanup_failure:
                cleanup_failures.append(("backend reset", cleanup_failure))

        if metrics_begin_attempted and not metrics_finished and metrics_session is not None:
            try:
                metrics_session.abort()
            except Exception as cleanup_failure:
                cleanup_failures.append(("metrics abort", cleanup_failure))

        if grammar_timing_session is not None and not grammar_timing_finished:
            try:
                grammar_timing_session.abort()
            except Exception as cleanup_failure:
                cleanup_failures.append(("grammar timing abort", cleanup_failure))

        if cleanup_failures:
            raise ConstrainedGenerationCleanupError(
                failure,
                cleanup_failures,
            ) from failure
        raise


def _validate_grammar_specification(
    grammar: GrammarSpecification,
) -> tuple[GrammarSpecification, GrammarType]:
    if isinstance(grammar, RegexGrammar):
        _validate_regex_pattern(grammar.pattern)
        return grammar, "regex"
    if isinstance(grammar, JsonSchemaGrammar):
        _validate_json_schema(grammar.schema)
        return grammar, "json_schema"
    raise TypeError("grammar must be a RegexGrammar or JsonSchemaGrammar")


def _validate_grammar_context(
    grammar_context: GrammarGenerationContext[LogitsT, StateT],
    *,
    vocab_size: int,
    expected_grammar_type: GrammarType | None,
) -> tuple[
    GrammarConstraint[StateT],
    GrammarLogitMask[LogitsT],
    int,
    GrammarTimingSession,
]:
    if not isinstance(grammar_context, GrammarGenerationContext):
        raise TypeError("grammar_context must be a GrammarGenerationContext")

    constraint = grammar_context.constraint
    if not isinstance(constraint, GrammarConstraint):
        raise ConstrainedGenerationInvariantError(
            "grammar context constraint must implement GrammarConstraint"
        )
    logit_mask = grammar_context.logit_mask
    if not isinstance(logit_mask, GrammarLogitMask):
        raise ConstrainedGenerationInvariantError(
            "grammar context mask must implement GrammarLogitMask"
        )
    timing_session = grammar_context.timing_session
    if not isinstance(timing_session, GrammarTimingSession):
        raise ConstrainedGenerationInvariantError(
            "grammar context timing_session must be a GrammarTimingSession"
        )

    constraint_vocab_size = _validated_component_vocab_size(
        constraint.vocab_size,
        label="constraint",
    )
    mask_vocab_size = _validated_component_vocab_size(
        logit_mask.vocab_size,
        label="mask",
    )
    if constraint_vocab_size != vocab_size or mask_vocab_size != vocab_size:
        raise ConstrainedGenerationInvariantError(
            "backend, constraint, and mask vocabulary sizes must match exactly: "
            f"backend={vocab_size}, constraint={constraint_vocab_size}, mask={mask_vocab_size}"
        )

    grammar_type = constraint.grammar_type
    if grammar_type not in {"regex", "json_schema"}:
        raise ConstrainedGenerationInvariantError(
            "constraint grammar_type must be 'regex' or 'json_schema'"
        )
    if expected_grammar_type is not None and grammar_type != expected_grammar_type:
        raise ConstrainedGenerationInvariantError(
            f"grammar specification type {expected_grammar_type!r} does not match returned "
            f"constraint type {grammar_type!r}"
        )

    eos_token_id = grammar_context.eos_token_id
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ConstrainedGenerationInvariantError("grammar EOS token ID must be an integer")
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise ConstrainedGenerationInvariantError(
            f"grammar EOS token ID {eos_token_id} is outside vocabulary range "
            f"[0, {vocab_size})"
        )
    return constraint, logit_mask, eos_token_id, timing_session


def _validated_component_vocab_size(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConstrainedGenerationInvariantError(
            f"{label} vocabulary size must be an integer"
        )
    if value <= 0:
        raise ConstrainedGenerationInvariantError(
            f"{label} vocabulary size must be greater than zero"
        )
    return value


def _validate_native_valid_token_ids(
    token_ids: tuple[int, ...],
    *,
    vocab_size: int,
) -> tuple[int, ...]:
    if not isinstance(token_ids, tuple):
        raise ConstrainedGenerationInvariantError(
            "constraint valid-token output must be a tuple"
        )
    previous = -1
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ConstrainedGenerationInvariantError(
                "constraint valid-token output must contain Python integers"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise ConstrainedGenerationInvariantError(
                f"constraint returned token ID {token_id} outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if token_id <= previous:
            raise ConstrainedGenerationInvariantError(
                "constraint valid-token output must be strictly increasing and unique"
            )
        previous = token_id
    return token_ids


def _effective_valid_token_ids(
    native_valid_token_ids: tuple[int, ...],
    *,
    is_match: bool,
    eos_token_id: int,
) -> tuple[int, ...]:
    insertion = bisect_left(native_valid_token_ids, eos_token_id)
    if insertion < len(native_valid_token_ids) and native_valid_token_ids[insertion] == eos_token_id:
        raise ConstrainedGenerationInvariantError(
            "native valid-token output must not advertise the empty-byte EOS token"
        )
    if not is_match:
        return native_valid_token_ids
    return (
        native_valid_token_ids[:insertion]
        + (eos_token_id,)
        + native_valid_token_ids[insertion:]
    )


def _validate_selected_token(
    token_id: int,
    *,
    vocab_size: int,
    effective_valid_token_ids: tuple[int, ...],
) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise ConstrainedGenerationInvariantError("selected token ID must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ConstrainedGenerationInvariantError(
            f"selected token ID {token_id} is outside vocabulary range [0, {vocab_size})"
        )
    position = bisect_left(effective_valid_token_ids, token_id)
    if (
        position >= len(effective_valid_token_ids)
        or effective_valid_token_ids[position] != token_id
    ):
        raise ConstrainedGenerationInvariantError(
            f"selected token ID {token_id} is outside the effective grammar support"
        )
    return token_id


def _require_state_boolean(value: bool, *, operation: str) -> bool:
    if not isinstance(value, bool):
        raise ConstrainedGenerationInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _match_eligible_stop_token_sequence(
    generated_token_ids: Sequence[int],
    stop_token_sequences: Sequence[tuple[int, ...]],
    prefix_matches: Sequence[bool],
) -> tuple[int, ...] | None:
    if len(prefix_matches) != len(generated_token_ids) + 1:
        raise ConstrainedGenerationInvariantError(
            "grammar prefix-match history is inconsistent with generated tokens"
        )
    for stop_sequence in stop_token_sequences:
        if len(generated_token_ids) < len(stop_sequence):
            continue
        if tuple(generated_token_ids[-len(stop_sequence) :]) != stop_sequence:
            continue
        visible_prefix_length = len(generated_token_ids) - len(stop_sequence)
        if prefix_matches[visible_prefix_length]:
            return stop_sequence
    return None
