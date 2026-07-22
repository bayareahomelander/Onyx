"""Framework-neutral target-only token generation."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from .backend import AutoregressiveBackend, BackendError, ModelStep
from .metrics import (
    TargetGenerationMetrics,
    TargetMetricsSession,
    create_target_metrics_session,
)
from .stop_sequences import (
    match_stop_token_sequence,
    normalize_stop_token_sequences,
    validate_token_id,
    validate_token_ids,
)


class GenerationInvariantError(BackendError):
    """Raised when a backend violates the generation contract."""


class GenerationCleanupError(BackendError):
    """Raised when incomplete generation state cannot be reset safely."""


FinishReason = Literal["length", "stop", "grammar_complete"]
LogitsT = TypeVar("LogitsT")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated token IDs and the verified backend state at termination."""

    model_id: str
    token_ids: tuple[int, ...]
    finish_reason: FinishReason
    prompt_tokens: int
    final_cache_length: int
    metrics: TargetGenerationMetrics
    matched_stop_token_ids: tuple[int, ...] | None = None
    grammar_completion_token_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, TargetGenerationMetrics):
            raise TypeError("generation metrics must be TargetGenerationMetrics")
        expected_throughput = len(self.token_ids) / self.metrics.generation_time
        if not math.isclose(
            self.metrics.tokens_per_second,
            expected_throughput,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "tokens_per_second must equal generated tokens divided by generation_time"
            )
        if self.finish_reason == "length":
            if self.matched_stop_token_ids is not None:
                raise ValueError("length results cannot report a matched stop sequence")
            if self.grammar_completion_token_id is not None:
                raise ValueError("length results cannot report a grammar completion token")
            return
        if self.finish_reason == "stop":
            if self.grammar_completion_token_id is not None:
                raise ValueError("stop results cannot report a grammar completion token")
            if not isinstance(self.matched_stop_token_ids, tuple):
                raise TypeError("stop results must report matched stop token IDs as a tuple")
            if not self.matched_stop_token_ids:
                raise ValueError("stop results must report a nonempty matched stop sequence")
            if len(self.matched_stop_token_ids) > len(self.token_ids) or tuple(
                self.token_ids[-len(self.matched_stop_token_ids) :]
            ) != self.matched_stop_token_ids:
                raise ValueError("matched stop token IDs must be a suffix of generated token IDs")
            return
        if self.finish_reason != "grammar_complete":
            raise ValueError(
                "finish_reason must be 'length', 'stop', or 'grammar_complete'"
            )
        if self.matched_stop_token_ids is not None:
            raise ValueError("grammar-complete results cannot report a matched stop sequence")
        if isinstance(self.grammar_completion_token_id, bool) or not isinstance(
            self.grammar_completion_token_id, int
        ):
            raise TypeError("grammar-complete results must report an integer completion token ID")
        if not self.token_ids or self.token_ids[-1] != self.grammar_completion_token_id:
            raise ValueError("grammar completion token ID must be the final sampled token ID")

    @property
    def generated_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def visible_token_ids(self) -> tuple[int, ...]:
        """Return sampled IDs after finish-reason-specific terminal trimming."""

        if self.matched_stop_token_ids is not None:
            return self.token_ids[: -len(self.matched_stop_token_ids)]
        if self.grammar_completion_token_id is not None:
            return self.token_ids[:-1]
        return self.token_ids


@dataclass(frozen=True, slots=True)
class TargetGenerationStep:
    """One selected token and an optional terminal generation result."""

    token_id: int
    result: GenerationResult | None = None

    def __post_init__(self) -> None:
        if isinstance(self.token_id, bool) or not isinstance(self.token_id, int):
            raise TypeError("target generation step token_id must be an integer")
        if self.result is not None and (
            not self.result.token_ids or self.result.token_ids[-1] != self.token_id
        ):
            raise ValueError("terminal generation result must end with the step token ID")


def generate_greedy(
    backend: AutoregressiveBackend[LogitsT],
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    stop_token_sequences: Sequence[Sequence[int]] = (),
    metrics_session: TargetMetricsSession | None = None,
) -> GenerationResult:
    """Generate token IDs from one autoregressive backend using an argmax selector.

    The injected selector keeps backend-native logits on their native device. Returned token IDs
    include a complete matched stop sequence; trimming belongs to the later tokenizer/API layer.
    The final selected token is not decoded back into the cache because no following logits are
    needed.
    """

    return generate_target(
        backend,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        select_token=select_token,
        stop_token_sequences=stop_token_sequences,
        metrics_session=metrics_session,
    )


def generate_target(
    backend: AutoregressiveBackend[LogitsT],
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    stop_token_sequences: Sequence[Sequence[int]] = (),
    metrics_session: TargetMetricsSession | None = None,
) -> GenerationResult:
    """Generate target tokens with one caller-provided selection session.

    Stop sequences are matched only against generated-token suffixes after each selection. The
    first configured complete match wins when multiple sequences complete on the same token.
    """

    for step in iterate_target(
        backend,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        select_token=select_token,
        stop_token_sequences=stop_token_sequences,
        metrics_session=metrics_session,
    ):
        if step.result is not None:
            return step.result
    raise GenerationInvariantError("target generation ended without a terminal result")


def iterate_target(
    backend: AutoregressiveBackend[LogitsT],
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    stop_token_sequences: Sequence[Sequence[int]] = (),
    metrics_session: TargetMetricsSession | None = None,
) -> Generator[TargetGenerationStep, None, None]:
    """Return a validated, cancellable iterator over target-token selections."""

    prompt, stops, vocab_size = _validate_generation_inputs(
        backend,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        select_token=select_token,
        stop_token_sequences=stop_token_sequences,
    )
    if metrics_session is None:
        metrics_session = create_target_metrics_session(cache_mode="unknown")
    if not isinstance(metrics_session, TargetMetricsSession):
        raise TypeError("metrics_session must be a TargetMetricsSession")
    return _iterate_validated_target(
        backend,
        prompt,
        max_new_tokens=max_new_tokens,
        select_token=select_token,
        stops=stops,
        vocab_size=vocab_size,
        metrics_session=metrics_session,
    )


def select_highest_logit(logits: Sequence[float]) -> int:
    """Return the first index containing the greatest score in a CPU sequence."""

    try:
        values = tuple(float(value) for value in logits)
    except (TypeError, ValueError) as exc:
        raise ValueError("logits must contain numeric values") from exc
    if not values:
        raise ValueError("logits cannot be empty")
    if any(math.isnan(value) for value in values):
        raise ValueError("logits cannot contain NaN")

    return max(range(len(values)), key=values.__getitem__)


def _validate_backend_step(
    backend: AutoregressiveBackend[LogitsT],
    step: ModelStep[LogitsT],
    expected_cache_length: int,
) -> None:
    if step.cache_length != expected_cache_length:
        raise GenerationInvariantError(
            f"backend step reported cache length {step.cache_length}; "
            f"expected {expected_cache_length}"
        )
    _validate_backend_cache_length(backend, expected_cache_length)


def _validate_backend_cache_length(
    backend: AutoregressiveBackend[LogitsT],
    expected_cache_length: int,
) -> int:
    cache_length = backend.cache_length
    if cache_length != expected_cache_length:
        raise GenerationInvariantError(
            f"backend state reported cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )
    return cache_length


def _validate_generation_inputs(
    backend: AutoregressiveBackend[LogitsT],
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    stop_token_sequences: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if not callable(select_token):
        raise TypeError("select_token must be callable")
    vocab_size = backend.vocab_size
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise GenerationInvariantError("backend vocab_size must be an integer")
    if vocab_size <= 0:
        raise GenerationInvariantError("backend vocab_size must be greater than zero")

    prompt = tuple(prompt_token_ids)
    validate_token_ids(prompt, vocab_size, label="prompt", allow_empty=False)
    stops = normalize_stop_token_sequences(stop_token_sequences, vocab_size)
    return prompt, stops, vocab_size


def _iterate_validated_target(
    backend: AutoregressiveBackend[LogitsT],
    prompt: tuple[int, ...],
    *,
    max_new_tokens: int,
    select_token: Callable[[LogitsT], int],
    stops: tuple[tuple[int, ...], ...],
    vocab_size: int,
    metrics_session: TargetMetricsSession,
) -> Generator[TargetGenerationStep, None, None]:
    completed = False
    failure: BaseException | None = None
    try:
        expected_cache_length = len(prompt)
        generated = []
        step: ModelStep[LogitsT] | None = None
        metrics_session.begin()

        while len(generated) < max_new_tokens:
            with metrics_session.active():
                if step is None:
                    step = backend.prefill(prompt)
                else:
                    expected_cache_length += 1
                    step = backend.decode(generated[-1])
                _validate_backend_step(backend, step, expected_cache_length)

                token_id = select_token(step.logits)
                validate_token_id(token_id, vocab_size, label="selected token")
                generated.append(token_id)
                if len(generated) == 1:
                    metrics_session.mark_first_token()

                matched_stop_token_ids = match_stop_token_sequence(generated, stops)
                finish_reason: FinishReason | None = None
                if matched_stop_token_ids is not None:
                    finish_reason = "stop"
                elif len(generated) >= max_new_tokens:
                    finish_reason = "length"

            if finish_reason is not None:
                metrics = metrics_session.finish(len(generated))
                result = GenerationResult(
                    model_id=backend.model_id,
                    token_ids=tuple(generated),
                    finish_reason=finish_reason,
                    prompt_tokens=len(prompt),
                    final_cache_length=backend.cache_length,
                    metrics=metrics,
                    matched_stop_token_ids=matched_stop_token_ids,
                )
                completed = True
                yield TargetGenerationStep(token_id=token_id, result=result)
                return

            yield TargetGenerationStep(token_id=token_id)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if not completed:
            reset_failure: Exception | None = None
            metrics_failure: Exception | None = None
            try:
                backend.reset()
            except Exception as cleanup_exc:
                reset_failure = cleanup_exc
            try:
                metrics_session.abort()
            except Exception as cleanup_exc:
                metrics_failure = cleanup_exc

            if reset_failure is not None and metrics_failure is None:
                if failure is not None and not isinstance(failure, GeneratorExit):
                    raise GenerationCleanupError(
                        f"target generation failed: {failure}; reset also failed: "
                        f"{reset_failure}"
                    ) from failure
                raise GenerationCleanupError(
                    f"incomplete target generation reset failed: {reset_failure}"
                ) from reset_failure
            if reset_failure is None and metrics_failure is not None:
                if failure is not None and not isinstance(failure, GeneratorExit):
                    raise GenerationCleanupError(
                        f"target generation failed: {failure}; metrics abort also failed: "
                        f"{metrics_failure}"
                    ) from failure
                raise GenerationCleanupError(
                    f"incomplete target generation metrics abort failed: {metrics_failure}"
                ) from metrics_failure
            if reset_failure is not None and metrics_failure is not None:
                if failure is not None and not isinstance(failure, GeneratorExit):
                    raise GenerationCleanupError(
                        f"target generation failed: {failure}; backend reset also failed: "
                        f"{reset_failure}; metrics abort also failed: {metrics_failure}"
                    ) from failure
                raise GenerationCleanupError(
                    f"incomplete target generation reset failed: {reset_failure}; "
                    f"metrics abort also failed: {metrics_failure}"
                ) from reset_failure
