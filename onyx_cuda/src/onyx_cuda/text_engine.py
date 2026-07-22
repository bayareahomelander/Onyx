"""Deterministic target-only text generation over framework-neutral contracts."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import Generic, TypeVar

from .backend import AutoregressiveBackend
from .constrained_generation import (
    ConstrainedGenerationCleanupError,
    ConstrainedGenerationError,
    GrammarGenerationContext,
    GrammarSpecification,
    _generate_constrained_target,
    _iterate_constrained_target,
    _validate_grammar_specification,
)
from .generation import (
    GenerationResult,
    TargetGenerationStep,
    generate_target,
    iterate_target,
)
from .grammar import GrammarType
from .metrics import TargetMetricsSession, create_target_metrics_session
from .selection import (
    GREEDY_SELECTION,
    GreedySelection,
    SelectionError,
    SelectionPolicy,
    TemperatureTopPSelection,
    validate_selection_policy,
)
from .tokenizer import TokenizerAdapter
from .stop_sequences import normalize_stop_token_sequences, validate_token_ids
from .streaming import (
    StreamingCleanupError,
    StreamingInvariantError,
    TextGenerationComplete,
    TextGenerationDelta,
    TextGenerationEvent,
    TextGenerationResult,
    _StableTextDecoder,
    _StopTokenBuffer,
)


class VocabularyMismatchError(ValueError):
    """Raised when a model backend and tokenizer report different vocabulary sizes."""


LogitsT = TypeVar("LogitsT")


class TargetTextEngine(Generic[LogitsT]):
    """Compose a tokenizer, target backend, and token selector into text generation."""

    def __init__(
        self,
        backend: AutoregressiveBackend[LogitsT],
        tokenizer: TokenizerAdapter,
        *,
        select_token: Callable[[LogitsT], int],
        create_sampling_selector: (
            Callable[[TemperatureTopPSelection], Callable[[LogitsT], int]] | None
        ) = None,
        create_metrics_session: Callable[[], TargetMetricsSession] | None = None,
        create_grammar_context: (
            Callable[
                [GrammarSpecification],
                GrammarGenerationContext[LogitsT, object],
            ]
            | None
        ) = None,
    ) -> None:
        if not callable(select_token):
            raise TypeError("select_token must be callable")
        if create_sampling_selector is not None and not callable(create_sampling_selector):
            raise TypeError("create_sampling_selector must be callable")
        if create_metrics_session is not None and not callable(create_metrics_session):
            raise TypeError("create_metrics_session must be callable")
        if create_grammar_context is not None and not callable(create_grammar_context):
            raise TypeError("create_grammar_context must be callable")

        backend_vocab_size = _validate_vocab_size(backend.vocab_size, label="backend")
        tokenizer_vocab_size = _validate_vocab_size(tokenizer.vocab_size, label="tokenizer")
        if backend_vocab_size != tokenizer_vocab_size:
            raise VocabularyMismatchError(
                f"backend vocabulary size {backend_vocab_size} does not match "
                f"tokenizer vocabulary size {tokenizer_vocab_size}"
            )

        self._backend = backend
        self._tokenizer = tokenizer
        self._select_token = select_token
        self._create_sampling_selector = create_sampling_selector
        if create_metrics_session is None:
            cache_mode = getattr(backend, "cache_mode", "unknown")
            self._create_metrics_session = lambda: create_target_metrics_session(
                cache_mode=cache_mode
            )
        else:
            self._create_metrics_session = create_metrics_session
        self._create_grammar_context = create_grammar_context

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer.tokenizer_id

    @property
    def vocab_size(self) -> int:
        return self._backend.vocab_size

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> TextGenerationResult:
        prompt_token_ids, select_token, stops = self._prepare_generation(
            prompt,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )
        _validate_max_new_tokens(max_new_tokens)

        generation = generate_target(
            self._backend,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            select_token=select_token,
            stop_token_sequences=stops,
            metrics_session=self._new_metrics_session(),
        )
        return self._build_result(generation)

    def generate_constrained(
        self,
        prompt: str,
        *,
        grammar: GrammarSpecification,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> TextGenerationResult:
        """Generate one non-streaming sequence through a fresh grammar context."""

        grammar, expected_grammar_type = _validate_grammar_specification(grammar)
        prompt_token_ids, select_token, stops = self._prepare_generation(
            prompt,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )
        _validate_max_new_tokens(max_new_tokens)
        if self._create_grammar_context is None:
            raise ConstrainedGenerationError(
                "constrained generation is not configured for this text engine"
            )
        grammar_context = self._create_grammar_context(grammar)
        generation = _generate_constrained_target(
            self._backend,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            select_token=select_token,
            grammar_context=grammar_context,
            stop_token_sequences=stops,
            metrics_session=None,
            expected_grammar_type=expected_grammar_type,
            create_metrics_session=self._new_metrics_session,
        )
        try:
            return self._build_result(generation)
        except BaseException as failure:
            try:
                self._backend.reset()
            except Exception as cleanup_failure:
                raise ConstrainedGenerationCleanupError(
                    failure,
                    (("backend reset", cleanup_failure),),
                ) from failure
            raise

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> Generator[TextGenerationEvent, None, None]:
        """Return validated incremental text events for one target generation."""

        prompt_token_ids, select_token, stops = self._prepare_generation(
            prompt,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )
        _validate_max_new_tokens(max_new_tokens)
        target_steps = iterate_target(
            self._backend,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            select_token=select_token,
            stop_token_sequences=stops,
            metrics_session=self._new_metrics_session(),
        )
        return self._stream_validated(target_steps, stops)

    def stream_constrained(
        self,
        prompt: str,
        *,
        grammar: GrammarSpecification,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> Generator[TextGenerationEvent, None, None]:
        """Return validated incremental text events through a fresh grammar context."""

        grammar, expected_grammar_type = _validate_grammar_specification(grammar)
        prompt_token_ids, select_token, stops = self._prepare_generation(
            prompt,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )
        _validate_max_new_tokens(max_new_tokens)
        create_grammar_context = self._create_grammar_context
        if create_grammar_context is None:
            raise ConstrainedGenerationError(
                "constrained generation is not configured for this text engine"
            )
        return self._stream_constrained_validated(
            grammar,
            expected_grammar_type=expected_grammar_type,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            select_token=select_token,
            stops=stops,
            create_grammar_context=create_grammar_context,
        )

    def _prepare_generation(
        self,
        prompt: str,
        *,
        stop_token_sequences: Sequence[Sequence[int]],
        selection: SelectionPolicy,
    ) -> tuple[tuple[int, ...], Callable[[LogitsT], int], tuple[tuple[int, ...], ...]]:
        policy = validate_selection_policy(selection)
        encoded_prompt = self._tokenizer.encode(prompt)
        try:
            prompt_token_ids = tuple(encoded_prompt)
        except TypeError as exc:
            raise TypeError("tokenizer encode must return a sequence of token IDs") from exc
        if not prompt_token_ids:
            raise ValueError("prompt must encode to at least one token")
        validate_token_ids(
            prompt_token_ids,
            self.vocab_size,
            label="prompt",
            allow_empty=False,
        )

        if isinstance(policy, GreedySelection):
            select_token = self._select_token
        else:
            if self._create_sampling_selector is None:
                raise SelectionError("temperature/top-p sampling is not configured")
            select_token = self._create_sampling_selector(policy)
            if not callable(select_token):
                raise TypeError("create_sampling_selector must return a callable")

        stops = normalize_stop_token_sequences(stop_token_sequences, self.vocab_size)
        return prompt_token_ids, select_token, stops

    def _new_metrics_session(self) -> TargetMetricsSession:
        session = self._create_metrics_session()
        if not isinstance(session, TargetMetricsSession):
            raise TypeError("create_metrics_session must return a TargetMetricsSession")
        return session

    def _stream_constrained_validated(
        self,
        grammar: GrammarSpecification,
        *,
        expected_grammar_type: GrammarType,
        prompt_token_ids: tuple[int, ...],
        max_new_tokens: int,
        select_token: Callable[[LogitsT], int],
        stops: tuple[tuple[int, ...], ...],
        create_grammar_context: Callable[
            [GrammarSpecification], GrammarGenerationContext[LogitsT, object]
        ],
    ) -> Generator[TextGenerationEvent, None, None]:
        grammar_context = create_grammar_context(grammar)
        target_steps = _iterate_constrained_target(
            self._backend,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            select_token=select_token,
            grammar_context=grammar_context,
            stop_token_sequences=stops,
            metrics_session=None,
            expected_grammar_type=expected_grammar_type,
            create_metrics_session=self._new_metrics_session,
        )
        yield from self._stream_validated(target_steps, stops)

    def _build_result(self, generation: GenerationResult) -> TextGenerationResult:
        output_token_ids = generation.visible_token_ids

        return TextGenerationResult(
            tokenizer_id=self._tokenizer.tokenizer_id,
            text=self._tokenizer.decode(output_token_ids),
            output_token_ids=output_token_ids,
            generation=generation,
        )

    def _stream_validated(
        self,
        target_steps: Generator[TargetGenerationStep, None, None],
        stops: tuple[tuple[int, ...], ...],
    ) -> Generator[TextGenerationEvent, None, None]:
        stop_buffer = _StopTokenBuffer(stops)
        decoder = _StableTextDecoder(self._tokenizer)
        terminal_seen = False
        completion_delivered = False
        failure: BaseException | None = None
        try:
            for step in target_steps:
                if step.result is None:
                    released = stop_buffer.push(step.token_id)
                    delta = decoder.append(released, final=False)
                else:
                    terminal_seen = True
                    released = stop_buffer.finish(step.token_id, step.result)
                    delta = decoder.append(released, final=True)

                if delta:
                    yield TextGenerationDelta(delta)

                if step.result is None:
                    continue

                result = self._build_result(step.result)
                if decoder.token_ids != result.output_token_ids or decoder.text != result.text:
                    raise StreamingInvariantError(
                        "incremental output does not match the completed generation result"
                    )
                completion_delivered = True
                yield TextGenerationComplete(result)
                return
            raise StreamingInvariantError("target generation ended without a terminal result")
        except BaseException as exc:
            failure = exc
            raise
        finally:
            cleanup_failure: Exception | None = None
            try:
                target_steps.close()
            except Exception as exc:
                cleanup_failure = exc

            if terminal_seen and not completion_delivered:
                try:
                    self._backend.reset()
                except Exception as exc:
                    cleanup_failure = cleanup_failure or exc

            if cleanup_failure is not None:
                if failure is not None and not isinstance(failure, GeneratorExit):
                    raise StreamingCleanupError(
                        f"text streaming failed: {failure}; cleanup also failed: "
                        f"{cleanup_failure}"
                    ) from failure
                raise StreamingCleanupError(
                    f"incomplete text stream cleanup failed: {cleanup_failure}"
                ) from cleanup_failure


def _validate_vocab_size(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} vocabulary size must be an integer")
    if value <= 0:
        raise ValueError(f"{label} vocabulary size must be greater than zero")
    return value


def _validate_max_new_tokens(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_new_tokens must be an integer")
    if value <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
