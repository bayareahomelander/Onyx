"""Lifecycle-owned production target-only text generation for CUDA."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Sequence
from threading import Lock
from typing import Any

from .backend import BackendError, BackendStateError
from .constrained_generation import (
    ConstrainedGenerationCleanupError,
    ConstrainedGenerationInvariantError,
    GrammarGenerationContext,
    GrammarSpecification,
    JsonSchemaGrammar,
    RegexGrammar,
    _TimedGrammarLogitMask,
)
from .metrics import (
    TargetMetricsSession,
    create_grammar_timing_session,
)
from .model_profile import DEFAULT_TARGET_PROFILE, QwenModelProfile
from .native_json import compile_native_json_schema
from .native_regex import compile_native_regex
from .production_tokenizer import QwenTokenizerAdapter, build_qwen_grammar_vocabulary
from .selection import GREEDY_SELECTION, SelectionPolicy, TemperatureTopPSelection
from .streaming import TextGenerationComplete, TextGenerationEvent
from .text_engine import TargetTextEngine, TextGenerationResult
from .torch_backend import (
    TorchCUDATargetBackend,
    load_torch_cuda_target,
    select_cuda_argmax,
)
from .torch_selection import create_cuda_sampler
from .torch_grammar_mask import create_cuda_grammar_logit_mask
from .torch_metrics import create_torch_metrics_session


class ProductionEngineError(BackendError):
    """Base error raised by the production target-only engine lifecycle."""


class ProductionEngineLoadError(ProductionEngineError):
    """Raised when loaded production components cannot be composed safely."""


class ProductionEngineCleanupError(ProductionEngineError):
    """Raised when stream and engine cleanup cannot both complete safely."""


class ProductionTargetTextStream(Iterator[TextGenerationEvent]):
    """Lifecycle-owned iterator for one production text generation."""

    def __init__(
        self,
        owner: ProductionTargetTextEngine,
        source: Generator[TextGenerationEvent, None, None],
    ) -> None:
        self._owner = owner
        self._source = source
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __iter__(self) -> ProductionTargetTextStream:
        return self

    def __next__(self) -> TextGenerationEvent:
        if self._closed:
            raise StopIteration
        try:
            event = next(self._source)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            try:
                self._finish()
            except Exception as cleanup_exc:
                raise ProductionEngineCleanupError(
                    f"production stream failed: {exc}; cleanup also failed: {cleanup_exc}"
                ) from exc
            raise

        if isinstance(event, TextGenerationComplete):
            self._finish()
        return event

    def close(self) -> None:
        """Cancel this stream and release its generation state; repeated calls are safe."""

        self._finish()

    def __enter__(self) -> ProductionTargetTextStream:
        if self._closed:
            raise BackendStateError("the production target text stream is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._source.close()
        finally:
            self._owner._release_stream(self)


class ProductionTargetTextEngine:
    """Own a pinned CUDA backend and its target-only text engine."""

    def __init__(
        self,
        backend: TorchCUDATargetBackend,
        *,
        create_sampling_selector: (
            Callable[[TemperatureTopPSelection], Callable[[Any], int]] | None
        ) = None,
        create_metrics_session: Callable[[], TargetMetricsSession] | None = None,
        create_grammar_context: (
            Callable[[GrammarSpecification], GrammarGenerationContext[Any, object]] | None
        ) = None,
    ) -> None:
        if create_metrics_session is None and isinstance(backend, TorchCUDATargetBackend):
            device_index = backend.device_index

            def create_backend_metrics_session() -> TargetMetricsSession:
                return create_torch_metrics_session(device_index=device_index)

            create_metrics_session = create_backend_metrics_session
        self._backend = backend
        self._closed = False
        self._model_id = backend.model_id
        self._tokenizer_id = backend.tokenizer.tokenizer_id
        self._vocab_size = backend.vocab_size
        self._active_stream: ProductionTargetTextStream | None = None
        self._create_grammar_context = create_grammar_context
        self._engine = TargetTextEngine(
            backend,
            backend.tokenizer,
            select_token=select_cuda_argmax,
            create_sampling_selector=create_sampling_selector,
            create_metrics_session=create_metrics_session,
            create_grammar_context=create_grammar_context,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def cache_length(self) -> int:
        if self._closed:
            return 0
        return self._backend.cache_length

    @property
    def is_closed(self) -> bool:
        return self._closed

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> TextGenerationResult:
        self._require_open()
        self._require_idle()
        return self._engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )

    def generate_constrained(
        self,
        prompt: str,
        *,
        grammar: GrammarSpecification,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> TextGenerationResult:
        self._require_open()
        self._require_idle()
        return self._engine.generate_constrained(
            prompt,
            grammar=grammar,
            max_new_tokens=max_new_tokens,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> ProductionTargetTextStream:
        """Create the single active lifecycle-owned incremental generation."""

        self._require_open()
        self._require_idle()
        source = self._engine.stream(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )
        stream = ProductionTargetTextStream(self, source)
        self._active_stream = stream
        return stream

    def stream_constrained(
        self,
        prompt: str,
        *,
        grammar: GrammarSpecification,
        max_new_tokens: int,
        stop_token_sequences: Sequence[Sequence[int]] = (),
        selection: SelectionPolicy = GREEDY_SELECTION,
    ) -> ProductionTargetTextStream:
        """Create the single active constrained incremental generation."""

        self._require_open()
        self._require_idle()
        source = self._engine.stream_constrained(
            prompt,
            grammar=grammar,
            max_new_tokens=max_new_tokens,
            stop_token_sequences=stop_token_sequences,
            selection=selection,
        )
        stream = ProductionTargetTextStream(self, source)
        self._active_stream = stream
        return stream

    def close(self) -> None:
        """Release the text engine, tokenizer, model, and cache; repeated calls are safe."""

        if self._closed:
            return
        self._closed = True
        stream_failure: Exception | None = None
        active_stream = self._active_stream
        if active_stream is not None:
            try:
                active_stream.close()
            except Exception as exc:
                stream_failure = exc
        self._active_stream = None
        self._engine = None
        self._create_grammar_context = None
        backend_failure: Exception | None = None
        try:
            self._backend.close()
        except Exception as exc:
            backend_failure = exc

        if stream_failure is not None and backend_failure is not None:
            raise ProductionEngineCleanupError(
                f"active stream cleanup failed: {stream_failure}; "
                f"backend cleanup also failed: {backend_failure}"
            ) from stream_failure
        if stream_failure is not None:
            raise stream_failure
        if backend_failure is not None:
            raise backend_failure

    def __enter__(self) -> ProductionTargetTextEngine:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise BackendStateError("the production target text engine is closed")

    def _require_idle(self) -> None:
        if self._active_stream is not None:
            raise BackendStateError("the production target text engine has an active stream")

    def _release_stream(self, stream: ProductionTargetTextStream) -> None:
        if self._active_stream is stream:
            self._active_stream = None


def load_production_target_engine(
    profile: QwenModelProfile = DEFAULT_TARGET_PROFILE,
    *,
    device_index: int = 0,
    local_files_only: bool = False,
) -> ProductionTargetTextEngine:
    """Load and compose the pinned target backend and tokenizer for target text generation."""

    backend = load_torch_cuda_target(
        profile,
        device_index=device_index,
        local_files_only=local_files_only,
    )
    try:
        create_grammar_context = _create_production_grammar_context_factory(
            backend,
            backend.tokenizer,
            device_index=getattr(backend, "device_index", device_index),
        )
        return ProductionTargetTextEngine(
            backend,
            create_sampling_selector=lambda policy: create_cuda_sampler(
                policy,
                device_index=device_index,
            ),
            create_metrics_session=lambda: create_torch_metrics_session(
                device_index=device_index,
            ),
            create_grammar_context=create_grammar_context,
        )
    except Exception as exc:
        try:
            backend.close()
        except Exception as cleanup_exc:
            raise ProductionEngineLoadError(
                f"production engine composition failed: {exc}; "
                f"cleanup also failed: {cleanup_exc}"
            ) from exc
        if isinstance(exc, ProductionEngineError):
            raise
        raise ProductionEngineLoadError(
            f"production engine composition failed: {exc}"
        ) from exc


def _create_production_grammar_context_factory(
    backend: TorchCUDATargetBackend,
    tokenizer: QwenTokenizerAdapter,
    *,
    device_index: int,
) -> Callable[[GrammarSpecification], GrammarGenerationContext[Any, object]]:
    """Create one lazy vocabulary/mask cache that compiles a fresh constraint per call."""

    lock = Lock()
    support: tuple[tuple[bytes, ...], Any] | None = None

    def create_context(
        grammar: GrammarSpecification,
    ) -> GrammarGenerationContext[Any, object]:
        nonlocal support
        with lock:
            if support is None:
                vocabulary = _validate_production_vocabulary(
                    build_qwen_grammar_vocabulary(tokenizer)
                )
                vocabulary_size = len(vocabulary)
                tokenizer_size = _validate_production_vocab_size(
                    tokenizer.vocab_size,
                    label="tokenizer",
                )
                backend_size = _validate_production_vocab_size(
                    backend.vocab_size,
                    label="backend",
                )
                if vocabulary_size != tokenizer_size or vocabulary_size != backend_size:
                    raise ConstrainedGenerationInvariantError(
                        "production grammar vocabulary size must match tokenizer and backend "
                        f"exactly: vocabulary={vocabulary_size}, tokenizer={tokenizer_size}, "
                        f"backend={backend_size}"
                    )
                eos_token_id = _validate_production_token_id(
                    tokenizer.eos_token_id,
                    vocab_size=vocabulary_size,
                    label="EOS",
                )
                if vocabulary[eos_token_id] != b"":
                    raise ConstrainedGenerationInvariantError(
                        "production EOS token must map to the native empty-byte transition"
                    )
                logit_mask = create_cuda_grammar_logit_mask(
                    vocabulary_size,
                    device_index=device_index,
                )
                mask_size = _validate_production_vocab_size(
                    logit_mask.vocab_size,
                    label="grammar mask",
                )
                if mask_size != vocabulary_size:
                    raise ConstrainedGenerationInvariantError(
                        "production grammar mask vocabulary size does not match the exact "
                        f"grammar vocabulary: mask={mask_size}, vocabulary={vocabulary_size}"
                    )
                mask_device_index = getattr(logit_mask, "device_index", device_index)
                if (
                    isinstance(mask_device_index, bool)
                    or not isinstance(mask_device_index, int)
                    or mask_device_index != device_index
                ):
                    raise ConstrainedGenerationInvariantError(
                        "production grammar mask must use the backend's exact CUDA device: "
                        f"backend={device_index}, mask={mask_device_index!r}"
                    )
                if not isinstance(logit_mask, _TimedGrammarLogitMask):
                    raise ConstrainedGenerationInvariantError(
                        "production grammar mask must provide completed "
                        "transfer/application timing"
                    )
                support = vocabulary, logit_mask

            vocabulary, logit_mask = support
            eos_token_id = _validate_production_token_id(
                tokenizer.eos_token_id,
                vocab_size=len(vocabulary),
                label="EOS",
            )
            if vocabulary[eos_token_id] != b"":
                raise ConstrainedGenerationInvariantError(
                    "production EOS token must map to the native empty-byte transition"
                )
            if isinstance(grammar, RegexGrammar):
                compile_constraint = compile_native_regex
                source = grammar.pattern
            elif isinstance(grammar, JsonSchemaGrammar):
                compile_constraint = compile_native_json_schema
                source = grammar.schema
            else:
                raise TypeError("grammar must be a RegexGrammar or JsonSchemaGrammar")

            timing_session = create_grammar_timing_session()
            constraint = None
            try:
                with timing_session.compilation():
                    constraint = compile_constraint(vocabulary, source)
                return GrammarGenerationContext(
                    constraint=constraint,
                    logit_mask=logit_mask,
                    eos_token_id=eos_token_id,
                    timing_session=timing_session,
                )
            except BaseException as failure:
                cleanup_failures: list[tuple[str, Exception]] = []
                if constraint is not None:
                    try:
                        constraint.reset()
                    except Exception as cleanup_failure:
                        cleanup_failures.append(
                            ("grammar constraint reset", cleanup_failure)
                        )
                try:
                    timing_session.abort()
                except Exception as cleanup_failure:
                    cleanup_failures.append(("grammar timing abort", cleanup_failure))
                if cleanup_failures:
                    raise ConstrainedGenerationCleanupError(
                        failure,
                        cleanup_failures,
                    ) from failure
                raise

    return create_context


def _validate_production_vocab_size(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConstrainedGenerationInvariantError(
            f"production {label} vocabulary size must be an integer"
        )
    if value <= 0:
        raise ConstrainedGenerationInvariantError(
            f"production {label} vocabulary size must be greater than zero"
        )
    return value


def _validate_production_vocabulary(value: tuple[bytes, ...]) -> tuple[bytes, ...]:
    if not isinstance(value, tuple):
        raise ConstrainedGenerationInvariantError(
            "production grammar vocabulary must be an immutable tuple"
        )
    if not value:
        raise ConstrainedGenerationInvariantError(
            "production grammar vocabulary cannot be empty"
        )
    for token_id, token_bytes in enumerate(value):
        if not isinstance(token_bytes, bytes):
            raise ConstrainedGenerationInvariantError(
                f"production grammar vocabulary entry {token_id} must be bytes"
            )
    return value


def _validate_production_token_id(
    value: int,
    *,
    vocab_size: int,
    label: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConstrainedGenerationInvariantError(
            f"production {label} token ID must be an integer"
        )
    if value < 0 or value >= vocab_size:
        raise ConstrainedGenerationInvariantError(
            f"production {label} token ID {value} is outside vocabulary range "
            f"[0, {vocab_size})"
        )
    return value
