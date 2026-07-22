"""Deterministic test implementations of Onyx framework-neutral contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import count

from .backend import BackendStateError, ModelStep
from .cache import CacheCheckpointStateError
from .grammar import (
    GrammarCompilationError,
    GrammarStateError,
    GrammarType,
    _normalize_grammar_vocabulary,
    _validate_grammar_token_id,
    _validate_json_schema,
    _validate_regex_pattern,
)
from .metrics import (
    GrammarTimingSession,
    TargetGenerationMetrics,
    TargetMetricsSession,
    create_grammar_timing_session,
    create_target_metrics_session,
)
from .tokenizer import UnknownTextTokenError
from .verification import BatchedTargetVerificationResult


class ScriptExhaustedError(BackendStateError):
    """Raised when a fake backend has no scripted logits left to return."""


@dataclass(frozen=True, slots=True)
class FakeCacheCheckpoint:
    """Opaque cache checkpoint owned by one fake backend sequence epoch."""

    owner_id: int
    epoch: int
    allocation_id: int
    cache_length: int

    def __post_init__(self) -> None:
        _validate_checkpoint_integer(self.owner_id, label="owner_id", minimum=1)
        _validate_checkpoint_integer(self.epoch, label="epoch", minimum=1)
        _validate_checkpoint_integer(self.allocation_id, label="allocation_id", minimum=1)
        _validate_checkpoint_integer(self.cache_length, label="cache_length", minimum=0)


@dataclass(frozen=True, slots=True)
class _FakeCacheSnapshot:
    checkpoint: FakeCacheCheckpoint
    cached_token_ids: tuple[int, ...]
    next_row: int


_FAKE_CACHE_OWNER_IDS = count(1)


class FakeAutoregressiveBackend:
    """A deterministic backend driven by a fixed sequence of logit rows.

    The first row is returned by ``prefill``. Each later row is returned by one ``decode`` call.
    Calling ``prefill`` again resets both the script and logical cache to start a new sequence.
    """

    def __init__(
        self,
        scripted_logits: Iterable[Sequence[float]],
        *,
        model_id: str = "fake-target",
    ) -> None:
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a string")
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")

        rows = []
        for row_number, row in enumerate(scripted_logits, start=1):
            try:
                normalized_row = tuple(float(value) for value in row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"scripted logit row {row_number} must be numeric") from exc
            if not normalized_row:
                raise ValueError(f"scripted logit row {row_number} cannot be empty")
            rows.append(normalized_row)

        if not rows:
            raise ValueError("scripted_logits must contain at least one row")

        vocab_size = len(rows[0])
        if any(len(row) != vocab_size for row in rows[1:]):
            raise ValueError("all scripted logit rows must have the same vocabulary size")

        self._model_id = model_id
        self._scripted_logits = tuple(rows)
        self._vocab_size = vocab_size
        self._owner_id = next(_FAKE_CACHE_OWNER_IDS)
        self._epoch = 0
        self.reset()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def cache_length(self) -> int:
        return self._cache_length

    @property
    def cache_mode(self) -> str:
        return "fake"

    @property
    def cached_token_ids(self) -> tuple[int, ...]:
        """Return the exact logical token prefix retained by this deterministic fake."""

        return self._cached_token_ids

    @property
    def active_checkpoint_count(self) -> int:
        """Return the number of live checkpoint handles in the current sequence epoch."""

        return len(self._cache_checkpoints)

    def prefill(self, prompt_token_ids: Sequence[int], /) -> ModelStep[tuple[float, ...]]:
        prompt = tuple(prompt_token_ids)
        if not prompt:
            raise ValueError("prompt_token_ids cannot be empty")
        for position, token_id in enumerate(prompt):
            self._validate_token(token_id, label=f"prompt token at position {position}")

        self.reset()
        logits = self._take_next_logits()
        self._cached_token_ids = prompt
        self._cache_length = len(prompt)
        return ModelStep(logits=logits, cache_length=self._cache_length)

    def decode(self, token_id: int, /) -> ModelStep[tuple[float, ...]]:
        if self._cache_length == 0:
            raise BackendStateError("prefill must be called before decode")
        self._validate_token(token_id, label="decode token")

        logits = self._take_next_logits()
        self._cached_token_ids += (token_id,)
        self._cache_length += 1
        return ModelStep(logits=logits, cache_length=self._cache_length)

    def verify_proposal(
        self,
        current_token_id: int,
        proposal_token_ids: Sequence[int],
        /,
    ) -> BatchedTargetVerificationResult[tuple[float, ...]]:
        """Consume a current token and nonempty proposal as one deterministic batch."""

        self._validate_token(current_token_id, label="current token")
        try:
            proposal = tuple(proposal_token_ids)
        except TypeError as exc:
            raise TypeError("proposal_token_ids must be a sequence of integers") from exc
        if not proposal:
            raise ValueError("proposal_token_ids cannot be empty")
        for position, token_id in enumerate(proposal):
            self._validate_token(token_id, label=f"proposal token at position {position}")

        self._validate_verification_state()
        input_suffix = (current_token_id, *proposal)
        row_count = len(input_suffix)
        logit_rows = self._stage_verification_rows(row_count)
        self._validate_staged_verification_rows(logit_rows, expected_count=row_count)

        cached_token_ids = self._cached_token_ids + input_suffix
        cache_length = self._cache_length + row_count
        next_row = self._next_row + row_count
        result = self._build_verification_result(
            logit_rows=logit_rows,
            cache_length=cache_length,
        )

        self._cached_token_ids = cached_token_ids
        self._cache_length = cache_length
        self._next_row = next_row
        return result

    def create_cache_checkpoint(self) -> FakeCacheCheckpoint:
        """Record the exact current fake-cache prefix and scripted execution position."""

        self._validate_active_cache_state()
        allocation_id = self._next_checkpoint_id
        if (
            isinstance(allocation_id, bool)
            or not isinstance(allocation_id, int)
            or allocation_id < 1
            or allocation_id in self._cache_checkpoints
        ):
            raise CacheCheckpointStateError("fake cache checkpoint allocation state is invalid")

        checkpoint = FakeCacheCheckpoint(
            owner_id=self._owner_id,
            epoch=self._epoch,
            allocation_id=allocation_id,
            cache_length=self._cache_length,
        )
        snapshot = _FakeCacheSnapshot(
            checkpoint=checkpoint,
            cached_token_ids=self._cached_token_ids,
            next_row=self._next_row,
        )
        self._next_checkpoint_id = allocation_id + 1
        self._cache_checkpoints[allocation_id] = snapshot
        return checkpoint

    def rollback_cache(self, checkpoint: FakeCacheCheckpoint, /) -> None:
        """Restore the exact prefix and script position recorded by ``checkpoint``."""

        self._validate_checkpoint_type_and_owner(checkpoint)
        if checkpoint.epoch != self._epoch:
            raise CacheCheckpointStateError("fake cache checkpoint belongs to a stale sequence")

        snapshot = self._cache_checkpoints.get(checkpoint.allocation_id)
        if snapshot is None:
            raise CacheCheckpointStateError(
                "fake cache checkpoint is unknown, released, or represents a discarded suffix"
            )
        if checkpoint != snapshot.checkpoint:
            raise CacheCheckpointStateError(
                "fake cache checkpoint metadata does not match its canonical allocation"
            )

        self._validate_active_cache_state()
        self._validate_cache_snapshot(snapshot)
        target_length = snapshot.checkpoint.cache_length
        if target_length > self._cache_length:
            raise CacheCheckpointStateError(
                "fake cache checkpoint is ahead of the current cache position"
            )
        if self._cached_token_ids[:target_length] != snapshot.cached_token_ids:
            raise CacheCheckpointStateError(
                "fake cache checkpoint prefix no longer matches the active sequence"
            )
        if self._next_row - snapshot.next_row != self._cache_length - target_length:
            raise CacheCheckpointStateError(
                "fake cache checkpoint script position is inconsistent with the active sequence"
            )

        discarded_allocations = []
        for allocation_id, active_snapshot in self._cache_checkpoints.items():
            self._validate_cache_snapshot(active_snapshot)
            active_length = active_snapshot.checkpoint.cache_length
            if active_length <= target_length:
                if snapshot.cached_token_ids[:active_length] != active_snapshot.cached_token_ids:
                    raise CacheCheckpointStateError(
                        "fake cache checkpoint registry contains inconsistent retained prefixes"
                    )
                if snapshot.next_row - active_snapshot.next_row != target_length - active_length:
                    raise CacheCheckpointStateError(
                        "fake cache checkpoint registry contains inconsistent script positions"
                    )
            else:
                discarded_allocations.append(allocation_id)

        self._cached_token_ids = snapshot.cached_token_ids
        self._cache_length = target_length
        self._next_row = snapshot.next_row
        for allocation_id in discarded_allocations:
            del self._cache_checkpoints[allocation_id]

    def release_cache_checkpoint(self, checkpoint: FakeCacheCheckpoint, /) -> None:
        """End one checkpoint lifetime without changing the fake cache."""

        self._validate_checkpoint_type_and_owner(checkpoint)
        if checkpoint.epoch != self._epoch:
            return

        snapshot = self._cache_checkpoints.get(checkpoint.allocation_id)
        if snapshot is None:
            return
        if checkpoint != snapshot.checkpoint:
            raise CacheCheckpointStateError(
                "fake cache checkpoint metadata does not match its canonical allocation"
            )
        del self._cache_checkpoints[checkpoint.allocation_id]

    def reset(self) -> None:
        self._next_row = 0
        self._cache_length = 0
        self._cached_token_ids: tuple[int, ...] = ()
        self._cache_checkpoints: dict[int, _FakeCacheSnapshot] = {}
        self._next_checkpoint_id = 1
        self._epoch += 1

    def _take_next_logits(self) -> tuple[float, ...]:
        if self._next_row >= len(self._scripted_logits):
            raise ScriptExhaustedError("the fake backend has no scripted logits remaining")

        logits = self._scripted_logits[self._next_row]
        self._next_row += 1
        return logits

    def _stage_verification_rows(
        self,
        row_count: int,
    ) -> tuple[tuple[float, ...], ...]:
        logit_rows = self._scripted_logits[self._next_row : self._next_row + row_count]
        if len(logit_rows) != row_count:
            raise ScriptExhaustedError(
                "the fake backend has insufficient scripted logits for the verification batch"
            )
        return logit_rows

    def _build_verification_result(
        self,
        *,
        logit_rows: tuple[tuple[float, ...], ...],
        cache_length: int,
    ) -> BatchedTargetVerificationResult[tuple[float, ...]]:
        return BatchedTargetVerificationResult(
            logit_rows=logit_rows,
            cache_length=cache_length,
        )

    def _validate_staged_verification_rows(
        self,
        logit_rows: tuple[tuple[float, ...], ...],
        *,
        expected_count: int,
    ) -> None:
        if type(logit_rows) is not tuple:
            raise TypeError("staged verification rows must be a tuple")
        if len(logit_rows) != expected_count:
            raise ValueError(
                f"staged verification returned {len(logit_rows)} rows; "
                f"expected {expected_count}"
            )
        for position, row in enumerate(logit_rows):
            if type(row) is not tuple:
                raise TypeError(
                    f"staged verification row at position {position} must be a tuple"
                )
            if len(row) != self._vocab_size:
                raise ValueError(
                    f"staged verification row at position {position} has vocabulary size "
                    f"{len(row)}; expected {self._vocab_size}"
                )

    def _validate_token(self, token_id: int, *, label: str) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{label} must be an integer")
        if token_id < 0 or token_id >= self._vocab_size:
            raise ValueError(
                f"{label} {token_id} is outside vocabulary range [0, {self._vocab_size})"
            )

    def _validate_verification_state(self) -> None:
        if self._cache_length == 0:
            raise BackendStateError("prefill must be called before verifying a proposal")
        if (
            isinstance(self._cache_length, bool)
            or not isinstance(self._cache_length, int)
            or self._cache_length < 1
            or type(self._cached_token_ids) is not tuple
            or len(self._cached_token_ids) != self._cache_length
        ):
            raise BackendStateError(
                "fake cache length does not match its exact cached token prefix"
            )
        if (
            isinstance(self._next_row, bool)
            or not isinstance(self._next_row, int)
            or self._next_row < 1
            or self._next_row > len(self._scripted_logits)
        ):
            raise BackendStateError("fake cache scripted-logits position is invalid")

    def _validate_active_cache_state(self) -> None:
        if self._cache_length == 0:
            raise CacheCheckpointStateError(
                "prefill must be called before creating or rolling back a cache checkpoint"
            )
        if len(self._cached_token_ids) != self._cache_length:
            raise CacheCheckpointStateError(
                "fake cache length does not match its exact cached token prefix"
            )
        if self._next_row < 1 or self._next_row > len(self._scripted_logits):
            raise CacheCheckpointStateError("fake cache scripted-logits position is invalid")

    def _validate_checkpoint_type_and_owner(
        self,
        checkpoint: FakeCacheCheckpoint,
    ) -> None:
        if not isinstance(checkpoint, FakeCacheCheckpoint):
            raise TypeError("checkpoint must be a FakeCacheCheckpoint")
        for value, minimum in (
            (checkpoint.owner_id, 1),
            (checkpoint.epoch, 1),
            (checkpoint.allocation_id, 1),
            (checkpoint.cache_length, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise CacheCheckpointStateError("fake cache checkpoint metadata is invalid")
        if checkpoint.owner_id != self._owner_id:
            raise CacheCheckpointStateError("fake cache checkpoint belongs to another backend")

    def _validate_cache_snapshot(self, snapshot: _FakeCacheSnapshot) -> None:
        checkpoint = snapshot.checkpoint
        if (
            checkpoint.owner_id != self._owner_id
            or checkpoint.epoch != self._epoch
            or len(snapshot.cached_token_ids) != checkpoint.cache_length
            or snapshot.next_row < 1
            or snapshot.next_row > len(self._scripted_logits)
        ):
            raise CacheCheckpointStateError("fake cache checkpoint registry is inconsistent")


def _validate_checkpoint_integer(value: int, *, label: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{label} cannot be negative")
        raise ValueError(f"{label} must be greater than zero")


class FakeCharacterTokenizer:
    """A deterministic tokenizer that assigns one token ID per Unicode character."""

    def __init__(
        self,
        vocabulary: Sequence[str],
        *,
        tokenizer_id: str = "fake-character",
    ) -> None:
        if not isinstance(tokenizer_id, str):
            raise TypeError("tokenizer_id must be a string")
        if not tokenizer_id.strip():
            raise ValueError("tokenizer_id cannot be empty")

        tokens = tuple(vocabulary)
        if not tokens:
            raise ValueError("vocabulary must contain at least one character")
        for token_id, token in enumerate(tokens):
            if not isinstance(token, str):
                raise TypeError(f"vocabulary entry {token_id} must be a string")
            if len(token) != 1:
                raise ValueError(
                    f"vocabulary entry {token_id} must contain exactly one character"
                )
        if len(set(tokens)) != len(tokens):
            raise ValueError("vocabulary characters must be unique")

        self._tokenizer_id = tokenizer_id
        self._tokens = tokens
        self._token_to_id = {token: token_id for token_id, token in enumerate(tokens)}

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    def encode(self, text: str, /) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        token_ids = []
        for position, character in enumerate(text):
            try:
                token_ids.append(self._token_to_id[character])
            except KeyError as exc:
                raise UnknownTextTokenError(
                    f"character {character!r} at position {position} is not in the vocabulary"
                ) from exc
        return tuple(token_ids)

    def decode(self, token_ids: Sequence[int], /) -> str:
        characters = []
        for position, token_id in enumerate(token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"token ID at position {position} must be an integer")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"token ID {token_id} at position {position} is outside vocabulary range "
                    f"[0, {self.vocab_size})"
                )
            characters.append(self._tokens[token_id])
        return "".join(characters)


class DeterministicMetricsClock:
    """Advance one second on every clock read for exact metrics assertions."""

    def __init__(self) -> None:
        self._value = -1.0

    def __call__(self) -> float:
        self._value += 1.0
        return self._value


def create_deterministic_metrics_session(
    *,
    cache_mode: str = "fake",
) -> TargetMetricsSession:
    """Create a fresh deterministic metrics session for one fake generation."""

    return create_target_metrics_session(
        cache_mode=cache_mode,
        clock=DeterministicMetricsClock(),
    )


def create_deterministic_grammar_timing_session() -> GrammarTimingSession:
    """Create a fresh deterministic grammar timing session for one fake request."""

    return create_grammar_timing_session(clock=DeterministicMetricsClock())


def deterministic_target_metrics(
    generated_tokens: int,
    *,
    cache_mode: str = "fake",
) -> TargetGenerationMetrics:
    """Return the metric record produced by ``DeterministicMetricsClock``."""

    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
        raise TypeError("generated_tokens must be an integer")
    if generated_tokens <= 0:
        raise ValueError("generated_tokens must be greater than zero")
    generation_time = float(generated_tokens + 1)
    return TargetGenerationMetrics(
        ttft=1.0,
        generation_time=generation_time,
        tokens_per_second=generated_tokens / generation_time,
        cache_mode=cache_mode,
    )


@dataclass(frozen=True, slots=True)
class FakeGrammarProgram:
    """Immutable logical-state graph used by the deterministic grammar fake."""

    initial_state: str
    transitions: tuple[tuple[str, int, str], ...]
    valid_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    match_states: frozenset[str] = frozenset()
    dead_states: frozenset[str] = frozenset()
    dead_transitions: tuple[tuple[str, int, str], ...] = ()

    def __post_init__(self) -> None:
        initial_state = _validate_fake_state_name(self.initial_state)
        transitions = tuple(self.transitions)
        dead_transitions = tuple(self.dead_transitions)
        valid_entries = tuple(self.valid_token_ids)
        match_states = frozenset(
            _validate_fake_state_name(state) for state in self.match_states
        )
        dead_states = frozenset(
            _validate_fake_state_name(state) for state in self.dead_states
        )
        if match_states & dead_states:
            raise ValueError("fake grammar states cannot be both matching and dead")

        transition_keys = set()
        normalized_transitions = []
        for position, transition in enumerate(transitions):
            try:
                source, token_id, target = transition
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fake grammar transition {position} must contain source, token ID, and target"
                ) from exc
            source = _validate_fake_state_name(source)
            target = _validate_fake_state_name(target)
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"fake grammar transition {position} token ID must be an integer")
            if token_id < 0:
                raise ValueError(f"fake grammar transition {position} token ID cannot be negative")
            key = (source, token_id)
            if key in transition_keys:
                raise ValueError(
                    f"fake grammar transition from {source!r} on token {token_id} is duplicated"
                )
            transition_keys.add(key)
            normalized_transitions.append((source, token_id, target))

        dead_transition_keys = set()
        normalized_dead_transitions = []
        for position, transition in enumerate(dead_transitions):
            try:
                source, token_id, target = transition
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fake grammar dead transition {position} must contain source, token ID, "
                    "and target"
                ) from exc
            source = _validate_fake_state_name(source)
            target = _validate_fake_state_name(target)
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(
                    f"fake grammar dead transition {position} token ID must be an integer"
                )
            if token_id < 0:
                raise ValueError(
                    f"fake grammar dead transition {position} token ID cannot be negative"
                )
            key = (source, token_id)
            if key in transition_keys or key in dead_transition_keys:
                raise ValueError(
                    f"fake grammar transition from {source!r} on token {token_id} is duplicated"
                )
            dead_transition_keys.add(key)
            normalized_dead_transitions.append((source, token_id, target))

        valid_states = set()
        normalized_valid_entries = []
        valid_by_state = {}
        for position, entry in enumerate(valid_entries):
            try:
                state, token_ids = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fake grammar valid-token entry {position} must contain state and token IDs"
                ) from exc
            state = _validate_fake_state_name(state)
            if state in valid_states:
                raise ValueError(f"fake grammar valid tokens for state {state!r} are duplicated")
            valid_states.add(state)
            normalized_ids = []
            for token_id in tuple(token_ids):
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    raise TypeError(
                        f"fake grammar valid token for state {state!r} must be an integer"
                    )
                if token_id < 0:
                    raise ValueError(
                        f"fake grammar valid token for state {state!r} cannot be negative"
                    )
                normalized_ids.append(token_id)
            if len(set(normalized_ids)) != len(normalized_ids):
                raise ValueError(f"fake grammar valid tokens for state {state!r} are duplicated")
            normalized = tuple(sorted(normalized_ids))
            valid_by_state[state] = normalized
            normalized_valid_entries.append((state, normalized))

        for source, token_id, _target in normalized_transitions:
            if token_id not in valid_by_state.get(source, ()):
                raise ValueError(
                    f"fake grammar transition token {token_id} is not valid for state {source!r}"
                )
        for source, token_id, target in normalized_dead_transitions:
            if token_id in valid_by_state.get(source, ()):
                raise ValueError(
                    f"fake grammar dead transition token {token_id} is advertised as valid for "
                    f"state {source!r}"
                )
            if target not in dead_states:
                raise ValueError(
                    f"fake grammar dead transition from {source!r} must target a dead state"
                )
        for state, token_ids in normalized_valid_entries:
            for token_id in token_ids:
                if (state, token_id) not in transition_keys:
                    raise ValueError(
                        f"fake grammar valid token {token_id} has no transition from state {state!r}"
                    )
        for state in dead_states:
            if valid_by_state.get(state):
                raise ValueError(f"dead fake grammar state {state!r} cannot have valid tokens")
        all_states = {initial_state, *match_states, *dead_states, *valid_by_state}
        for source, _token_id, target in normalized_transitions:
            all_states.add(source)
            all_states.add(target)
        for source, _token_id, target in normalized_dead_transitions:
            all_states.add(source)
            all_states.add(target)
        for state in all_states:
            if (
                not valid_by_state.get(state)
                and state not in match_states
                and state not in dead_states
            ):
                raise ValueError(
                    f"fake grammar state {state!r} without valid tokens must be matching or dead"
                )

        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "transitions", tuple(normalized_transitions))
        object.__setattr__(self, "valid_token_ids", tuple(normalized_valid_entries))
        object.__setattr__(self, "match_states", match_states)
        object.__setattr__(self, "dead_states", dead_states)
        object.__setattr__(self, "dead_transitions", tuple(normalized_dead_transitions))


@dataclass(frozen=True, slots=True)
class FakeGrammarStateHandle:
    """Opaque state owned by one fake constraint and one reset epoch."""

    owner_id: int
    epoch: int
    value: int


_FAKE_GRAMMAR_OWNER_IDS = count(1)


class FakeGrammarConstraint:
    """Deterministic explicit-state grammar constraint backed by a scripted graph."""

    def __init__(
        self,
        vocabulary: Sequence[bytes],
        *,
        grammar_type: GrammarType,
        program: FakeGrammarProgram,
    ) -> None:
        self._vocabulary = _normalize_grammar_vocabulary(vocabulary)
        if grammar_type not in {"regex", "json_schema"}:
            raise ValueError("grammar_type must be 'regex' or 'json_schema'")
        if not isinstance(program, FakeGrammarProgram):
            raise TypeError("program must be a FakeGrammarProgram")
        for source, token_id, _target in program.transitions:
            try:
                _validate_grammar_token_id(token_id, len(self._vocabulary))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fake grammar transition from {source!r} has invalid token ID {token_id}"
                ) from exc
        for source, token_id, _target in program.dead_transitions:
            try:
                _validate_grammar_token_id(token_id, len(self._vocabulary))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fake grammar dead transition from {source!r} has invalid token ID "
                    f"{token_id}"
                ) from exc

        self._grammar_type = grammar_type
        self._program = program
        self._transitions = {
            (source, token_id): target for source, token_id, target in program.transitions
        }
        self._dead_transitions = {
            (source, token_id): target for source, token_id, target in program.dead_transitions
        }
        self._valid_token_ids = dict(program.valid_token_ids)
        self._owner_id = next(_FAKE_GRAMMAR_OWNER_IDS)
        self._epoch = 0
        self.reset()

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def grammar_type(self) -> GrammarType:
        return self._grammar_type

    def init_state(self) -> FakeGrammarStateHandle:
        return self._insert_state(self._program.initial_state)

    def advance_state(
        self,
        state: FakeGrammarStateHandle,
        token_id: int,
        /,
    ) -> FakeGrammarStateHandle:
        state_name = self._state_name(state)
        _validate_grammar_token_id(token_id, self.vocab_size)
        if token_id in self._valid_token_ids.get(state_name, ()):
            target = self._transitions[(state_name, token_id)]
            return self._insert_state(target)
        if self._vocabulary[token_id] == b"":
            return self._insert_state(state_name)
        target = self._dead_transitions.get((state_name, token_id))
        if target is None:
            raise GrammarStateError(
                f"token ID {token_id} is not valid from fake grammar state {state_name!r}"
            )
        return self._insert_state(target)

    def get_valid_token_ids(self, state: FakeGrammarStateHandle, /) -> tuple[int, ...]:
        return self._valid_token_ids.get(self._state_name(state), ())

    def is_match_state(self, state: FakeGrammarStateHandle, /) -> bool:
        return self._state_name(state) in self._program.match_states

    def is_dead_state(self, state: FakeGrammarStateHandle, /) -> bool:
        return self._state_name(state) in self._program.dead_states

    def release_state(self, state: FakeGrammarStateHandle, /) -> None:
        self._validate_handle_owner(state)
        if state.epoch == self._epoch:
            self._states.pop(state.value, None)

    def release_states(self, states: Sequence[FakeGrammarStateHandle], /) -> None:
        try:
            normalized = tuple(states)
        except TypeError as exc:
            raise TypeError("fake grammar states must be a sequence") from exc
        for state in normalized:
            self._validate_handle_owner(state)
        for state in normalized:
            if state.epoch == self._epoch:
                self._states.pop(state.value, None)

    def reset(self) -> None:
        self._epoch += 1
        self._next_state_id = 1
        self._states: dict[int, str] = {}

    @property
    def active_state_count(self) -> int:
        return len(self._states)

    def _insert_state(self, state_name: str) -> FakeGrammarStateHandle:
        value = self._next_state_id
        self._next_state_id += 1
        self._states[value] = state_name
        return FakeGrammarStateHandle(
            owner_id=self._owner_id,
            epoch=self._epoch,
            value=value,
        )

    def _state_name(self, state: FakeGrammarStateHandle) -> str:
        self._validate_handle_owner(state)
        if state.epoch != self._epoch or state.value not in self._states:
            raise GrammarStateError("fake grammar state handle is unknown or has been released")
        return self._states[state.value]

    def _validate_handle_owner(self, state: FakeGrammarStateHandle) -> None:
        if not isinstance(state, FakeGrammarStateHandle):
            raise TypeError("fake grammar state must be a FakeGrammarStateHandle")
        if state.owner_id != self._owner_id:
            raise GrammarStateError("fake grammar state belongs to another constraint")


class FakeGrammarCompiler:
    """Compile registered regex/JSON sources into fresh scripted constraints."""

    def __init__(
        self,
        *,
        regex_programs: Mapping[str, FakeGrammarProgram] | None = None,
        json_schema_programs: Mapping[str, FakeGrammarProgram] | None = None,
    ) -> None:
        self._regex_programs = self._normalize_programs(
            regex_programs,
            source_label="regex pattern",
            validate_source=_validate_regex_pattern,
        )
        self._json_schema_programs = self._normalize_programs(
            json_schema_programs,
            source_label="JSON Schema",
            validate_source=_validate_json_schema,
        )

    def compile_regex(
        self,
        vocabulary: Sequence[bytes],
        pattern: str,
        /,
    ) -> FakeGrammarConstraint:
        pattern = _validate_regex_pattern(pattern)
        return self._compile(
            vocabulary,
            source=pattern,
            programs=self._regex_programs,
            grammar_type="regex",
        )

    def compile_json_schema(
        self,
        vocabulary: Sequence[bytes],
        schema: str,
        /,
    ) -> FakeGrammarConstraint:
        schema = _validate_json_schema(schema)
        return self._compile(
            vocabulary,
            source=schema,
            programs=self._json_schema_programs,
            grammar_type="json_schema",
        )

    @staticmethod
    def _normalize_programs(
        programs: Mapping[str, FakeGrammarProgram] | None,
        *,
        source_label: str,
        validate_source,
    ) -> dict[str, FakeGrammarProgram]:
        if programs is None:
            return {}
        if not isinstance(programs, Mapping):
            raise TypeError(f"fake {source_label} programs must be a mapping")
        normalized = {}
        for source, program in programs.items():
            validate_source(source)
            if not isinstance(program, FakeGrammarProgram):
                raise TypeError(f"fake {source_label} program must be a FakeGrammarProgram")
            normalized[source] = program
        return normalized

    @staticmethod
    def _compile(
        vocabulary: Sequence[bytes],
        *,
        source: str,
        programs: Mapping[str, FakeGrammarProgram],
        grammar_type: GrammarType,
    ) -> FakeGrammarConstraint:
        try:
            program = programs[source]
        except KeyError as exc:
            raise GrammarCompilationError(
                f"no fake {grammar_type} program is registered for the requested source"
            ) from exc
        return FakeGrammarConstraint(
            vocabulary,
            grammar_type=grammar_type,
            program=program,
        )


def _validate_fake_state_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("fake grammar state names must be strings")
    if not value:
        raise ValueError("fake grammar state names cannot be empty")
    return value
