"""Cached token generation."""

import math
import time
from collections.abc import Iterator
from typing import Literal, NamedTuple

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from onyx_cuda.cache import CacheState
from onyx_cuda.masking import TokenMask, apply_grammar_mask, grammar_argmax
from onyx_cuda.prefill import prefill
from onyx_cuda.vocabulary import TokenByteVocabulary


class GenerationResult(NamedTuple):
    token_ids: list[int]
    past_key_values: Cache | None
    finish_reason: Literal["eos", "stop", "length"]
    timings: "GenerationTimings | None" = None
    speculation: "SpeculationStats | None" = None


class AcceptedTokenEvent(NamedTuple):
    token_id: int


class GenerationFinishedEvent(NamedTuple):
    result: GenerationResult


class _TextStop(Exception):
    """Ask a suspended token generator to finalize at a decoded-text stop.

    The prefix includes the token completing the stop, which can also contain
    visible text. Throwing this signal at a token yield skips buffered tokens
    and further inference, while letting the producer finalize its metadata.
    """

    def __init__(self, token_ids: list[int]):
        super().__init__("Decoded text reached a stop")
        self.token_ids = list(token_ids)


def _take_ready_tokens(pending: list[int], retain: int = 0) -> list[int]:
    count = max(len(pending) - retain, 0)
    ready = pending[:count]
    del pending[:count]
    return ready


class GenerationTimings(NamedTuple):
    """End-to-end fields are always set; stage and grammar timings need measure."""

    time_to_first_token_seconds: float
    decode_tokens_per_second: float | None
    total_seconds: float
    grammar_compile_seconds: float | None = None
    valid_token_enumeration_seconds: float | None = None
    mask_transfer_seconds: float | None = None
    proposed_token_count: int | None = None
    accepted_proposal_count: int | None = None
    acceptance_rate: float | None = None
    speculative_iteration_count: int | None = None
    draft_seconds: float | None = None
    verify_seconds: float | None = None
    mask_seconds: float | None = None
    verification_replays: int | None = None
    canonical_replay_tokens: int | None = None
    replay_stats: dict | None = None


class SpeculationStats(NamedTuple):
    """Speculative counters; unlike timings, collecting them adds no synchronization."""

    proposed_token_count: int
    accepted_proposal_count: int
    speculative_iteration_count: int
    replay_stats: dict


def _matched_stop_length(token_ids: list[int], stop_sequences: list[list[int]]) -> int:
    return max(
        (
            len(sequence)
            for sequence in stop_sequences
            if sequence and token_ids[-len(sequence) :] == sequence
        ),
        default=0,
    )


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)

    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
        sorted_probabilities.masked_fill_(
            cumulative_probabilities - sorted_probabilities >= top_p, 0
        )
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_index = torch.multinomial(sorted_probabilities, 1, generator=generator)
        return sorted_indices.gather(-1, sampled_index).squeeze(-1)

    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def _validate_generation_options(
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> None:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _validate_grammar_request(
    regex: str | None,
    token_byte_vocabulary: TokenByteVocabulary | None,
    json_schema: str | None,
) -> bool:
    if regex is not None and json_schema is not None:
        raise ValueError("regex and json_schema are mutually exclusive")
    grammar_requested = regex is not None or json_schema is not None
    if grammar_requested and token_byte_vocabulary is None:
        raise ValueError("token_byte_vocabulary is required when a grammar is set")
    if json_schema is not None:
        from onyx_cuda import _rust

        _rust.validate_json_schema(json_schema)
    return grammar_requested


def _validate_json_result(
    json_schema: str, token_byte_vocabulary: TokenByteVocabulary, token_ids: list[int]
) -> None:
    from onyx_cuda import _rust

    text = b"".join(token_byte_vocabulary.token_bytes[token_id] for token_id in token_ids).decode(
        "utf-8"
    )
    _rust.validate_json_output(json_schema, text)


def _grammar_choices(constraint, grammar_state: int, eos_token_ids: list[int]) -> list[int]:
    """Return the token IDs selectable after grammar_state.

    A complete match that can still grow (``[0-9]+`` after one digit) also
    permits EOS, so the model rather than the first match decides where output
    ends. An empty list means the match is complete and cannot be extended.
    EOS tokens decode to no bytes, so the grammar never lists them itself.
    Native constraints on CUDA return the same set as a cached device mask.
    """
    choices = getattr(constraint, "choices", None)
    if choices is not None:
        return choices(grammar_state, eos_token_ids)
    valid_token_ids = constraint.get_valid_token_ids(grammar_state)
    if not constraint.is_match_state(grammar_state):
        if not valid_token_ids:
            raise ValueError("Grammar constraint has no valid token continuation")
        return valid_token_ids
    if not valid_token_ids:
        return []
    return [*valid_token_ids, *eos_token_ids]


class _MaskedGrammar:
    """A native constraint whose selectable tokens are cached device masks.

    The engine caches scans by grammar state and names each exact token set by
    a scan id, so a mask is uploaded once per set, not once per position. The
    cache is bounded; a mask is one byte per vocabulary token.
    """

    _MASK_CAPACITY = 32

    def __init__(self, constraint, device: torch.device):
        self._constraint = constraint
        self._device = device
        self._masks: dict[tuple, torch.Tensor] = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._constraint, name)

    def choices(self, grammar_state: int, eos_token_ids: list[int]):
        scan_id, count = self._constraint.scan_valid_tokens(grammar_state)
        # Same rules and list length as _grammar_choices for plain constraints.
        match = self._constraint.is_match_state(grammar_state)
        if not match and not count:
            raise ValueError("Grammar constraint has no valid token continuation")
        if match and not count:
            return []
        eos = tuple(eos_token_ids) if match else ()
        if len(set(eos)) != len(eos):
            # A repeated EOS counts twice in the list form; a mask cannot.
            return [*self._constraint.get_valid_token_ids(grammar_state), *eos]
        key = (scan_id, eos)
        blocked = self._masks.pop(key, None)
        if blocked is None:
            host = torch.frombuffer(self._constraint.blocked_token_mask(grammar_state), dtype=torch.bool)
            blocked = host.to(self._device)
            if eos:
                blocked[list(eos)] = False
            if len(self._masks) >= self._MASK_CAPACITY:
                del self._masks[next(iter(self._masks))]
        self._masks[key] = blocked
        return TokenMask(blocked, count + len(eos))

    def prefetch(self, grammar_state: int) -> None:
        """Fill the engine's scan cache now, typically while the GPU is busy."""
        self._constraint.scan_valid_tokens(grammar_state)


def _initialize_grammar_constraint(
    logits_vocab_size: int,
    regex: str | None,
    token_byte_vocabulary: TokenByteVocabulary,
    json_schema: str | None,
    device: torch.device | None = None,
):
    from onyx_cuda import _rust

    token_bytes = token_byte_vocabulary.token_bytes
    if len(token_bytes) != logits_vocab_size:
        raise ValueError("token_byte_vocabulary must match the model logits width")
    constraint = _rust.GrammarConstraint(token_bytes)
    if json_schema is not None:
        constraint.compile_json_schema(json_schema)
    else:
        constraint.compile_regex(regex)
    if device is not None and device.type == "cuda":
        constraint = _MaskedGrammar(constraint, device)
    return constraint, constraint.init_state()


def generate_token_events(
    model: PreTrainedModel,
    prompt_token_ids: list[int],
    max_tokens: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    measure: bool = False,
    regex: str | None = None,
    token_byte_vocabulary: TokenByteVocabulary | None = None,
    json_schema: str | None = None,
) -> Iterator[AcceptedTokenEvent | GenerationFinishedEvent]:
    """Generate at most max_tokens with greedy or top-p sampling."""
    _validate_generation_options(max_tokens, temperature, top_p, seed)
    grammar_requested = _validate_grammar_request(regex, token_byte_vocabulary, json_schema)

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []

    generated: list[int] = []
    pending: list[int] = []
    retain = max((len(stop) for stop in stop_sequences if stop), default=1) - 1
    finish_reason: Literal["eos", "stop", "length"] = "length"
    time_to_first_token = None
    grammar_compile_seconds = None
    valid_token_enumeration_seconds = None
    mask_transfer_seconds = None
    # End-to-end timings are always recorded: reading each selected token
    # already synchronizes. Only measure adds the per-stage synchronizations.
    if measure:
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    constraint = None
    grammar_state = None
    grammar_choices = None

    def choices_after(state):
        nonlocal valid_token_enumeration_seconds
        enumeration_started_at = time.perf_counter() if measure else None
        choices = _grammar_choices(constraint, state, eos_token_ids)
        if enumeration_started_at is not None:
            valid_token_enumeration_seconds += time.perf_counter() - enumeration_started_at
        return choices

    try:
        result = prefill(model, prompt_token_ids)
        logits = result.logits
        cache = CacheState.from_prefill(result.past_key_values, result.logits.device)
        if grammar_requested:
            compile_started_at = time.perf_counter() if measure else None
            constraint, grammar_state = _initialize_grammar_constraint(
                logits.shape[-1],
                regex,
                token_byte_vocabulary,
                json_schema,
                logits.device,
            )
            if compile_started_at is not None:
                grammar_compile_seconds = time.perf_counter() - compile_started_at
                valid_token_enumeration_seconds = 0.0
                mask_transfer_seconds = 0.0

        generator = None
        if temperature > 0 and seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(seed)

        for step in range(max_tokens):
            token_id = None
            if constraint is not None:
                if grammar_choices is None:
                    grammar_choices = choices_after(grammar_state)
                if not grammar_choices:
                    finish_reason = "stop"
                    break
                if measure:
                    torch.cuda.synchronize(logits.device)
                    mask_started_at = time.perf_counter()
                if temperature == 0:
                    token_id = grammar_argmax(logits, grammar_choices)
                else:
                    logits = apply_grammar_mask(logits, grammar_choices)
                grammar_choices = None
                if measure:
                    torch.cuda.synchronize(logits.device)
                    mask_transfer_seconds += time.perf_counter() - mask_started_at

            if token_id is None:
                token_id = _sample_token(logits, temperature, top_p, generator)
            token = token_id.item()
            if time_to_first_token is None:
                time_to_first_token = time.perf_counter() - started_at
            generated.append(token)
            pending.append(token)

            if constraint is not None and token not in eos_token_ids:
                previous_state = grammar_state
                grammar_state = constraint.advance_state(grammar_state, token)
                constraint.release_state(previous_state)
                if constraint.is_match_state(grammar_state):
                    # Decide now whether this complete match can grow, so a
                    # finished output does not pay for another forward pass.
                    grammar_choices = choices_after(grammar_state)
                    if not grammar_choices:
                        finish_reason = "stop"
                        break

            matched_stop_length = _matched_stop_length(generated, stop_sequences)
            if matched_stop_length:
                del generated[-matched_stop_length:]
                del pending[-matched_stop_length:]
                finish_reason = "stop"
                break
            if token in eos_token_ids:
                finish_reason = "eos"
                break
            for ready_token in _take_ready_tokens(pending, retain):
                yield AcceptedTokenEvent(ready_token)
            if step + 1 == max_tokens:
                break

            with torch.inference_mode():
                logits = cache.extend(model, token_id[:, None])[:, -1, :]
        for ready_token in _take_ready_tokens(pending):
            yield AcceptedTokenEvent(ready_token)
    except _TextStop as stop:
        generated = stop.token_ids
        finish_reason = "stop"
        # As in ordinary generation, the last returned token is not in KV yet.
        cache.crop(len(prompt_token_ids) + len(generated) - 1)
    finally:
        if constraint is not None and grammar_state is not None:
            constraint.release_state(grammar_state)

    if json_schema is not None and finish_reason != "length":
        _validate_json_result(json_schema, token_byte_vocabulary, generated)

    timings = None
    if time_to_first_token is not None:
        # The last token was read from the GPU; no generation work is queued.
        total_seconds = time.perf_counter() - started_at
        decode_seconds = total_seconds - time_to_first_token
        decode_token_count = max(len(generated) - 1, 0)
        decode_tokens_per_second = (
            decode_token_count / decode_seconds
            if decode_token_count and decode_seconds > 0
            else None
        )
        timings = GenerationTimings(
            time_to_first_token_seconds=time_to_first_token,
            decode_tokens_per_second=decode_tokens_per_second,
            total_seconds=total_seconds,
            grammar_compile_seconds=grammar_compile_seconds,
            valid_token_enumeration_seconds=(valid_token_enumeration_seconds),
            mask_transfer_seconds=mask_transfer_seconds,
        )

    yield GenerationFinishedEvent(
        GenerationResult(generated, cache.past_key_values, finish_reason, timings)
    )


def generate_tokens(
    model: PreTrainedModel,
    prompt_token_ids: list[int],
    max_tokens: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    measure: bool = False,
    regex: str | None = None,
    token_byte_vocabulary: TokenByteVocabulary | None = None,
    json_schema: str | None = None,
) -> GenerationResult:
    """Collect the same incremental loop used by sampled streaming."""
    events = generate_token_events(
        model,
        prompt_token_ids,
        max_tokens,
        eos_token_ids,
        stop_sequences=stop_sequences,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        measure=measure,
        regex=regex,
        token_byte_vocabulary=token_byte_vocabulary,
        json_schema=json_schema,
    )
    try:
        for event in events:
            if isinstance(event, GenerationFinishedEvent):
                return event.result
        raise RuntimeError("Generation ended without a terminal event")
    finally:
        events.close()
