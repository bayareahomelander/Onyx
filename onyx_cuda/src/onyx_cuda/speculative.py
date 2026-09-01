"""Fixed-gamma speculative token generation."""

import json
import time
from collections.abc import Iterable, Iterator
from typing import NamedTuple

import torch
from transformers import PreTrainedModel

from onyx_cuda.cache import CacheState
from onyx_cuda.generation import (
    GenerationResult,
    GenerationTimings,
    _initialize_grammar_constraint,
    _matched_stop_length,
    _validate_grammar_request,
    _validate_generation_options,
    generate_tokens,
)
from onyx_cuda.masking import apply_grammar_mask
from onyx_cuda.prefill import prefill
from onyx_cuda.vocabulary import TokenByteVocabulary


class ProposalResult(NamedTuple):
    token_ids: list[int]
    draft_cache_length_before: int
    draft_cache_length_after: int


class VerificationResult(NamedTuple):
    token_ids: list[int]
    accepted_proposal_count: int


class AcceptedTokenEvent(NamedTuple):
    token_id: int


class GenerationFinishedEvent(NamedTuple):
    result: GenerationResult


class TextDeltaEvent(NamedTuple):
    text: str


def _take_ready_tokens(pending: list[int], retain: int = 0) -> list[int]:
    count = max(len(pending) - retain, 0)
    ready = pending[:count]
    del pending[:count]
    return ready


def _flush_stream_text(pending: str, stop: list[str] | None) -> tuple[str, str, bool]:
    active_stops = [sequence for sequence in (stop or []) if sequence]
    if not active_stops:
        return pending, "", False

    positions = [pending.find(sequence) for sequence in active_stops]
    positions = [position for position in positions if position >= 0]
    if positions:
        return pending[: min(positions)], "", True

    retain = max(len(sequence) for sequence in active_stops) - 1
    if retain <= 0 or len(pending) > retain:
        split = max(len(pending) - retain, 0)
        return pending[:split], pending[split:], False
    return "", pending, False


def _synchronize_device(device) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _mask_grammar_logits(
    constraint,
    grammar_state: int,
    logits: torch.Tensor,
    mask_times: list[float] | None = None,
):
    if mask_times is not None:
        if logits.device.type == "cuda":
            torch.cuda.synchronize(logits.device)
        started_at = time.perf_counter()
    valid_token_ids = constraint.get_valid_token_ids(grammar_state)
    if not valid_token_ids:
        raise ValueError("Grammar constraint has no valid token continuation")
    masked_logits = apply_grammar_mask(logits, valid_token_ids)
    if mask_times is not None:
        if logits.device.type == "cuda":
            torch.cuda.synchronize(logits.device)
        mask_times.append(time.perf_counter() - started_at)
    return masked_logits


def _advance_grammar_state(
    constraint,
    grammar_state: int,
    token_id: int,
    live_grammar_states: set[int],
) -> int:
    next_state = constraint.advance_state(grammar_state, token_id)
    live_grammar_states.add(next_state)
    return next_state


def _release_grammar_states(constraint, states: list[int], live_grammar_states: set[int]) -> None:
    states = list(dict.fromkeys(states))
    if states:
        constraint.release_states(states)
        live_grammar_states.difference_update(states)


def propose_tokens(
    draft_model: PreTrainedModel,
    draft_cache: CacheState,
    generated_token_ids: list[int],
    gamma: int,
    remaining_tokens: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
    *,
    grammar_constraint=None,
    grammar_state: int | None = None,
    live_grammar_states: set[int] | None = None,
    mask_times: list[float] | None = None,
) -> ProposalResult:
    """Greedily propose tokens after the target-selected current token."""
    if not generated_token_ids:
        raise ValueError("draft proposal requires a target-selected token")
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []
    start_length = draft_cache.length
    if grammar_constraint is not None:
        if grammar_state is None or live_grammar_states is None:
            raise ValueError("grammar state tracking is required")
        if grammar_constraint.is_match_state(grammar_state):
            return ProposalResult([], start_length, start_length)
    if generated_token_ids[-1] in eos_token_ids or _matched_stop_length(
        generated_token_ids, stop_sequences
    ):
        return ProposalResult([], start_length, start_length)

    proposed: list[int] = []
    input_ids = torch.tensor(
        [[generated_token_ids[-1]]],
        device=draft_cache.attention_mask.device,
    )
    draft_grammar_state = grammar_state
    with torch.inference_mode():
        for _ in range(min(gamma, remaining_tokens)):
            logits = draft_cache.extend(draft_model, input_ids)[:, -1, :]
            if grammar_constraint is not None:
                logits = _mask_grammar_logits(
                    grammar_constraint,
                    draft_grammar_state,
                    logits,
                    mask_times,
                )
            token_id = logits.argmax(dim=-1)
            token = token_id.item()
            proposed.append(token)
            if grammar_constraint is not None:
                draft_grammar_state = _advance_grammar_state(
                    grammar_constraint,
                    draft_grammar_state,
                    token,
                    live_grammar_states,
                )
                if grammar_constraint.is_match_state(draft_grammar_state):
                    break
            if token in eos_token_ids or _matched_stop_length(
                generated_token_ids + proposed, stop_sequences
            ):
                break
            input_ids = token_id[:, None]

    return ProposalResult(proposed, start_length, draft_cache.length)


def _verify_proposal(
    draft_model: PreTrainedModel,
    draft_cache: CacheState,
    target_model: PreTrainedModel,
    target_cache: CacheState,
    current_token_id: int,
    proposal: ProposalResult,
    *,
    grammar_constraint=None,
    grammar_state: int | None = None,
    live_grammar_states: set[int] | None = None,
    mask_times: list[float] | None = None,
) -> tuple[VerificationResult, list[int]]:
    """Verify every proposal position in one target forward."""
    input_ids = torch.tensor(
        [[current_token_id, *proposal.token_ids]],
        device=target_cache.attention_mask.device,
    )
    target_length_before = target_cache.length
    with torch.inference_mode():
        target_logits = target_cache.extend(target_model, input_ids)

    verified_token_ids: list[int]
    verified_grammar_states: list[int] = []
    if grammar_constraint is None:
        target_token_ids = target_logits.argmax(dim=-1)[0].tolist()
        accepted = 0
        for proposed, target in zip(proposal.token_ids, target_token_ids):
            if proposed != target:
                break
            accepted += 1
        verified_token_ids = proposal.token_ids[:accepted] + [target_token_ids[accepted]]
    else:
        if grammar_state is None or live_grammar_states is None:
            raise ValueError("grammar state tracking is required")
        accepted = 0
        verify_grammar_state = grammar_state
        verified_token_ids = []
        for position, proposed in enumerate(proposal.token_ids):
            logits = _mask_grammar_logits(
                grammar_constraint,
                verify_grammar_state,
                target_logits[:, position, :],
                mask_times,
            )
            target = logits.argmax(dim=-1).item()
            token = proposed if proposed == target else target
            verified_token_ids.append(token)
            verify_grammar_state = _advance_grammar_state(
                grammar_constraint,
                verify_grammar_state,
                token,
                live_grammar_states,
            )
            verified_grammar_states.append(verify_grammar_state)
            if proposed != target:
                break
            accepted += 1
            if grammar_constraint.is_match_state(verify_grammar_state):
                break
        else:
            logits = _mask_grammar_logits(
                grammar_constraint,
                verify_grammar_state,
                target_logits[:, len(proposal.token_ids), :],
                mask_times,
            )
            target = logits.argmax(dim=-1).item()
            verified_token_ids.append(target)
            verify_grammar_state = _advance_grammar_state(
                grammar_constraint,
                verify_grammar_state,
                target,
                live_grammar_states,
            )
            verified_grammar_states.append(verify_grammar_state)

    if accepted == len(proposal.token_ids):
        draft_token_id = proposal.token_ids[-1] if proposal.token_ids else current_token_id
        with torch.inference_mode():
            draft_cache.extend(
                draft_model,
                torch.tensor(
                    [[draft_token_id]],
                    device=draft_cache.attention_mask.device,
                ),
            )

    target_cache.crop(target_length_before + accepted + 1)
    draft_cache.crop(proposal.draft_cache_length_before + accepted + 1)
    return (
        VerificationResult(verified_token_ids, accepted),
        verified_grammar_states,
    )


def verify_proposal(
    draft_model: PreTrainedModel,
    draft_cache: CacheState,
    target_model: PreTrainedModel,
    target_cache: CacheState,
    current_token_id: int,
    proposal: ProposalResult,
) -> VerificationResult:
    """Verify every proposal position in one target forward."""
    return _verify_proposal(
        draft_model,
        draft_cache,
        target_model,
        target_cache,
        current_token_id,
        proposal,
    )[0]


def generate_speculative_events(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    prompt_token_ids: list[int],
    max_tokens: int,
    gamma: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    *,
    measure: bool = False,
    regex: str | None = None,
    token_byte_vocabulary: TokenByteVocabulary | None = None,
    json_schema: str | None = None,
) -> Iterator[AcceptedTokenEvent | GenerationFinishedEvent]:
    """Yield accepted tokens and one terminal result from one generation loop."""
    if gamma < 1:
        raise ValueError("gamma must be at least 1")
    _validate_generation_options(max_tokens, temperature, top_p, seed)
    grammar_requested = _validate_grammar_request(regex, token_byte_vocabulary, json_schema)
    if temperature > 0:
        result = generate_tokens(
            target_model,
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
        for token_id in result.token_ids:
            yield AcceptedTokenEvent(token_id)
        yield GenerationFinishedEvent(result)
        return

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []
    constraint = None
    grammar_state = None
    live_grammar_states: set[int] = set()
    generated: list[int] = []
    pending_events: list[int] = []
    event_retain = (
        max(
            (len(sequence) for sequence in stop_sequences if sequence),
            default=1,
        )
        - 1
    )
    finish_reason = "length"
    finished = False
    started_at = None
    time_to_first_token = None
    grammar_compile_seconds = None
    mask_times: list[float] | None = [] if measure else None
    proposed_token_count = 0
    accepted_proposal_count = 0
    speculative_iteration_count = 0
    draft_seconds = 0.0
    verify_seconds = 0.0
    measurement_device = None
    if measure:
        measurement_device = next(target_model.parameters()).device
        _synchronize_device(measurement_device)
        started_at = time.perf_counter()

    try:
        draft_prefill = prefill(draft_model, prompt_token_ids)
        target_prefill = prefill(target_model, prompt_token_ids)
        draft_cache = CacheState.from_prefill(
            draft_prefill.past_key_values, draft_prefill.logits.device
        )
        target_cache = CacheState.from_prefill(
            target_prefill.past_key_values, target_prefill.logits.device
        )
        if grammar_requested:
            compile_started_at = time.perf_counter() if measure else None
            constraint, grammar_state = _initialize_grammar_constraint(
                target_prefill.logits.shape[-1],
                regex,
                token_byte_vocabulary,
                json_schema,
            )
            if compile_started_at is not None:
                grammar_compile_seconds = time.perf_counter() - compile_started_at
            live_grammar_states.add(grammar_state)
            if constraint.is_match_state(grammar_state):
                finish_reason = "stop"
                finished = True

        if not finished:
            if constraint is None:
                first_token = target_prefill.token_id
            else:
                first_token_logits = _mask_grammar_logits(
                    constraint,
                    grammar_state,
                    target_prefill.logits,
                    mask_times,
                )
                first_token = first_token_logits.argmax(dim=-1)
            generated.append(first_token.item())
            pending_events.append(generated[-1])
            if started_at is not None:
                _synchronize_device(measurement_device)
                time_to_first_token = time.perf_counter() - started_at

            if constraint is not None:
                previous_state = grammar_state
                grammar_state = _advance_grammar_state(
                    constraint,
                    grammar_state,
                    generated[-1],
                    live_grammar_states,
                )
                _release_grammar_states(constraint, [previous_state], live_grammar_states)
                if constraint.is_match_state(grammar_state):
                    finish_reason = "stop"
                    finished = True

            if not finished:
                matched_stop_length = _matched_stop_length(generated, stop_sequences)
                if matched_stop_length:
                    del generated[-matched_stop_length:]
                    del pending_events[-matched_stop_length:]
                    finish_reason = "stop"
                    finished = True
                elif generated[-1] in eos_token_ids:
                    finish_reason = "eos"
                    finished = True

            if not finished:
                for token_id in _take_ready_tokens(pending_events, event_retain):
                    yield AcceptedTokenEvent(token_id)

        while not finished and len(generated) < max_tokens:
            speculative_iteration_count += 1
            if measurement_device is not None:
                _synchronize_device(measurement_device)
                draft_started_at = time.perf_counter()
                mask_seconds_before = sum(mask_times)
            states_before_draft = set(live_grammar_states)
            proposal = propose_tokens(
                draft_model,
                draft_cache,
                generated,
                gamma,
                max_tokens - len(generated) - 1,
                eos_token_ids,
                stop_sequences,
                grammar_constraint=constraint,
                grammar_state=grammar_state,
                live_grammar_states=(live_grammar_states if constraint is not None else None),
                mask_times=mask_times,
            )
            proposed_token_count += len(proposal.token_ids)
            if measurement_device is not None:
                _synchronize_device(measurement_device)
                draft_seconds += max(
                    time.perf_counter()
                    - draft_started_at
                    - (sum(mask_times) - mask_seconds_before),
                    0.0,
                )
            draft_grammar_states = list(live_grammar_states - states_before_draft)
            if measurement_device is not None:
                _synchronize_device(measurement_device)
                verify_started_at = time.perf_counter()
                mask_seconds_before = sum(mask_times)
            verified, verified_grammar_states = _verify_proposal(
                draft_model,
                draft_cache,
                target_model,
                target_cache,
                generated[-1],
                proposal,
                grammar_constraint=constraint,
                grammar_state=grammar_state,
                live_grammar_states=(live_grammar_states if constraint is not None else None),
                mask_times=mask_times,
            )
            accepted_proposal_count += verified.accepted_proposal_count
            if measurement_device is not None:
                _synchronize_device(measurement_device)
                verify_seconds += max(
                    time.perf_counter()
                    - verify_started_at
                    - (sum(mask_times) - mask_seconds_before),
                    0.0,
                )

            retained_grammar_state = None
            for position, token_id in enumerate(verified.token_ids):
                generated.append(token_id)
                pending_events.append(token_id)
                if constraint is not None and constraint.is_match_state(
                    verified_grammar_states[position]
                ):
                    retained_grammar_state = verified_grammar_states[position]
                    finish_reason = "stop"
                    finished = True
                    break

                matched_stop_length = _matched_stop_length(generated, stop_sequences)
                if matched_stop_length:
                    del generated[-matched_stop_length:]
                    del pending_events[-matched_stop_length:]
                    if constraint is not None:
                        retained_grammar_state = verified_grammar_states[position]
                    finish_reason = "stop"
                    finished = True
                    break
                if token_id in eos_token_ids:
                    if constraint is not None:
                        retained_grammar_state = verified_grammar_states[position]
                    finish_reason = "eos"
                    finished = True
                    break

                for ready_token_id in _take_ready_tokens(pending_events, event_retain):
                    yield AcceptedTokenEvent(ready_token_id)

            if constraint is not None:
                if retained_grammar_state is None:
                    retained_grammar_state = verified_grammar_states[-1]
                _release_grammar_states(
                    constraint,
                    [
                        grammar_state,
                        *draft_grammar_states,
                        *(
                            state
                            for state in verified_grammar_states
                            if state != retained_grammar_state
                        ),
                    ],
                    live_grammar_states,
                )
                grammar_state = retained_grammar_state

        for token_id in _take_ready_tokens(pending_events):
            yield AcceptedTokenEvent(token_id)

        if json_schema is not None:
            json.loads(
                b"".join(
                    token_byte_vocabulary.token_bytes[token_id] for token_id in generated
                ).decode("utf-8")
            )
        past_key_values = target_cache.past_key_values
    finally:
        if constraint is not None and live_grammar_states:
            constraint.release_states(list(live_grammar_states))

    timings = None
    if started_at is not None and time_to_first_token is not None:
        _synchronize_device(measurement_device)
        total_seconds = time.perf_counter() - started_at
        decode_seconds = total_seconds - time_to_first_token
        decode_token_count = max(len(generated) - 1, 0)
        timings = GenerationTimings(
            time_to_first_token_seconds=time_to_first_token,
            decode_tokens_per_second=(
                decode_token_count / decode_seconds
                if decode_token_count and decode_seconds > 0
                else None
            ),
            total_seconds=total_seconds,
            grammar_compile_seconds=grammar_compile_seconds,
            proposed_token_count=proposed_token_count,
            accepted_proposal_count=accepted_proposal_count,
            acceptance_rate=(
                accepted_proposal_count / proposed_token_count if proposed_token_count else 0.0
            ),
            speculative_iteration_count=speculative_iteration_count,
            draft_seconds=draft_seconds,
            verify_seconds=verify_seconds,
            mask_seconds=sum(mask_times),
        )
    yield GenerationFinishedEvent(
        GenerationResult(generated, past_key_values, finish_reason, timings)
    )


def generate_speculative(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    prompt_token_ids: list[int],
    max_tokens: int,
    gamma: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    *,
    measure: bool = False,
    regex: str | None = None,
    token_byte_vocabulary: TokenByteVocabulary | None = None,
    json_schema: str | None = None,
) -> GenerationResult:
    """Collect the shared event stream into the established result shape."""
    accepted_token_ids = []
    events = generate_speculative_events(
        draft_model,
        target_model,
        prompt_token_ids,
        max_tokens,
        gamma,
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
            if isinstance(event, AcceptedTokenEvent):
                accepted_token_ids.append(event.token_id)
                continue
            if accepted_token_ids != event.result.token_ids:
                raise RuntimeError("Accepted-token events do not match the final result")
            return event.result
        raise RuntimeError("Generation ended without a terminal event")
    finally:
        events.close()


def decode_speculative_events(
    events: Iterable[AcceptedTokenEvent | GenerationFinishedEvent],
    tokenizer,
    stop: list[str] | None = None,
) -> Iterator[TextDeltaEvent | GenerationFinishedEvent]:
    """Incrementally decode token events and preserve one terminal event."""
    token_ids: list[int] = []
    decoded = ""
    emitted = ""
    pending = ""
    stopped = False
    terminal_seen = False
    event_iterator = iter(events)
    try:
        for event in event_iterator:
            if terminal_seen:
                raise RuntimeError("Generation emitted events after completion")
            if isinstance(event, AcceptedTokenEvent):
                token_ids.append(event.token_id)
                if stopped:
                    continue
                # ponytail: cumulative decode is bounded by max_tokens; use
                # raw token bytes only if profiling shows this helper matters.
                current = tokenizer.decode(token_ids, skip_special_tokens=True)
                replacement = current.find("\ufffd")
                stable = current if replacement < 0 else current[:replacement]
                if not stable.startswith(decoded):
                    raise RuntimeError("Tokenizer changed already-decoded text")
                pending += stable[len(decoded) :]
                decoded = stable
                text, pending, stopped = _flush_stream_text(pending, stop)
                if text:
                    emitted += text
                    yield TextDeltaEvent(text)
                continue

            terminal_seen = True
            final_text = tokenizer.decode(event.result.token_ids, skip_special_tokens=True)
            positions = [final_text.find(sequence) for sequence in (stop or []) if sequence]
            positions = [position for position in positions if position >= 0]
            if positions:
                final_text = final_text[: min(positions)]
            if not final_text.startswith(emitted):
                raise RuntimeError("Streamed text does not match the final result")
            remaining = final_text[len(emitted) :]
            if remaining:
                yield TextDeltaEvent(remaining)
            yield event

        if not terminal_seen:
            raise RuntimeError("Generation ended without a terminal event")
    finally:
        close = getattr(event_iterator, "close", None)
        if close is not None:
            close()
