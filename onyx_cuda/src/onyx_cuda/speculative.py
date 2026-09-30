"""Fixed-gamma speculative token generation through one event loop."""

import time
from collections.abc import Iterable, Iterator, Sequence
from typing import NamedTuple

import torch
from transformers import PreTrainedModel

from onyx_cuda.cache import CacheState
from onyx_cuda.numerics import GreedyCheckpoint, ambiguous_logits
from onyx_cuda.generation import (
    AcceptedTokenEvent,
    GenerationFinishedEvent,
    _take_ready_tokens,
    GenerationResult,
    GenerationTimings,
    SpeculationStats,
    _grammar_choices,
    _initialize_grammar_constraint,
    _matched_stop_length,
    _validate_grammar_request,
    _validate_generation_options,
    _validate_json_result,
    _TextStop,
    generate_token_events,
)
from onyx_cuda.masking import grammar_argmax
from onyx_cuda.prefill import prefill
from onyx_cuda.vocabulary import TokenByteVocabulary


class ProposalResult(NamedTuple):
    token_ids: list[int]
    # Draft length with every token before the current one consumed; a
    # rejection crops to this plus the accepted tokens and the current one.
    draft_cache_length_before: int
    draft_cache_length_after: int
    # Grammar state after the last proposed token, when it is not EOS.
    grammar_state: int | None = None


class VerificationResult(NamedTuple):
    token_ids: list[int]
    accepted_proposal_count: int


class TextDeltaEvent(NamedTuple):
    text: str


def _flush_stream_text(pending: str, stop: list[str] | None) -> tuple[str, str, bool]:
    active_stops = [sequence for sequence in (stop or []) if sequence]
    if not active_stops:
        return pending, "", False

    positions = [pending.find(sequence) for sequence in active_stops]
    positions = [position for position in positions if position >= 0]
    if positions:
        return pending[: min(positions)], "", True

    # Only a suffix matching a stop's prefix can become a stop in a later chunk.
    retain = 0
    for sequence in active_stops:
        for length in range(min(len(pending), len(sequence) - 1), retain, -1):
            if pending.endswith(sequence[:length]):
                retain = length
                break
    split = len(pending) - retain
    return pending[:split], pending[split:], False


def _synchronize_device(device) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_grammar_choices(
    constraint,
    grammar_state: int,
    eos_token_ids: list[int],
    mask_times: list[float] | None = None,
) -> list[int]:
    started_at = time.perf_counter() if mask_times is not None else None
    choices = _grammar_choices(constraint, grammar_state, eos_token_ids)
    if started_at is not None:
        mask_times.append(time.perf_counter() - started_at)
    return choices


def _grammar_token(
    choices: list[int],
    logits: torch.Tensor,
    mask_times: list[float] | None = None,
):
    if mask_times is not None:
        if logits.device.type == "cuda":
            torch.cuda.synchronize(logits.device)
        started_at = time.perf_counter()
    token_id = grammar_argmax(logits, choices)
    if mask_times is not None:
        if logits.device.type == "cuda":
            torch.cuda.synchronize(logits.device)
        mask_times.append(time.perf_counter() - started_at)
    return token_id


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
    grammar_choices: list[int] | None = None,
    unconsumed: int = 1,
) -> ProposalResult:
    """Greedily propose tokens after the target-selected current token.

    grammar_choices, when given, are the selectable tokens after grammar_state.
    unconsumed counts the trailing generated tokens the draft cache has not
    consumed: the current token, preceded by the last proposed token when the
    previous verification accepted every proposal. The first draft step
    consumes them together.
    """
    if not generated_token_ids:
        raise ValueError("draft proposal requires a target-selected token")
    if not 1 <= unconsumed <= len(generated_token_ids):
        raise ValueError("unconsumed must count from one to every generated token")
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []
    start_length = draft_cache.length
    before = start_length + unconsumed - 1
    if grammar_constraint is not None:
        if grammar_state is None or live_grammar_states is None:
            raise ValueError("grammar state tracking is required")
        if grammar_choices is None:
            grammar_choices = _timed_grammar_choices(
                grammar_constraint, grammar_state, eos_token_ids, mask_times
            )
        if not grammar_choices:
            return ProposalResult([], before, start_length)
    if generated_token_ids[-1] in eos_token_ids or _matched_stop_length(
        generated_token_ids, stop_sequences
    ):
        return ProposalResult([], before, start_length)

    proposed: list[int] = []
    input_ids = torch.tensor(
        [generated_token_ids[-unconsumed:]],
        device=draft_cache.device,
    )
    draft_grammar_state = grammar_state
    choices = grammar_choices
    with torch.inference_mode():
        for _ in range(min(gamma, remaining_tokens)):
            logits = draft_cache.extend(draft_model, input_ids)[:, -1, :]
            if grammar_constraint is not None:
                if choices is None:
                    choices = _timed_grammar_choices(
                        grammar_constraint, draft_grammar_state, eos_token_ids, mask_times
                    )
                token_id = _grammar_token(choices, logits, mask_times)
                choices = None
            else:
                token_id = logits.argmax(dim=-1)
            token = token_id.item()
            proposed.append(token)
            if token in eos_token_ids:
                break
            if grammar_constraint is not None:
                draft_grammar_state = _advance_grammar_state(
                    grammar_constraint,
                    draft_grammar_state,
                    token,
                    live_grammar_states,
                )
                if grammar_constraint.is_match_state(draft_grammar_state):
                    choices = _timed_grammar_choices(
                        grammar_constraint, draft_grammar_state, eos_token_ids, mask_times
                    )
                    if not choices:
                        break
            if _matched_stop_length(generated_token_ids + proposed, stop_sequences):
                break
            input_ids = token_id[:, None]

    final_state = (draft_grammar_state if grammar_constraint is not None and proposed
                   and proposed[-1] not in eos_token_ids else None)
    return ProposalResult(proposed, before, draft_cache.length, final_state)


def _verify_proposal(
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
    checkpoint: GreedyCheckpoint | None = None,
    generated_token_ids: list[int] | None = None,
    eos_token_ids: Sequence[int] = (),
    grammar_choices: list[int] | None = None,
) -> tuple[VerificationResult, list[int]]:
    """Verify every proposal position in one target forward.

    Grammar states are returned for each verified token except a final EOS.
    """
    input_ids = torch.tensor(
        [[current_token_id, *proposal.token_ids]],
        device=target_cache.attention_mask.device,
    )
    target_length_before = target_cache.length
    if checkpoint is not None:
        checkpoint.before_forward(input_ids)
    with torch.inference_mode():
        target_logits = target_cache.extend(target_model, input_ids)
    prefetch = getattr(grammar_constraint, "prefetch", None)
    if prefetch is not None and proposal.grammar_state is not None:
        # The draft never scans the state after its last token, which the final
        # verification position needs. Scanning it now overlaps the target
        # forward; the verifier's own scan of that state is then a cache hit.
        started_at = time.perf_counter() if mask_times is not None else None
        prefetch(proposal.grammar_state)
        if started_at is not None:
            mask_times.append(time.perf_counter() - started_at)

    replayed = False
    for attempt in range(2):
        verified_token_ids: list[int]
        verified_grammar_states: list[int] = []
        position_choices = None
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
            position_choices = []
            choices = grammar_choices
            # The final position selects the correction or bonus token.
            for position in range(len(proposal.token_ids) + 1):
                if choices is None:
                    choices = _timed_grammar_choices(
                        grammar_constraint, verify_grammar_state, eos_token_ids, mask_times
                    )
                    if position == 0:
                        grammar_choices = choices
                if not choices:
                    break
                target = _grammar_token(
                    choices, target_logits[:, position, :], mask_times
                ).item()
                position_choices.append(choices)
                choices = None
                verified_token_ids.append(target)
                proposed = (
                    proposal.token_ids[position] if position < len(proposal.token_ids) else None
                )
                if target in eos_token_ids:
                    # EOS ends the output and never advances the grammar.
                    if target == proposed:
                        accepted += 1
                    break
                verify_grammar_state = _advance_grammar_state(
                    grammar_constraint,
                    verify_grammar_state,
                    target,
                    live_grammar_states,
                )
                verified_grammar_states.append(verify_grammar_state)
                if target != proposed:
                    break
                accepted += 1
                if grammar_constraint.is_match_state(verify_grammar_state):
                    choices = _timed_grammar_choices(
                        grammar_constraint, verify_grammar_state, eos_token_ids, mask_times
                    )
        if attempt == 0 and checkpoint is not None and not checkpoint.target_is_canonical and ambiguous_logits(
            target_logits, len(verified_token_ids), position_choices,
        ):
            if grammar_constraint is not None:
                _release_grammar_states(grammar_constraint, verified_grammar_states, live_grammar_states)
            target_logits = checkpoint.replay(target_cache, input_ids, generated_token_ids)
            replayed = True
            continue
        break

    target_cache.crop(target_length_before + accepted + 1)
    # After a full acceptance the draft has not consumed the last proposed
    # token (or, for an empty proposal, the current one). The next proposal
    # feeds it with the new current token in one draft step instead of a
    # separate catch-up step here, which the final round would also waste.
    if accepted < len(proposal.token_ids):
        draft_cache.crop(proposal.draft_cache_length_before + accepted + 1)
    if replayed:
        checkpoint.commit(target_cache, input_ids, accepted, target_logits)
    return (
        VerificationResult(verified_token_ids, accepted),
        verified_grammar_states,
    )


def verify_proposal(
    draft_cache: CacheState,
    target_model: PreTrainedModel,
    target_cache: CacheState,
    current_token_id: int,
    proposal: ProposalResult,
) -> VerificationResult:
    """Verify every proposal position in one target forward."""
    return _verify_proposal(
        draft_cache,
        target_model,
        target_cache,
        current_token_id,
        proposal,
    )[0]


def _start_draft_cache(draft_model, prompt_token_ids, max_tokens):
    """Prefill the draft; use its graph backend's static cache when available."""
    draft_prefill = prefill(draft_model, prompt_token_ids)
    backend = getattr(draft_model, "_onyx_draft_backend", None)
    cache = (backend.start(draft_prefill.past_key_values, len(prompt_token_ids) + max_tokens)
             if backend is not None else None)
    if cache is None:
        cache = CacheState.from_prefill(draft_prefill.past_key_values, draft_prefill.logits.device)
    return cache


def generate_speculative_events(
    draft_model: PreTrainedModel | None,
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
    if isinstance(gamma, bool) or not isinstance(gamma, int) or gamma < 0:
        raise ValueError("gamma must be a nonnegative integer (0 disables speculation)")
    _validate_generation_options(max_tokens, temperature, top_p, seed)
    grammar_requested = _validate_grammar_request(regex, token_byte_vocabulary, json_schema)
    if gamma == 0 or temperature > 0:
        yield from generate_token_events(
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
        return

    if draft_model is None:
        raise ValueError("draft_model is required when speculation is enabled")

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []
    constraint = None
    grammar_state = None
    grammar_choices = None
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
    time_to_first_token = None
    grammar_compile_seconds = None
    mask_times: list[float] | None = [] if measure else None
    proposed_token_count = 0
    accepted_proposal_count = 0
    speculative_iteration_count = 0
    draft_seconds = 0.0
    verify_seconds = 0.0
    # End-to-end timings are always recorded: reading each selected token
    # already synchronizes. Only measure adds the stage synchronizations.
    measurement_device = None
    if measure:
        measurement_device = next(target_model.parameters()).device
        _synchronize_device(measurement_device)
    started_at = time.perf_counter()

    draft_cache = None
    try:
        target_prefill = prefill(target_model, prompt_token_ids)
        device = target_prefill.logits.device
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
                target_prefill.logits.device,
            )
            if compile_started_at is not None:
                grammar_compile_seconds = time.perf_counter() - compile_started_at
            live_grammar_states.add(grammar_state)
            grammar_choices = _timed_grammar_choices(
                constraint, grammar_state, eos_token_ids, mask_times
            )
            if not grammar_choices:
                finish_reason = "stop"
                finished = True

        checkpoint = GreedyCheckpoint(target_model, prompt_token_ids, constraint,
                                      live_grammar_states,
                                      eos_token_ids=eos_token_ids, measure=measure)
        if not finished:
            if constraint is None:
                first_token = target_prefill.token_id
            else:
                first_token = _grammar_token(
                    grammar_choices, target_prefill.logits, mask_times
                )
                grammar_choices = None
            generated.append(first_token.item())
            pending_events.append(generated[-1])
            time_to_first_token = time.perf_counter() - started_at

            if constraint is not None and generated[-1] not in eos_token_ids:
                previous_state = grammar_state
                grammar_state = _advance_grammar_state(
                    constraint,
                    grammar_state,
                    generated[-1],
                    live_grammar_states,
                )
                _release_grammar_states(constraint, [previous_state], live_grammar_states)
                if constraint.is_match_state(grammar_state):
                    grammar_choices = _timed_grammar_choices(
                        constraint, grammar_state, eos_token_ids, mask_times
                    )
                    if not grammar_choices:
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

        # A numerical repair can replace target_cache's underlying KV object;
        # do not retain the obsolete object through the prefill result.
        if not finished and len(generated) < max_tokens:
            checkpoint.seed(target_cache, target_prefill.logits)
            # The first token needs only the target, so the draft prefill runs
            # after that token's yield point and is skipped when generation ends there.
            draft_cache = _start_draft_cache(draft_model, prompt_token_ids, max_tokens)
            # Generated tokens are consumed by the draft from this length on.
            draft_start_length = draft_cache.length
        del target_prefill
        while not finished and len(generated) < max_tokens:
            ready_events = []
            if constraint is not None and grammar_choices is None:
                # Shared by the draft's first proposal and the target's first check.
                grammar_choices = _timed_grammar_choices(
                    constraint, grammar_state, eos_token_ids, mask_times
                )
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
                grammar_choices=grammar_choices,
                unconsumed=len(generated) - (draft_cache.length - draft_start_length),
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
                draft_cache,
                target_model,
                target_cache,
                generated[-1],
                proposal,
                grammar_constraint=constraint,
                grammar_state=grammar_state,
                live_grammar_states=(live_grammar_states if constraint is not None else None),
                mask_times=mask_times,
                checkpoint=checkpoint,
                generated_token_ids=generated,
                eos_token_ids=eos_token_ids,
                grammar_choices=grammar_choices,
            )
            grammar_choices = None
            accepted_proposal_count += verified.accepted_proposal_count
            if measurement_device is not None:
                _synchronize_device(measurement_device)
                verify_seconds += max(
                    time.perf_counter()
                    - verify_started_at
                    - (sum(mask_times) - mask_seconds_before),
                    0.0,
                )

            # Verification stops at a complete match that cannot grow, so only
            # the final token can end the grammar. Check it before stop
            # sequences, in the same order as target-only generation.
            grammar_complete = False
            if (constraint is not None
                    and len(verified_grammar_states) == len(verified.token_ids)
                    and constraint.is_match_state(verified_grammar_states[-1])):
                grammar_choices = _timed_grammar_choices(
                    constraint, verified_grammar_states[-1], eos_token_ids, mask_times
                )
                grammar_complete = not grammar_choices
            for position, token_id in enumerate(verified.token_ids):
                generated.append(token_id)
                pending_events.append(token_id)
                if grammar_complete and position == len(verified.token_ids) - 1:
                    finish_reason = "stop"
                    finished = True
                    break

                matched_stop_length = _matched_stop_length(generated, stop_sequences)
                if matched_stop_length:
                    del generated[-matched_stop_length:]
                    del pending_events[-matched_stop_length:]
                    finish_reason = "stop"
                    finished = True
                    break
                if token_id in eos_token_ids:
                    finish_reason = "eos"
                    finished = True
                    break

                ready_events.extend(_take_ready_tokens(pending_events, event_retain))

            if constraint is not None:
                # Once finished, any retained state is released by the final cleanup.
                retained_grammar_state = (
                    verified_grammar_states[-1] if verified_grammar_states else grammar_state
                )
                _release_grammar_states(
                    constraint,
                    [
                        state
                        for state in (grammar_state, *draft_grammar_states, *verified_grammar_states)
                        if state != retained_grammar_state
                    ],
                    live_grammar_states,
                )
                grammar_state = retained_grammar_state

            for ready_token_id in ready_events:
                yield AcceptedTokenEvent(ready_token_id)

        for token_id in _take_ready_tokens(pending_events):
            yield AcceptedTokenEvent(token_id)

        if json_schema is not None and finish_reason != "length":
            _validate_json_result(json_schema, token_byte_vocabulary, generated)
        past_key_values = target_cache.past_key_values
    except _TextStop as stop:
        generated = stop.token_ids
        finish_reason = "stop"
        # Verification may have computed later tokens in this same batch.
        # Return only the consumed prefix, with its last token still outside KV.
        target_cache.crop(len(prompt_token_ids) + len(generated) - 1)
        past_key_values = target_cache.past_key_values
        if json_schema is not None:
            _validate_json_result(json_schema, token_byte_vocabulary, generated)
    finally:
        # A graph draft cache is shared model state; free it for the next generation.
        release = getattr(draft_cache, "release", None)
        if release is not None:
            release()
        if constraint is not None and live_grammar_states:
            constraint.release_states(list(live_grammar_states))

    speculation = SpeculationStats(proposed_token_count, accepted_proposal_count,
                                   speculative_iteration_count, checkpoint.report())
    timings = None
    if time_to_first_token is not None:
        # Cache work after the last token read (such as a recovery snapshot)
        # may still be queued; wait so it is counted.
        _synchronize_device(device)
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
            draft_seconds=draft_seconds if measure else None,
            verify_seconds=verify_seconds if measure else None,
            mask_seconds=sum(mask_times) if measure else None,
            verification_replays=checkpoint.replays,
            canonical_replay_tokens=checkpoint.replayed_tokens,
            replay_stats=speculation.replay_stats,
        )
    yield GenerationFinishedEvent(
        GenerationResult(generated, past_key_values, finish_reason, timings, speculation)
    )


def generate_speculative(
    draft_model: PreTrainedModel | None,
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


def _finish_text_stop(event_iterator, token_ids: list[int]) -> GenerationFinishedEvent:
    """Finalize native generators; foreign iterators have no cache or timings."""
    signal = _TextStop(token_ids)
    throw = getattr(event_iterator, "throw", None)
    if throw is not None:
        try:
            terminal = throw(signal)
        except _TextStop as escaped:
            if escaped is not signal:
                raise
        else:
            if (not isinstance(terminal, GenerationFinishedEvent)
                    or terminal.result.token_ids != token_ids
                    or terminal.result.finish_reason != "stop"):
                raise RuntimeError("Generation did not finalize at the decoded text stop")
            return terminal
    return GenerationFinishedEvent(GenerationResult(list(token_ids), None, "stop"))


def decode_speculative_events(
    events: Iterable[AcceptedTokenEvent | GenerationFinishedEvent],
    tokenizer,
    stop: list[str] | None = None,
) -> Iterator[TextDeltaEvent | GenerationFinishedEvent]:
    """Decode events, stopping the producer as soon as a text stop is detected.

    Native producers retain cache and timing metadata when stopped. For a plain
    iterable, the early terminal result has only the consumed token IDs; cache
    and timings are unavailable. Stop-containing tokens remain in this prefix.
    """
    token_ids: list[int] = []
    decoded = ""
    emitted = ""
    pending = ""
    terminal_seen = False
    source_closed = False
    event_iterator = iter(events)
    try:
        for event in event_iterator:
            if terminal_seen:
                raise RuntimeError("Generation emitted events after completion")
            if isinstance(event, AcceptedTokenEvent):
                token_ids.append(event.token_id)
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
                if stopped:
                    terminal = _finish_text_stop(event_iterator, token_ids)
                    close = getattr(event_iterator, "close", None)
                    source_closed = True
                    if close is not None:
                        close()
                    if text:
                        yield TextDeltaEvent(text)
                    yield terminal
                    return
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
                event = GenerationFinishedEvent(event.result._replace(finish_reason="stop"))
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
        if close is not None and not source_closed:
            close()
