from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import (
    GenerationResult,
    StreamingCleanupError,
    StreamingInvariantError,
    TargetTextEngine,
    TemperatureTopPSelection,
    TextGenerationComplete,
    TextGenerationDelta,
    TextGenerationResult,
    create_reference_sampler,
    select_highest_logit,
)
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    FakeCharacterTokenizer,
    create_deterministic_metrics_session,
    deterministic_target_metrics,
)
from onyx_cuda.testing import ScriptExhaustedError


VOCABULARY = ("P", "a", "b", "c", "!")
SCRIPT = (
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
)


def make_engine(script=SCRIPT, *, tokenizer=None, create_sampling_selector=None):
    backend = FakeAutoregressiveBackend(script, model_id="stream-target")
    tokenizer = tokenizer or FakeCharacterTokenizer(
        VOCABULARY,
        tokenizer_id="stream-tokenizer",
    )
    engine = TargetTextEngine(
        backend,
        tokenizer,
        select_token=select_highest_logit,
        create_sampling_selector=create_sampling_selector,
        create_metrics_session=create_deterministic_metrics_session,
    )
    return engine, backend


def collect_stream(engine, prompt="P", **kwargs):
    events = list(engine.stream(prompt, **kwargs))
    deltas = tuple(
        event.text for event in events if isinstance(event, TextGenerationDelta)
    )
    completions = tuple(
        event.result for event in events if isinstance(event, TextGenerationComplete)
    )
    assert len(completions) == 1
    assert isinstance(events[-1], TextGenerationComplete)
    return "".join(deltas), completions[0], events


def test_greedy_stream_content_and_completion_equal_non_streaming_result():
    engine, backend = make_engine()
    expected = engine.generate("P", max_new_tokens=3)

    text, completed, events = collect_stream(engine, max_new_tokens=3)

    assert text == expected.text == "abc"
    assert completed == expected
    assert [type(event) for event in events] == [
        TextGenerationDelta,
        TextGenerationDelta,
        TextGenerationDelta,
        TextGenerationComplete,
    ]
    assert backend.cache_length == expected.generation.final_cache_length == 3


def test_seeded_sampling_stream_replays_and_equals_non_streaming_result():
    engine, _ = make_engine(create_sampling_selector=create_reference_sampler)
    policy = TemperatureTopPSelection(temperature=0.7, top_p=0.9, seed=17)
    expected = engine.generate("P", max_new_tokens=3, selection=policy)

    first_text, first_result, _ = collect_stream(
        engine,
        max_new_tokens=3,
        selection=policy,
    )
    second_text, second_result, _ = collect_stream(
        engine,
        max_new_tokens=3,
        selection=policy,
    )

    assert first_text == second_text == expected.text
    assert first_result == second_result == expected


def test_multi_token_stop_prefix_is_buffered_and_never_emitted():
    script = (
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    engine, _ = make_engine(script)

    text, result, events = collect_stream(
        engine,
        max_new_tokens=3,
        stop_token_sequences=((2, 4),),
    )

    assert text == result.text == "a"
    assert [event.text for event in events if isinstance(event, TextGenerationDelta)] == [
        "a"
    ]
    assert result.output_token_ids == (1,)
    assert result.sampled_token_ids == (1, 2, 4)
    assert result.generation.matched_stop_token_ids == (2, 4)


def test_ordered_overlapping_stop_releases_only_nonmatching_prefix():
    script = (
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    engine, _ = make_engine(script)

    text, result, _ = collect_stream(
        engine,
        max_new_tokens=3,
        stop_token_sequences=((2, 4), (1, 2, 4)),
    )

    assert text == "a"
    assert result.generation.matched_stop_token_ids == (2, 4)


def test_incomplete_stop_prefix_flushes_at_length_boundary():
    engine, _ = make_engine(SCRIPT[:2])

    text, result, events = collect_stream(
        engine,
        max_new_tokens=2,
        stop_token_sequences=((2, 3),),
    )

    assert text == result.text == "ab"
    assert [event.text for event in events if isinstance(event, TextGenerationDelta)] == [
        "a",
        "b",
    ]
    assert result.generation.finish_reason == "length"


def test_immediate_stop_yields_only_completion_with_empty_visible_output():
    engine, backend = make_engine(((0.0, 0.0, 0.0, 0.0, 1.0),))

    text, result, events = collect_stream(
        engine,
        max_new_tokens=2,
        stop_token_sequences=((4,),),
    )

    assert text == result.text == ""
    assert result.output_token_ids == ()
    assert result.sampled_token_ids == (4,)
    assert events == [TextGenerationComplete(result)]
    assert backend.cache_length == 1


class SplitUtf8Tokenizer:
    tokenizer_id = "split-utf8"
    vocab_size = 4

    def encode(self, text, /):
        if text != "P":
            raise ValueError("expected test prompt")
        return (0,)

    def decode(self, token_ids, /):
        decoded = {
            (): "",
            (1,): "\ufffd",
            (1, 2): "\ufffd",
            (1, 2, 3): "\u4f60",
        }
        return decoded[tuple(token_ids)]


def test_split_utf8_text_is_withheld_until_cumulative_decode_is_stable():
    script = (
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    engine, _ = make_engine(script, tokenizer=SplitUtf8Tokenizer())

    text, result, events = collect_stream(engine, max_new_tokens=3)

    assert text == result.text == "\u4f60"
    assert [event.text for event in events if isinstance(event, TextGenerationDelta)] == [
        "\u4f60"
    ]


class NonMonotonicTokenizer(SplitUtf8Tokenizer):
    tokenizer_id = "non-monotonic"

    def decode(self, token_ids, /):
        decoded = {(): "", (1,): "a", (1, 2): "b"}
        return decoded[tuple(token_ids)]


def test_non_monotonic_tokenizer_decode_fails_and_resets_backend():
    script = ((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    engine, backend = make_engine(script, tokenizer=NonMonotonicTokenizer())
    stream = engine.stream("P", max_new_tokens=2)

    assert next(stream) == TextGenerationDelta("a")
    with pytest.raises(StreamingInvariantError, match="already emitted"):
        next(stream)

    assert backend.cache_length == 0


def test_backend_failure_during_stream_resets_partial_generation_state():
    engine, backend = make_engine(SCRIPT[:1])
    stream = engine.stream("P", max_new_tokens=2)

    assert next(stream) == TextGenerationDelta("a")
    with pytest.raises(ScriptExhaustedError):
        next(stream)

    assert backend.cache_length == 0


def test_early_consumer_cancellation_resets_backend_and_is_idempotent():
    engine, backend = make_engine()
    stream = engine.stream("P", max_new_tokens=3)

    assert next(stream) == TextGenerationDelta("a")
    assert backend.cache_length == 1
    stream.close()
    stream.close()

    assert backend.cache_length == 0


def test_cancellation_after_final_delta_but_before_completion_resets_backend():
    engine, backend = make_engine(SCRIPT[:1])
    stream = engine.stream("P", max_new_tokens=1)

    assert next(stream) == TextGenerationDelta("a")
    assert backend.cache_length == 1
    stream.close()

    assert backend.cache_length == 0


class FailingResetBackend(FakeAutoregressiveBackend):
    fail_reset = False

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset failed")
        super().reset()


def test_cancellation_reports_reset_failure():
    backend = FailingResetBackend(SCRIPT, model_id="stream-target")
    tokenizer = FakeCharacterTokenizer(VOCABULARY)
    engine = TargetTextEngine(backend, tokenizer, select_token=select_highest_logit)
    stream = engine.stream("P", max_new_tokens=3)

    assert next(stream) == TextGenerationDelta("a")
    backend.fail_reset = True
    with pytest.raises(StreamingCleanupError, match="reset failed"):
        stream.close()


def test_stream_events_and_terminal_result_are_immutable():
    generation = GenerationResult(
        "target",
        (1,),
        "length",
        1,
        1,
        deterministic_target_metrics(1),
    )
    result = TextGenerationResult("tokenizer", "a", (1,), generation)
    delta = TextGenerationDelta("a")
    complete = TextGenerationComplete(result)

    with pytest.raises(FrozenInstanceError):
        delta.text = "b"
    with pytest.raises(FrozenInstanceError):
        complete.result = result
    with pytest.raises(ValueError, match="cannot be empty"):
        TextGenerationDelta("")
    with pytest.raises(TypeError, match="must be TextGenerationResult"):
        TextGenerationComplete(None)


@pytest.mark.parametrize("max_new_tokens", [0, -1, True, 1.5])
def test_stream_validates_generation_limit_before_returning_iterator(max_new_tokens):
    engine, backend = make_engine()

    with pytest.raises((TypeError, ValueError)):
        engine.stream("P", max_new_tokens=max_new_tokens)

    assert backend.cache_length == 0
