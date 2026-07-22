from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import (
    GenerationResult,
    SelectionError,
    TargetTextEngine,
    TemperatureTopPSelection,
    TextGenerationResult,
    UnknownTextTokenError,
    VocabularyMismatchError,
    create_reference_sampler,
    select_highest_logit,
)
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    FakeCharacterTokenizer,
    create_deterministic_metrics_session,
    deterministic_target_metrics,
)


VOCABULARY = ("P", "a", "b", "c", "!")
SCRIPT = (
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
)


def make_engine(script=SCRIPT, *, create_sampling_selector=None):
    backend = FakeAutoregressiveBackend(script, model_id="text-target")
    tokenizer = FakeCharacterTokenizer(VOCABULARY, tokenizer_id="text-tokenizer")
    return (
        TargetTextEngine(
            backend,
            tokenizer,
            select_token=select_highest_logit,
            create_sampling_selector=create_sampling_selector,
            create_metrics_session=create_deterministic_metrics_session,
        ),
        backend,
    )


class ReportedVocabBackend(FakeAutoregressiveBackend):
    def __init__(self, reported_vocab_size):
        super().__init__(((0.0,),))
        self.reported_vocab_size = reported_vocab_size

    @property
    def vocab_size(self):
        return self.reported_vocab_size


class ReportedVocabTokenizer:
    tokenizer_id = "reported-vocab-tokenizer"

    def __init__(self, reported_vocab_size):
        self.reported_vocab_size = reported_vocab_size

    @property
    def vocab_size(self):
        return self.reported_vocab_size

    def encode(self, text, /):
        raise AssertionError("configuration validation must not encode text")

    def decode(self, token_ids, /):
        raise AssertionError("configuration validation must not decode tokens")


def test_generates_text_and_preserves_token_metadata():
    engine, backend = make_engine()

    result = engine.generate("P", max_new_tokens=3)

    assert result == TextGenerationResult(
        tokenizer_id="text-tokenizer",
        text="abc",
        output_token_ids=(1, 2, 3),
        generation=GenerationResult(
            model_id="text-target",
            token_ids=(1, 2, 3),
            finish_reason="length",
            prompt_tokens=1,
            final_cache_length=3,
            metrics=deterministic_target_metrics(3),
        ),
    )
    assert result.sampled_token_ids == (1, 2, 3)
    assert result.generated_tokens == 3
    assert backend.cache_length == 3
    assert engine.model_id == "text-target"
    assert engine.tokenizer_id == "text-tokenizer"
    assert engine.vocab_size == len(VOCABULARY)


def test_trims_terminal_single_token_stop_but_preserves_sampled_token():
    script = (
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    engine, _ = make_engine(script)

    result = engine.generate("P", max_new_tokens=3, stop_token_sequences=((4,),))

    assert result.text == "a"
    assert result.output_token_ids == (1,)
    assert result.sampled_token_ids == (1, 4)
    assert result.generation.finish_reason == "stop"
    assert result.generation.matched_stop_token_ids == (4,)
    assert result.generated_tokens == 2


def test_first_generated_stop_token_produces_empty_text():
    script = ((0.0, 0.0, 0.0, 0.0, 1.0),)
    engine, backend = make_engine(script)

    result = engine.generate("P", max_new_tokens=2, stop_token_sequences=((4,),))

    assert result.text == ""
    assert result.output_token_ids == ()
    assert result.sampled_token_ids == (4,)
    assert result.generation.final_cache_length == 1
    assert backend.cache_length == 1


def test_trims_complete_multi_token_stop_from_visible_output():
    script = (
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    engine, backend = make_engine(script)

    result = engine.generate(
        "P",
        max_new_tokens=3,
        stop_token_sequences=((2, 4),),
    )

    assert result.text == "a"
    assert result.output_token_ids == (1,)
    assert result.sampled_token_ids == (1, 2, 4)
    assert result.generation.matched_stop_token_ids == (2, 4)
    assert result.generation.final_cache_length == 3
    assert backend.cache_length == 3


@pytest.mark.parametrize(
    ("stops", "expected_text", "expected_output"),
    [
        (((2, 4), (1, 2, 4)), "a", (1,)),
        (((1, 2, 4), (2, 4)), "", ()),
    ],
)
def test_overlapping_stop_order_controls_exact_visible_trim(
    stops,
    expected_text,
    expected_output,
):
    script = (
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    engine, _ = make_engine(script)

    result = engine.generate(
        "P",
        max_new_tokens=3,
        stop_token_sequences=stops,
    )

    assert result.text == expected_text
    assert result.output_token_ids == expected_output
    assert result.generation.matched_stop_token_ids == stops[0]


def test_incomplete_stop_prefix_remains_visible_at_length_boundary():
    engine, _ = make_engine(SCRIPT[:2])

    result = engine.generate(
        "P",
        max_new_tokens=2,
        stop_token_sequences=((2, 3),),
    )

    assert result.text == "ab"
    assert result.output_token_ids == (1, 2)
    assert result.sampled_token_ids == (1, 2)
    assert result.generation.finish_reason == "length"
    assert result.generation.matched_stop_token_ids is None


def test_repeated_calls_restart_deterministically():
    engine, _ = make_engine()

    first = engine.generate("P", max_new_tokens=3)
    second = engine.generate("P", max_new_tokens=3)

    assert first == second


def test_seeded_sampling_restarts_exactly_and_preserves_metadata():
    engine, backend = make_engine(create_sampling_selector=create_reference_sampler)
    policy = TemperatureTopPSelection(temperature=0.7, top_p=0.9, seed=17)

    first = engine.generate("P", max_new_tokens=3, selection=policy)
    second = engine.generate("P", max_new_tokens=3, selection=policy)

    assert first == second
    assert first.text == "Pc!"
    assert first.sampled_token_ids == (0, 3, 4)
    assert first.generation.finish_reason == "length"
    assert first.generation.final_cache_length == 3
    assert backend.cache_length == 3


def test_seeded_sampling_preserves_current_stop_and_final_cache_semantics():
    engine, _ = make_engine(create_sampling_selector=create_reference_sampler)
    policy = TemperatureTopPSelection(temperature=0.7, top_p=0.9, seed=17)

    result = engine.generate(
        "P",
        max_new_tokens=3,
        stop_token_sequences=((4,),),
        selection=policy,
    )

    assert result.text == "Pc"
    assert result.output_token_ids == (0, 3)
    assert result.sampled_token_ids == (0, 3, 4)
    assert result.generation.finish_reason == "stop"
    assert result.generation.matched_stop_token_ids == (4,)
    assert result.generation.final_cache_length == 3


def test_sampling_must_be_configured_before_backend_prefill():
    engine, backend = make_engine()

    with pytest.raises(SelectionError, match="not configured"):
        engine.generate(
            "P",
            max_new_tokens=1,
            selection=TemperatureTopPSelection(1.0, 1.0, 0),
        )

    assert backend.cache_length == 0


def test_sampling_factory_must_return_callable_before_backend_prefill():
    engine, backend = make_engine(create_sampling_selector=lambda policy: None)

    with pytest.raises(TypeError, match="must return a callable"):
        engine.generate(
            "P",
            max_new_tokens=1,
            selection=TemperatureTopPSelection(1.0, 1.0, 0),
        )

    assert backend.cache_length == 0


def test_unknown_selection_policy_fails_before_prompt_or_prefill():
    engine, backend = make_engine()

    with pytest.raises(TypeError, match="selection must be"):
        engine.generate("P", max_new_tokens=1, selection=None)

    assert backend.cache_length == 0


def test_rejects_backend_and_tokenizer_vocabulary_mismatch():
    backend = FakeAutoregressiveBackend(((0.0, 1.0),))
    tokenizer = FakeCharacterTokenizer(("a", "b", "c"))

    with pytest.raises(VocabularyMismatchError, match="2 does not match.*3"):
        TargetTextEngine(backend, tokenizer, select_token=select_highest_logit)

    assert backend.cache_length == 0


def test_rejects_non_callable_selector_during_configuration():
    backend = FakeAutoregressiveBackend(((0.0, 1.0),))
    tokenizer = FakeCharacterTokenizer(("a", "b"))

    with pytest.raises(TypeError, match="select_token must be callable"):
        TargetTextEngine(backend, tokenizer, select_token=None)


def test_rejects_empty_prompt_before_backend_prefill():
    engine, backend = make_engine()

    with pytest.raises(ValueError, match="at least one token"):
        engine.generate("", max_new_tokens=1)

    assert backend.cache_length == 0


def test_propagates_unknown_prompt_character_before_backend_prefill():
    engine, backend = make_engine()

    with pytest.raises(UnknownTextTokenError, match="position 1"):
        engine.generate("Px", max_new_tokens=1)

    assert backend.cache_length == 0


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("backend", 0, ValueError),
        ("backend", True, TypeError),
        ("tokenizer", -1, ValueError),
        ("tokenizer", 5.0, TypeError),
    ],
)
def test_rejects_invalid_component_vocabulary_sizes(field, value, error_type):
    if field == "backend":
        backend = ReportedVocabBackend(value)
        tokenizer = FakeCharacterTokenizer(("a",))
    else:
        backend = FakeAutoregressiveBackend(((0.0,),))
        tokenizer = ReportedVocabTokenizer(value)

    with pytest.raises(error_type, match=f"{field} vocabulary size"):
        TargetTextEngine(backend, tokenizer, select_token=select_highest_logit)


def test_text_generation_result_validates_stop_trimming_and_is_immutable():
    generation = GenerationResult(
        "target",
        (1, 4),
        "stop",
        1,
        2,
        deterministic_target_metrics(2),
        (4,),
    )

    with pytest.raises(ValueError, match="finish reason"):
        TextGenerationResult("tokenizer", "a!", (1, 4), generation)

    result = TextGenerationResult("tokenizer", "a", (1,), generation)
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"
