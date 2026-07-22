from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import (
    GenerationCleanupError,
    GenerationInvariantError,
    GenerationResult,
    ModelStep,
    TemperatureTopPSelection,
    create_reference_sampler,
    generate_greedy as _generate_greedy,
    generate_target as _generate_target,
    select_highest_logit,
)
from onyx_cuda.generation import TargetGenerationStep, iterate_target as _iterate_target
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    ScriptExhaustedError,
    create_deterministic_metrics_session,
    deterministic_target_metrics,
)


SCRIPT = (
    (0.1, 0.2, 0.7),
    (0.8, 0.1, 0.1),
    (0.2, 0.6, 0.2),
)


def generate_greedy(*args, **kwargs):
    kwargs.setdefault("metrics_session", create_deterministic_metrics_session())
    return _generate_greedy(*args, **kwargs)


def generate_target(*args, **kwargs):
    kwargs.setdefault("metrics_session", create_deterministic_metrics_session())
    return _generate_target(*args, **kwargs)


def iterate_target(*args, **kwargs):
    kwargs.setdefault("metrics_session", create_deterministic_metrics_session())
    return _iterate_target(*args, **kwargs)


def test_generates_greedy_tokens_until_length_limit():
    backend = FakeAutoregressiveBackend(SCRIPT, model_id="target-test")

    result = generate_greedy(
        backend,
        [1, 2],
        max_new_tokens=3,
        select_token=select_highest_logit,
    )

    assert result == GenerationResult(
        model_id="target-test",
        token_ids=(2, 0, 1),
        finish_reason="length",
        prompt_tokens=2,
        final_cache_length=4,
        metrics=deterministic_target_metrics(3),
    )
    assert result.generated_tokens == 3


def test_general_target_loop_accepts_one_seeded_sampling_session():
    backend = FakeAutoregressiveBackend(SCRIPT, model_id="sampled-target")
    select_token = create_reference_sampler(
        TemperatureTopPSelection(temperature=0.7, top_p=0.9, seed=17)
    )

    result = generate_target(
        backend,
        [1, 2],
        max_new_tokens=3,
        select_token=select_token,
    )

    assert result == GenerationResult(
        model_id="sampled-target",
        token_ids=(2, 2, 2),
        finish_reason="length",
        prompt_tokens=2,
        final_cache_length=4,
        metrics=deterministic_target_metrics(3),
    )


def test_target_iterator_yields_each_selection_and_terminal_result():
    backend = FakeAutoregressiveBackend(SCRIPT, model_id="iterated-target")

    steps = list(
        iterate_target(
            backend,
            [1, 2],
            max_new_tokens=3,
            select_token=select_highest_logit,
        )
    )

    assert steps[:-1] == [TargetGenerationStep(2), TargetGenerationStep(0)]
    assert steps[-1] == TargetGenerationStep(
        1,
        GenerationResult(
            model_id="iterated-target",
            token_ids=(2, 0, 1),
            finish_reason="length",
            prompt_tokens=2,
            final_cache_length=4,
            metrics=deterministic_target_metrics(3),
        ),
    )
    assert backend.cache_length == 4


def test_closing_target_iterator_early_resets_backend():
    backend = FakeAutoregressiveBackend(SCRIPT)
    steps = iterate_target(
        backend,
        [0],
        max_new_tokens=3,
        select_token=select_highest_logit,
    )

    assert next(steps) == TargetGenerationStep(2)
    assert backend.cache_length == 1
    steps.close()

    assert backend.cache_length == 0


def test_stops_on_first_selected_single_token_sequence_without_decode():
    backend = FakeAutoregressiveBackend([SCRIPT[0]])

    result = generate_greedy(
        backend,
        [0, 1],
        max_new_tokens=3,
        select_token=select_highest_logit,
        stop_token_sequences=((2,),),
    )

    assert result.token_ids == (2,)
    assert result.finish_reason == "stop"
    assert result.matched_stop_token_ids == (2,)
    assert result.final_cache_length == 2


def test_stops_after_later_generated_single_token_sequence():
    backend = FakeAutoregressiveBackend(SCRIPT)

    result = generate_greedy(
        backend,
        [1],
        max_new_tokens=3,
        select_token=select_highest_logit,
        stop_token_sequences=((0,),),
    )

    assert result.token_ids == (2, 0)
    assert result.finish_reason == "stop"
    assert result.matched_stop_token_ids == (0,)
    assert result.final_cache_length == 2


def test_multi_token_stop_completing_at_length_boundary_finishes_as_stop():
    backend = FakeAutoregressiveBackend(SCRIPT)

    result = generate_target(
        backend,
        [1],
        max_new_tokens=2,
        select_token=select_highest_logit,
        stop_token_sequences=((2, 0),),
    )

    assert result == GenerationResult(
        model_id="fake-target",
        token_ids=(2, 0),
        finish_reason="stop",
        prompt_tokens=1,
        final_cache_length=2,
        metrics=deterministic_target_metrics(2),
        matched_stop_token_ids=(2, 0),
    )


def test_incomplete_stop_prefix_at_length_boundary_remains_a_length_finish():
    backend = FakeAutoregressiveBackend(SCRIPT)

    result = generate_target(
        backend,
        [1],
        max_new_tokens=2,
        select_token=select_highest_logit,
        stop_token_sequences=((2, 0, 1),),
    )

    assert result.token_ids == (2, 0)
    assert result.finish_reason == "length"
    assert result.matched_stop_token_ids is None
    assert result.final_cache_length == 2


def test_prefix_sharing_stops_as_soon_as_a_complete_sequence_matches():
    backend = FakeAutoregressiveBackend(SCRIPT[:2])

    result = generate_target(
        backend,
        [1],
        max_new_tokens=3,
        select_token=select_highest_logit,
        stop_token_sequences=((2, 0, 1), (2, 0)),
    )

    assert result.token_ids == (2, 0)
    assert result.matched_stop_token_ids == (2, 0)
    assert result.final_cache_length == 2


def test_first_configured_complete_suffix_wins_for_overlapping_stops():
    shorter_first = FakeAutoregressiveBackend(SCRIPT)
    longer_first = FakeAutoregressiveBackend(SCRIPT)

    shorter = generate_target(
        shorter_first,
        [1],
        max_new_tokens=3,
        select_token=select_highest_logit,
        stop_token_sequences=((0, 1), (2, 0, 1)),
    )
    longer = generate_target(
        longer_first,
        [1],
        max_new_tokens=3,
        select_token=select_highest_logit,
        stop_token_sequences=((2, 0, 1), (0, 1)),
    )

    assert shorter.token_ids == longer.token_ids == (2, 0, 1)
    assert shorter.matched_stop_token_ids == (0, 1)
    assert longer.matched_stop_token_ids == (2, 0, 1)
    assert shorter.final_cache_length == longer.final_cache_length == 3


def test_one_token_limit_does_not_request_decode():
    backend = FakeAutoregressiveBackend([SCRIPT[0]])

    result = generate_greedy(
        backend,
        [0],
        max_new_tokens=1,
        select_token=select_highest_logit,
    )

    assert result.token_ids == (2,)
    assert result.final_cache_length == 1


def test_reference_selector_uses_first_index_for_ties_and_supports_infinity():
    assert select_highest_logit([-3.0, 2.0, 2.0]) == 1
    assert select_highest_logit([float("-inf"), float("inf")]) == 1


@pytest.mark.parametrize("logits", [[], [0.1, float("nan")], [0.1, "invalid"]])
def test_reference_selector_rejects_invalid_logits(logits):
    with pytest.raises(ValueError):
        select_highest_logit(logits)


@pytest.mark.parametrize("max_new_tokens", [0, -1, True, 1.5])
def test_rejects_invalid_generation_limits(max_new_tokens):
    backend = FakeAutoregressiveBackend(SCRIPT)

    with pytest.raises((TypeError, ValueError)):
        generate_greedy(
            backend,
            [0],
            max_new_tokens=max_new_tokens,
            select_token=select_highest_logit,
        )


@pytest.mark.parametrize("prompt", [[], [-1], [3], [True], ["1"]])
def test_rejects_invalid_prompt_tokens(prompt):
    backend = FakeAutoregressiveBackend(SCRIPT)

    with pytest.raises((TypeError, ValueError)):
        generate_greedy(
            backend,
            prompt,
            max_new_tokens=1,
            select_token=select_highest_logit,
        )


@pytest.mark.parametrize(
    "stops",
    [
        None,
        {(1,)},
        (1,),
        ((),),
        ({1},),
        ((-1,),),
        ((3,),),
        ((True,),),
        (("1",),),
    ],
)
def test_rejects_invalid_stop_token_sequences_before_prefill(stops):
    backend = FakeAutoregressiveBackend(SCRIPT)

    with pytest.raises((TypeError, ValueError)):
        generate_greedy(
            backend,
            [0],
            max_new_tokens=1,
            select_token=select_highest_logit,
            stop_token_sequences=stops,
        )

    assert backend.cache_length == 0


@pytest.mark.parametrize("selected_token", [-1, 3, True, "1"])
def test_rejects_invalid_selector_results(selected_token):
    backend = FakeAutoregressiveBackend(SCRIPT)

    with pytest.raises((TypeError, ValueError)):
        generate_greedy(
            backend,
            [0],
            max_new_tokens=1,
            select_token=lambda logits: selected_token,
        )

    assert backend.cache_length == 0


def test_rejects_non_callable_selector_before_prefill():
    backend = FakeAutoregressiveBackend(SCRIPT)

    with pytest.raises(TypeError, match="select_token must be callable"):
        generate_greedy(backend, [0], max_new_tokens=1, select_token=None)

    assert backend.cache_length == 0


class IncorrectCacheBackend(FakeAutoregressiveBackend):
    def prefill(self, prompt_token_ids, /):
        step = super().prefill(prompt_token_ids)
        return ModelStep(logits=step.logits, cache_length=step.cache_length + 1)


class InvalidVocabBackend(FakeAutoregressiveBackend):
    def __init__(self, invalid_vocab_size):
        super().__init__(SCRIPT)
        self.invalid_vocab_size = invalid_vocab_size

    @property
    def vocab_size(self):
        return self.invalid_vocab_size


class FailingResetBackend(FakeAutoregressiveBackend):
    fail_reset = False

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset failed")
        super().reset()


def test_detects_backend_cache_contract_violation():
    backend = IncorrectCacheBackend(SCRIPT)

    with pytest.raises(GenerationInvariantError, match="expected 1"):
        generate_greedy(
            backend,
            [0],
            max_new_tokens=1,
            select_token=select_highest_logit,
        )

    assert backend.cache_length == 0


def test_target_iterator_reports_cancellation_reset_failure():
    backend = FailingResetBackend(SCRIPT)
    steps = iterate_target(
        backend,
        [0],
        max_new_tokens=3,
        select_token=select_highest_logit,
    )
    assert next(steps) == TargetGenerationStep(2)
    backend.fail_reset = True

    with pytest.raises(GenerationCleanupError, match="reset failed"):
        steps.close()


def test_target_iterator_reports_execution_and_reset_failures_together():
    backend = FailingResetBackend(SCRIPT)

    def fail_selection(logits):
        backend.fail_reset = True
        raise RuntimeError("selection failed")

    steps = iterate_target(
        backend,
        [0],
        max_new_tokens=3,
        select_token=fail_selection,
    )
    with pytest.raises(
        GenerationCleanupError,
        match="selection failed.*reset also failed.*reset failed",
    ):
        next(steps)


@pytest.mark.parametrize("vocab_size", [0, -1, True, 3.0, "3"])
def test_detects_invalid_backend_vocabulary_size_before_prefill(vocab_size):
    backend = InvalidVocabBackend(vocab_size)

    with pytest.raises(GenerationInvariantError, match="vocab_size"):
        generate_greedy(
            backend,
            [0],
            max_new_tokens=1,
            select_token=select_highest_logit,
        )

    assert backend.cache_length == 0


def test_backend_exhaustion_propagates_without_extra_cache_growth():
    backend = FakeAutoregressiveBackend([SCRIPT[0]])

    with pytest.raises(ScriptExhaustedError):
        generate_greedy(
            backend,
            [0],
            max_new_tokens=2,
            select_token=select_highest_logit,
        )

    assert backend.cache_length == 0


def test_generation_result_is_immutable():
    result = GenerationResult(
        "target",
        (1,),
        "length",
        2,
        2,
        deterministic_target_metrics(1),
    )

    with pytest.raises(FrozenInstanceError):
        result.finish_reason = "stop"


def test_generation_result_validates_matched_stop_metadata():
    metrics = deterministic_target_metrics(1)
    with pytest.raises(ValueError, match="cannot report"):
        GenerationResult("target", (1,), "length", 2, 2, metrics, (1,))
    with pytest.raises(TypeError, match="as a tuple"):
        GenerationResult("target", (1,), "stop", 2, 2, metrics, [1])
    with pytest.raises(ValueError, match="nonempty"):
        GenerationResult("target", (1,), "stop", 2, 2, metrics, ())
    with pytest.raises(ValueError, match="suffix"):
        GenerationResult(
            "target",
            (1, 2),
            "stop",
            2,
            3,
            deterministic_target_metrics(2),
            (1,),
        )
