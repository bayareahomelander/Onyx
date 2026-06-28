import os
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import probe_cuda_real_logits
import onyx_cuda.real_logits_handoff as real_logits_handoff
from onyx_cuda.real_logits_handoff import (
    DEFAULT_REGEX,
    run_real_logits_handoff,
    select_and_advance_one_token,
    validate_real_logits,
)


class FakeTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def decode(self, token_ids):
        return b"".join(self.vocabulary[token_id] for token_id in token_ids).decode("utf-8")


class RecordingGrammar:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.released_states = []
        self.states = {1: b""}

    def compile_regex(self, pattern):
        assert pattern == DEFAULT_REGEX

    def init_state(self):
        return 1

    def get_valid_token_ids(self, state):
        assert state == 1
        return [1, 2]

    def advance_state(self, state, token_id):
        assert state == 1
        self.states[2] = self.vocabulary[token_id]
        return 2

    def is_match_state(self, state):
        return state == 2 and self.states[state] in (b"1", b"2")

    def release_states(self, states):
        self.released_states.extend(states)


def test_one_step_selection_excludes_invalid_raw_argmax_and_releases_states():
    torch = pytest.importorskip("torch")
    vocabulary = [b"x", b"1", b"2"]
    grammar = RecordingGrammar(vocabulary)
    logits = torch.tensor([10.0, 1.0, 3.0])

    def reference_selector(values, valid_ids, *, check_inputs):
        assert check_inputs is True
        selected = max(valid_ids, key=lambda token_id: float(values[token_id]))
        return torch.tensor([selected])

    result = select_and_advance_one_token(
        logits,
        vocabulary=vocabulary,
        tokenizer=FakeTokenizer(vocabulary),
        regex=DEFAULT_REGEX,
        grammar_factory=lambda _vocabulary: grammar,
        selector=reference_selector,
    )

    assert result.raw_argmax_token_id == 0
    assert result.raw_argmax_was_valid is False
    assert result.selected_token_id == 2
    assert result.selected_token_text == "2"
    assert result.selected_token_bytes_hex == "32"
    assert result.selected_token_was_valid is True
    assert result.invalid_raw_argmax_excluded is True
    assert result.grammar_matched_after_selection is True
    assert grammar.released_states == [1, 2]


def test_one_step_selection_releases_initial_state_when_selector_raises():
    torch = pytest.importorskip("torch")
    vocabulary = [b"x", b"1", b"2"]
    grammar = RecordingGrammar(vocabulary)

    def failing_selector(*_args, **_kwargs):
        raise RuntimeError("synthetic selector failure")

    with pytest.raises(RuntimeError, match="synthetic selector failure"):
        select_and_advance_one_token(
            torch.tensor([10.0, 1.0, 3.0]),
            vocabulary=vocabulary,
            tokenizer=FakeTokenizer(vocabulary),
            regex=DEFAULT_REGEX,
            grammar_factory=lambda _vocabulary: grammar,
            selector=failing_selector,
        )

    assert grammar.released_states == [1]


class FakeLogits:
    is_cuda = True
    ndim = 3

    def __init__(self, dtype, shape=(1, 8, 16)):
        self.dtype = dtype
        self.shape = shape


class FakeTorch:
    float16 = object()
    float32 = object()
    bfloat16 = object()

    @staticmethod
    def is_tensor(_value):
        return True


def test_real_logits_contract_checks_observed_vocabulary_width():
    logits = FakeLogits(FakeTorch.float16)

    assert validate_real_logits(logits, expected_width=16, torch=FakeTorch) == (1, 8, 16)

    with pytest.raises(ValueError, match="does not match configured width"):
        validate_real_logits(logits, expected_width=15, torch=FakeTorch)


def test_real_logits_contract_rejects_unsupported_bfloat16():
    logits = FakeLogits(FakeTorch.bfloat16)

    with pytest.raises(ValueError, match="must use float16 or float32"):
        validate_real_logits(logits, expected_width=16, torch=FakeTorch)


@pytest.mark.parametrize(
    ("selection_mode", "cleanup_fails"),
    [
        ("simple_failure", False),
        ("chained_failure", False),
        ("chained_failure", True),
        ("success", True),
    ],
)
def test_real_logits_handoff_releases_allocations_and_preserves_failures(
    monkeypatch, selection_mode, cleanup_fails
):
    references = {}
    cleanup_calls = []
    selection_failure = RuntimeError("synthetic post-forward selection failure")
    cleanup_failure = RuntimeError("synthetic empty-cache failure")

    class FakeDevice:
        def __str__(self):
            return "cuda:0"

    class FakeCuda:
        @staticmethod
        def device(_device):
            return nullcontext()

        @staticmethod
        def memory_allocated(_device):
            return 0

        @staticmethod
        def memory_reserved(_device):
            return 0

        @staticmethod
        def max_memory_allocated(_device):
            return 0

        @staticmethod
        def reset_peak_memory_stats():
            cleanup_calls.append("reset_peak_memory_stats")

        @staticmethod
        def empty_cache():
            assert set(references) == {
                "model",
                "encoded",
                "outputs",
                "logits",
                "last_logits",
            }
            alive = {name for name, reference in references.items() if reference() is not None}
            assert not alive, f"resources still alive before empty_cache: {sorted(alive)}"
            cleanup_calls.append("empty_cache")
            if cleanup_fails:
                raise cleanup_failure

        @staticmethod
        def synchronize(_device):
            cleanup_calls.append("synchronize")

    class FakeRuntimeTorch:
        cuda = FakeCuda()
        float16 = object()
        float32 = object()

        @staticmethod
        def device(_kind, _index):
            return FakeDevice()

        @staticmethod
        def is_tensor(value):
            return isinstance(value, FakeTensor)

        @staticmethod
        def no_grad():
            return nullcontext()

    class FakeTensor:
        is_cuda = True
        device = FakeDevice()

        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = FakeRuntimeTorch.float16

        def __getitem__(self, _key):
            last_logits = FakeTensor((1, self.shape[-1]))
            references["last_logits"] = weakref.ref(last_logits)
            return last_logits

        def contiguous(self):
            return self

    class FakeEncoded(dict):
        def __init__(self):
            super().__init__(input_ids=SimpleNamespace(shape=(1, 2)))

        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            encoded = FakeEncoded()
            references["encoded"] = weakref.ref(encoded)
            return encoded

    class FakeOutputs:
        def __init__(self):
            self.logits = FakeTensor((1, 2, 4))
            references["logits"] = weakref.ref(self.logits)

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **_kwargs):
            outputs = FakeOutputs()
            references["outputs"] = weakref.ref(outputs)
            return outputs

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            model = FakeModel()
            references["model"] = weakref.ref(model)
            return model

    def load_metadata(*_args, **_kwargs):
        return SimpleNamespace(vocab_size=4), FakeTokenizer(), "resolved-revision"

    def inspect_tokenizer(*_args, **_kwargs):
        return SimpleNamespace(compatible=True, errors=[]), [b"x", b"1", b"2", b"3"]

    def fail_selection(*_args, **_kwargs):
        if selection_mode == "success":
            return SimpleNamespace()
        if selection_mode == "chained_failure":
            try:
                raise ValueError("synthetic chained cause")
            except ValueError as cause:
                raise selection_failure from cause
        raise selection_failure

    original_collect = real_logits_handoff.gc.collect

    def recording_collect():
        cleanup_calls.append("gc.collect")
        return original_collect()

    monkeypatch.setattr(
        real_logits_handoff,
        "_require_runtime",
        lambda: (FakeRuntimeTorch, FakeModelLoader, lambda **_kwargs: object(), object()),
    )
    monkeypatch.setattr(real_logits_handoff, "_load_tokenizer_metadata", load_metadata)
    monkeypatch.setattr(real_logits_handoff, "inspect_loaded_tokenizer", inspect_tokenizer)
    monkeypatch.setattr(real_logits_handoff, "select_and_advance_one_token", fail_selection)
    monkeypatch.setattr(real_logits_handoff.gc, "collect", recording_collect)

    expected_failure = cleanup_failure if selection_mode == "success" else selection_failure
    with pytest.raises(RuntimeError, match=str(expected_failure)) as raised:
        run_real_logits_handoff()

    assert raised.value is expected_failure
    assert set(references) == {"model", "encoded", "outputs", "logits", "last_logits"}
    assert all(reference() is None for reference in references.values())
    assert cleanup_calls[-3:] == ["gc.collect", "empty_cache", "synchronize"]
    if selection_mode == "chained_failure":
        assert isinstance(selection_failure.__cause__, ValueError)
        assert selection_failure.__cause__.__traceback__ is None
    if cleanup_fails:
        assert getattr(expected_failure, "_onyx_cleanup_failures") == (
            "torch.cuda.empty_cache: RuntimeError: synthetic empty-cache failure",
        )


def test_cli_returns_dependency_error_without_writing_report(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(probe_cuda_real_logits, "run_real_logits_handoff", fail)

    assert probe_cuda_real_logits.main([]) == 2


def test_real_logits_handoff_rejects_custom_model_before_runtime_loading(monkeypatch):
    def unexpected_runtime_load():
        pytest.fail("runtime loading must not start for an unsupported model")

    monkeypatch.setattr(real_logits_handoff, "_require_runtime", unexpected_runtime_load)

    with pytest.raises(ValueError, match="only supports the validated model"):
        run_real_logits_handoff("other/model")


def test_cli_uses_fixed_qwen_model_and_pinned_revision(monkeypatch):
    calls = []

    def record(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(probe_cuda_real_logits, "run_real_logits_handoff", record)
    monkeypatch.setattr(probe_cuda_real_logits, "format_real_logits_report", lambda _report: "ok")

    assert probe_cuda_real_logits.main([]) == 0
    assert calls == [
        (
            real_logits_handoff.DEFAULT_MODEL_ID,
            {
                "revision": real_logits_handoff.DEFAULT_MODEL_REVISION,
                "prompt": real_logits_handoff.DEFAULT_PROMPT,
                "regex": real_logits_handoff.DEFAULT_REGEX,
                "local_files_only": False,
                "device_index": 0,
            },
        )
    ]
    assert not hasattr(probe_cuda_real_logits.build_parser().parse_args([]), "model")


@pytest.mark.skipif(
    os.environ.get("ONYX_RUN_REAL_MODEL_TEST") != "1",
    reason="set ONYX_RUN_REAL_MODEL_TEST=1 to load the pinned quantized model",
)
def test_live_quantized_real_logits_handoff():
    report = run_real_logits_handoff(local_files_only=True)

    assert report.passed is True
    assert report.expected_logits_width == 151_936
    assert report.observed_logits_width == 151_936
    assert report.selection.invalid_raw_argmax_excluded is True
    assert report.selection.grammar_matched_after_selection is True
    assert report.memory_snapshots[-1].phase == "after_cleanup"
