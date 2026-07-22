from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import onyx_cuda.candidate_qualification as candidate_module
from onyx_cuda import (
    DEFAULT_TARGET_PROFILE,
    QWEN_3B_CANDIDATE_PROFILE,
    CandidateQualificationCleanupError,
    CandidateQualificationExecutionError,
    CandidateQualificationImportError,
    CandidateQualificationUnavailableError,
    CandidateTargetQualification,
    ModelStep,
    QwenTokenizerFingerprint,
    qualify_qwen_3b_candidate,
)


class FakeDevice:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self

    def item(self):
        return self.value


class FakeLogits:
    dtype = "torch.float16"
    is_cuda = True

    def __init__(self, size, device):
        self.shape = (size,)
        self.device = device

    def isnan(self):
        return FakeScalar(False)


class FakeProcess:
    def __init__(self):
        self.rss = 1_000
        self.peak = 2_000

    def memory_info(self):
        return SimpleNamespace(rss=self.rss, peak_wset=self.peak)


class FakePsutil:
    __version__ = "7.2.2"

    def __init__(self):
        self.process = FakeProcess()

    def Process(self):
        return self.process


class FakeCuda:
    def __init__(self, *, available=True, device_count=1):
        self.available = available
        self.detected_device_count = device_count
        self.allocated = 10
        self.reserved = 20
        self.peak = 10
        self.total = 6_141 * 1024 * 1024
        self.reset_calls = 0
        self.synchronize_calls = 0
        self.empty_cache_calls = 0

    def is_available(self):
        return self.available

    def device_count(self):
        return self.detected_device_count

    def get_device_properties(self, device):
        return SimpleNamespace(name="Fake RTX 4050")

    def reset_peak_memory_stats(self, device):
        self.reset_calls += 1
        self.peak = self.allocated

    def synchronize(self, device):
        self.synchronize_calls += 1

    def empty_cache(self):
        self.empty_cache_calls += 1

    def mem_get_info(self, device):
        return self.total - self.reserved, self.total

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return self.reserved

    def max_memory_allocated(self, device):
        return self.peak


class FakeTorch:
    __version__ = "2.6.0+cu124"

    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCuda()

    def device(self, name):
        return FakeDevice(name)


class FakeTokenizer:
    tokenizer_id = QWEN_3B_CANDIDATE_PROFILE.pinned_id
    vocab_size = 6

    def encode(self, text):
        assert text == "Hello world"
        return (1, 2)


class FakeBackend:
    model_id = QWEN_3B_CANDIDATE_PROFILE.pinned_id
    vocab_size = 6
    model_vocab_size = 8
    padded_vocab_rows = 2
    model_memory_footprint_bytes = 1_234

    def __init__(self, cuda, *, close_residue=(10, 20), prefill_error=None, close_error=None):
        self.cuda = cuda
        self.tokenizer = FakeTokenizer()
        self.cache_length = 0
        self.close_residue = close_residue
        self.prefill_error = prefill_error
        self.close_error = close_error
        self.closed = False
        self.prefill_lengths = []

    @property
    def is_closed(self):
        return self.closed

    def prefill(self, token_ids):
        if self.prefill_error is not None:
            raise self.prefill_error
        self.cache_length = len(token_ids)
        self.prefill_lengths.append(self.cache_length)
        if self.cache_length > 2:
            self.cuda.allocated = 700
            self.cuda.reserved = 900
            self.cuda.peak = 750
        else:
            self.cuda.allocated = 500
            self.cuda.reserved = 650
            self.cuda.peak = 550
        return ModelStep(FakeLogits(self.vocab_size, FakeDevice("cuda:0")), self.cache_length)

    def decode(self, token_id):
        self.cache_length += 1
        self.cuda.allocated = 720
        self.cuda.peak = 780
        return ModelStep(FakeLogits(self.vocab_size, FakeDevice("cuda:0")), self.cache_length)

    def reset(self):
        self.cache_length = 0

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error
        self.cuda.allocated, self.cuda.reserved = self.close_residue


def make_fingerprint(profile, *, vocabulary="vocab", special="special", chat="chat"):
    return QwenTokenizerFingerprint(
        tokenizer_id=profile.pinned_id,
        vocab_size=6,
        base_vocab_size=4,
        eos_token_id=4,
        pad_token_id=5,
        vocabulary_sha256=vocabulary * 64,
        special_tokens_sha256=special * 64,
        chat_template_sha256=chat * 64,
    )


def install_tokenizer_loader(monkeypatch, *, candidate_changes=None):
    calls = []

    def load(profile, *, local_files_only):
        calls.append((profile, local_files_only))
        values = candidate_changes if profile == QWEN_3B_CANDIDATE_PROFILE else None
        fingerprint = make_fingerprint(profile, **(values or {}))
        tokenizer = SimpleNamespace(compatibility_fingerprint=lambda: fingerprint)
        return SimpleNamespace(tokenizer=tokenizer, load_seconds=0.1)

    monkeypatch.setattr(candidate_module, "load_qwen_tokenizer", load)
    return calls


def install_modules(monkeypatch, torch_module, psutil_module):
    modules = {
        "torch": torch_module,
        "transformers": SimpleNamespace(__version__="4.57.6"),
        "bitsandbytes": SimpleNamespace(__version__="0.49.2"),
        "psutil": psutil_module,
    }
    monkeypatch.setattr(candidate_module.importlib, "import_module", modules.__getitem__)


def install_backend_factory(monkeypatch, cuda, *, residues=((10, 20), (10, 20))):
    calls = []
    backends = []

    def load(profile, *, device_index, local_files_only):
        cycle = len(backends)
        calls.append((profile, device_index, local_files_only))
        cuda.allocated = 400
        cuda.reserved = 600
        cuda.peak = 450
        backend = FakeBackend(cuda, close_residue=residues[cycle])
        backends.append(backend)
        return backend

    monkeypatch.setattr(candidate_module, "load_torch_cuda_target", load)
    return calls, backends


def test_candidate_profile_is_separate_and_immutable():
    assert QWEN_3B_CANDIDATE_PROFILE.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert QWEN_3B_CANDIDATE_PROFILE.revision == (
        "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    )
    assert DEFAULT_TARGET_PROFILE.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    with pytest.raises(FrozenInstanceError):
        QWEN_3B_CANDIDATE_PROFILE.revision = "changed"


def test_qualifies_two_complete_candidate_lifecycles(monkeypatch):
    torch_module = FakeTorch()
    psutil_module = FakePsutil()
    install_modules(monkeypatch, torch_module, psutil_module)
    tokenizer_calls = install_tokenizer_loader(monkeypatch)
    backend_calls, backends = install_backend_factory(monkeypatch, torch_module.cuda)

    result = qualify_qwen_3b_candidate(
        device_index=0,
        local_files_only=True,
        total_cache_tokens=8,
    )

    assert isinstance(result, CandidateTargetQualification)
    assert result.model_id == QWEN_3B_CANDIDATE_PROFILE.model_id
    assert result.revision == QWEN_3B_CANDIDATE_PROFILE.revision
    assert result.reference_tokenizer_id == DEFAULT_TARGET_PROFILE.pinned_id
    assert result.quantization == "bitsandbytes-nf4-double-quant"
    assert result.device_name == "Fake RTX 4050"
    assert result.total_cache_tokens == 8
    assert result.tokenizer_compatibility.fully_compatible
    assert len(result.lifecycles) == 2
    assert [item.lifecycle_number for item in result.lifecycles] == [1, 2]
    assert all(item.first_forward_cache_length == 2 for item in result.lifecycles)
    assert all(item.ceiling_prefill_tokens == 7 for item in result.lifecycles)
    assert all(item.ceiling_prefill_cache_length == 7 for item in result.lifecycles)
    assert all(item.ceiling_decode_cache_length == 8 for item in result.lifecycles)
    assert all(item.logits_shape == (6,) for item in result.lifecycles)
    assert all(item.model_memory_footprint_bytes == 1_234 for item in result.lifecycles)
    assert result.peak_vram_bytes == 900
    assert result.peak_process_ram_bytes == 2_000
    assert tokenizer_calls == [
        (DEFAULT_TARGET_PROFILE, True),
        (QWEN_3B_CANDIDATE_PROFILE, True),
    ]
    assert backend_calls == [
        (QWEN_3B_CANDIDATE_PROFILE, 0, True),
        (QWEN_3B_CANDIDATE_PROFILE, 0, True),
    ]
    assert all(backend.closed for backend in backends)
    assert all(backend.prefill_lengths == [2, 7] for backend in backends)
    assert torch_module.cuda.empty_cache_calls == 2
    with pytest.raises(FrozenInstanceError):
        result.device_name = "changed"


def test_tokenizer_compatibility_reports_each_mismatch_independently(monkeypatch):
    install_tokenizer_loader(
        monkeypatch,
        candidate_changes={"vocabulary": "other", "special": "other", "chat": "other"},
    )

    result = candidate_module._compare_tokenizers(local_files_only=True)

    assert not result.token_ids_equal
    assert not result.special_tokens_equal
    assert not result.chat_template_equal
    assert not result.fully_compatible


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"device_index": -1}, ValueError),
        ({"device_index": True}, TypeError),
        ({"local_files_only": None}, TypeError),
        ({"total_cache_tokens": 1}, ValueError),
        ({"total_cache_tokens": 2049}, ValueError),
        ({"total_cache_tokens": True}, TypeError),
    ],
)
def test_invalid_inputs_fail_before_optional_import(monkeypatch, kwargs, error_type):
    monkeypatch.setattr(
        candidate_module.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import of {name}"),
    )

    with pytest.raises(error_type):
        qualify_qwen_3b_candidate(**kwargs)


@pytest.mark.parametrize("missing_name", ["torch", "transformers", "bitsandbytes", "psutil"])
def test_reports_each_missing_optional_dependency(monkeypatch, missing_name):
    def import_module(name):
        if name == missing_name:
            raise ModuleNotFoundError(f"{name} missing")
        return SimpleNamespace()

    monkeypatch.setattr(candidate_module.importlib, "import_module", import_module)

    with pytest.raises(CandidateQualificationImportError, match=f"{missing_name} missing"):
        qualify_qwen_3b_candidate(total_cache_tokens=8)


def test_reports_cuda_unavailable_before_loading_tokenizers(monkeypatch):
    torch_module = FakeTorch(FakeCuda(available=False))
    install_modules(monkeypatch, torch_module, FakePsutil())
    monkeypatch.setattr(
        candidate_module,
        "load_qwen_tokenizer",
        lambda *args, **kwargs: pytest.fail("tokenizer must not load"),
    )

    with pytest.raises(CandidateQualificationUnavailableError, match="CUDA unavailable"):
        qualify_qwen_3b_candidate(total_cache_tokens=8)


def test_reports_unavailable_device_index(monkeypatch):
    torch_module = FakeTorch(FakeCuda(device_count=1))
    install_modules(monkeypatch, torch_module, FakePsutil())

    with pytest.raises(CandidateQualificationUnavailableError, match="detected 1"):
        qualify_qwen_3b_candidate(device_index=1, total_cache_tokens=8)


def test_second_lifecycle_allocator_growth_is_rejected(monkeypatch):
    torch_module = FakeTorch()
    install_modules(monkeypatch, torch_module, FakePsutil())
    install_tokenizer_loader(monkeypatch)
    install_backend_factory(
        monkeypatch,
        torch_module.cuda,
        residues=((10, 20), (11, 21)),
    )

    with pytest.raises(CandidateQualificationCleanupError, match="second candidate lifecycle"):
        qualify_qwen_3b_candidate(total_cache_tokens=8)


def test_execution_and_cleanup_failures_are_both_reported(monkeypatch):
    torch_module = FakeTorch()
    install_modules(monkeypatch, torch_module, FakePsutil())
    install_tokenizer_loader(monkeypatch)
    backend = FakeBackend(
        torch_module.cuda,
        prefill_error=RuntimeError("forward failed"),
        close_error=RuntimeError("close failed"),
    )
    monkeypatch.setattr(candidate_module, "load_torch_cuda_target", lambda *args, **kwargs: backend)

    with pytest.raises(CandidateQualificationCleanupError, match="forward failed.*close failed"):
        qualify_qwen_3b_candidate(total_cache_tokens=8)


def test_successful_execution_close_failure_is_a_cleanup_error(monkeypatch):
    torch_module = FakeTorch()
    install_modules(monkeypatch, torch_module, FakePsutil())
    install_tokenizer_loader(monkeypatch)
    backend = FakeBackend(torch_module.cuda, close_error=RuntimeError("close failed"))
    monkeypatch.setattr(candidate_module, "load_torch_cuda_target", lambda *args, **kwargs: backend)

    with pytest.raises(CandidateQualificationCleanupError, match="cleanup failed: close failed"):
        qualify_qwen_3b_candidate(total_cache_tokens=8)


def test_backend_cache_state_mismatch_is_rejected_and_cleaned_up(monkeypatch):
    class InconsistentBackend(FakeBackend):
        def prefill(self, token_ids):
            step = super().prefill(token_ids)
            self.cache_length += 1
            return step

    torch_module = FakeTorch()
    install_modules(monkeypatch, torch_module, FakePsutil())
    install_tokenizer_loader(monkeypatch)
    backend = InconsistentBackend(torch_module.cuda)
    monkeypatch.setattr(candidate_module, "load_torch_cuda_target", lambda *args, **kwargs: backend)

    with pytest.raises(CandidateQualificationExecutionError, match="backend state"):
        qualify_qwen_3b_candidate(total_cache_tokens=8)

    assert backend.closed


def test_memory_measurement_requires_windows_peak_working_set():
    cuda = FakeCuda()
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100))

    with pytest.raises(CandidateQualificationExecutionError, match="peak_wset"):
        candidate_module._memory_snapshot(cuda, process, FakeDevice("cuda:0"))
