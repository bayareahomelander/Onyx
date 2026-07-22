import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import onyx_cuda.production_tokenizer as production_tokenizer_module
from onyx_cuda import TokenizerAdapter
from onyx_cuda.model_profile import DEFAULT_TARGET_PROFILE, QwenModelProfile
from onyx_cuda.production_tokenizer import (
    ProductionTokenizerError,
    ProductionTokenizerLoadError,
    QwenTokenizerAdapter,
    TransformersImportError,
    build_qwen_grammar_vocabulary,
    load_qwen_tokenizer,
)


class FakeTokenizer:
    vocab_size = 4
    eos_token_id = 4
    pad_token_id = 5

    def __init__(self):
        self.encode_calls = []
        self.decode_calls = []
        self.chat_template = "{{ messages }}"
        self.special_tokens_map_extended = {"eos_token": "<eos>", "pad_token": "<pad>"}
        self.all_special_ids = [4, 5]
        self.all_special_tokens = ["<eos>", "<pad>"]

    def __len__(self):
        return 6

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return [0, 4]

    def decode(self, token_ids, **kwargs):
        self.decode_calls.append((token_ids, kwargs))
        return "decoded"

    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 2, "d": 3, "<eos>": 4, "<pad>": 5}

    def get_added_vocab(self):
        return {"<eos>": 4, "<pad>": 5}


class FakeFastTokenizer:
    vocabulary = {"a": 0, "b": 1, "c": 2, "d": 3, "<eos>": 4, "<pad>": 5}

    def __init__(self):
        self.encode_calls = []
        self.decode_calls = []

    def get_vocab_size(self, *, with_added_tokens):
        return 6 if with_added_tokens else 4

    def token_to_id(self, token):
        return self.vocabulary.get(token)

    def encode(self, text, *, add_special_tokens):
        self.encode_calls.append((text, add_special_tokens))
        return SimpleNamespace(ids=[0, 4])

    def decode(self, token_ids, *, skip_special_tokens):
        self.decode_calls.append((token_ids, skip_special_tokens))
        return "decoded"

    def get_vocab(self, *, with_added_tokens):
        assert with_added_tokens
        return dict(self.vocabulary)

    def get_added_tokens_decoder(self):
        def metadata(content):
            return SimpleNamespace(
                content=content,
                lstrip=False,
                normalized=False,
                rstrip=False,
                single_word=False,
                special=True,
            )

        return {4: metadata("<eos>"), 5: metadata("<pad>")}


class RecordingHub:
    def __init__(self, paths, error=None):
        self.paths = paths
        self.error = error
        self.calls = []

    def hf_hub_download(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.paths[kwargs["filename"]]


class RecordingTokenizerFactory:
    def __init__(self, tokenizer=None, error=None):
        self.tokenizer = tokenizer or FakeFastTokenizer()
        self.error = error
        self.calls = []

    def from_file(self, path):
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        return self.tokenizer


def fake_tokenizer_configuration():
    def added_token(content):
        return {
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    return {
        "added_tokens_decoder": {
            "4": added_token("<eos>"),
            "5": added_token("<pad>"),
        },
        "additional_special_tokens": [],
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "chat_template": "{{ messages }}",
    }


def fake_tokenizer_asset():
    def added(token_id, content):
        return {
            "id": token_id,
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    return {
        "model": {
            "type": "BPE",
            "byte_fallback": False,
            "vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
        },
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [added(4, "<eos>"), added(5, "<pad>")],
    }


def install_fake_tokenizer_dependencies(
    monkeypatch,
    tmp_path,
    *,
    configuration=None,
    tokenizer_asset=None,
    hub_error=None,
    tokenizer_error=None,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    configuration_path = tmp_path / "tokenizer_config.json"
    tokenizer_path.write_text(
        json.dumps(tokenizer_asset) if tokenizer_asset is not None else "{}",
        encoding="utf-8",
    )
    configuration_path.write_text(
        json.dumps(configuration or fake_tokenizer_configuration()),
        encoding="utf-8",
    )
    hub = RecordingHub(
        {
            "tokenizer.json": tokenizer_path,
            "tokenizer_config.json": configuration_path,
        },
        error=hub_error,
    )
    factory = RecordingTokenizerFactory(error=tokenizer_error)
    requested = []
    modules = {
        "huggingface_hub": hub,
        "tokenizers": SimpleNamespace(Tokenizer=factory),
    }

    def import_module(name):
        requested.append(name)
        return modules[name]

    monkeypatch.setattr(
        "onyx_cuda.production_tokenizer.importlib.import_module",
        import_module,
    )
    return hub, factory, requested


def test_default_profile_is_pinned_and_immutable():
    assert DEFAULT_TARGET_PROFILE.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert DEFAULT_TARGET_PROFILE.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert DEFAULT_TARGET_PROFILE.pinned_id.endswith(f"@{DEFAULT_TARGET_PROFILE.revision}")


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("model_id", "", ValueError),
        ("model_id", None, TypeError),
        ("revision", "  ", ValueError),
        ("revision", 1, TypeError),
    ],
)
def test_profile_rejects_invalid_identity_fields(field, value, error_type):
    values = {"model_id": "Qwen/test", "revision": "abc123"}
    values[field] = value

    with pytest.raises(error_type, match=field):
        QwenModelProfile(**values)


def test_loads_pinned_fast_tokenizer_without_tensor_runtime(monkeypatch, tmp_path):
    hub, factory, requested = install_fake_tokenizer_dependencies(monkeypatch, tmp_path)

    result = load_qwen_tokenizer(local_files_only=True)

    assert isinstance(result.tokenizer, TokenizerAdapter)
    assert result.tokenizer.tokenizer_id == DEFAULT_TARGET_PROFILE.pinned_id
    assert result.tokenizer.vocab_size == 6
    assert result.tokenizer.base_vocab_size == 4
    assert result.tokenizer.eos_token_id == 4
    assert result.tokenizer.pad_token_id == 5
    assert result.tokenizer.encode("hello") == (0, 4)
    assert result.tokenizer.decode((0, 4)) == "decoded"
    assert result.load_seconds >= 0
    assert requested == ["huggingface_hub", "tokenizers"]
    assert "torch" not in requested
    assert "transformers" not in requested
    assert hub.calls == [
        {
            "repo_id": DEFAULT_TARGET_PROFILE.model_id,
            "filename": "tokenizer.json",
            "revision": DEFAULT_TARGET_PROFILE.revision,
            "local_files_only": True,
        },
        {
            "repo_id": DEFAULT_TARGET_PROFILE.model_id,
            "filename": "tokenizer_config.json",
            "revision": DEFAULT_TARGET_PROFILE.revision,
            "local_files_only": True,
        },
    ]
    assert factory.calls == [str(tmp_path / "tokenizer.json")]


def test_builds_grammar_vocabulary_lazily_without_isolated_decode(monkeypatch, tmp_path):
    _, factory, _ = install_fake_tokenizer_dependencies(
        monkeypatch,
        tmp_path,
        tokenizer_asset=fake_tokenizer_asset(),
    )
    loaded = load_qwen_tokenizer(local_files_only=True)

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("grammar-vocabulary construction must not decode token IDs")

    factory.tokenizer.decode = forbidden_decode
    first = build_qwen_grammar_vocabulary(loaded.tokenizer)
    second = build_qwen_grammar_vocabulary(loaded.tokenizer)

    assert first == second == (b"a", b"b", b"c", b"d", b"", b"")
    assert first is not second
    assert factory.tokenizer.decode_calls == []


def test_loading_does_not_parse_or_build_grammar_vocabulary(monkeypatch, tmp_path):
    install_fake_tokenizer_dependencies(monkeypatch, tmp_path)

    loaded = load_qwen_tokenizer()

    with pytest.raises(ProductionTokenizerError, match="model must be an object") as error:
        build_qwen_grammar_vocabulary(loaded.tokenizer)
    assert error.value.__cause__ is not None


def test_manually_constructed_adapter_fails_builder_but_remains_usable():
    adapter = QwenTokenizerAdapter(FakeTokenizer(), DEFAULT_TARGET_PROFILE)

    assert adapter.encode("hello") == (0, 4)
    assert adapter.decode((0, 4)) == "decoded"
    with pytest.raises(ProductionTokenizerError, match="no production tokenizer.json") as error:
        build_qwen_grammar_vocabulary(adapter)
    assert error.value.__cause__ is not None


def test_builder_wraps_runtime_metadata_access_failures(monkeypatch, tmp_path):
    _, factory, _ = install_fake_tokenizer_dependencies(
        monkeypatch,
        tmp_path,
        tokenizer_asset=fake_tokenizer_asset(),
    )
    loaded = load_qwen_tokenizer()

    def fail_get_vocab(*, with_added_tokens):
        raise RuntimeError("runtime vocabulary unavailable")

    factory.tokenizer.get_vocab = fail_get_vocab

    with pytest.raises(ProductionTokenizerError, match="runtime vocabulary unavailable") as error:
        build_qwen_grammar_vocabulary(loaded.tokenizer)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_builder_wraps_file_json_and_conversion_failures(monkeypatch, tmp_path):
    _, factory, _ = install_fake_tokenizer_dependencies(
        monkeypatch,
        tmp_path,
        tokenizer_asset=fake_tokenizer_asset(),
    )
    loaded = load_qwen_tokenizer()
    source = loaded.tokenizer._grammar_vocabulary_source

    loaded.tokenizer._grammar_vocabulary_source = (
        production_tokenizer_module._QwenGrammarVocabularySource(
            tokenizer_json_path=tmp_path / "missing-tokenizer.json",
            runtime_tokenizer=source.runtime_tokenizer,
        )
    )
    with pytest.raises(ProductionTokenizerError, match="failed to read") as file_error:
        build_qwen_grammar_vocabulary(loaded.tokenizer)
    assert file_error.value.__cause__ is not None

    source.tokenizer_json_path.write_text("{", encoding="utf-8")
    loaded.tokenizer._grammar_vocabulary_source = source
    with pytest.raises(ProductionTokenizerError, match="failed to parse") as json_error:
        build_qwen_grammar_vocabulary(loaded.tokenizer)
    assert json_error.value.__cause__ is not None

    asset = fake_tokenizer_asset()
    asset["added_tokens"][0]["content"] = "\ud800"
    asset["added_tokens"][0]["special"] = False
    source.tokenizer_json_path.write_text(json.dumps(asset), encoding="utf-8")
    factory.tokenizer.vocabulary = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "\ud800": 4,
        "<pad>": 5,
    }
    original_added = factory.tokenizer.get_added_tokens_decoder()
    original_added[4].content = "\ud800"
    original_added[4].special = False
    factory.tokenizer.get_added_tokens_decoder = lambda: original_added

    with pytest.raises(ProductionTokenizerError, match="cannot be encoded") as conversion_error:
        build_qwen_grammar_vocabulary(loaded.tokenizer)
    assert conversion_error.value.__cause__ is not None
    assert conversion_error.value.__cause__.__cause__ is not None


def test_builder_rejects_non_qwen_adapter_before_source_access(monkeypatch):
    def unexpected_build(*args, **kwargs):
        raise AssertionError("invalid input must fail before construction")

    monkeypatch.setattr(
        production_tokenizer_module,
        "_build_qwen_grammar_vocabulary_from_source",
        unexpected_build,
    )

    with pytest.raises(TypeError, match="QwenTokenizerAdapter"):
        build_qwen_grammar_vocabulary(SimpleNamespace())


def test_adapter_uses_stable_encode_and_decode_options():
    wrapped = FakeTokenizer()
    adapter = QwenTokenizerAdapter(wrapped, DEFAULT_TARGET_PROFILE)

    assert adapter.encode("hello") == (0, 4)
    assert adapter.decode((0, 4)) == "decoded"
    assert wrapped.encode_calls == [("hello", {"add_special_tokens": False})]
    assert wrapped.decode_calls == [
        (
            (0, 4),
            {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            },
        )
    ]


def test_adapter_produces_deterministic_compatibility_fingerprint():
    adapter = QwenTokenizerAdapter(FakeTokenizer(), DEFAULT_TARGET_PROFILE)

    first = adapter.compatibility_fingerprint()
    second = adapter.compatibility_fingerprint()

    assert first == second
    assert first.tokenizer_id == DEFAULT_TARGET_PROFILE.pinned_id
    assert first.vocab_size == 6
    assert first.base_vocab_size == 4
    assert first.eos_token_id == 4
    assert first.pad_token_id == 5
    assert len(first.vocabulary_sha256) == 64
    assert len(first.special_tokens_sha256) == 64
    assert len(first.chat_template_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        first.vocab_size = 7


def test_compatibility_fingerprint_keeps_chat_template_separate():
    first_tokenizer = FakeTokenizer()
    second_tokenizer = FakeTokenizer()
    second_tokenizer.chat_template = "different"

    first = QwenTokenizerAdapter(
        first_tokenizer, DEFAULT_TARGET_PROFILE
    ).compatibility_fingerprint()
    second = QwenTokenizerAdapter(
        second_tokenizer, DEFAULT_TARGET_PROFILE
    ).compatibility_fingerprint()

    assert first.vocabulary_sha256 == second.vocabulary_sha256
    assert first.special_tokens_sha256 == second.special_tokens_sha256
    assert first.chat_template_sha256 != second.chat_template_sha256


def test_compatibility_fingerprint_rejects_malformed_vocabulary():
    tokenizer = FakeTokenizer()
    tokenizer.get_vocab = lambda: {"a": 0}
    adapter = QwenTokenizerAdapter(tokenizer, DEFAULT_TARGET_PROFILE)

    with pytest.raises(ProductionTokenizerError, match="contains 1 entries; expected 6"):
        adapter.compatibility_fingerprint()


@pytest.mark.parametrize("missing_name", ["huggingface_hub", "tokenizers"])
def test_reports_missing_tokenizer_dependency(monkeypatch, missing_name):
    def missing_module(name):
        if name == missing_name:
            raise ModuleNotFoundError(f"{name} is missing")
        return SimpleNamespace()

    monkeypatch.setattr(
        "onyx_cuda.production_tokenizer.importlib.import_module",
        missing_module,
    )

    with pytest.raises(TransformersImportError, match=f"{missing_name} is missing"):
        load_qwen_tokenizer()


def test_wraps_pinned_file_load_failures(monkeypatch, tmp_path):
    install_fake_tokenizer_dependencies(
        monkeypatch,
        tmp_path,
        hub_error=OSError("snapshot unavailable"),
    )

    with pytest.raises(ProductionTokenizerLoadError, match="snapshot unavailable"):
        load_qwen_tokenizer()


def test_wraps_fast_tokenizer_load_failures(monkeypatch, tmp_path):
    install_fake_tokenizer_dependencies(
        monkeypatch,
        tmp_path,
        tokenizer_error=ValueError("invalid tokenizer payload"),
    )

    with pytest.raises(ProductionTokenizerLoadError, match="invalid tokenizer payload"):
        load_qwen_tokenizer()


def test_rejects_malformed_tokenizer_configuration(monkeypatch, tmp_path):
    configuration = fake_tokenizer_configuration()
    configuration["eos_token"] = None
    install_fake_tokenizer_dependencies(
        monkeypatch,
        tmp_path,
        configuration=configuration,
    )

    with pytest.raises(ProductionTokenizerLoadError, match="eos_token must be a string"):
        load_qwen_tokenizer()


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_rejects_non_boolean_local_files_only_before_import(monkeypatch, value):
    def unexpected_import(name):
        raise AssertionError("invalid input must fail before importing tokenizer dependencies")

    monkeypatch.setattr(
        "onyx_cuda.production_tokenizer.importlib.import_module",
        unexpected_import,
    )

    with pytest.raises(TypeError, match="boolean"):
        load_qwen_tokenizer(local_files_only=value)


@pytest.mark.parametrize("token_ids", [(-1,), (6,), (True,), ("1",)])
def test_adapter_rejects_invalid_decode_ids(token_ids):
    adapter = QwenTokenizerAdapter(FakeTokenizer(), DEFAULT_TARGET_PROFILE)

    with pytest.raises((TypeError, ValueError), match="position 0"):
        adapter.decode(token_ids)


def test_adapter_wraps_execution_errors_and_non_string_decode():
    class BrokenTokenizer(FakeTokenizer):
        def encode(self, text, **kwargs):
            raise RuntimeError("encode failed")

        def decode(self, token_ids, **kwargs):
            return None

    adapter = QwenTokenizerAdapter(BrokenTokenizer(), DEFAULT_TARGET_PROFILE)

    with pytest.raises(ProductionTokenizerError, match="encode failed"):
        adapter.encode("hello")
    with pytest.raises(ProductionTokenizerError, match="non-string"):
        adapter.decode((0,))


def test_rejects_malformed_loaded_tokenizer_metadata():
    malformed = FakeTokenizer()
    malformed.eos_token_id = 6

    with pytest.raises(ProductionTokenizerLoadError, match="EOS token"):
        QwenTokenizerAdapter(malformed, DEFAULT_TARGET_PROFILE)
