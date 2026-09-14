from types import SimpleNamespace

import pytest

import onyx_cuda.model as model_module
from onyx_cuda.model import LoadedModel


def _loaded_model(
    *,
    width=2,
    token_bytes=None,
    special_ids=None,
    eos_id=0,
    prompt="same",
):
    tokenizer = SimpleNamespace(
        token_bytes=token_bytes or [b"a", b"b"],
        all_special_ids=special_ids or [0],
        eos_token_id=eos_id,
        prompt=prompt,
        get_vocab=lambda: {"a": 0, "b": 1},
        convert_ids_to_tokens=lambda i: str(i),
    )
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=width))
    return LoadedModel(model, tokenizer, "revision")


def test_model_pair_rejects_each_compatibility_mismatch(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "build_token_byte_vocabulary",
        lambda tokenizer, width: SimpleNamespace(token_bytes=tokenizer.token_bytes),
    )
    monkeypatch.setattr(
        model_module,
        "format_prompt",
        lambda tokenizer, messages: tokenizer.prompt,
    )
    draft = _loaded_model()
    model_module._require_compatible_models(draft, _loaded_model())

    cases = [
        (_loaded_model(width=3), "vocabulary sizes differ"),
        (_loaded_model(token_bytes=[b"a", b"c"]), "token bytes differ"),
        (_loaded_model(special_ids=[1]), "special token IDs differ"),
        (_loaded_model(eos_id=1), "EOS token IDs differ"),
    ]
    for target, message in cases:
        with pytest.raises(RuntimeError, match=message):
            model_module._require_compatible_models(draft, target)


def test_target_only_loading_never_loads_or_validates_a_draft(monkeypatch):
    calls = []
    target = _loaded_model()

    def load(model_id, *, revision):
        assert revision == model_module.MODEL_REVISIONS[model_id]
        calls.append(model_id)
        return target

    monkeypatch.setattr(model_module, "load_model", load)
    monkeypatch.setattr(
        model_module,
        "_require_compatible_models",
        lambda *_: pytest.fail("draft validation is unnecessary"),
    )
    pair = model_module.load_model_pair(include_draft=False)
    assert calls == [model_module.TARGET_MODEL_ID]
    assert pair.draft is None
    assert pair.target is target


def test_target_additions_in_unused_draft_slots_and_target_template_are_supported(monkeypatch):
    monkeypatch.setattr(model_module, "build_token_byte_vocabulary",
                        lambda tokenizer, width: SimpleNamespace(token_bytes=tokenizer.token_bytes))
    draft = _loaded_model(width=3, token_bytes=[b"a", b"b", b""])
    target = _loaded_model(width=3, token_bytes=[b"a", b"b", b"<think>"], prompt="target template")
    target.tokenizer.get_vocab = lambda: {"a": 0, "b": 1, "<think>": 2}
    model_module._require_compatible_models(draft, target)
    # A reassigned existing draft ID is not a harmless target extension.
    draft.tokenizer.get_vocab = lambda: {"a": 0, "b": 1, "other": 2}
    with pytest.raises(RuntimeError, match="token bytes differ at token ID 2"):
        model_module._require_compatible_models(draft, target)


def test_pair_uses_target_tokenizer_after_compatibility_check(monkeypatch):
    draft, target = _loaded_model(), _loaded_model(prompt="target")
    monkeypatch.setattr(model_module, "load_model", lambda name, **kw:
                        target if name == model_module.TARGET_MODEL_ID else draft)
    checked = []
    monkeypatch.setattr(model_module, "_require_compatible_models", lambda d,t: checked.append((d,t)))
    pair = model_module.load_model_pair()
    assert checked == [(draft, target)]
    assert pair.draft.model is draft.model
    assert pair.draft.tokenizer is pair.target.tokenizer
    assert draft.tokenizer is not target.tokenizer
