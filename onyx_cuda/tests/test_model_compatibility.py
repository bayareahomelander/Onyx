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
        (_loaded_model(prompt="different"), "chat-template output differs"),
    ]
    for target, message in cases:
        with pytest.raises(RuntimeError, match=message):
            model_module._require_compatible_models(draft, target)
