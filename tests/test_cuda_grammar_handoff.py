import pytest


def _cuda_grammar_runtime_or_skip():
    import importlib

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    try:
        from onyx._rust import GrammarConstraint
    except ImportError:
        pytest.skip("onyx Rust extension is not available")

    module = importlib.import_module("onyx_cuda.grammar_handoff")
    return torch, GrammarConstraint, module


def test_masked_argmax_from_grammar_state_uses_rust_valid_ids():
    torch, GrammarConstraint, module = _cuda_grammar_runtime_or_skip()

    vocab = [b"The", b" year", b" is "]
    vocab.extend(str(digit).encode("ascii") for digit in range(10))
    vocab.extend([b" invalid", b" invalid high"])
    token_to_id = {token: index for index, token in enumerate(vocab)}

    constraint = GrammarConstraint(vocab)
    constraint.compile_regex("The year is [0-9]")

    state = constraint.init_state()
    for token in (b"The", b" year", b" is "):
        state = constraint.advance_state(state, token_to_id[token])

    valid_ids = constraint.get_valid_token_ids(state)
    assert token_to_id[b"7"] in valid_ids
    assert token_to_id[b" invalid high"] not in valid_ids

    logits = torch.zeros(len(vocab), device="cuda", dtype=torch.float32)
    logits[token_to_id[b" invalid high"]] = 100.0
    logits[token_to_id[b"7"]] = 10.0

    actual = module.masked_argmax_from_grammar_state(
        logits,
        constraint,
        state,
        check_inputs=False,
    )

    assert int(actual.item()) == token_to_id[b"7"]


def test_masked_argmax_from_grammar_state_rejects_empty_valid_ids():
    _, _, module = _cuda_grammar_runtime_or_skip()

    class EmptyGrammar:
        def get_valid_token_ids(self, state):
            return []

    with pytest.raises(ValueError, match="no valid token ids"):
        module.masked_argmax_from_grammar_state(None, EmptyGrammar(), 0)
