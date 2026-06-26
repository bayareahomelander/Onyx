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

    valid_id_cache = module.CudaValidIdCache(constraint)
    cached_actual = module.masked_argmax_from_cached_grammar_state(
        logits,
        valid_id_cache,
        state,
        check_inputs=False,
    )

    assert int(cached_actual.item()) == token_to_id[b"7"]


def test_cuda_valid_id_cache_reuses_tensor_for_state_and_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    from onyx_cuda.grammar_handoff import CudaValidIdCache

    class CountingGrammar:
        def __init__(self):
            self.calls = 0

        def get_valid_token_ids(self, state):
            self.calls += 1
            return [1, 3, 5]

    grammar = CountingGrammar()
    cache = CudaValidIdCache(grammar)

    current_device = torch.cuda.current_device()

    first = cache.get(9, torch.device("cuda"))
    second = cache.get(9, torch.device(f"cuda:{current_device}"))

    assert grammar.calls == 2
    assert first.data_ptr() == second.data_ptr()
    assert first.cpu().tolist() == [1, 3, 5]
    assert cache.num_entries == 1

    cache.discard(9)
    assert cache.num_entries == 0

    third = cache.get(9, torch.device("cuda"))
    assert grammar.calls == 3
    assert third.cpu().tolist() == [1, 3, 5]

    cache.clear()
    assert cache.num_entries == 0


def test_cuda_valid_id_cache_refreshes_reused_handle_after_grammar_reset():
    torch, GrammarConstraint, module = _cuda_grammar_runtime_or_skip()

    vocab = [b"a", b"b", b"c"]
    constraint = GrammarConstraint(vocab)
    constraint.compile_regex("(ab|bc)")
    cache = module.CudaValidIdCache(constraint)

    initial = constraint.init_state()
    after_a = constraint.advance_state(initial, 0)
    first_valid = cache.get(after_a, torch.device("cuda"))
    assert first_valid.cpu().tolist() == [1]

    constraint.reset()
    reset_initial = constraint.init_state()
    after_b = constraint.advance_state(reset_initial, 1)
    assert after_b == after_a

    refreshed_valid = cache.get(after_b, torch.device("cuda"))
    assert refreshed_valid.cpu().tolist() == [2]
    assert cache.num_entries == 1


def test_masked_argmax_from_grammar_state_rejects_empty_valid_ids():
    _, _, module = _cuda_grammar_runtime_or_skip()

    class EmptyGrammar:
        def get_valid_token_ids(self, state):
            return []

    with pytest.raises(ValueError, match="no valid token ids"):
        module.masked_argmax_from_grammar_state(None, EmptyGrammar(), 0)
