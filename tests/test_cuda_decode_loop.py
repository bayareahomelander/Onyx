import pytest


def _cuda_decode_runtime_or_skip():
    import importlib

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    try:
        from onyx._rust import GrammarConstraint
    except ImportError:
        pytest.skip("onyx Rust extension is not available")

    masked_argmax = importlib.import_module("onyx_cuda.masked_argmax")
    masked_argmax._load_extension()

    from onyx_cuda.decode_loop import decode_greedy_from_logits

    return torch, GrammarConstraint, decode_greedy_from_logits


def _build_decode_case(torch, GrammarConstraint):
    vocab = [bytes([letter]) for letter in range(ord("A"), ord("Z") + 1)]
    vocab.append(b"-")
    vocab.extend(str(digit).encode("ascii") for digit in range(10))
    vocab.extend([b" invalid", b" invalid high"])
    token_to_id = {token: index for index, token in enumerate(vocab)}

    constraint = GrammarConstraint(vocab)
    constraint.compile_regex("[A-Z]{3}-[0-9]{4}")

    expected_tokens = [b"O", b"N", b"Y", b"-", b"2", b"0", b"2", b"6"]
    invalid_id = token_to_id[b" invalid high"]
    logits_steps = []
    for token in expected_tokens:
        logits = torch.zeros(len(vocab), device="cuda", dtype=torch.float32)
        logits[invalid_id] = 100.0
        logits[token_to_id[token]] = 10.0
        logits_steps.append(logits)

    expected_ids = tuple(token_to_id[token] for token in expected_tokens)
    return constraint, vocab, logits_steps, expected_ids


def test_decode_greedy_from_logits_rejects_invalid_max_steps():
    from onyx_cuda.decode_loop import decode_greedy_from_logits

    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        decode_greedy_from_logits([], object(), 0, max_steps=0)


def test_cuda_decode_loop_completes_multi_token_regex():
    torch, GrammarConstraint, decode = _cuda_decode_runtime_or_skip()
    constraint, vocab, logits_steps, expected_ids = _build_decode_case(torch, GrammarConstraint)
    initial_state = constraint.init_state()

    result = decode(
        logits_steps,
        constraint,
        initial_state,
        max_steps=16,
        check_inputs=False,
    )

    assert result.token_ids == expected_ids
    assert b"".join(vocab[token_id] for token_id in result.token_ids) == b"ONY-2026"
    assert result.steps == 8
    assert result.matched is True
    assert result.termination_reason == "grammar_complete"
    assert constraint.is_match_state(result.final_state) is True
    assert result.timings.pre_cleanup_us > 0

    with pytest.raises(ValueError, match="Unknown grammar state handle"):
        constraint.get_valid_token_ids(initial_state)


def test_cuda_decode_loop_reports_logits_exhaustion():
    torch, GrammarConstraint, decode = _cuda_decode_runtime_or_skip()
    constraint, _, logits_steps, expected_ids = _build_decode_case(torch, GrammarConstraint)
    initial_state = constraint.init_state()

    result = decode(
        logits_steps[:2],
        constraint,
        initial_state,
        max_steps=16,
        check_inputs=False,
    )

    assert result.token_ids == expected_ids[:2]
    assert result.matched is False
    assert result.termination_reason == "logits_exhausted"
    with pytest.raises(ValueError, match="Unknown grammar state handle"):
        constraint.get_valid_token_ids(initial_state)
    assert constraint.get_valid_token_ids(result.final_state)


def test_cuda_decode_loop_honors_step_limit():
    torch, GrammarConstraint, decode = _cuda_decode_runtime_or_skip()
    constraint, _, logits_steps, expected_ids = _build_decode_case(torch, GrammarConstraint)

    result = decode(
        logits_steps,
        constraint,
        constraint.init_state(),
        max_steps=3,
        check_inputs=False,
    )

    assert result.token_ids == expected_ids[:3]
    assert result.matched is False
    assert result.termination_reason == "max_steps"


def test_cuda_decode_loop_rejects_batched_rows():
    torch, GrammarConstraint, decode = _cuda_decode_runtime_or_skip()
    constraint, vocab, _, _ = _build_decode_case(torch, GrammarConstraint)
    batched_logits = torch.zeros((2, len(vocab)), device="cuda", dtype=torch.float32)
    initial_state = constraint.init_state()

    with pytest.raises(ValueError, match="one logits row per step"):
        decode(
            [batched_logits],
            constraint,
            initial_state,
            max_steps=1,
        )

    assert constraint.get_valid_token_ids(initial_state)


def test_cuda_decode_loop_restores_initial_state_ownership_after_progress_error():
    torch, GrammarConstraint, decode = _cuda_decode_runtime_or_skip()
    constraint, _, logits_steps, _ = _build_decode_case(torch, GrammarConstraint)

    class RecordingGrammar:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.created_states = []
            self.released_states = []

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def advance_state(self, state, token_id):
            next_state = self.wrapped.advance_state(state, token_id)
            self.created_states.append(next_state)
            return next_state

        def release_states(self, states):
            self.released_states.extend(states)
            self.wrapped.release_states(states)

    recording_grammar = RecordingGrammar(constraint)
    initial_state = recording_grammar.init_state()

    def failing_logits():
        yield logits_steps[0]
        raise RuntimeError("synthetic logits failure")

    with pytest.raises(RuntimeError, match="synthetic logits failure"):
        decode(
            failing_logits(),
            recording_grammar,
            initial_state,
            max_steps=4,
            check_inputs=False,
        )

    assert recording_grammar.get_valid_token_ids(initial_state)
    assert recording_grammar.created_states
    assert initial_state not in recording_grammar.released_states
    for created_state in recording_grammar.created_states:
        with pytest.raises(ValueError, match="Unknown grammar state handle"):
            recording_grammar.get_valid_token_ids(created_state)


def test_cuda_decode_loop_bounds_and_clears_internal_cache(monkeypatch):
    import importlib

    torch, GrammarConstraint, _ = _cuda_decode_runtime_or_skip()
    constraint, _, logits_steps, _ = _build_decode_case(torch, GrammarConstraint)
    decode_module = importlib.import_module("onyx_cuda.decode_loop")
    original_cache = decode_module.CudaValidIdCache
    instances = []

    class TrackingCache(original_cache):
        def __init__(self, grammar_constraint):
            super().__init__(grammar_constraint)
            self.max_entries = 0
            instances.append(self)

        def get(self, state, device):
            result = super().get(state, device)
            self.max_entries = max(self.max_entries, self.num_entries)
            return result

    monkeypatch.setattr(decode_module, "CudaValidIdCache", TrackingCache)

    result = decode_module.decode_greedy_from_logits(
        logits_steps,
        constraint,
        constraint.init_state(),
        max_steps=16,
        check_inputs=False,
    )

    assert result.matched is True
    assert len(instances) == 1
    assert instances[0].max_entries == 1
    assert instances[0].num_entries == 0
