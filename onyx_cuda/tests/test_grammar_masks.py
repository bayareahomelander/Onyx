"""Cached native token masks must behave exactly like the ID lists they replace."""

import json

import pytest
import torch

from onyx_cuda import _rust
from onyx_cuda.generation import _MaskedGrammar, _grammar_choices
from onyx_cuda.masking import TokenMask, apply_grammar_mask, grammar_argmax
from onyx_cuda.numerics import ambiguous_logits

VOCABULARY = [bytes([byte]) for byte in range(128)] + [b'"}', b"ab", b""] + [b""] * 5
EOS = [len(VOCABULARY) - 2, len(VOCABULARY) - 1]


def _constraint(grammar):
    constraint = _rust.GrammarConstraint(VOCABULARY)
    if isinstance(grammar, dict):
        constraint.compile_json_schema(json.dumps(grammar))
    else:
        constraint.compile_regex(grammar)
    return constraint


@pytest.mark.parametrize("grammar,document", [
    ({"type": "object", "properties": {"text": {"type": "string"}}}, '{"text":"ab"}'),
    ("[0-9]+", "123"),
])
def test_masked_choices_match_list_choices(grammar, document):
    plain = _constraint(grammar)
    masked = _MaskedGrammar(_constraint(grammar), torch.device("cpu"))
    plain_state, masked_state = plain.init_state(), masked.init_state()
    for byte in [*document.encode(), None]:
        expected = _grammar_choices(plain, plain_state, EOS)
        choices = _grammar_choices(masked, masked_state, EOS)
        if expected:
            assert isinstance(choices, TokenMask)
            assert len(choices) == len(expected)
            assert list(choices) == sorted(expected)
        else:
            assert choices == []
        if byte is not None:
            plain_state = plain.advance_state(plain_state, byte)
            masked_state = masked.advance_state(masked_state, byte)


def test_masked_choices_reuse_masks_and_keep_list_errors():
    masked = _MaskedGrammar(_constraint({"type": "object", "properties": {"text": {"type": "string"}}}),
                            torch.device("cpu"))
    state = masked.init_state()
    for byte in b'{"text":"':
        state = masked.advance_state(state, byte)
    first = _grammar_choices(masked, state, EOS)
    state = masked.advance_state(state, ord("a"))
    assert _grammar_choices(masked, state, EOS).blocked is first.blocked
    masked.prefetch(state)

    dead = _MaskedGrammar(_constraint("a"), torch.device("cpu"))
    state = dead.advance_state(dead.init_state(), ord("b"))
    with pytest.raises(ValueError, match="no valid token continuation"):
        _grammar_choices(dead, state, EOS)
    # A repeated EOS keeps its list length, as for plain constraints.
    grown = _MaskedGrammar(_constraint("a+"), torch.device("cpu"))
    state = grown.advance_state(grown.init_state(), ord("a"))
    assert _grammar_choices(grown, state, [EOS[0], EOS[0]]) == [ord("a"), EOS[0], EOS[0]]


def _masks(width, sets):
    masks = []
    for valid in sets:
        blocked = torch.ones(width, dtype=torch.bool)
        blocked[valid] = False
        masks.append(TokenMask(blocked, len(valid)))
    return masks


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_guard_matches_per_position_guard(dtype):
    generator = torch.Generator().manual_seed(3)
    width = 64
    for trial in range(200):
        logits = torch.randn((1, 4, width), generator=generator).to(dtype)
        if trial % 2:
            # Near ties among eligible tokens exercise the tolerance boundary.
            logits[0, :, 7] = logits[0, :, 9] + torch.finfo(dtype).eps * (trial % 3)
        sets = [sorted(torch.randperm(width, generator=generator)[:count].tolist())
                for count in (1, 2, 5, width)]
        sets[trial % 4] = [7, 9, 11]
        for count in range(1, 5):
            expected = ambiguous_logits(logits, count, sets)
            assert ambiguous_logits(logits, count, _masks(width, sets)) == expected


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mask_selection_is_bitwise_identical_to_id_selection(dtype):
    generator = torch.Generator(device="cuda").manual_seed(5)
    width = 1024
    for trial in range(50):
        logits = torch.randn((1, width), device="cuda", dtype=dtype, generator=generator)
        logits[0, 10:20] = logits[0, 10]  # ties choose the lowest valid index
        valid = sorted(set(torch.randint(0, width, (1 + trial * 10,), generator=generator,
                                         device="cuda").tolist()) | {12, 15})
        blocked = torch.ones(width, dtype=torch.bool, device="cuda")
        blocked[valid] = False
        mask = TokenMask(blocked, len(valid))
        assert torch.equal(apply_grammar_mask(logits, mask), apply_grammar_mask(logits, valid))
        assert torch.equal(grammar_argmax(logits, mask),
                           grammar_argmax(logits, valid))
