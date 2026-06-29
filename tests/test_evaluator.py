import pytest

from onyx.evaluator import validate_grammar_completion


def test_validate_grammar_completion_accepts_complete_full_match():
    validate_grammar_completion("2026", {"finish_reason": "grammar_complete"}, r"[0-9]{4}")


def test_validate_grammar_completion_rejects_incomplete_generation():
    with pytest.raises(RuntimeError, match="did not complete"):
        validate_grammar_completion("20", {"finish_reason": "length"}, r"[0-9]{4}")


def test_validate_grammar_completion_rejects_nonmatching_output():
    with pytest.raises(RuntimeError, match="does not fully match"):
        validate_grammar_completion("year", {"finish_reason": "grammar_complete"}, r"[0-9]{4}")
