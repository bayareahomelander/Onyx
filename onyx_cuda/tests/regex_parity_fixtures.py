"""Frozen 2026-07-14 black-box outcomes from the read-only root regex runtime."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegexStateExpectation:
    token_ids: tuple[int, ...]
    valid_token_ids: tuple[int, ...]
    is_match: bool
    is_dead: bool


@dataclass(frozen=True, slots=True)
class RegexParityCase:
    name: str
    vocabulary: tuple[bytes, ...]
    pattern: str
    states: tuple[RegexStateExpectation, ...]


REGEX_PARITY_CASES = (
    RegexParityCase(
        name="literal_and_empty_token",
        vocabulary=(b"a", b"b", b"ab", b"x", b"", b"abc", b"c"),
        pattern="ab",
        states=(
            RegexStateExpectation((), (0, 2, 5), False, False),
            RegexStateExpectation((0,), (1,), False, False),
            RegexStateExpectation((2,), (0, 1, 3, 6), True, False),
            RegexStateExpectation((3,), (), False, True),
            RegexStateExpectation((4,), (0, 2, 5), False, False),
            RegexStateExpectation((5,), (), False, False),
        ),
    ),
    RegexParityCase(
        name="alternation",
        vocabulary=(b"a", b"b", b"c", b"bc", b"x", b""),
        pattern="(?:a|bc)",
        states=(
            RegexStateExpectation((), (0, 1, 3), False, False),
            RegexStateExpectation((0,), (0, 1, 2, 4), True, False),
            RegexStateExpectation((1,), (2,), False, False),
            RegexStateExpectation((3,), (0, 1, 2, 4), True, False),
            RegexStateExpectation((4,), (), False, True),
            RegexStateExpectation((5,), (0, 1, 3), False, False),
        ),
    ),
    RegexParityCase(
        name="bounded_repetition",
        vocabulary=(b"1", b"2", b"12", b"123", b"1234", b"a", b""),
        pattern="[0-9]{2,3}",
        states=(
            RegexStateExpectation((), (0, 1, 2, 3, 4), False, False),
            RegexStateExpectation((0,), (0, 1, 2, 3), False, False),
            RegexStateExpectation((2,), (0, 1, 2, 5), True, False),
            RegexStateExpectation((3,), (0, 1, 5), True, False),
            RegexStateExpectation((4,), (), False, False),
            RegexStateExpectation((5,), (), False, True),
            RegexStateExpectation((6,), (0, 1, 2, 3, 4), False, False),
        ),
    ),
    RegexParityCase(
        name="character_class",
        vocabulary=(b"A", b"B", b"C", b"D", b"AB", b"", b"-"),
        pattern="[A-C]+",
        states=(
            RegexStateExpectation((), (0, 1, 2, 4), False, False),
            RegexStateExpectation((0,), (0, 1, 2, 3, 4, 6), True, False),
            RegexStateExpectation((4,), (0, 1, 2, 3, 4, 6), True, False),
            RegexStateExpectation((3,), (), False, True),
            RegexStateExpectation((5,), (0, 1, 2, 4), False, False),
        ),
    ),
    RegexParityCase(
        name="split_utf8",
        vocabulary=(b"\xc3", b"\xa9", b"\xc3\xa9", b"!", b"", b"x"),
        pattern=r"\x{00E9}!",
        states=(
            RegexStateExpectation((), (0, 2), False, False),
            RegexStateExpectation((0,), (1,), False, False),
            RegexStateExpectation((2,), (3,), False, False),
            RegexStateExpectation((0, 1), (3,), False, False),
            RegexStateExpectation((2, 3), (0, 1, 3, 5), True, False),
            RegexStateExpectation((4,), (0, 2), False, False),
            RegexStateExpectation((5,), (), False, True),
        ),
    ),
    RegexParityCase(
        name="empty_match",
        vocabulary=(b"a", b"aa", b"b", b""),
        pattern="a*",
        states=(
            RegexStateExpectation((), (0, 1, 2), True, False),
            RegexStateExpectation((0,), (0, 1, 2), True, False),
            RegexStateExpectation((1,), (0, 1, 2), True, False),
            RegexStateExpectation((2,), (), False, False),
            RegexStateExpectation((3,), (0, 1, 2), True, False),
        ),
    ),
)


INVALID_REGEX_PATTERNS = ("(", "[z-a]")
