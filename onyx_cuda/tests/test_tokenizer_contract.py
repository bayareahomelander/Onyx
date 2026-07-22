import pytest

from onyx_cuda import TokenizerAdapter, UnknownTextTokenError
from onyx_cuda.testing import FakeCharacterTokenizer


def test_fake_tokenizer_satisfies_runtime_contract():
    tokenizer = FakeCharacterTokenizer(("a", "b", "c"), tokenizer_id="test-tokenizer")

    assert isinstance(tokenizer, TokenizerAdapter)
    assert tokenizer.tokenizer_id == "test-tokenizer"
    assert tokenizer.vocab_size == 3


def test_encodes_and_decodes_deterministically():
    tokenizer = FakeCharacterTokenizer(("a", "b", "c"))

    assert tokenizer.encode("cab") == (2, 0, 1)
    assert tokenizer.decode((2, 0, 1)) == "cab"
    assert tokenizer.decode(tokenizer.encode("abcabc")) == "abcabc"


def test_round_trips_unicode_characters():
    tokenizer = FakeCharacterTokenizer(("a", "é", "世"))

    assert tokenizer.encode("世éa") == (2, 1, 0)
    assert tokenizer.decode((2, 1, 0)) == "世éa"


def test_empty_text_and_token_sequences_are_supported():
    tokenizer = FakeCharacterTokenizer(("a",))

    assert tokenizer.encode("") == ()
    assert tokenizer.decode(()) == ""


@pytest.mark.parametrize(
    ("vocabulary", "error_type", "message"),
    [
        ((), ValueError, "at least one character"),
        (("a", "a"), ValueError, "must be unique"),
        (("",), ValueError, "exactly one character"),
        (("ab",), ValueError, "exactly one character"),
        ((1,), TypeError, "must be a string"),
    ],
)
def test_rejects_invalid_vocabularies(vocabulary, error_type, message):
    with pytest.raises(error_type, match=message):
        FakeCharacterTokenizer(vocabulary)


@pytest.mark.parametrize(
    ("tokenizer_id", "error_type"),
    [("", ValueError), ("  ", ValueError), (None, TypeError)],
)
def test_rejects_invalid_tokenizer_identifiers(tokenizer_id, error_type):
    with pytest.raises(error_type, match="tokenizer_id"):
        FakeCharacterTokenizer(("a",), tokenizer_id=tokenizer_id)


def test_reports_unknown_input_character_and_position():
    tokenizer = FakeCharacterTokenizer(("a", "b"))

    with pytest.raises(UnknownTextTokenError, match="'x' at position 1"):
        tokenizer.encode("axb")


def test_encode_rejects_non_string_input():
    tokenizer = FakeCharacterTokenizer(("a",))

    with pytest.raises(TypeError, match="text must be a string"):
        tokenizer.encode(["a"])


@pytest.mark.parametrize("token_ids", [(-1,), (2,), (True,), ("1",)])
def test_decode_rejects_invalid_token_ids(token_ids):
    tokenizer = FakeCharacterTokenizer(("a", "b"))

    with pytest.raises((TypeError, ValueError), match="position 0"):
        tokenizer.decode(token_ids)
