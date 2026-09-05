import pytest
from types import SimpleNamespace
from tokenizers.decoders import ByteLevel

from onyx_cuda.vocabulary import build_token_byte_vocabulary


class FakeTokenizer:
    all_special_ids = [2]
    backend_tokenizer = SimpleNamespace(decoder=ByteLevel())

    def get_vocab(self):
        return {"a": 0, "Ã": 1, "special": 2}


def test_build_token_byte_vocabulary_is_deterministic_and_counts_empty_ids():
    first = build_token_byte_vocabulary(FakeTokenizer(), 4)
    second = build_token_byte_vocabulary(FakeTokenizer(), 4)

    assert first == second
    assert first.token_bytes == [b"a", b"\xc3", b"", b""]
    assert first.special_token_count == 1
    assert first.empty_token_count == 2


def test_unsupported_tokenizer_fails_instead_of_guessing_bytes():
    tokenizer = FakeTokenizer()
    tokenizer.backend_tokenizer = None
    with pytest.raises(ValueError, match="ByteLevel tokenizer"):
        build_token_byte_vocabulary(tokenizer, 4)


def test_build_token_byte_vocabulary_rejects_tokenizer_model_mismatch():
    tokenizer = FakeTokenizer()
    tokenizer.all_special_ids = [4]

    with pytest.raises(ValueError, match="exceeds model logits vocabulary size 4"):
        build_token_byte_vocabulary(tokenizer, 4)
