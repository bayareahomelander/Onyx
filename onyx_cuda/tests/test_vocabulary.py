import pytest

from onyx_cuda.vocabulary import build_token_byte_vocabulary


class FakeTokenizer:
    all_special_ids = [2]

    def get_vocab(self):
        return {"a": 0, "broken": 1, "special": 2}

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        values = {0: "a", 1: "\ufffd", 2: "<special>"}
        return values[token_ids[0]]


def test_build_token_byte_vocabulary_is_deterministic_and_counts_empty_ids():
    first = build_token_byte_vocabulary(FakeTokenizer(), 4)
    second = build_token_byte_vocabulary(FakeTokenizer(), 4)

    assert first == second
    assert first.token_bytes == [b"a", b"", b"<special>", b""]
    assert first.special_token_count == 1
    assert first.empty_token_count == 2


def test_build_token_byte_vocabulary_rejects_tokenizer_model_mismatch():
    tokenizer = FakeTokenizer()
    tokenizer.all_special_ids = [4]

    with pytest.raises(ValueError, match="exceeds model logits vocabulary size 4"):
        build_token_byte_vocabulary(tokenizer, 4)
