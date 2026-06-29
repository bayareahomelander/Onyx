import pytest

from onyx.tokenizer_compat import (
    build_grammar_vocabulary,
    byte_level_token_for_bytes,
    validate_byte_reconstruction,
    validate_tokenizer_pair,
)


class ByteTokenizer:
    def __init__(self, *, swapped_ids=False, special_ids=(), extra_tokens=()):
        vocabulary = {byte_level_token_for_bytes(bytes([value])): value for value in range(256)}
        vocabulary.update(dict(extra_tokens))
        if swapped_ids:
            zero_token = byte_level_token_for_bytes(b"\x00")
            one_token = byte_level_token_for_bytes(b"\x01")
            vocabulary[zero_token], vocabulary[one_token] = 1, 0
        self._vocabulary = vocabulary
        self.all_special_ids = list(special_ids)
        self.added_tokens_decoder = {}

    def get_vocab(self):
        return dict(self._vocabulary)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def test_identical_tokenizers_and_byte_vocabulary_pass_compatibility_checks():
    draft = ByteTokenizer()
    target = ByteTokenizer()

    compatible_ids = validate_tokenizer_pair(
        draft,
        target,
        draft_logits_width=256,
        target_logits_width=256,
    )
    vocabulary, stats, errors = build_grammar_vocabulary(draft, 256)
    validate_byte_reconstruction(draft, vocabulary)

    assert errors == []
    assert stats["populated_token_ids"] == 256
    assert compatible_ids == tuple(range(256))


def test_tokenizer_pair_rejects_same_size_with_different_id_mapping():
    with pytest.raises(ValueError, match="Tokenizer mismatch at token ID"):
        validate_tokenizer_pair(
            ByteTokenizer(),
            ByteTokenizer(swapped_ids=True),
            draft_logits_width=256,
            target_logits_width=256,
        )


def test_tokenizer_pair_rejects_different_model_widths():
    with pytest.raises(ValueError, match="model vocabulary widths differ"):
        validate_tokenizer_pair(
            ByteTokenizer(),
            ByteTokenizer(),
            draft_logits_width=256,
            target_logits_width=257,
        )


def test_tokenizer_pair_rejects_different_special_token_ids():
    with pytest.raises(ValueError, match="special token IDs differ"):
        validate_tokenizer_pair(
            ByteTokenizer(special_ids=(0,)),
            ByteTokenizer(special_ids=(1,)),
            draft_logits_width=256,
            target_logits_width=256,
        )


def test_tokenizer_pair_allows_target_only_ids_outside_draft_vocabulary():
    compatible_ids = validate_tokenizer_pair(
        ByteTokenizer(),
        ByteTokenizer(extra_tokens=(("<tool_response>", 256),)),
        draft_logits_width=257,
        target_logits_width=257,
    )

    assert compatible_ids == tuple(range(256))


def test_byte_reconstruction_rejects_misaligned_vocabulary():
    tokenizer = ByteTokenizer()
    vocabulary, _stats, errors = build_grammar_vocabulary(tokenizer, 256)
    assert errors == []
    vocabulary[ord("O")] = b"X"

    with pytest.raises(ValueError, match="byte reconstruction failed"):
        validate_byte_reconstruction(tokenizer, vocabulary, samples=("ONYX",))
