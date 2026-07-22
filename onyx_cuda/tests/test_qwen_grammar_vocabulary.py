import copy
import json
from types import SimpleNamespace

import pytest

from onyx_cuda.qwen_grammar_vocabulary import (
    _QwenGrammarVocabularyError,
    _QwenGrammarVocabularyRuntimeMetadata,
    _build_qwen_grammar_vocabulary_from_source,
    _decode_added_token_content,
    _decode_byte_level_piece,
    _AddedTokenRecord,
)


def byte_level_characters_by_byte():
    byte_values = [
        *range(ord("!"), ord("~") + 1),
        *range(0xA1, 0xAC + 1),
        *range(0xAE, 0xFF + 1),
    ]
    code_points = list(byte_values)
    next_code_point = 256
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(next_code_point)
            next_code_point += 1
    return {byte: chr(code_point) for byte, code_point in zip(byte_values, code_points)}


BYTE_LEVEL_CHARACTERS = byte_level_characters_by_byte()


def added_token(token_id, content, *, special):
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": special,
    }


def default_asset():
    return {
        "model": {
            "type": "BPE",
            "byte_fallback": False,
            "vocab": {"a": 0, "b": 1},
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "added_tokens": [
            added_token(2, "<special>", special=True),
            added_token(3, "<tool>", special=False),
        ],
    }


def runtime_metadata_for(asset):
    base_vocabulary = dict(asset["model"]["vocab"])
    complete_vocabulary = dict(base_vocabulary)
    runtime_added_tokens = {}
    for token in asset["added_tokens"]:
        complete_vocabulary[token["content"]] = token["id"]
        runtime_added_tokens[token["id"]] = SimpleNamespace(
            content=token["content"],
            single_word=token["single_word"],
            lstrip=token["lstrip"],
            rstrip=token["rstrip"],
            normalized=token["normalized"],
            special=token["special"],
        )
    return _QwenGrammarVocabularyRuntimeMetadata(
        base_vocab_size=len(base_vocabulary),
        vocab_size=len(complete_vocabulary),
        vocabulary=complete_vocabulary,
        added_tokens=runtime_added_tokens,
    )


def write_asset(tmp_path, asset):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(asset, ensure_ascii=True), encoding="utf-8")
    return path


def build_asset(tmp_path, asset, runtime=None):
    return _build_qwen_grammar_vocabulary_from_source(
        write_asset(tmp_path, asset),
        runtime or runtime_metadata_for(asset),
        expected_base_vocab_size=len(asset["model"]["vocab"]),
        expected_vocab_size=(len(asset["model"]["vocab"]) + len(asset["added_tokens"])),
    )


def test_canonical_byte_level_alphabet_round_trips_all_256_bytes():
    piece = "".join(BYTE_LEVEL_CHARACTERS[byte] for byte in range(256))

    assert _decode_byte_level_piece(piece, label="complete alphabet") == bytes(range(256))
    assert len(set(piece)) == 256


def test_builds_ascii_space_control_partial_and_complete_multibyte_pieces(tmp_path):
    asset = {
        "model": {
            "type": "BPE",
            "byte_fallback": False,
            "vocab": {
                "A": 0,
                f"{BYTE_LEVEL_CHARACTERS[0x20]}hello": 1,
                BYTE_LEVEL_CHARACTERS[0x0A]: 2,
                BYTE_LEVEL_CHARACTERS[0x09]: 3,
                "ä½ł": 4,
                "ä¸ª": 5,
            },
        },
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [],
    }

    vocabulary = build_asset(tmp_path, asset)

    assert vocabulary == (
        b"A",
        b" hello",
        b"\n",
        b"\t",
        bytes.fromhex("e4bda0"),
        bytes.fromhex("e4b8aa"),
    )


def test_preserves_characterized_partial_utf8_token_bytes():
    assert _decode_byte_level_piece("ä", label="base token 160") == bytes.fromhex("e4")
    assert _decode_byte_level_piece("½", label="base token 121") == bytes.fromhex("bd")
    assert _decode_byte_level_piece("ł", label="base token 254") == bytes.fromhex("a0")
    assert (
        b"".join(
            _decode_byte_level_piece(piece, label=f"base token {token_id}")
            for token_id, piece in ((160, "ä"), (121, "½"), (254, "ł"))
        )
        == "你".encode()
    )
    assert _decode_byte_level_piece("ä¸ª", label="base token 18947") == "个".encode()


def test_applies_special_non_special_and_intentional_empty_added_token_policy(tmp_path):
    asset = {
        "model": {"type": "BPE", "byte_fallback": False, "vocab": {"a": 0}},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            added_token(1, "<special>", special=True),
            added_token(2, "<tool>", special=False),
            added_token(3, "", special=False),
            added_token(4, f"{BYTE_LEVEL_CHARACTERS[0x20]}x", special=False),
            added_token(5, f"{BYTE_LEVEL_CHARACTERS[0x20]}你", special=False),
        ],
    }

    vocabulary = build_asset(tmp_path, asset)

    assert vocabulary == (
        b"a",
        b"",
        b"<tool>",
        b"",
        b" x",
        f"{BYTE_LEVEL_CHARACTERS[0x20]}你".encode(),
    )


def test_construction_is_deterministic_and_returns_immutable_bytes(tmp_path):
    asset = default_asset()
    path = write_asset(tmp_path, asset)
    runtime = runtime_metadata_for(asset)

    first = _build_qwen_grammar_vocabulary_from_source(
        path,
        runtime,
        expected_base_vocab_size=2,
        expected_vocab_size=4,
    )
    second = _build_qwen_grammar_vocabulary_from_source(
        path,
        runtime,
        expected_base_vocab_size=2,
        expected_vocab_size=4,
    )

    assert first == second == (b"a", b"b", b"", b"<tool>")
    assert isinstance(first, tuple)
    assert all(isinstance(token, bytes) for token in first)
    with pytest.raises(TypeError):
        first[0] = b"changed"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_model", "model must be an object"),
        ("model_not_object", "model must be an object"),
        ("unsupported_model", "model type must be 'BPE'"),
        ("vocab_not_object", "vocab must be an object"),
        ("missing_decoder", "decoder must be an object"),
        ("decoder_not_object", "decoder must be an object"),
        ("unsupported_decoder", "decoder type must be 'ByteLevel'"),
        ("byte_fallback", "byte_fallback must be absent or false"),
        ("added_not_array", "added_tokens must be an array"),
        ("empty_base_piece", "empty base piece"),
        ("invalid_base_character", "canonical ByteLevel alphabet"),
        ("added_not_object", "must be an object"),
        ("added_content", "content must be a string"),
        ("added_boolean", "special must be a boolean"),
        ("duplicate_content", "duplicate token content"),
    ],
)
def test_rejects_malformed_or_unsupported_asset_structures(tmp_path, case, message):
    asset = default_asset()
    expected_base_vocab_size = 2
    expected_vocab_size = 4
    if case == "missing_model":
        asset.pop("model")
    elif case == "model_not_object":
        asset["model"] = []
    elif case == "unsupported_model":
        asset["model"]["type"] = "Unigram"
    elif case == "vocab_not_object":
        asset["model"]["vocab"] = []
    elif case == "missing_decoder":
        asset.pop("decoder")
    elif case == "decoder_not_object":
        asset["decoder"] = []
    elif case == "unsupported_decoder":
        asset["decoder"]["type"] = "WordPiece"
    elif case == "byte_fallback":
        asset["model"]["byte_fallback"] = True
    elif case == "added_not_array":
        asset["added_tokens"] = {}
    elif case == "empty_base_piece":
        asset["model"]["vocab"] = {"": 0, "b": 1}
    elif case == "invalid_base_character":
        asset["model"]["vocab"] = {"你": 0, "b": 1}
    elif case == "added_not_object":
        asset["added_tokens"][0] = None
    elif case == "added_content":
        asset["added_tokens"][0]["content"] = 1
    elif case == "added_boolean":
        asset["added_tokens"][0]["special"] = 1
    elif case == "duplicate_content":
        asset["added_tokens"][1]["content"] = "<special>"

    with pytest.raises(_QwenGrammarVocabularyError, match=message):
        _build_qwen_grammar_vocabulary_from_source(
            write_asset(tmp_path, asset),
            runtime_metadata_for(default_asset()),
            expected_base_vocab_size=expected_base_vocab_size,
            expected_vocab_size=expected_vocab_size,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("base_boolean", "base token .* ID must be an integer"),
        ("base_negative", "outside base range"),
        ("base_out_of_range", "outside base range"),
        ("base_duplicate", "duplicate base token ID"),
        ("added_boolean", "added token .* ID must be an integer"),
        ("added_negative", "outside added-token range"),
        ("added_overlap", "outside added-token range"),
        ("added_out_of_range", "outside added-token range"),
        ("added_duplicate", "duplicate token ID"),
        ("missing_base", "contains 1 entries; expected 2"),
        ("missing_added", "contains 1 entries; expected 2"),
    ],
)
def test_rejects_invalid_missing_duplicate_overlapping_or_noncontiguous_ids(
    tmp_path,
    case,
    message,
):
    asset = default_asset()
    if case == "base_boolean":
        asset["model"]["vocab"]["a"] = True
    elif case == "base_negative":
        asset["model"]["vocab"]["a"] = -1
    elif case == "base_out_of_range":
        asset["model"]["vocab"]["b"] = 2
    elif case == "base_duplicate":
        asset["model"]["vocab"]["b"] = 0
    elif case == "added_boolean":
        asset["added_tokens"][0]["id"] = True
    elif case == "added_negative":
        asset["added_tokens"][0]["id"] = -1
    elif case == "added_overlap":
        asset["added_tokens"][0]["id"] = 1
    elif case == "added_out_of_range":
        asset["added_tokens"][1]["id"] = 4
    elif case == "added_duplicate":
        asset["added_tokens"][1]["id"] = 2
    elif case == "missing_base":
        asset["model"]["vocab"].pop("b")
    elif case == "missing_added":
        asset["added_tokens"].pop()

    with pytest.raises(_QwenGrammarVocabularyError, match=message):
        _build_qwen_grammar_vocabulary_from_source(
            write_asset(tmp_path, asset),
            runtime_metadata_for(default_asset()),
            expected_base_vocab_size=2,
            expected_vocab_size=4,
        )


def test_rejects_duplicate_json_object_keys(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(
        '{"model":{"type":"BPE","vocab":{"a":0,"a":1}},'
        '"decoder":{"type":"ByteLevel"},"added_tokens":[]}',
        encoding="utf-8",
    )

    with pytest.raises(_QwenGrammarVocabularyError, match="duplicate JSON object key"):
        _build_qwen_grammar_vocabulary_from_source(
            path,
            _QwenGrammarVocabularyRuntimeMetadata(2, 2, {"a": 0, "b": 1}, {}),
            expected_base_vocab_size=2,
            expected_vocab_size=2,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("base_size", "runtime base vocabulary size"),
        ("complete_size", "runtime vocabulary size"),
        ("vocabulary_type", "runtime vocabulary must be a mapping"),
        ("vocabulary_count", "runtime vocabulary contains"),
        ("vocabulary_boolean", "ID must be an integer"),
        ("vocabulary_duplicate_id", "duplicate token ID"),
        ("token_content", "runtime token ID 0"),
        ("added_type", "added-token decoder must be a mapping"),
        ("added_count", "added-token decoder contains"),
        ("added_content", "added-token metadata"),
        ("added_special", "added-token metadata"),
        ("added_missing_field", "does not expose special metadata"),
    ],
)
def test_rejects_runtime_asset_and_adapter_disagreements(tmp_path, case, message):
    asset = default_asset()
    runtime = runtime_metadata_for(asset)
    values = {
        "base_vocab_size": runtime.base_vocab_size,
        "vocab_size": runtime.vocab_size,
        "vocabulary": copy.deepcopy(runtime.vocabulary),
        "added_tokens": copy.deepcopy(runtime.added_tokens),
    }
    if case == "base_size":
        values["base_vocab_size"] = 1
    elif case == "complete_size":
        values["vocab_size"] = 3
    elif case == "vocabulary_type":
        values["vocabulary"] = []
    elif case == "vocabulary_count":
        values["vocabulary"].pop("b")
    elif case == "vocabulary_boolean":
        values["vocabulary"]["a"] = True
    elif case == "vocabulary_duplicate_id":
        values["vocabulary"]["b"] = 0
    elif case == "token_content":
        values["vocabulary"].pop("a")
        values["vocabulary"]["changed"] = 0
    elif case == "added_type":
        values["added_tokens"] = []
    elif case == "added_count":
        values["added_tokens"].pop(3)
    elif case == "added_content":
        values["added_tokens"][3].content = "changed"
    elif case == "added_special":
        values["added_tokens"][2].special = False
    elif case == "added_missing_field":
        values["added_tokens"][2] = SimpleNamespace(
            content="<special>",
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
        )

    with pytest.raises(_QwenGrammarVocabularyError, match=message):
        _build_qwen_grammar_vocabulary_from_source(
            write_asset(tmp_path, asset),
            _QwenGrammarVocabularyRuntimeMetadata(**values),
            expected_base_vocab_size=2,
            expected_vocab_size=4,
        )


def test_rejects_file_read_invalid_json_and_non_object_root(tmp_path):
    runtime = runtime_metadata_for(default_asset())
    with pytest.raises(_QwenGrammarVocabularyError, match="failed to read") as missing:
        _build_qwen_grammar_vocabulary_from_source(
            tmp_path / "missing.json",
            runtime,
            expected_base_vocab_size=2,
            expected_vocab_size=4,
        )
    assert isinstance(missing.value.__cause__, OSError)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(_QwenGrammarVocabularyError, match="failed to parse") as malformed:
        _build_qwen_grammar_vocabulary_from_source(
            invalid,
            runtime,
            expected_base_vocab_size=2,
            expected_vocab_size=4,
        )
    assert malformed.value.__cause__ is not None

    root_array = tmp_path / "root-array.json"
    root_array.write_text("[]", encoding="utf-8")
    with pytest.raises(_QwenGrammarVocabularyError, match="root must be an object"):
        _build_qwen_grammar_vocabulary_from_source(
            root_array,
            runtime,
            expected_base_vocab_size=2,
            expected_vocab_size=4,
        )


def test_conversion_failures_never_become_empty_tokens(tmp_path):
    asset = {
        "model": {"type": "BPE", "byte_fallback": False, "vocab": {"a": 0}},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [added_token(1, "\ud800", special=False)],
    }

    with pytest.raises(_QwenGrammarVocabularyError, match="cannot be encoded") as error:
        build_asset(tmp_path, asset)

    assert isinstance(error.value.__cause__, UnicodeError)


def test_added_decoder_falls_back_for_the_whole_content_when_one_character_is_unknown():
    content = f"{BYTE_LEVEL_CHARACTERS[0x20]}你"
    token = _AddedTokenRecord(
        token_id=4,
        content=content,
        single_word=False,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=False,
    )

    assert _decode_added_token_content(token) == content.encode("utf-8")
    assert _decode_added_token_content(token) != b" " + "你".encode("utf-8")
