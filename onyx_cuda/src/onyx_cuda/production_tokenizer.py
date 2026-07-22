"""Lazy production tokenizer loading for the pinned Qwen target profile."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .model_profile import DEFAULT_TARGET_PROFILE, QwenModelProfile
from .qwen_grammar_vocabulary import (
    _QwenGrammarVocabularyError,
    _QwenGrammarVocabularyRuntimeMetadata,
    _build_qwen_grammar_vocabulary_from_source,
)


class ProductionTokenizerError(RuntimeError):
    """Base error raised by production tokenizer loading or execution."""


class TransformersImportError(ProductionTokenizerError):
    """Raised when an optional Hugging Face tokenizer dependency cannot be imported.

    The historical name remains part of the public API even though tokenizer-only loading no
    longer imports the top-level Transformers package, which imports PyTorch in current releases.
    """


class ProductionTokenizerLoadError(ProductionTokenizerError):
    """Raised when the pinned tokenizer cannot be loaded or validated."""


@dataclass(frozen=True, slots=True)
class ProductionTokenizerLoad:
    """A loaded tokenizer adapter and reproducible load metadata."""

    tokenizer: QwenTokenizerAdapter
    load_seconds: float

    def __post_init__(self) -> None:
        if self.load_seconds < 0:
            raise ValueError("load_seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class QwenTokenizerFingerprint:
    """Stable hashes of token IDs, special tokens, and chat-template metadata."""

    tokenizer_id: str
    vocab_size: int
    base_vocab_size: int
    eos_token_id: int
    pad_token_id: int
    vocabulary_sha256: str
    special_tokens_sha256: str
    chat_template_sha256: str


@dataclass(frozen=True, slots=True)
class _AddedTokenMetadata:
    """Transformers-compatible metadata reconstructed from tokenizer_config.json."""

    content: str
    lstrip: bool
    normalized: bool
    rstrip: bool
    single_word: bool
    special: bool


@dataclass(frozen=True, slots=True)
class _QwenGrammarVocabularySource:
    """The exact tokenizer asset and Tokenizers runtime used by the production adapter."""

    tokenizer_json_path: Path
    runtime_tokenizer: Any


class _QwenFastTokenizer:
    """Small compatibility facade over the framework-free Tokenizers runtime."""

    def __init__(
        self,
        tokenizer: Any,
        configuration: Mapping[str, Any],
        tokenizer_json_path: Any,
    ) -> None:
        self._tokenizer = tokenizer
        try:
            source_path = Path(tokenizer_json_path)
        except (TypeError, ValueError) as exc:
            raise ProductionTokenizerLoadError(
                f"the pinned tokenizer.json path is invalid: {exc}"
            ) from exc
        self._grammar_vocabulary_source = _QwenGrammarVocabularySource(
            tokenizer_json_path=source_path,
            runtime_tokenizer=tokenizer,
        )
        try:
            self._vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)
            self._complete_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        except Exception as exc:
            raise ProductionTokenizerLoadError(
                f"the fast tokenizer does not expose vocabulary sizes: {exc}"
            ) from exc

        self._vocab_size = _validate_positive_int(
            self._vocab_size,
            label="base vocabulary size",
        )
        self._complete_vocab_size = _validate_positive_int(
            self._complete_vocab_size,
            label="tokenizer size",
        )
        if self._vocab_size > self._complete_vocab_size:
            raise ProductionTokenizerLoadError(
                "base vocabulary size cannot exceed the complete tokenizer size"
            )

        metadata_by_token, added_vocabulary = _parse_added_token_metadata(
            configuration,
            tokenizer,
            self._complete_vocab_size,
        )
        expected_added_tokens = self._complete_vocab_size - self._vocab_size
        if len(added_vocabulary) != expected_added_tokens:
            raise ProductionTokenizerLoadError(
                f"tokenizer configuration contains {len(added_vocabulary)} added tokens; "
                f"expected {expected_added_tokens}"
            )
        eos_token = _required_config_string(configuration, "eos_token")
        pad_token = _required_config_string(configuration, "pad_token")
        additional_special_tokens = _required_config_strings(
            configuration,
            "additional_special_tokens",
        )
        self.chat_template = _required_config_string(configuration, "chat_template")

        self.eos_token_id = _configured_token_id(
            tokenizer,
            eos_token,
            self._complete_vocab_size,
            label="EOS token",
        )
        self.pad_token_id = _configured_token_id(
            tokenizer,
            pad_token,
            self._complete_vocab_size,
            label="pad token",
        )
        try:
            eos_metadata = metadata_by_token[eos_token]
            pad_metadata = metadata_by_token[pad_token]
        except KeyError as exc:
            raise ProductionTokenizerLoadError(
                f"configured special token {exc.args[0]!r} has no added-token metadata"
            ) from exc

        all_special_tokens = []
        for token in (eos_token, pad_token, *additional_special_tokens):
            _configured_token_id(
                tokenizer,
                token,
                self._complete_vocab_size,
                label="special token",
            )
            if token not in all_special_tokens:
                all_special_tokens.append(token)

        self._added_vocabulary = added_vocabulary
        self.special_tokens_map_extended = {
            "eos_token": eos_metadata,
            "pad_token": pad_metadata,
            "additional_special_tokens": list(additional_special_tokens),
        }
        self.all_special_tokens = all_special_tokens
        self.all_special_ids = [
            _configured_token_id(
                tokenizer,
                token,
                self._complete_vocab_size,
                label="special token",
            )
            for token in all_special_tokens
        ]

        if not callable(getattr(tokenizer, "encode", None)):
            raise ProductionTokenizerLoadError("the fast tokenizer does not provide encode()")
        if not callable(getattr(tokenizer, "decode", None)):
            raise ProductionTokenizerLoadError("the fast tokenizer does not provide decode()")
        if not callable(getattr(tokenizer, "get_vocab", None)):
            raise ProductionTokenizerLoadError("the fast tokenizer does not provide get_vocab()")

    def __len__(self) -> int:
        return self._complete_vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]:
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        try:
            return encoded.ids
        except AttributeError as exc:
            raise ProductionTokenizerLoadError(
                "the fast tokenizer encode result does not expose token IDs"
            ) from exc

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del clean_up_tokenization_spaces
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def get_vocab(self) -> Mapping[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def get_added_vocab(self) -> Mapping[str, int]:
        return dict(self._added_vocabulary)


class QwenTokenizerAdapter:
    """Adapt a Hugging Face Qwen tokenizer to the framework-neutral contract."""

    def __init__(self, tokenizer: Any, profile: QwenModelProfile) -> None:
        self._profile = profile
        self._tokenizer = tokenizer
        self._grammar_vocabulary_source = (
            tokenizer._grammar_vocabulary_source
            if isinstance(tokenizer, _QwenFastTokenizer)
            else None
        )

        try:
            tokenizer_size = len(tokenizer)
            base_vocab_size = tokenizer.vocab_size
            eos_token_id = tokenizer.eos_token_id
            pad_token_id = tokenizer.pad_token_id
        except (AttributeError, TypeError) as exc:
            raise ProductionTokenizerLoadError(
                "the loaded tokenizer does not expose the required Qwen metadata"
            ) from exc

        self._vocab_size = _validate_positive_int(tokenizer_size, label="tokenizer size")
        self._base_vocab_size = _validate_positive_int(
            base_vocab_size, label="base vocabulary size"
        )
        if self._base_vocab_size > self._vocab_size:
            raise ProductionTokenizerLoadError(
                "base vocabulary size cannot exceed the complete tokenizer size"
            )
        self._eos_token_id = _validate_token_id(
            eos_token_id, self._vocab_size, label="EOS token"
        )
        self._pad_token_id = _validate_token_id(
            pad_token_id, self._vocab_size, label="pad token"
        )

        if not callable(getattr(tokenizer, "encode", None)):
            raise ProductionTokenizerLoadError("the loaded tokenizer does not provide encode()")
        if not callable(getattr(tokenizer, "decode", None)):
            raise ProductionTokenizerLoadError("the loaded tokenizer does not provide decode()")

    @property
    def tokenizer_id(self) -> str:
        return self._profile.pinned_id

    @property
    def profile(self) -> QwenModelProfile:
        return self._profile

    @property
    def vocab_size(self) -> int:
        """Return all token IDs understood by the tokenizer, including added tokens."""

        return self._vocab_size

    @property
    def base_vocab_size(self) -> int:
        return self._base_vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def compatibility_fingerprint(self) -> QwenTokenizerFingerprint:
        """Return deterministic metadata for exact cross-profile tokenizer comparison."""

        try:
            vocabulary = self._tokenizer.get_vocab()
            added_vocabulary = self._tokenizer.get_added_vocab()
            special_tokens = self._tokenizer.special_tokens_map_extended
            all_special_ids = self._tokenizer.all_special_ids
            all_special_tokens = self._tokenizer.all_special_tokens
            chat_template = self._tokenizer.chat_template
        except (AttributeError, TypeError) as exc:
            raise ProductionTokenizerError(
                "Qwen tokenizer does not expose compatibility metadata"
            ) from exc

        vocabulary_payload = _canonical_vocabulary(vocabulary, self._vocab_size)
        special_payload = _canonical_json(
            {
                "added_vocabulary": added_vocabulary,
                "all_special_ids": all_special_ids,
                "all_special_tokens": all_special_tokens,
                "special_tokens": special_tokens,
            }
        )
        chat_template_payload = _canonical_json(chat_template)
        return QwenTokenizerFingerprint(
            tokenizer_id=self.tokenizer_id,
            vocab_size=self.vocab_size,
            base_vocab_size=self.base_vocab_size,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vocabulary_sha256=_sha256(vocabulary_payload),
            special_tokens_sha256=_sha256(special_payload),
            chat_template_sha256=_sha256(chat_template_payload),
        )

    def encode(self, text: str, /) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        try:
            encoded = self._tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise ProductionTokenizerError(f"Qwen tokenizer encode failed: {exc}") from exc

        token_ids = tuple(encoded)
        _validate_token_ids(token_ids, self._vocab_size)
        return token_ids

    def decode(self, token_ids: Sequence[int], /) -> str:
        normalized = tuple(token_ids)
        _validate_token_ids(normalized, self._vocab_size)

        try:
            decoded = self._tokenizer.decode(
                normalized,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise ProductionTokenizerError(f"Qwen tokenizer decode failed: {exc}") from exc
        if not isinstance(decoded, str):
            raise ProductionTokenizerError("Qwen tokenizer decode returned a non-string result")
        return decoded


def build_qwen_grammar_vocabulary(
    tokenizer: QwenTokenizerAdapter,
    /,
) -> tuple[bytes, ...]:
    """Construct the exact token-ID-indexed byte vocabulary for a loaded Qwen tokenizer."""

    if not isinstance(tokenizer, QwenTokenizerAdapter):
        raise TypeError("tokenizer must be a QwenTokenizerAdapter")

    try:
        source = tokenizer._grammar_vocabulary_source
        if source is None:
            raise _QwenGrammarVocabularyError(
                "Qwen tokenizer adapter has no production tokenizer.json source"
            )
        runtime = source.runtime_tokenizer
        runtime_metadata = _QwenGrammarVocabularyRuntimeMetadata(
            base_vocab_size=runtime.get_vocab_size(with_added_tokens=False),
            vocab_size=runtime.get_vocab_size(with_added_tokens=True),
            vocabulary=runtime.get_vocab(with_added_tokens=True),
            added_tokens=runtime.get_added_tokens_decoder(),
        )
        return _build_qwen_grammar_vocabulary_from_source(
            source.tokenizer_json_path,
            runtime_metadata,
            expected_base_vocab_size=tokenizer.base_vocab_size,
            expected_vocab_size=tokenizer.vocab_size,
        )
    except Exception as exc:
        raise ProductionTokenizerError(
            f"failed to build Qwen grammar vocabulary: {exc}"
        ) from exc


def load_qwen_tokenizer(
    profile: QwenModelProfile = DEFAULT_TARGET_PROFILE,
    *,
    local_files_only: bool = False,
) -> ProductionTokenizerLoad:
    """Load the pinned fast tokenizer without importing a tensor or CUDA runtime."""

    if not isinstance(profile, QwenModelProfile):
        raise TypeError("profile must be a QwenModelProfile")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean")

    modules = {}
    for module_name in ("huggingface_hub", "tokenizers"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise TransformersImportError(
                f"tokenizer dependency {module_name} could not be imported: {exc}"
            ) from exc

    start = time.perf_counter()
    try:
        tokenizer_path = modules["huggingface_hub"].hf_hub_download(
            repo_id=profile.model_id,
            filename="tokenizer.json",
            revision=profile.revision,
            local_files_only=local_files_only,
        )
        configuration_path = modules["huggingface_hub"].hf_hub_download(
            repo_id=profile.model_id,
            filename="tokenizer_config.json",
            revision=profile.revision,
            local_files_only=local_files_only,
        )
        configuration = _read_tokenizer_configuration(configuration_path)
        tokenizer_factory = getattr(modules["tokenizers"], "Tokenizer", None)
        from_file = getattr(tokenizer_factory, "from_file", None)
        if not callable(from_file):
            raise ProductionTokenizerLoadError(
                "the tokenizers dependency does not provide Tokenizer.from_file()"
            )
        tokenizer = _QwenFastTokenizer(
            from_file(str(tokenizer_path)),
            configuration,
            tokenizer_path,
        )
        adapter = QwenTokenizerAdapter(tokenizer, profile)
    except ProductionTokenizerLoadError:
        raise
    except Exception as exc:
        raise ProductionTokenizerLoadError(
            f"failed to load tokenizer {profile.pinned_id}: {exc}"
        ) from exc

    return ProductionTokenizerLoad(
        tokenizer=adapter,
        load_seconds=time.perf_counter() - start,
    )


def _read_tokenizer_configuration(path: Any) -> Mapping[str, Any]:
    try:
        payload = Path(path).read_text(encoding="utf-8")
        configuration = json.loads(payload)
    except (OSError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionTokenizerLoadError(
            f"failed to read pinned tokenizer configuration: {exc}"
        ) from exc
    if not isinstance(configuration, Mapping):
        raise ProductionTokenizerLoadError("tokenizer configuration must be a JSON object")
    return configuration


def _parse_added_token_metadata(
    configuration: Mapping[str, Any],
    tokenizer: Any,
    vocab_size: int,
) -> tuple[dict[str, _AddedTokenMetadata], dict[str, int]]:
    raw_decoder = configuration.get("added_tokens_decoder")
    if not isinstance(raw_decoder, Mapping):
        raise ProductionTokenizerLoadError(
            "tokenizer configuration added_tokens_decoder must be an object"
        )

    metadata_by_token = {}
    added_vocabulary = {}
    for raw_token_id, raw_metadata in raw_decoder.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as exc:
            raise ProductionTokenizerLoadError(
                f"added-token ID {raw_token_id!r} is not an integer"
            ) from exc
        _validate_token_id(token_id, vocab_size, label="added token")
        if not isinstance(raw_metadata, Mapping):
            raise ProductionTokenizerLoadError(
                f"added-token metadata for ID {token_id} must be an object"
            )
        content = _required_config_string(raw_metadata, "content")
        if content in metadata_by_token:
            raise ProductionTokenizerLoadError(
                f"tokenizer configuration contains duplicate added token {content!r}"
            )
        actual_token_id = _configured_token_id(
            tokenizer,
            content,
            vocab_size,
            label="added token",
        )
        if actual_token_id != token_id:
            raise ProductionTokenizerLoadError(
                f"added token {content!r} has ID {actual_token_id}; expected {token_id}"
            )
        metadata = _AddedTokenMetadata(
            content=content,
            lstrip=_required_config_bool(raw_metadata, "lstrip"),
            normalized=_required_config_bool(raw_metadata, "normalized"),
            rstrip=_required_config_bool(raw_metadata, "rstrip"),
            single_word=_required_config_bool(raw_metadata, "single_word"),
            special=_required_config_bool(raw_metadata, "special"),
        )
        metadata_by_token[content] = metadata
        added_vocabulary[content] = token_id
    return metadata_by_token, added_vocabulary


def _required_config_string(configuration: Mapping[str, Any], field: str) -> str:
    value = configuration.get(field)
    if not isinstance(value, str):
        raise ProductionTokenizerLoadError(
            f"tokenizer configuration {field} must be a string"
        )
    return value


def _required_config_strings(
    configuration: Mapping[str, Any],
    field: str,
) -> tuple[str, ...]:
    value = configuration.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProductionTokenizerLoadError(
            f"tokenizer configuration {field} must be a list of strings"
        )
    return tuple(value)


def _required_config_bool(configuration: Mapping[str, Any], field: str) -> bool:
    value = configuration.get(field)
    if not isinstance(value, bool):
        raise ProductionTokenizerLoadError(
            f"tokenizer configuration {field} must be a boolean"
        )
    return value


def _configured_token_id(
    tokenizer: Any,
    token: str,
    vocab_size: int,
    *,
    label: str,
) -> int:
    try:
        token_id = tokenizer.token_to_id(token)
    except Exception as exc:
        raise ProductionTokenizerLoadError(
            f"{label} {token!r} could not be resolved: {exc}"
        ) from exc
    if token_id is None:
        raise ProductionTokenizerLoadError(f"{label} {token!r} is absent from the tokenizer")
    return _validate_token_id(token_id, vocab_size, label=label)


def _validate_positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionTokenizerLoadError(f"{label} must be an integer")
    if value <= 0:
        raise ProductionTokenizerLoadError(f"{label} must be greater than zero")
    return value


def _validate_token_id(value: int, vocab_size: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionTokenizerLoadError(f"{label} ID must be an integer")
    if value < 0 or value >= vocab_size:
        raise ProductionTokenizerLoadError(
            f"{label} ID {value} is outside tokenizer range [0, {vocab_size})"
        )
    return value


def _validate_token_ids(token_ids: Sequence[int], vocab_size: int) -> None:
    for position, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"token ID at position {position} must be an integer")
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"token ID {token_id} at position {position} is outside tokenizer range "
                f"[0, {vocab_size})"
            )


def _canonical_vocabulary(vocabulary: Any, vocab_size: int) -> bytes:
    if not isinstance(vocabulary, Mapping):
        raise ProductionTokenizerError("Qwen tokenizer vocabulary must be a mapping")

    normalized = []
    seen_ids = set()
    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            raise ProductionTokenizerError("Qwen tokenizer vocabulary keys must be strings")
        _validate_token_id(token_id, vocab_size, label="vocabulary token")
        if token_id in seen_ids:
            raise ProductionTokenizerError(
                f"Qwen tokenizer vocabulary contains duplicate token ID {token_id}"
            )
        seen_ids.add(token_id)
        normalized.append((token_id, token))
    if len(normalized) != vocab_size:
        raise ProductionTokenizerError(
            f"Qwen tokenizer vocabulary contains {len(normalized)} entries; "
            f"expected {vocab_size}"
        )
    return _canonical_json(sorted(normalized))


def _canonical_json(value: Any) -> bytes:
    normalized = _normalize_metadata(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProductionTokenizerError(
            "Qwen tokenizer compatibility metadata is not serializable"
        ) from exc
    return text.encode("utf-8")


def _normalize_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalize_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata(item) for item in value]
    if hasattr(value, "content"):
        return {
            "content": str(value.content),
            "lstrip": bool(getattr(value, "lstrip", False)),
            "normalized": bool(getattr(value, "normalized", False)),
            "rstrip": bool(getattr(value, "rstrip", False)),
            "single_word": bool(getattr(value, "single_word", False)),
            "special": bool(getattr(value, "special", False)),
        }
    raise ProductionTokenizerError(
        f"unsupported tokenizer compatibility metadata type: {type(value).__name__}"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
