"""CUDA model loading and draft/target compatibility."""

from typing import NamedTuple
import re

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from onyx_cuda.device import require_cuda
from onyx_cuda.config import (
    DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL, ModelSelection, resolve_model_selection, validate_model_id,
)
from onyx_cuda.prompt import format_prompt
from onyx_cuda.revisions import MODEL_REVISIONS
from onyx_cuda.vocabulary import build_token_byte_vocabulary, get_token_byte_vocabulary

MODEL_ID = DEFAULT_DRAFT_MODEL
TARGET_MODEL_ID = DEFAULT_TARGET_MODEL
_COMPATIBILITY_MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Reply with CUDA ready."},
]


class LoadedModel(NamedTuple):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    revision: str

    @property
    def model_id(self) -> str | None:
        return getattr(self.model, "_onyx_model_id", None) or getattr(
            getattr(self.model, "config", None), "_name_or_path", None
        )


class LoadedModelPair(NamedTuple):
    draft: LoadedModel | None
    target: LoadedModel


class ModelMetadata(NamedTuple):
    config: object
    tokenizer: PreTrainedTokenizerBase
    revision: str


def inspect_model(model_id: str, *, revision: str | None = None) -> ModelMetadata:
    """Resolve and validate configuration/tokenizer without loading weights or CUDA."""
    validate_model_id(model_id)
    if revision is not None and (not isinstance(revision, str) or not revision.strip() or revision != revision.strip()):
        raise ValueError("Model revision must be a nonempty string without surrounding whitespace")
    requested_revision = revision or MODEL_REVISIONS.get(model_id)
    config = AutoConfig.from_pretrained(model_id, revision=requested_revision, trust_remote_code=False)
    revision = getattr(config, "_commit_hash", None)
    if not revision:
        raise RuntimeError(f"Could not resolve a revision for {model_id}")
    if requested_revision and re.fullmatch(r"[a-fA-F0-9]{40}", requested_revision) and revision != requested_revision:
        raise RuntimeError(f"Resolved revision for {model_id} differs from the validation pin")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise RuntimeError(f"Could not resolve an immutable revision for {model_id}")

    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("Prequantized models are unsupported; this loader uses FP16 weights")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    _validate_tokenizer(tokenizer, config)
    return ModelMetadata(config, tokenizer, revision)


def load_model(model_id: str = MODEL_ID, *, revision: str | None = None) -> LoadedModel:
    """Load a pinned bundled model, or resolve an explicitly selected model/revision."""
    device = require_cuda()
    config, tokenizer, revision = inspect_model(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        config=config,
        dtype=torch.float16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    model._onyx_model_id = model_id
    loaded = LoadedModel(model, tokenizer, revision)
    _validate_runtime(loaded)
    return loaded


def _validate_tokenizer(tokenizer, config) -> None:
    get_token_byte_vocabulary(tokenizer, config.vocab_size)
    prompt = format_prompt(tokenizer, [{"role": "user", "content": "Hi"}])
    if not prompt.token_ids or tokenizer.eos_token_id is None:
        raise ValueError("A nonempty chat template and EOS token are required")


def _validate_runtime(loaded: LoadedModel) -> None:
    """Smoke-test the actual forward/cache/rollback contract before serving."""
    from transformers.cache_utils import Cache
    from onyx_cuda.cache import CacheState

    try:
        device = next(loaded.model.parameters()).device
        prompt = format_prompt(loaded.tokenizer, [{"role": "user", "content": "Hi"}])
        ids = torch.tensor([prompt.token_ids[:3]], device=device, dtype=torch.long)
        with torch.inference_mode():
            output = loaded.model(input_ids=ids, use_cache=True)
            if not isinstance(output.past_key_values, Cache):
                raise ValueError("Transformers Cache output is required")
            cache = CacheState.from_prefill(output.past_key_values, device)
            start = ids.shape[1]
            if cache.length != start:
                raise ValueError("Prefill cache length is incorrect")
            token = output.logits[:, -1, :].argmax(-1)[:, None]
            expected = cache.extend(loaded.model, token).clone()
            if cache.length != start + 1:
                raise ValueError("Cache extension is unsupported")
            cache.crop(start)
            if cache.length != start:
                raise ValueError("Cache rollback is unsupported")
            replay = cache.extend(loaded.model, token)
            if (cache.length != start + 1 or replay.shape != (1, 1, loaded.model.config.vocab_size)
                    or not torch.isfinite(replay).all()
                    or not torch.allclose(expected, replay, rtol=1e-3, atol=1e-3)):
                raise ValueError("Cache replay changed logits or produced invalid output")
    except Exception as error:
        raise RuntimeError(f"Model {loaded.model_id} failed CUDA forward/cache validation: {error}") from error


def describe_model_pair(pair) -> dict:
    """Report observed identities, including explicitly injected engines."""
    def describe(loaded):
        if loaded is None:
            return None
        model = getattr(loaded, "model", None)
        config = getattr(model, "config", None)
        dtype = getattr(model, "dtype", None)
        model_id = getattr(loaded, "model_id", None) or getattr(config, "_name_or_path", None)
        revision = getattr(loaded, "revision", None)
        return {"id": model_id, "revision": revision, "precision": str(dtype) if dtype is not None else None,
                "validated_snapshot": model_id in MODEL_REVISIONS and revision == MODEL_REVISIONS[model_id]}
    return {"target": describe(getattr(pair, "target", None)), "draft": describe(getattr(pair, "draft", None))}


def _require_compatible_models(draft: LoadedModel, target: LoadedModel) -> None:
    require_compatible_metadata(
        ModelMetadata(draft.model.config, draft.tokenizer, draft.revision),
        ModelMetadata(target.model.config, target.tokenizer, target.revision),
    )


def require_compatible_metadata(draft: ModelMetadata, target: ModelMetadata) -> None:
    """Apply the same pair contract before or after loading weights."""
    draft_width = draft.config.vocab_size
    target_width = target.config.vocab_size
    if draft_width != target_width:
        raise RuntimeError(
            f"Draft and target logits vocabulary sizes differ: "
            f"{draft_width} != {target_width}"
        )

    draft_bytes = build_token_byte_vocabulary(
        draft.tokenizer, draft_width
    ).token_bytes
    target_bytes = build_token_byte_vocabulary(
        target.tokenizer, target_width
    ).token_bytes
    if draft_bytes != target_bytes:
        mismatch_id = next(
            token_id
            for token_id, values in enumerate(zip(draft_bytes, target_bytes))
            if values[0] != values[1]
        )
        raise RuntimeError(
            f"Draft and target token bytes differ at token ID {mismatch_id}"
        )

    if draft.tokenizer.all_special_ids != target.tokenizer.all_special_ids:
        raise RuntimeError("Draft and target special token IDs differ")
    if draft.tokenizer.eos_token_id != target.tokenizer.eos_token_id:
        raise RuntimeError("Draft and target EOS token IDs differ")
    if format_prompt(draft.tokenizer, _COMPATIBILITY_MESSAGES) != format_prompt(
        target.tokenizer, _COMPATIBILITY_MESSAGES
    ):
        raise RuntimeError("Draft and target chat-template output differs")


def load_model_pair(*, include_draft: bool = True, selection: ModelSelection | None = None) -> LoadedModelPair:
    """Load the target and, when requested, a compatible draft on cuda:0."""
    selection = selection if selection is not None else resolve_model_selection()
    target = load_model(selection.target_model, revision=selection.target_revision)
    if not include_draft:
        return LoadedModelPair(None, target)
    draft = load_model(selection.draft_model, revision=selection.draft_revision)
    _require_compatible_models(draft, target)
    return LoadedModelPair(draft, target)
