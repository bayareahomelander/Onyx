"""CUDA model loading and draft/target compatibility."""

from typing import NamedTuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from onyx_cuda.device import require_cuda
from onyx_cuda.prompt import format_prompt
from onyx_cuda.vocabulary import build_token_byte_vocabulary

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_COMPATIBILITY_MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Reply with CUDA ready."},
]


class LoadedModel(NamedTuple):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    revision: str


class LoadedModelPair(NamedTuple):
    draft: LoadedModel
    target: LoadedModel


def load_model(model_id: str = MODEL_ID) -> LoadedModel:
    """Load one tokenizer and FP16 model on cuda:0 at a resolved revision."""
    device = require_cuda()
    config = AutoConfig.from_pretrained(model_id)
    revision = getattr(config, "_commit_hash", None)
    if not revision:
        raise RuntimeError(f"Could not resolve a revision for {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        config=config,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    return LoadedModel(model, tokenizer, revision)


def _require_compatible_models(draft: LoadedModel, target: LoadedModel) -> None:
    draft_width = draft.model.config.vocab_size
    target_width = target.model.config.vocab_size
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


def load_model_pair() -> LoadedModelPair:
    """Load and validate the fixed draft/target model pair on cuda:0."""
    draft = load_model(MODEL_ID)
    target = load_model(TARGET_MODEL_ID)
    _require_compatible_models(draft, target)
    return LoadedModelPair(draft, target)
