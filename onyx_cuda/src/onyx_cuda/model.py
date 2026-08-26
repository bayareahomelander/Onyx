"""Single-model CUDA loading."""

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

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class LoadedModel(NamedTuple):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    revision: str


def load_model() -> LoadedModel:
    """Load the baseline tokenizer and FP16 model on cuda:0."""
    device = require_cuda()
    config = AutoConfig.from_pretrained(MODEL_ID)
    revision = getattr(config, "_commit_hash", None)
    if not revision:
        raise RuntimeError(f"Could not resolve a revision for {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        config=config,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    return LoadedModel(model, tokenizer, revision)
