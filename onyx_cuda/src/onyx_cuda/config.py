"""Runtime settings shared by the API, generators, and validation tools."""

import os
import re
from dataclasses import dataclass

from onyx_cuda.revisions import MODEL_REVISIONS

DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ENVIRONMENT_VARIABLES = (
    "ONYX_TARGET_MODEL", "ONYX_TARGET_REVISION", "ONYX_DRAFT_MODEL", "ONYX_DRAFT_REVISION",
)


def validate_model_id(value: str) -> str:
    """Accept repository IDs, never a local directory or arbitrary URL."""
    if (not isinstance(value, str) or not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", value, re.ASCII)
            or any(part.startswith((".", "-")) or part.endswith((".", "-"))
                   or ".." in part or "--" in part or len(part) > 96 for part in value.split("/"))
            or os.path.isdir(value)):
        raise ValueError("Models must be Hugging Face repository IDs, not paths or URLs")
    return value


def _setting(explicit: str | None, name: str, default: str | None = None) -> str | None:
    value = explicit if explicit is not None else os.environ.get(name, default)
    if value is not None and (not isinstance(value, str) or not value.strip() or value != value.strip()):
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


@dataclass(frozen=True)
class ModelSelection:
    target_model: str
    target_revision: str | None
    draft_model: str
    draft_revision: str | None


def resolve_model_selection(*, target_model: str | None = None, target_revision: str | None = None,
                            draft_model: str | None = None, draft_revision: str | None = None) -> ModelSelection:
    target = validate_model_id(_setting(target_model, "ONYX_TARGET_MODEL", DEFAULT_TARGET_MODEL))
    draft = validate_model_id(_setting(draft_model, "ONYX_DRAFT_MODEL", DEFAULT_DRAFT_MODEL))
    return ModelSelection(
        target, _setting(target_revision, "ONYX_TARGET_REVISION", MODEL_REVISIONS.get(target)),
        draft, _setting(draft_revision, "ONYX_DRAFT_REVISION", MODEL_REVISIONS.get(draft)),
    )


def resolve_greedy_backend(backend: str | None = None) -> str:
    value = backend if backend is not None else os.environ.get("ONYX_GREEDY_BACKEND", "torch")
    if value not in ("torch", "cuda"):
        raise ValueError("ONYX_GREEDY_BACKEND must be 'torch' or 'cuda'")
    return value


def initialize_greedy_backend(backend: str) -> None:
    """Check optional dependencies and compile kernels before accepting requests."""
    if backend == "cuda":
        import torch
        from onyx_cuda.masking import grammar_argmax

        for dtype in (torch.float16, torch.float32):
            logits = torch.tensor([[0.0, 1.0]], device="cuda", dtype=dtype)
            if grammar_argmax(logits, [1], backend=backend).item() != 1:
                raise RuntimeError("CUDA greedy selector startup check failed")
