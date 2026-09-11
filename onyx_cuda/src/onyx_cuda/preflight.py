"""Check selected model metadata and estimate memory without loading weights or using a GPU."""

import argparse
import sys
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM

from onyx_cuda._validation_report import (
    add_selection_arguments, check, evidence, positive_gib, reserve_report, selected_models, write_report,
)
from onyx_cuda.model import inspect_model, require_compatible_metadata


def estimate_memory(config, context_tokens):
    """Instantiate only a meta skeleton: no pretrained weights or device allocations."""
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=False, dtype=torch.float16)
    tensors = list(skeleton.parameters()) + list(skeleton.buffers())
    if any(tensor.device.type != "meta" for tensor in tensors):
        raise ValueError("Model construction did not honor the meta device")
    parameter_count = sum(parameter.numel() for parameter in skeleton.parameters())
    weights = parameter_count * 2
    buffers = sum(buffer.numel() * buffer.element_size() for buffer in skeleton.buffers())
    # Only use the conventional KV formula for architectures whose cache layout
    # is known here. Other supported Transformers architectures can still precheck.
    kv_bytes = None
    if config.model_type in ("qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3_text"):
        layers = getattr(config, "num_hidden_layers", None)
        heads = getattr(config, "num_attention_heads", None)
        kv_heads = getattr(config, "num_key_value_heads", heads)
        hidden = getattr(config, "hidden_size", None)
        head_dim = getattr(config, "head_dim", None) or (hidden // heads if hidden and heads else None)
        if all(isinstance(value, int) and value > 0 for value in (layers, kv_heads, head_dim)):
            kv_bytes = 2 * layers * kv_heads * head_dim * context_tokens * 2
    return {"parameter_count": parameter_count, "fp16_weight_bytes": weights,
            "buffer_bytes": buffers, "full_attention_kv_bytes": kv_bytes}


def precheck(selection, *, gamma=0, context_tokens=2048, vram_gib=None, report=None):
    if not 1 <= context_tokens <= 2048:
        raise ValueError("context_tokens must be between 1 and the API limit of 2048")
    if isinstance(gamma, bool) or not isinstance(gamma, int) or gamma < 0:
        raise ValueError("gamma must be a nonnegative integer")
    report = report if report is not None else evidence("model-preflight")
    report["settings"] = {"gamma": gamma, "context_tokens": context_tokens,
                          "vram_gib": vram_gib, "precision": "float16"}
    report["models"] = {}
    metadata = {}
    for role in ("target", "draft") if gamma > 0 else ("target",):
        model_id = getattr(selection, f"{role}_model")
        item = report["models"][role] = {"id": model_id, "revision": None}
        loaded = check(report, f"{role}.metadata", lambda: inspect_model(
            model_id, revision=getattr(selection, f"{role}_revision")))
        metadata[role] = loaded
        item.update(revision=loaded.revision, architecture=loaded.config.model_type,
                    max_position_embeddings=getattr(loaded.config, "max_position_embeddings", None))
        # Construction also checks whether the installed Transformers can create
        # this causal LM without custom remote code. A failure is a failed precheck,
        # not a claim that the model could never work with a different runtime.
        item["memory"] = check(report, f"{role}.architecture_and_memory",
                               lambda: estimate_memory(loaded.config, context_tokens + gamma))
    if gamma > 0:
        check(report, "pair.tokenizer_compatibility", lambda: require_compatible_metadata(
            metadata["draft"], metadata["target"]))
    weights = sum(item["memory"]["fp16_weight_bytes"] for item in report["models"].values())
    kv_values = [item["memory"]["full_attention_kv_bytes"] for item in report["models"].values()]
    report["memory"] = {
        "fp16_weight_bytes": weights,
        "full_attention_kv_bytes": sum(kv_values) if all(value is not None for value in kv_values) else None,
        "capacity_assessment": "weights_exceed_capacity" if vram_gib and weights >= vram_gib * 2**30 else "fit_not_established",
        "assumptions": "Batch 1; KV estimate uses context_tokens + gamma and full attention in FP16. "
                       "Sliding/hybrid caches may differ. Weights exclude runtime overhead. "
                       "Activations, attention workspaces, allocator reserve, and other GPU users are unestimated.",
    }
    report.update(status="passed", support_level="prechecked")
    report["limitations"] = ["No weights loaded, CUDA forward executed, or generation tested.",
                             "Memory estimates do not establish fit or recommend a model.",
                             "Run validate_model on the intended GPU and OS with these resolved revisions."]
    return replace(selection, target_revision=metadata["target"].revision,
                   draft_revision=metadata["draft"].revision if gamma > 0 else selection.draft_revision)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--context-tokens", type=int, choices=range(1, 2049), metavar="1..2048", default=2048)
    parser.add_argument("--vram-gib", type=positive_gib, help="Optional hypothetical GPU capacity in GiB")
    args = parser.parse_args(argv)
    reserve_report(args.output)
    report = evidence("model-preflight")
    try:
        selection = check(report, "selection", lambda: selected_models(args))
        precheck(selection, gamma=args.gamma, context_tokens=args.context_tokens,
                 vram_gib=args.vram_gib, report=report)
    except Exception as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 1
    finally:
        write_report(args.output, report)
    print(f"Prechecked; CUDA validation still required. Report: {args.output}")
    if report["memory"]["capacity_assessment"] == "weights_exceed_capacity":
        print("FP16 weights alone exceed the supplied VRAM capacity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
