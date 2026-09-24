"""Optional, model-owned exact-block recovery for the validated Qwen3 runtime.

Ordinary model forwards are never patched. Graph buffers are serialized across
callers/streams; output and cache state remain owned by each generation.
"""

import os
import threading
import time
import weakref
from types import SimpleNamespace

import torch
import transformers
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen3 import modeling_qwen3 as qwen
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from onyx_cuda.cache import CacheState, snapshot_cache
from onyx_cuda.revisions import MODEL_REVISIONS

_lifecycle_lock = threading.RLock()
_REPLAY_WIDTHS = (2, 3, 8)


def resolve_replay_backend(value=None):
    value = value if value is not None else os.environ.get("ONYX_REPLAY_BACKEND", "scalar")
    if value not in ("scalar", "graph"):
        raise ValueError("ONYX_REPLAY_BACKEND must be scalar or graph")
    return value


def _unsupported_reason(model):
    if type(model) is not qwen.Qwen3ForCausalLM:
        return "requires the pinned Qwen3-8B target"
    config = model.config
    if (getattr(model, "_onyx_model_id", None) != "Qwen/Qwen3-8B"
            or getattr(config, "_commit_hash", None) != MODEL_REVISIONS["Qwen/Qwen3-8B"]):
        return "requires the pinned Qwen3-8B target"
    if torch.__version__.split("+")[0] != "2.6.0" or transformers.__version__ != "4.57.6":
        return "requires PyTorch 2.6.0 and Transformers 4.57.6"
    if (model.training or config._attn_implementation != "sdpa"
            or model.model.has_sliding_layers or config.rope_scaling is not None):
        return "requires evaluation mode, full SDPA attention, and default rotary embeddings"
    parameters = list(model.parameters())
    device = parameters[0].device
    if device.type != "cuda" or any(p.device != device or p.dtype != torch.float16 for p in parameters):
        return "requires one CUDA device with FP16 weights"
    if torch.cuda.get_device_capability(device) != (7, 5):
        return "graph recovery is validated only on CUDA capability 7.5"
    return None


def _tiled_linear(inputs, weight, output):
    width, _ = inputs.shape
    size = weight.shape[0]
    for start in range(0, size, 256):
        end = min(start + 256, size)
        torch.bmm(inputs[:, None, :], weight[start:end].T[None].expand(width, -1, -1),
                  out=output[:, None, start:end])
    return output


class GraphReplayBackend:
    def __init__(self, model):
        self._model = weakref.ref(model)
        self.device = next(model.parameters()).device
        self._lock = threading.RLock()
        self._graphs = {}
        self._event = None
        self.closed = False
        self.fallback_reason = None
        started = time.perf_counter()
        try:
            with torch.cuda.device(self.device), torch.inference_mode():
                for module in model.modules():
                    if type(module) is torch.nn.Linear:
                        for width in _REPLAY_WIDTHS:
                            self._build(module, width)
                torch.cuda.synchronize(self.device)
        except BaseException:
            self.close()
            raise
        self.setup_seconds = time.perf_counter() - started

    def _build(self, module, width):
        inputs = torch.zeros((width, module.weight.shape[1]), device=self.device, dtype=torch.float16)
        # Allocate persistent outputs before capture to avoid a private memory
        # pool allocation for every linear/width graph. The model owns both
        # buffers; extend serializes their use and _linear clones each result.
        output = torch.empty((width, module.weight.shape[0]), device=self.device, dtype=torch.float16)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                _tiled_linear(inputs, module.weight, output)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _tiled_linear(inputs, module.weight, output)
        self._graphs[id(module), width] = inputs, output, graph

    def _linear(self, module, inputs):
        width = inputs.shape[1]
        buffer, output, graph = self._graphs[id(module), width]
        buffer.copy_(inputs.reshape_as(buffer))
        graph.replay()
        return output.clone().unsqueeze(0)

    @staticmethod
    def _norm(module, inputs):
        return torch.cat([module(inputs[:, i:i + 1].contiguous())
                          for i in range(inputs.shape[1])], dim=1)

    def supports(self, cache, ids):
        model = self._model()
        return (not self.closed and model is not None and not model.training
                and type(cache) is CacheState and type(cache.past_key_values) is DynamicCache
                and not cache.past_key_values.offloading
                and all(type(layer) is DynamicLayer for layer in cache.past_key_values.layers)
                and ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] in _REPLAY_WIDTHS
                and ids.device == self.device and ids.dtype == torch.long
                and cache.length + ids.shape[1] <= 8192
                and cache.attention_mask.shape == (1, cache.length)
                and bool(torch.all(cache.attention_mask == 1).item()))

    def extend(self, cache, ids):
        """Return None for scalar fallback, without changing an unsupported cache."""
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            if not self.supports(cache, ids):
                return None
            if self._event is not None:
                torch.cuda.current_stream(self.device).wait_event(self._event)
            trial = snapshot_cache(cache)
            exhausted = False
            try:
                result = trial.extend(self, ids)
            except torch.OutOfMemoryError:
                exhausted = True
                result = None
            finally:
                self._event = torch.cuda.Event()
                self._event.record(torch.cuda.current_stream(self.device))
            if exhausted:
                # Never publish a partially extended KV cache. Release model
                # graphs as well as temporaries so scalar recovery regains its
                # original memory budget for this and subsequent requests.
                self.fallback_reason = "CUDA memory exhausted during graph recovery"
                self.close()
                del trial
                torch.cuda.empty_cache()
                return None
            cache.past_key_values = trial.past_key_values
            cache.attention_mask = trial.attention_mask
            cache.cache_position = trial.cache_position
            return result

    def __call__(self, *, input_ids, past_key_values, cache_position, **kwargs):
        model = self._model()
        hidden = model.model.embed_tokens(input_ids)
        cos, sin = model.model.rotary_emb(hidden, cache_position.unsqueeze(0))
        width = input_ids.shape[1]
        for layer in model.model.layers:
            residual = hidden
            normalized = self._norm(layer.input_layernorm, hidden)
            attention = layer.self_attn
            shape = (1, width, -1, attention.head_dim)
            query = self._norm(attention.q_norm, self._linear(attention.q_proj, normalized).view(shape)).transpose(1, 2)
            key = self._norm(attention.k_norm, self._linear(attention.k_proj, normalized).view(shape)).transpose(1, 2)
            value = self._linear(attention.v_proj, normalized).view(shape).transpose(1, 2)
            query, key = qwen.apply_rotary_pos_emb(query, key, cos, sin)
            key, value = past_key_values.update(key, value, attention.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})
            rows = []
            for i in range(width):
                end = key.shape[2] - width + i + 1
                row, _ = sdpa_attention_forward(attention, query[:, :, i:i + 1].contiguous(),
                    key[:, :, :end].contiguous(), value[:, :, :end].contiguous(), None,
                    dropout=0.0, scaling=attention.scaling, sliding_window=attention.sliding_window)
                rows.append(row)
            attended = torch.cat(rows, 1).reshape(1, width, -1).contiguous()
            hidden = residual + self._linear(attention.o_proj, attended)
            residual = hidden
            normalized = self._norm(layer.post_attention_layernorm, hidden)
            mlp = layer.mlp
            hidden = residual + self._linear(mlp.down_proj,
                mlp.act_fn(self._linear(mlp.gate_proj, normalized)) * self._linear(mlp.up_proj, normalized))
        logits = self._linear(model.lm_head, self._norm(model.model.norm, hidden))
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if self._event is not None:
                self._event.synchronize()
            self._graphs.clear()
            self._event = None


def close_replay_backend(model):
    with _lifecycle_lock:
        backend = getattr(model, "_onyx_replay_backend", None)
        if backend is not None:
            backend.close()
            delattr(model, "_onyx_replay_backend")


def replay_backend_status(model, configuration):
    backend = getattr(model, "_onyx_replay_backend", None)
    if configuration["active"] == "graph" and (backend is None or backend.closed):
        return {**configuration, "active": "scalar",
                "reason": getattr(backend, "fallback_reason", None) or "graph backend was closed"}
    return dict(configuration)


def prepare_replay_backend(model, mode="graph"):
    """Explicit setup once per loaded model; unsupported targets remain scalar.

    Weights and model configuration must remain immutable until this backend is
    closed. Setup failures propagate after releasing partially built graphs.
    """
    with _lifecycle_lock:
        return _prepare(model, resolve_replay_backend(mode))


def _prepare(model, mode):
    existing = getattr(model, "_onyx_replay_backend", None)
    if mode == "scalar":
        close_replay_backend(model)
        return {"requested": mode, "active": "scalar", "setup_seconds": 0.0}
    reason = _unsupported_reason(model)
    if reason:
        close_replay_backend(model)
        return {"requested": mode, "active": "scalar", "reason": reason, "setup_seconds": 0.0}
    if existing is None or existing.closed:
        existing = GraphReplayBackend(model)
        model._onyx_replay_backend = existing
    return {"requested": mode, "active": "graph", "setup_seconds": existing.setup_seconds}
