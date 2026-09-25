"""Model-owned CUDA-graph decoding for the pinned Qwen2.5 draft (default; eager opts out).

The draft only proposes tokens and the target verifies every one, so this
backend changes speed, never output. Each single-token draft step replays one
CUDA graph that reads the model's own weights: fewer kernels than the ordinary
forward, a chunk-major static KV cache, and attention over only the 256-token
chunks in use. Prompts are prefilled with the ordinary forward and copied in.

One generation owns the static cache at a time. start() returns None while it
is owned, when the required length exceeds capacity, or for an unsupported
cache, and the caller keeps the ordinary dynamic cache.
"""

import math
import os
import threading
import time
import weakref

import torch
import transformers
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen2 import modeling_qwen2 as qwen2

from onyx_cuda.revisions import MODEL_REVISIONS

_lifecycle_lock = threading.RLock()
_DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_CHUNK = 256


def resolve_draft_backend(value=None):
    value = value if value is not None else os.environ.get("ONYX_DRAFT_BACKEND", "graph")
    if value not in ("eager", "graph"):
        raise ValueError("ONYX_DRAFT_BACKEND must be eager or graph")
    return value


def _unsupported_reason(model):
    if type(model) is not qwen2.Qwen2ForCausalLM:
        return "requires the pinned Qwen2.5-0.5B-Instruct draft"
    config = model.config
    if (getattr(model, "_onyx_model_id", None) != _DRAFT_ID
            or getattr(config, "_commit_hash", None) != MODEL_REVISIONS[_DRAFT_ID]):
        return "requires the pinned Qwen2.5-0.5B-Instruct draft"
    if torch.__version__.split("+")[0] != "2.6.0" or transformers.__version__ != "4.57.6":
        return "requires PyTorch 2.6.0 and Transformers 4.57.6"
    if (model.training or config.rope_scaling is not None or config.hidden_act != "silu"
            or config.use_sliding_window
            or any(kind != "full_attention" for kind in getattr(config, "layer_types", None) or ())):
        return "requires evaluation mode, default rotary embeddings, and full attention"
    parameters = list(model.parameters())
    device = parameters[0].device
    if device.type != "cuda" or any(p.device != device or p.dtype != torch.float16 for p in parameters):
        return "requires one CUDA device with FP16 weights"
    return None


class GraphDraftCache:
    """Draft cache interface used by speculation: length, device, extend, crop."""

    def __init__(self, backend, length):
        self._backend = backend
        self._length = length
        self.released = False

    @property
    def length(self) -> int:
        return self._length

    @property
    def device(self):
        return self._backend.device

    def extend(self, model, input_ids):
        return self._backend._extend(self, input_ids)

    def crop(self, length: int) -> None:
        if length < 0 or length > self._length:
            raise ValueError(f"cache length must be between 0 and {self._length}")
        # Positions beyond the length are masked and rewritten before reuse.
        self._length = length

    def release(self) -> None:
        self._backend._release(self)


class GraphDraftBackend:
    def __init__(self, model, capacity):
        self._model = weakref.ref(model)
        config = model.config
        self.device = next(model.parameters()).device
        self.capacity = math.ceil(capacity / _CHUNK) * _CHUNK
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.layer_count = config.num_hidden_layers
        self._lock = threading.RLock()
        self._owner = None
        self._event = None
        self._graphs = {}
        self.closed = False
        started = time.perf_counter()
        try:
            with torch.cuda.device(self.device), torch.inference_mode():
                self._allocate(model)
                self._capture()
                torch.cuda.synchronize(self.device)
        except BaseException:
            self.close()
            raise
        self.setup_seconds = time.perf_counter() - started

    def _allocate(self, model):
        chunks, hk, d = self.capacity // _CHUNK, self.kv_heads, self.head_dim
        root = math.sqrt(model.config.hidden_size)
        # RMSNorm is applied as x / ||x|| times sqrt(hidden) * weight; the scaled
        # weights are small per-layer vectors, and projections read model weights.
        self._layers = [{
            "attention": layer.self_attn, "mlp": layer.mlp,
            "input_norm": (layer.input_layernorm.weight.float() * root).half(),
            "post_norm": (layer.post_attention_layernorm.weight.float() * root).half(),
            "keys": torch.zeros((chunks, hk, _CHUNK, d), dtype=torch.float16, device=self.device),
            "values": torch.zeros((chunks, hk, _CHUNK, d), dtype=torch.float16, device=self.device),
        } for layer in model.model.layers]
        self._final_norm = (model.model.norm.weight.float() * root).half()
        rotary = model.model.rotary_emb
        positions = torch.arange(self.capacity, device=self.device, dtype=torch.float32)
        angles = positions[:, None] * rotary.inv_freq.float()[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        # rotate_half(x) * sin == roll(x, d / 2) * [-sin_first_half, sin_second_half].
        sign = torch.cat((-torch.ones(d // 2), torch.ones(d // 2))).to(self.device)
        self._cos = (angles.cos() * rotary.attention_scaling).half()
        self._sin = (angles.sin() * rotary.attention_scaling * sign).half()
        self._key_positions = torch.arange(self.capacity, device=self.device)
        self._ids = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._position = torch.zeros((1,), dtype=torch.long, device=self.device)

    @staticmethod
    def _normalize(hidden, weight):
        norm = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True, dtype=torch.float32)
        return (hidden / norm).half() * weight

    def _step(self, chunks):
        model = self._model()
        hq, hk, d = self.heads, self.kv_heads, self.head_dim
        group, span = hq // hk, chunks * _CHUNK
        hidden = model.model.embed_tokens.weight.index_select(0, self._ids)
        cos = self._cos.index_select(0, self._position)
        sin = self._sin.index_select(0, self._position)
        chunk = torch.div(self._position, _CHUNK, rounding_mode="floor")
        slot = self._position - chunk * _CHUNK
        mask = torch.where(self._key_positions[:span] <= self._position, 0.0, float("-inf")).half()
        mask = mask.view(chunks, 1, 1, _CHUNK).expand(chunks, hk, 1, _CHUNK).reshape(chunks * hk, 1, _CHUNK)
        for layer in self._layers:
            attention, mlp = layer["attention"], layer["mlp"]
            normalized = self._normalize(hidden, layer["input_norm"])
            qkv = torch.empty((1, (hq + 2 * hk) * d), dtype=torch.float16, device=self.device)
            for projection, start, end in ((attention.q_proj, 0, hq * d),
                                           (attention.k_proj, hq * d, (hq + hk) * d),
                                           (attention.v_proj, (hq + hk) * d, (hq + 2 * hk) * d)):
                torch.addmm(projection.bias, normalized, projection.weight.t(), out=qkv[:, start:end])
            qk = qkv[:, :(hq + hk) * d].view(hq + hk, d)
            qk = torch.addcmul(qk * cos, qk.roll(d // 2, dims=-1), sin)
            layer["keys"][chunk, :, slot] = qk[hq:][None]
            layer["values"][chunk, :, slot] = qkv[:, (hq + hk) * d:].view(1, hk, d)
            keys = layer["keys"][:chunks].view(chunks * hk, _CHUNK, d)
            values = layer["values"][:chunks].view(chunks * hk, _CHUNK, d)
            query = qk[:hq].view(1, hk, group, d).expand(chunks, hk, group, d).reshape(chunks * hk, group, d)
            scores = torch.baddbmm(mask, query, keys.transpose(1, 2), alpha=attention.scaling)
            # Softmax spans every used chunk of each head, then chunks reduce in parallel.
            scores = scores.view(chunks, hk, group, _CHUNK).permute(1, 2, 0, 3).reshape(hk, group, span)
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = probabilities.view(hk, group, chunks, _CHUNK).permute(2, 0, 1, 3)
            attended = torch.bmm(probabilities.reshape(chunks * hk, group, _CHUNK), values)
            attended = attended.view(chunks, hq * d).sum(0, keepdim=True)
            hidden = torch.addmm(hidden, attended, attention.o_proj.weight.t())
            normalized = self._normalize(hidden, layer["post_norm"])
            gate = normalized @ mlp.gate_proj.weight.t()
            up = normalized @ mlp.up_proj.weight.t()
            hidden = torch.addmm(hidden, torch.nn.functional.silu(gate) * up, mlp.down_proj.weight.t())
        return self._normalize(hidden, self._final_norm) @ model.lm_head.weight.t()

    def _capture(self):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for chunks in (1, self.capacity // _CHUNK):
                for _ in range(2):
                    self._step(chunks)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        pool = None
        for chunks in range(1, self.capacity // _CHUNK + 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                logits = self._step(chunks)
            pool = graph.pool()
            self._graphs[chunks] = graph, logits

    def _supports(self, past_key_values):
        return (type(past_key_values) is DynamicCache and not past_key_values.offloading
                and len(past_key_values.layers) == self.layer_count
                and all(type(layer) is DynamicLayer and layer.keys is not None
                        and layer.keys.shape[:2] == (1, self.kv_heads) and layer.keys.shape[-1] == self.head_dim
                        and layer.keys.dtype == torch.float16 and layer.keys.device == self.device
                        for layer in past_key_values.layers))

    def _fence(self):
        if self._event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._event)

    def _record(self):
        self._event = torch.cuda.Event()
        self._event.record(torch.cuda.current_stream(self.device))

    def start(self, past_key_values, required_length):
        """Copy a prefilled dynamic cache into the static cache and return its owner."""
        with self._lock:
            owner = self._owner() if self._owner is not None else None
            if (self.closed or (owner is not None and not owner.released)
                    or required_length > self.capacity or not self._supports(past_key_values)):
                return None
            length = past_key_values.get_seq_length()
            if length > required_length:
                return None
            with torch.cuda.device(self.device), torch.inference_mode():
                self._fence()
                for layer, source in zip(self._layers, past_key_values.layers):
                    for start in range(0, length, _CHUNK):
                        end = min(start + _CHUNK, length)
                        layer["keys"][start // _CHUNK, :, :end - start] = source.keys[0, :, start:end]
                        layer["values"][start // _CHUNK, :, :end - start] = source.values[0, :, start:end]
                self._record()
            cache = GraphDraftCache(self, length)
            self._owner = weakref.ref(cache)
            return cache

    def _extend(self, cache, input_ids):
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            if self.closed or self._owner is None or self._owner() is not cache or cache.released:
                raise RuntimeError("Draft graph cache is no longer active")
            if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
                    or input_ids.device != self.device or input_ids.dtype != torch.long):
                raise ValueError("draft graph decoding requires one sequence of token IDs on the draft device")
            width = input_ids.shape[1]
            if cache.length + width > self.capacity:
                raise ValueError("draft graph cache capacity exceeded")
            self._fence()
            rows = []
            for i in range(width):
                position = cache.length + i
                graph, logits = self._graphs[position // _CHUNK + 1]
                self._ids.copy_(input_ids[0, i:i + 1])
                self._position.fill_(position)
                graph.replay()
                rows.append(logits.clone())
            self._record()
            cache._length += width
            return rows[0].unsqueeze(1) if width == 1 else torch.stack(rows, dim=1)

    def _release(self, cache):
        with self._lock:
            cache.released = True
            if self._owner is not None and self._owner() is cache:
                self._owner = None

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if self._event is not None:
                self._event.synchronize()
            self._graphs.clear()
            self._layers = []
            self._cos = self._sin = self._key_positions = self._ids = self._position = self._final_norm = None
            self._event = None


def close_draft_backend(model):
    with _lifecycle_lock:
        backend = getattr(model, "_onyx_draft_backend", None)
        if backend is not None:
            backend.close()
            delattr(model, "_onyx_draft_backend")


def draft_backend_status(model, configuration):
    backend = getattr(model, "_onyx_draft_backend", None)
    if configuration["active"] == "graph" and (backend is None or backend.closed):
        return {**configuration, "active": "eager", "reason": "draft graph backend was closed"}
    return dict(configuration)


def prepare_draft_backend(model, mode="graph", *, capacity=8192):
    """Explicit setup once per loaded draft; unsupported drafts remain eager.

    capacity is the longest prompt-plus-output length decoded with graphs;
    longer generations use the ordinary cache. Weights and configuration must
    remain unchanged until this backend is closed. Setup failures propagate
    after releasing partially built graphs.
    """
    with _lifecycle_lock:
        mode = resolve_draft_backend(mode)
        existing = getattr(model, "_onyx_draft_backend", None)
        if mode == "eager":
            close_draft_backend(model)
            return {"requested": mode, "active": "eager", "setup_seconds": 0.0}
        reason = _unsupported_reason(model)
        if reason:
            close_draft_backend(model)
            return {"requested": mode, "active": "eager", "reason": reason, "setup_seconds": 0.0}
        if existing is not None and (existing.closed or existing.capacity < capacity):
            close_draft_backend(model)
            existing = None
        if existing is None:
            existing = GraphDraftBackend(model, capacity)
            model._onyx_draft_backend = existing
        return {"requested": mode, "active": "graph", "capacity": existing.capacity,
                "setup_seconds": existing.setup_seconds}
