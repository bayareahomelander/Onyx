# single-model baseline engine for benchmarking. production uses SpeculativeEngine.

from abc import ABC, abstractmethod
from typing import Optional, Generator, List, Callable, Any, Tuple
import time

import mlx.core as mx
import mlx.nn as nn

import onyx

_GrammarConstraint = None
if onyx.RUST_AVAILABLE:
    try:
        from onyx._rust import GrammarConstraint as _GrammarConstraint
    except ImportError:
        pass


class BaseKVCache(ABC):
    
    @abstractmethod
    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        pass
    
    @abstractmethod
    def size(self) -> int:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass


class NaiveKVCache(BaseKVCache):
    def __init__(self, step: int = 256):
        self.step = step
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
    
    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        prev = self.offset
        
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, seq_len, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
        
        self.offset += keys.shape[2]
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values
        
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
    
    def size(self) -> int:
        return self.offset
    
    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.offset = 0
    
    @property
    def state(self) -> Optional[Tuple[mx.array, mx.array]]:
        if self.keys is None:
            return None
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]


class KVCacheManager:
    def __init__(self, num_layers: int, cache_class: type = NaiveKVCache, **kwargs):
        self.num_layers = num_layers
        self.cache_class = cache_class
        self.cache_kwargs = kwargs
        self.caches: List[BaseKVCache] = self._create_caches()
    
    def _create_caches(self) -> List[BaseKVCache]:
        return [self.cache_class(**self.cache_kwargs) for _ in range(self.num_layers)]
    
    def reset(self) -> None:
        for cache in self.caches:
            cache.reset()
    
    def total_size(self) -> int:
        if not self.caches:
            return 0
        return self.caches[0].size()
    
    def as_list(self) -> List[BaseKVCache]:
        return self.caches


class OnyxEngine:
    DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        lazy_load: bool = False,
    ):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.cache_manager: Optional[KVCacheManager] = None
        
        self.vocab_bytes: Optional[List[bytes]] = None
        
        self._load_time: Optional[float] = None
        
        self._verify_backend()
        
        if not lazy_load:
            self.load_model()
    
    def _verify_backend(self) -> None:
        if not onyx.RUST_AVAILABLE:
            import warnings
            warnings.warn(
                "Rust backend not available. Run 'maturin develop' to build it. "
                "Falling back to Python implementation."
            )
    
    def load_model(self) -> float:
        from mlx_lm import load
        
        start = time.perf_counter()
        self.model, self.tokenizer = load(self.model_path)
        self._load_time = time.perf_counter() - start
        
        num_layers = len(self.model.layers)
        self.cache_manager = KVCacheManager(num_layers, NaiveKVCache)
        
        self._build_vocab_bytes()
        
        return self._load_time
    
    def _build_vocab_bytes(self) -> None:
        vocab_size = self.tokenizer.vocab_size
        self.vocab_bytes = []
        
        for token_id in range(vocab_size):
            try:
                token_str = self.tokenizer.decode([token_id])
                token_bytes = token_str.encode('utf-8')
                self.vocab_bytes.append(token_bytes)
            except Exception:
                self.vocab_bytes.append(b"")
    
    def _create_attention_mask(
        self, 
        seq_len: int, 
        cache_offset: int = 0
    ) -> Optional[mx.array]:
        if seq_len <= 1:
            return None
            
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
        
        if cache_offset > 0:
            prefix = mx.zeros((seq_len, cache_offset))
            mask = mx.concatenate([prefix, mask], axis=1)
        
        return mask
    
    def _sample_token(
        self,
        logits: mx.array,
        temperature: float = 0.0,
    ) -> mx.array:
        if temperature == 0.0:
            return mx.argmax(logits, axis=-1)
        else:
            scaled_logits = logits / temperature
            return mx.random.categorical(scaled_logits)
    
    def _apply_grammar_mask(
        self,
        logits: mx.array,
        valid_token_ids: List[int],
    ) -> mx.array:

        actual_vocab_size = logits.shape[-1]
        
        valid_mask = mx.zeros((actual_vocab_size,), dtype=mx.bool_)
        
        if valid_token_ids:
            valid_ids = [tid for tid in valid_token_ids if tid < actual_vocab_size]
            if valid_ids:
                valid_indices = mx.array(valid_ids)
                valid_mask = valid_mask.at[valid_indices].add(True)
        
        mask = mx.where(valid_mask, mx.zeros((actual_vocab_size,)), mx.full((actual_vocab_size,), float('-inf')))
        
        return logits + mask
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_tokens: Optional[List[int]] = None,
        regex: Optional[str] = None,
    ) -> Tuple[str, dict]:
        if self.model is None:
            self.load_model()
        
        self.cache_manager.reset()
        
        if stop_tokens is None:
            stop_tokens = []
            if hasattr(self.tokenizer, 'eos_token_id'):
                eos = self.tokenizer.eos_token_id
                if isinstance(eos, int):
                    stop_tokens.append(eos)
                elif isinstance(eos, list):
                    stop_tokens.extend(eos)
        
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([prompt_tokens])
        
        generated_tokens = []
        metrics = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": 0,
            "ttft": None,
            "generation_time": 0.0,
            "tokens_per_second": 0.0,
            "grammar_constrained": regex is not None,
            "mask_time_total": 0.0,
            "mask_time_avg": 0.0,
        }
        
        grammar_constraint = None
        grammar_state = None
        
        if regex is not None:
            if _GrammarConstraint is None:
                raise RuntimeError(
                    "Grammar constraints require the Rust backend. "
                    "Run 'maturin develop' to build it."
                )
            
            grammar_constraint = _GrammarConstraint(self.vocab_bytes)
            
            compile_start = time.perf_counter()
            grammar_constraint.compile_regex(regex)
            metrics["grammar_compile_time"] = time.perf_counter() - compile_start
            
            grammar_state = grammar_constraint.init_state()
        
        generation_start = time.perf_counter()
        first_token_time = None
        mask_times = []
        
        cache_list = self.cache_manager.as_list()
        
        logits = self.model(input_ids, cache=cache_list)
        mx.eval(logits)
        
        last_logits = logits[:, -1, :]
        
        if grammar_constraint is not None:
            mask_start = time.perf_counter()
            valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
            mask_time = time.perf_counter() - mask_start
            mask_times.append(mask_time)
            
            if valid_tokens:
                last_logits = self._apply_grammar_mask(last_logits, valid_tokens)
            else:
                raise ValueError(
                    f"Grammar constraint has no valid tokens at initial state. "
                    f"Pattern: {regex}"
                )
        
        next_token = self._sample_token(last_logits, temperature)
        mx.eval(next_token)
        
        first_token_time = time.perf_counter()
        metrics["ttft"] = first_token_time - generation_start
        
        token_id = next_token.item()
        generated_tokens.append(token_id)
        
        grammar_complete = False
        if grammar_constraint is not None:
            grammar_state = grammar_constraint.advance_state(grammar_state, token_id)
            if grammar_constraint.is_match_state(grammar_state):
                grammar_complete = True
        
        should_continue = token_id not in stop_tokens
        
        if grammar_complete:
            should_continue = False
        
        if should_continue:
            for _ in range(max_tokens - 1):
                input_ids = mx.array([[token_id]])
                
                logits = self.model(input_ids, cache=cache_list)
                
                last_logits = logits[:, -1, :]
                
                if grammar_constraint is not None:
                    mask_start = time.perf_counter()
                    valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
                    mask_time = time.perf_counter() - mask_start
                    mask_times.append(mask_time)
                    
                    if valid_tokens:
                        last_logits = self._apply_grammar_mask(last_logits, valid_tokens)
                    else:
                        break
                
                next_token = self._sample_token(last_logits, temperature)
                mx.eval(next_token)
                
                token_id = next_token.item()
                generated_tokens.append(token_id)
                
                if grammar_constraint is not None:
                    grammar_state = grammar_constraint.advance_state(grammar_state, token_id)
                    if grammar_constraint.is_match_state(grammar_state):
                        break
                
                if token_id in stop_tokens:
                    break
        
        generation_end = time.perf_counter()
        
        output_text = self.tokenizer.decode(generated_tokens)
        
        metrics["generated_tokens"] = len(generated_tokens)
        metrics["generation_time"] = generation_end - generation_start
        if metrics["generated_tokens"] > 0:
            metrics["tokens_per_second"] = (
                metrics["generated_tokens"] / metrics["generation_time"]
            )
        
        if mask_times:
            metrics["mask_time_total"] = sum(mask_times)
            metrics["mask_time_avg"] = sum(mask_times) / len(mask_times)
            metrics["mask_calls"] = len(mask_times)
        
        return output_text, metrics
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_tokens: Optional[List[int]] = None,
        regex: Optional[str] = None,
    ) -> Generator[Tuple[str, Optional[dict]], None, None]:
        if self.model is None:
            self.load_model()
        
        self.cache_manager.reset()
        
        if stop_tokens is None:
            stop_tokens = []
            if hasattr(self.tokenizer, 'eos_token_id'):
                eos = self.tokenizer.eos_token_id
                if isinstance(eos, int):
                    stop_tokens.append(eos)
                elif isinstance(eos, list):
                    stop_tokens.extend(eos)
        
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([prompt_tokens])
        
        metrics = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": 0,
            "ttft": None,
            "generation_time": 0.0,
            "tokens_per_second": 0.0,
            "grammar_constrained": regex is not None,
            "mask_time_total": 0.0,
            "mask_time_avg": 0.0,
        }
        
        grammar_constraint = None
        grammar_state = None
        mask_times = []
        
        if regex is not None:
            if _GrammarConstraint is None:
                raise RuntimeError(
                    "Grammar constraints require the Rust backend. "
                    "Run 'maturin develop' to build it."
                )
            
            grammar_constraint = _GrammarConstraint(self.vocab_bytes)
            compile_start = time.perf_counter()
            grammar_constraint.compile_regex(regex)
            metrics["grammar_compile_time"] = time.perf_counter() - compile_start
            grammar_state = grammar_constraint.init_state()
        
        generation_start = time.perf_counter()
        cache_list = self.cache_manager.as_list()
        
        logits = self.model(input_ids, cache=cache_list)
        mx.eval(logits)
        
        last_logits = logits[:, -1, :]
        
        if grammar_constraint is not None:
            mask_start = time.perf_counter()
            valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
            mask_times.append(time.perf_counter() - mask_start)
            
            if valid_tokens:
                last_logits = self._apply_grammar_mask(last_logits, valid_tokens)
            else:
                raise ValueError(f"Grammar constraint has no valid tokens at initial state.")
        
        next_token = self._sample_token(last_logits, temperature)
        mx.eval(next_token)
        
        metrics["ttft"] = time.perf_counter() - generation_start
        
        token_id = next_token.item()
        metrics["generated_tokens"] += 1
        
        grammar_complete = False
        if grammar_constraint is not None:
            grammar_state = grammar_constraint.advance_state(grammar_state, token_id)
            if grammar_constraint.is_match_state(grammar_state):
                grammar_complete = True
        
        token_text = self.tokenizer.decode([token_id])
        yield token_text, None
        
        should_continue = token_id not in stop_tokens
        if grammar_complete:
            should_continue = False
        
        if should_continue:
            for _ in range(max_tokens - 1):
                input_ids = mx.array([[token_id]])
                logits = self.model(input_ids, cache=cache_list)
                
                last_logits = logits[:, -1, :]
                
                if grammar_constraint is not None:
                    mask_start = time.perf_counter()
                    valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
                    mask_times.append(time.perf_counter() - mask_start)
                    
                    if not valid_tokens:
                        break
                    
                    last_logits = self._apply_grammar_mask(last_logits, valid_tokens)
                
                next_token = self._sample_token(last_logits, temperature)
                mx.eval(next_token)
                
                token_id = next_token.item()
                metrics["generated_tokens"] += 1
                
                if grammar_constraint is not None:
                    grammar_state = grammar_constraint.advance_state(grammar_state, token_id)
                
                token_text = self.tokenizer.decode([token_id])
                yield token_text, None
                
                if grammar_constraint is not None and grammar_constraint.is_match_state(grammar_state):
                    break
                if token_id in stop_tokens:
                    break
        
        generation_end = time.perf_counter()
        metrics["generation_time"] = generation_end - generation_start
        if metrics["generated_tokens"] > 0:
            metrics["tokens_per_second"] = (
                metrics["generated_tokens"] / metrics["generation_time"]
            )
        
        if mask_times:
            metrics["mask_time_total"] = sum(mask_times)
            metrics["mask_time_avg"] = sum(mask_times) / len(mask_times)
            metrics["mask_calls"] = len(mask_times)
        
        yield "", metrics
    
    def validate(self, text: str, grammar_type: str = "json") -> bool:
        return onyx.validate_grammar(text, grammar_type)
    
    @property
    def load_time(self) -> Optional[float]:
        return self._load_time


def get_device_info() -> dict:
    return {
        "device": str(mx.default_device()),
        "mlx_available": True,
    }
