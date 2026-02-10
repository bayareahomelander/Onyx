"""
onyx speculative decoding engine

this implements speculative decoding with a draft-verify pattern to achieve
faster generation by using a smaller draft model to propose tokens
that are verified by a larger target model.

"""

from typing import Optional, List, Tuple, Union, Callable
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

from onyx.cache import PagedKVCache, make_paged_cache

import onyx
_GrammarConstraint = None
if onyx.RUST_AVAILABLE:
    try:
        from onyx._rust import GrammarConstraint as _GrammarConstraint
    except ImportError:
        pass


def sample_argmax(logits: mx.array) -> mx.array:
    return mx.argmax(logits[:, -1, :], axis=-1)


_compiled_sample_argmax = mx.compile(sample_argmax)


class SpeculativeEngine:
    """
    speculative decoding engine using draft-verify pattern
    
    this engine loads two models:
    - draft model: smaller, faster model for proposing tokens
    - target model: larger, more capable model for verification
    
    the speculative decoding loop:
    1. draft: generate gamma tokens with the draft model
    2. verify: run target model on all draft tokens at once
    3. accept: keep tokens up to first mismatch
    4. rollback: reset caches to valid position
    
    supports two cache modes:
    - "naive": uses mlx_lm's default KVCache (array slicing for rollback)
    - "paged": uses PagedKVCache with O(1) zero-copy rollback
    """
    
    DRAFT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    TARGET_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    
    def __init__(
        self,
        draft_model_path: Optional[str] = None,
        target_model_path: Optional[str] = None,
        cache_mode: str = "paged",
        block_size: int = 16,
        lazy_load: bool = False,
        use_compile: bool = True,
    ):
        """
        initialize the speculative decoding engine
        
        args:
            draft_model_path: path to draft model (default: Qwen2.5-0.5B-4bit)
            target_model_path: path to target model (default: Qwen2.5-1.5B-4bit)
            cache_mode: "naive" for mlx_lm KVCache, "paged" for PagedKVCache
            block_size: block size for paged cache (default 16)
            lazy_load: if True, defer model loading until first use
            use_compile: if True, use JIT compilation for model forward passes
        """
        self.draft_model_path = draft_model_path or self.DRAFT_MODEL
        self.target_model_path = target_model_path or self.TARGET_MODEL
        self.cache_mode = cache_mode
        self.block_size = block_size
        self.use_compile = use_compile
        
        self.draft_model: Optional[nn.Module] = None
        self.target_model: Optional[nn.Module] = None
        self.tokenizer = None
        
        self.draft_cache: Optional[List] = None
        self.target_cache: Optional[List] = None
        self.vocab_bytes: Optional[List[bytes]] = None
        
        self._draft_load_time: Optional[float] = None
        self._target_load_time: Optional[float] = None
        
        if not lazy_load:
            self.load_models()
    
    def load_models(self) -> Tuple[float, float]:
        """load both models and return a tuple of (draft_load_time, target_load_time)"""
        start = time.perf_counter()
        self.draft_model, self.tokenizer = load(self.draft_model_path)
        self._draft_load_time = time.perf_counter() - start
        print(f"  Draft loaded in {self._draft_load_time:.2f}s")
        
        start = time.perf_counter()
        self.target_model, target_tokenizer = load(self.target_model_path)
        self._target_load_time = time.perf_counter() - start
        print(f"  Target loaded in {self._target_load_time:.2f}s")
        
        draft_vocab = self.draft_model.model.embed_tokens.weight.shape[0]
        target_vocab = self.target_model.model.embed_tokens.weight.shape[0]
        if draft_vocab != target_vocab:
            min_vocab = min(draft_vocab, target_vocab)
            max_vocab = max(draft_vocab, target_vocab)
            if (max_vocab - min_vocab) / min_vocab > 0.01:
                raise ValueError(
                    f"Tokenizer mismatch: draft vocab={draft_vocab}, target vocab={target_vocab}"
                )
            print(f"  Note: vocab size differs (draft={draft_vocab}, target={target_vocab})")
            print(f"        Using common vocabulary subset ({min_vocab} tokens)")
        
        print(f"  Cache mode: {self.cache_mode}" + 
              (f" (block_size={self.block_size})" if self.cache_mode == "paged" else ""))
        
        self._build_vocab_bytes()
        
        if self.use_compile:
            print("  JIT optimization: Using compiled sampling functions")
        
        return self._draft_load_time, self._target_load_time
    
    def _build_vocab_bytes(self) -> None:
        """build the vocabulary byte representation for grammar constraints"""
        vocab_size = self.tokenizer.vocab_size
        self.vocab_bytes = []
        
        for token_id in range(vocab_size):
            try:
                token_str = self.tokenizer.decode([token_id])
                token_bytes = token_str.encode('utf-8')
                self.vocab_bytes.append(token_bytes)
            except Exception:
                self.vocab_bytes.append(b"")
    
    def _reset_caches(self) -> None:
        """reset KV caches for both models"""
        if self.cache_mode == "paged":
            self.draft_cache = make_paged_cache(self.draft_model, self.block_size)
            self.target_cache = make_paged_cache(self.target_model, self.block_size)
        else:
            self.draft_cache = make_prompt_cache(self.draft_model)
            self.target_cache = make_prompt_cache(self.target_model)
    
    def _rollback_cache(
        self, 
        cache: List[Union[KVCache, PagedKVCache]], 
        valid_length: int
    ) -> None:
        """
        rollback cache to a valid length.
        
        for PagedKVCache: O(1) operation via rollback_to()
        for NaiveKVCache: adjusts offset
        """
        for layer_cache in cache:
            if isinstance(layer_cache, PagedKVCache):
                # O(1) rollback - just update counters and discard block pointers
                layer_cache.rollback_to(valid_length)
            elif hasattr(layer_cache, 'offset'):
                layer_cache.offset = valid_length
    
    def _get_cache_size(self, cache: List[Union[KVCache, PagedKVCache]]) -> int:
        """get the current size of the cache"""
        if cache and len(cache) > 0:
            layer_cache = cache[0]
            if isinstance(layer_cache, PagedKVCache):
                return layer_cache.size()
            elif hasattr(layer_cache, 'offset'):
                return layer_cache.offset
        return 0
    
    def _sample_greedy(self, logits: mx.array) -> mx.array:
        """greedy sampling"""
        return mx.argmax(logits, axis=-1)
    
    def _sample_from_logits(self, logits: mx.array) -> mx.array:
        """sample from full model output logits"""
        if self.use_compile:
            if logits.ndim == 3:
                return _compiled_sample_argmax(logits)
            return mx.argmax(logits, axis=-1)
        return mx.argmax(logits[:, -1, :], axis=-1) if logits.ndim == 3 else mx.argmax(logits, axis=-1)
    
    def _apply_grammar_mask(
        self,
        logits: mx.array,
        valid_token_ids: List[int],
    ) -> mx.array:
        """
        apply grammar constraint mask to logits and sets invalid token logits to -inf so they have 0 probability after softmax
        
        args:
            logits: model output logits [batch, vocab_size]
            valid_token_ids: list of token IDs that are grammatically valid
        """
        actual_vocab_size = logits.shape[-1]
        
        valid_mask = mx.zeros((actual_vocab_size,), dtype=mx.bool_)
        
        if valid_token_ids:
            valid_ids = [tid for tid in valid_token_ids if tid < actual_vocab_size]
            if valid_ids:
                valid_indices = mx.array(valid_ids)
                valid_mask = valid_mask.at[valid_indices].add(True)
        
        mask = mx.where(
            valid_mask, 
            mx.zeros((actual_vocab_size,)), 
            mx.full((actual_vocab_size,), float('-inf'))
        )
        
        return logits + mask
    
    def _get_cache_stats(self) -> dict:
        """get statistics from the caches (for paged mode)"""
        if self.cache_mode != "paged" or not self.draft_cache:
            return {}
        
        draft_stats = {
            "draft_" + k: v for k, v in self.draft_cache[0].stats.items()
        }
        target_stats = {
            "target_" + k: v for k, v in self.target_cache[0].stats.items()
        }
        
        draft_rollbacks = sum(c._rollback_count for c in self.draft_cache)
        target_rollbacks = sum(c._rollback_count for c in self.target_cache)
        
        return {
            **draft_stats,
            **target_stats,
            "total_draft_rollbacks": draft_rollbacks,
            "total_target_rollbacks": target_rollbacks,
        }
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        gamma: int = 4,
        stop_tokens: Optional[List[int]] = None,
        regex: Optional[str] = None,
        json_schema: Optional[str] = None,
        draft_grammar_aware: bool = True,
    ) -> Tuple[str, dict]:
        """
        generate text using speculative decoding with optional grammar constraints and return a tuple of (generated_text, metrics_dict)
        
        args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            gamma: Number of speculative (draft) tokens per iteration
            stop_tokens: Token IDs that stop generation
            regex: Optional regex pattern to constrain generation
            json_schema: Optional json schema string to constrain generation
            draft_grammar_aware: If True and grammar is provided, the draft model also uses grammar constraints;
                                 if False, draft is unconstrained for comparison benchmarks.
        """
        if self.draft_model is None or self.target_model is None:
            self.load_models()
        
        self._reset_caches()
        
        if stop_tokens is None:
            stop_tokens = []
            if hasattr(self.tokenizer, 'eos_token_id'):
                eos = self.tokenizer.eos_token_id
                if isinstance(eos, int):
                    stop_tokens.append(eos)
                elif isinstance(eos, list):
                    stop_tokens.extend(eos)
        
        prompt_tokens = self.tokenizer.encode(prompt)
        
        grammar_active = regex is not None or json_schema is not None
        
        metrics = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "acceptance_rate": 0.0,
            "speculative_iterations": 0,
            "ttft": None,
            "generation_time": 0.0,
            "tokens_per_second": 0.0,
            "cache_mode": self.cache_mode,
            "jit_compiled": self.use_compile,
            "grammar_constrained": grammar_active,
            "draft_grammar_aware": draft_grammar_aware and grammar_active,
            "mask_time_total": 0.0,
            "mask_time_avg": 0.0,
        }
        
        grammar_constraint = None
        grammar_state = None
        
        if grammar_active:
            if _GrammarConstraint is None:
                raise RuntimeError(
                    "Grammar constraints require the Rust backend"
                )
            
            grammar_constraint = _GrammarConstraint(self.vocab_bytes)
            
            compile_start = time.perf_counter()
            if json_schema is not None:
                grammar_constraint.compile_json_schema(json_schema)
            else:
                grammar_constraint.compile_regex(regex)
            metrics["grammar_compile_time"] = time.perf_counter() - compile_start
            
            grammar_state = grammar_constraint.init_state()
        
        generated_tokens = []
        generation_start = time.perf_counter()
        first_token_time = None
        mask_times = []
        
        input_ids = mx.array([prompt_tokens])
        
        draft_logits = self.draft_model(input_ids, cache=self.draft_cache)
        target_logits = self.target_model(input_ids, cache=self.target_cache)
        
        mx.eval(draft_logits, target_logits)
        
        first_token_logits = target_logits[:, -1, :]
        
        if grammar_constraint is not None:
            mask_start = time.perf_counter()
            valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
            mask_times.append(time.perf_counter() - mask_start)
            
            if valid_tokens:
                first_token_logits = self._apply_grammar_mask(first_token_logits, valid_tokens)
            else:
                raise ValueError(
                    f"no valid tokens at initial state. "
                    f"Pattern: {regex}"
                )
        
        first_token = self._sample_greedy(first_token_logits)
        mx.eval(first_token)
        
        first_token_time = time.perf_counter()
        metrics["ttft"] = first_token_time - generation_start
        
        token_id = first_token.item()
        generated_tokens.append(token_id)
        
        grammar_complete = False
        if grammar_constraint is not None:
            grammar_state = grammar_constraint.advance_state(grammar_state, token_id)
            if grammar_constraint.is_match_state(grammar_state):
                grammar_complete = True
        
        if token_id in stop_tokens or grammar_complete:
            pass
        else:
            while len(generated_tokens) < max_tokens:
                metrics["speculative_iterations"] += 1
                
                cache_position_before_draft = self._get_cache_size(self.draft_cache)
                target_cache_position = self._get_cache_size(self.target_cache)
                
                draft_tokens = []
                current_token = token_id
                
                draft_grammar_state = grammar_state
                
                draft_input = mx.array([[current_token]])
                draft_logits = self.draft_model(draft_input, cache=self.draft_cache)
                mx.eval(draft_logits)
                
                for _ in range(gamma):
                    draft_last_logits = draft_logits[:, -1, :]
                    
                    if grammar_constraint is not None and draft_grammar_aware:
                        mask_start = time.perf_counter()
                        valid_tokens = grammar_constraint.get_valid_token_ids(draft_grammar_state)
                        mask_times.append(time.perf_counter() - mask_start)
                        
                        if not valid_tokens:
                            break
                        
                        draft_last_logits = self._apply_grammar_mask(draft_last_logits, valid_tokens)
                    
                    next_draft = self._sample_greedy(draft_last_logits)
                    mx.eval(next_draft)
                    draft_token = next_draft.item()
                    draft_tokens.append(draft_token)
                    
                    if grammar_constraint is not None and draft_grammar_aware:
                        draft_grammar_state = grammar_constraint.advance_state(
                            draft_grammar_state, draft_token
                        )
                        if grammar_constraint.is_match_state(draft_grammar_state):
                            break
                    
                    if draft_token in stop_tokens:
                        break
                    
                    draft_input = mx.array([[draft_token]])
                    draft_logits = self.draft_model(draft_input, cache=self.draft_cache)
                    mx.eval(draft_logits)
                
                metrics["draft_tokens_proposed"] += len(draft_tokens)
                
                if not draft_tokens:
                    break
                
                verify_sequence = [current_token] + draft_tokens
                verify_input = mx.array([verify_sequence])
                
                target_logits = self.target_model(verify_input, cache=self.target_cache)
                mx.eval(target_logits)
                
                accepted_count = 0
                verify_grammar_state = grammar_state
                
                for i, draft_token in enumerate(draft_tokens):
                    target_pos_logits = target_logits[:, i:i+1, :].squeeze(1)
                    
                    if grammar_constraint is not None:
                        mask_start = time.perf_counter()
                        valid_tokens = grammar_constraint.get_valid_token_ids(verify_grammar_state)
                        mask_times.append(time.perf_counter() - mask_start)
                        
                        if not valid_tokens:
                            break
                        
                        target_pos_logits = self._apply_grammar_mask(target_pos_logits, valid_tokens)
                    
                    target_pred = self._sample_greedy(target_pos_logits).item()
                    
                    if draft_token == target_pred:
                        accepted_count += 1
                        generated_tokens.append(draft_token)
                        
                        if grammar_constraint is not None:
                            verify_grammar_state = grammar_constraint.advance_state(
                                verify_grammar_state, draft_token
                            )
                            if grammar_constraint.is_match_state(verify_grammar_state):
                                grammar_complete = True
                                break
                        
                        if draft_token in stop_tokens:
                            break
                    else:
                        generated_tokens.append(target_pred)
                        
                        if grammar_constraint is not None:
                            verify_grammar_state = grammar_constraint.advance_state(
                                verify_grammar_state, target_pred
                            )
                            if grammar_constraint.is_match_state(verify_grammar_state):
                                grammar_complete = True
                        break
                
                metrics["draft_tokens_accepted"] += accepted_count
                
                if grammar_constraint is not None:
                    grammar_state = verify_grammar_state
                
                tokens_added = min(accepted_count + 1, len(draft_tokens))
                if generated_tokens and generated_tokens[-1] in stop_tokens:
                    tokens_added = accepted_count + (1 if accepted_count < len(draft_tokens) else 0)
                
                valid_draft_length = cache_position_before_draft + 1 + accepted_count
                self._rollback_cache(self.draft_cache, valid_draft_length)
                
                valid_target_length = target_cache_position + tokens_added
                self._rollback_cache(self.target_cache, valid_target_length)
                
                if generated_tokens:
                    token_id = generated_tokens[-1]
                    
                    if token_id in stop_tokens or grammar_complete:
                        break
                
                if len(generated_tokens) >= max_tokens:
                    break
        
        generation_end = time.perf_counter()
        
        output_text = self.tokenizer.decode(generated_tokens)
        
        metrics["generated_tokens"] = len(generated_tokens)
        metrics["generation_time"] = generation_end - generation_start
        
        if metrics["generated_tokens"] > 0:
            metrics["tokens_per_second"] = (
                metrics["generated_tokens"] / metrics["generation_time"]
            )
        
        if metrics["draft_tokens_proposed"] > 0:
            metrics["acceptance_rate"] = (
                metrics["draft_tokens_accepted"] / metrics["draft_tokens_proposed"]
            ) * 100
        
        if mask_times:
            metrics["mask_time_total"] = sum(mask_times)
            metrics["mask_time_avg"] = sum(mask_times) / len(mask_times)
            metrics["mask_calls"] = len(mask_times)
        
        if self.cache_mode == "paged":
            cache_stats = self._get_cache_stats()
            metrics["cache_stats"] = cache_stats
        
        return output_text, metrics
    
    def generate_baseline(
        self,
        prompt: str,
        max_tokens: int = 50,
        stop_tokens: Optional[List[int]] = None,
        regex: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        generate text using only the target model. standard autoregressive decoding without speculation
        but with optional grammar constraints for fair comparison.
        
        args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            stop_tokens: Token IDs that stop generation
            regex: Optional regex pattern to constrain generation
            json_schema: Optional json schema string to constrain generation
            
        returns a tuple of (generated_text, metrics_dict)
        """
        if self.target_model is None:
            self.load_models()
        
        if self.cache_mode == "paged":
            self.target_cache = make_paged_cache(self.target_model, self.block_size)
        else:
            self.target_cache = make_prompt_cache(self.target_model)
        
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
        
        grammar_active = regex is not None or json_schema is not None
        
        metrics = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": 0,
            "ttft": None,
            "generation_time": 0.0,
            "tokens_per_second": 0.0,
            "cache_mode": self.cache_mode,
            "jit_compiled": self.use_compile,
            "grammar_constrained": grammar_active,
            "mask_time_total": 0.0,
            "mask_time_avg": 0.0,
        }
        
        grammar_constraint = None
        grammar_state = None
        mask_times = []
        
        if grammar_active:
            if _GrammarConstraint is None:
                raise RuntimeError(
                    "Grammar constraints require the Rust backend"
                )
            
            grammar_constraint = _GrammarConstraint(self.vocab_bytes)
            compile_start = time.perf_counter()
            if json_schema is not None:
                grammar_constraint.compile_json_schema(json_schema)
            else:
                grammar_constraint.compile_regex(regex)
            metrics["grammar_compile_time"] = time.perf_counter() - compile_start
            grammar_state = grammar_constraint.init_state()
        
        generated_tokens = []
        generation_start = time.perf_counter()
        
        logits = self.target_model(input_ids, cache=self.target_cache)
        mx.eval(logits)
        
        last_logits = logits[:, -1, :]
        
        if grammar_constraint is not None:
            mask_start = time.perf_counter()
            valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
            mask_times.append(time.perf_counter() - mask_start)
            
            if valid_tokens:
                last_logits = self._apply_grammar_mask(last_logits, valid_tokens)
            else:
                raise ValueError(
                    f"Grammar constraint has no valid tokens at initial state. "
                    f"Pattern: {regex}"
                )
        
        next_token = self._sample_greedy(last_logits)
        mx.eval(next_token)
        
        metrics["ttft"] = time.perf_counter() - generation_start
        
        token_id = next_token.item()
        generated_tokens.append(token_id)
        
        grammar_complete = False
        if grammar_constraint is not None:
            grammar_state = grammar_constraint.advance_state(grammar_state, token_id)
            if grammar_constraint.is_match_state(grammar_state):
                grammar_complete = True
        
        if token_id not in stop_tokens and not grammar_complete:
            for _ in range(max_tokens - 1):
                input_ids = mx.array([[token_id]])
                logits = self.target_model(input_ids, cache=self.target_cache)
                last_logits = logits[:, -1, :]
                
                if grammar_constraint is not None:
                    mask_start = time.perf_counter()
                    valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
                    mask_times.append(time.perf_counter() - mask_start)
                    
                    if not valid_tokens:
                        break
                    
                    last_logits = self._apply_grammar_mask(last_logits, valid_tokens)
                
                next_token = self._sample_greedy(last_logits)
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
    
    @property
    def draft_load_time(self) -> Optional[float]:
        return self._draft_load_time
    
    @property
    def target_load_time(self) -> Optional[float]:
        return self._target_load_time
