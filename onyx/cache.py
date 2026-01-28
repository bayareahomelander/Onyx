"""
onyx paged kv cache

memory-efficient kv cache implementation using fixed-size blocks (pages)
to support zero-copy rollbacks for speculative decoding.

the key innovation is that rollback is o(1) - just update counters and
discard block pointers, no memory copies required.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class CacheBlock:
    """
    a fixed-size block holding key and value tensors for a set of tokens.
    
    each block stores up to `block_size` tokens worth of kv states.
    
    attributes:
        keys: key tensor [batch, n_heads, block_size, head_dim]
        values: value tensor [batch, n_heads, block_size, head_dim]
        num_tokens: number of valid tokens in this block (0 to block_size)
    """
    keys: mx.array
    values: mx.array
    num_tokens: int = 0
    
    @property
    def is_full(self) -> bool:
        """check if this block is full."""
        return self.num_tokens >= self.keys.shape[2]
    
    @property
    def capacity(self) -> int:
        """return the block's capacity (block_size)."""
        return self.keys.shape[2]
    
    @property
    def space_remaining(self) -> int:
        """return how many more tokens can fit."""
        return self.capacity - self.num_tokens


class PagedKVCache:
    """
    paged kv cache with o(1) rollback support.
    
    instead of storing all kv states in one monolithic tensor that requires
    copying on rollback, this implementation uses fixed-size blocks (pages).
    
    key features:
    - fixed block size (default 16 tokens per block)
    - page table maintains list of allocated blocks
    - rollback is o(1): update valid_length counter and discard block pointers
    - no memory copies on rollback
    
    this is critical for speculative decoding where we frequently need to
    roll back rejected draft tokens.
    """
    
    def __init__(self, block_size: int = 16):
        """
        initialize the paged kv cache.
        
        args:
            block_size: number of tokens per block (default 16)
        """
        self.block_size = block_size
        
        self.page_table: List[CacheBlock] = []
        
        self._valid_length: int = 0
        
        self._batch_size: Optional[int] = None
        self._n_heads: Optional[int] = None
        self._k_head_dim: Optional[int] = None
        self._v_head_dim: Optional[int] = None
        self._dtype = None
        
        self._total_blocks_allocated: int = 0
        self._rollback_count: int = 0
    
    def _allocate_block(self) -> CacheBlock:
        """
        allocate a new empty cacheblock.
        
        returns a new cacheblock with zeroed tensors
        """
        k_shape = (self._batch_size, self._n_heads, self.block_size, self._k_head_dim)
        v_shape = (self._batch_size, self._n_heads, self.block_size, self._v_head_dim)
        
        block = CacheBlock(
            keys=mx.zeros(k_shape, dtype=self._dtype),
            values=mx.zeros(v_shape, dtype=self._dtype),
            num_tokens=0,
        )
        
        self._total_blocks_allocated += 1
        return block
    
    def _initialize_metadata(self, keys: mx.array, values: mx.array) -> None:
        """initialize cache metadata from first keys/values."""
        self._batch_size = keys.shape[0]
        self._n_heads = keys.shape[1]
        self._k_head_dim = keys.shape[3]
        self._v_head_dim = values.shape[3]
        self._dtype = keys.dtype
    
    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        update the cache with new keys/values and return the full cache.
        
        this method fills blocks sequentially, allocating new blocks as needed.
        
        args:
            keys: new key tensors [batch, n_heads, seq_len, head_dim]
            values: new value tensors [batch, n_heads, seq_len, head_dim]
            
        returns tuple of (all_keys, all_values) as contiguous tensors
        """
        if self._batch_size is None:
            self._initialize_metadata(keys, values)
        
        seq_len = keys.shape[2]
        tokens_to_add = seq_len
        token_offset = 0
        
        while tokens_to_add > 0:
            if not self.page_table or self.page_table[-1].is_full:
                self.page_table.append(self._allocate_block())
            
            current_block = self.page_table[-1]
            
            space = current_block.space_remaining
            tokens_this_block = min(space, tokens_to_add)
            
            block_start = current_block.num_tokens
            block_end = block_start + tokens_this_block
            src_start = token_offset
            src_end = token_offset + tokens_this_block
            
            current_block.keys[..., block_start:block_end, :] = keys[..., src_start:src_end, :]
            current_block.values[..., block_start:block_end, :] = values[..., src_start:src_end, :]
            current_block.num_tokens += tokens_this_block
            
            tokens_to_add -= tokens_this_block
            token_offset += tokens_this_block
        
        self._valid_length += seq_len
        
        return self._get_contiguous_cache()
    
    def _get_contiguous_cache(self) -> Tuple[mx.array, mx.array]:
        """
        return all cached keys/values as contiguous tensors.
        
        this concatenates all blocks up to the valid length.
        """
        if not self.page_table:
            k_shape = (self._batch_size or 1, self._n_heads or 1, 0, self._k_head_dim or 1)
            v_shape = (self._batch_size or 1, self._n_heads or 1, 0, self._v_head_dim or 1)
            return mx.zeros(k_shape), mx.zeros(v_shape)
        
        k_parts = []
        v_parts = []
        tokens_remaining = self._valid_length
        
        for block in self.page_table:
            if tokens_remaining <= 0:
                break
            
            tokens_from_block = min(block.num_tokens, tokens_remaining)
            if tokens_from_block > 0:
                k_parts.append(block.keys[..., :tokens_from_block, :])
                v_parts.append(block.values[..., :tokens_from_block, :])
                tokens_remaining -= tokens_from_block
        
        if not k_parts:
            k_shape = (self._batch_size, self._n_heads, 0, self._k_head_dim)
            v_shape = (self._batch_size, self._n_heads, 0, self._v_head_dim)
            return mx.zeros(k_shape), mx.zeros(v_shape)
        
        all_keys = mx.concatenate(k_parts, axis=2)
        all_values = mx.concatenate(v_parts, axis=2)
        
        return all_keys, all_values
    
    def rollback(self, num_tokens: int) -> None:
        """
        roll back the cache by discarding the last num_tokens.
        
        this is o(1) operation - just update counters and potentially
        discard block pointers. no memory copies!
        
        args:
            num_tokens: number of tokens to remove from the end
        """
        self._rollback_count += 1
        
        if num_tokens <= 0:
            return
        
        if num_tokens >= self._valid_length:
            self.reset()
            return
        
        new_length = self._valid_length - num_tokens
        self._valid_length = new_length
        
        tokens_counted = 0
        blocks_to_keep = 0
        
        for i, block in enumerate(self.page_table):
            if tokens_counted >= new_length:
                break
            
            tokens_in_block = min(block.num_tokens, new_length - tokens_counted)
            tokens_counted += tokens_in_block
            blocks_to_keep = i + 1
            
            if tokens_counted >= new_length:
                block.num_tokens = tokens_in_block
        
        if blocks_to_keep < len(self.page_table):
            self.page_table = self.page_table[:blocks_to_keep]
    
    def rollback_to(self, valid_length: int) -> None:
        """
        roll back to a specific valid length.
        
        args:
            valid_length: target number of valid tokens
        """
        if valid_length >= self._valid_length:
            return
        
        tokens_to_remove = self._valid_length - valid_length
        self.rollback(tokens_to_remove)
    
    def size(self) -> int:
        """return the current number of valid cached tokens."""
        return self._valid_length
    
    @property
    def offset(self) -> int:
        """compatibility property for mlx_lm interface."""
        return self._valid_length
    
    @offset.setter
    def offset(self, value: int) -> None:
        """compatibility setter - performs rollback if needed."""
        if value < self._valid_length:
            self.rollback_to(value)
        elif value > self._valid_length:
            pass
    
    def reset(self) -> None:
        """clear the cache completely."""
        self.page_table = []
        self._valid_length = 0
    
    @property
    def num_blocks(self) -> int:
        """return the number of allocated blocks."""
        return len(self.page_table)
    
    @property
    def stats(self) -> dict:
        """return cache statistics."""
        return {
            "valid_length": self._valid_length,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "total_blocks_allocated": self._total_blocks_allocated,
            "rollback_count": self._rollback_count,
            "memory_efficiency": self._valid_length / max(1, self.num_blocks * self.block_size),
        }


def make_paged_cache(model, block_size: int = 16) -> List[PagedKVCache]:
    """
    create a list of pagedkvcache objects for each layer of the model.
    
    this is a drop-in replacement for mlx_lm's make_prompt_cache.
    
    args:
        model: the language model
        block_size: tokens per block (default 16)
        
    returns list of pagedkvcache objects, one per layer
    """
    num_layers = len(model.layers)
    return [PagedKVCache(block_size=block_size) for _ in range(num_layers)]


class PagedCacheManager:
    """
    manages a list of pagedkvcache objects for all transformer layers.
    
    provides a unified interface for cache operations across all layers.
    """
    
    def __init__(self, num_layers: int, block_size: int = 16):
        """
        initialize the cache manager.
        
        args:
            num_layers: number of transformer layers
            block_size: tokens per block
        """
        self.num_layers = num_layers
        self.block_size = block_size
        self.caches: List[PagedKVCache] = [
            PagedKVCache(block_size=block_size) for _ in range(num_layers)
        ]
    
    def reset(self) -> None:
        """reset all caches."""
        for cache in self.caches:
            cache.reset()
    
    def rollback(self, num_tokens: int) -> None:
        """roll back all caches by num_tokens."""
        for cache in self.caches:
            cache.rollback(num_tokens)
    
    def rollback_to(self, valid_length: int) -> None:
        """roll back all caches to valid_length."""
        for cache in self.caches:
            cache.rollback_to(valid_length)
    
    def size(self) -> int:
        """return the valid length (from first cache)."""
        if self.caches:
            return self.caches[0].size()
        return 0
    
    def as_list(self) -> List[PagedKVCache]:
        """return caches as a list for model forward pass."""
        return self.caches
    
    @property
    def stats(self) -> dict:
        """return aggregated stats from all caches."""
        if not self.caches:
            return {}
        
        total_blocks = sum(c.num_blocks for c in self.caches)
        total_rollbacks = sum(c._rollback_count for c in self.caches)
        
        return {
            "num_layers": self.num_layers,
            "block_size": self.block_size,
            "valid_length": self.size(),
            "total_blocks": total_blocks,
            "avg_blocks_per_layer": total_blocks / self.num_layers if self.num_layers > 0 else 0,
            "total_rollbacks": total_rollbacks,
        }
