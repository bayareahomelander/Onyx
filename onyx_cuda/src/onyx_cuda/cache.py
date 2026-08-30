"""Transformers cache state used by CUDA generation."""

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache


@dataclass
class CacheState:
    past_key_values: Cache
    attention_mask: torch.Tensor
    cache_position: torch.Tensor

    @classmethod
    def from_prefill(cls, past_key_values: Cache, device: torch.device):
        length = past_key_values.get_seq_length()
        return cls(
            past_key_values,
            torch.ones((1, length), dtype=torch.long, device=device),
            torch.arange(length, device=device),
        )

    @property
    def length(self) -> int:
        return self.past_key_values.get_seq_length()

    def extend(
        self, model: PreTrainedModel, input_ids: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(
            self.length,
            self.length + input_ids.shape[1],
            device=input_ids.device,
        )
        attention_mask = torch.cat(
            (self.attention_mask, torch.ones_like(input_ids)), dim=1
        )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            cache_position=positions,
            use_cache=True,
        )
        self.past_key_values = output.past_key_values
        self.attention_mask = attention_mask
        self.cache_position = torch.cat((self.cache_position, positions))
        return output.logits

    def crop(self, length: int) -> None:
        if length < 0 or length > self.length:
            raise ValueError(f"cache length must be between 0 and {self.length}")
        self.past_key_values.crop(length)
        self.attention_mask = self.attention_mask[:, :length]
        self.cache_position = self.cache_position[:length]
