"""Canonical cache replay for ambiguous low-precision greedy verification."""

import copy

import torch

from onyx_cuda.cache import CacheState
from onyx_cuda.masking import grammar_argmax
from onyx_cuda.prefill import prefill


def ambiguous_logits(logits, token_count, constraint=None, grammar_states=()):
    """Check the rounding scale of the two strongest eligible token scores.

    This is a numerical guard, not a proof of an error bound through every
    transformer layer. Exact output comparison remains a required release gate.
    """
    if logits.dtype not in (torch.float16, torch.bfloat16):
        return False
    if constraint is None:
        values = logits[0, :token_count].topk(2).values.float()
        tolerance = torch.finfo(logits.dtype).eps * values.abs().amax(-1).clamp_min(1)
        return bool(torch.any(values[:, 0] - values[:, 1] <= tolerance).item())
    for position, state in enumerate(grammar_states[:token_count]):
        valid = constraint.get_valid_token_ids(state)
        if len(valid) < 2:
            continue
        eligible = logits[0, position].index_select(
            0, torch.tensor(valid, device=logits.device, dtype=torch.long))
        values = eligible.topk(2).values.float()
        tolerance = torch.finfo(logits.dtype).eps * values.abs().max().clamp_min(1)
        if (values[0] - values[1] <= tolerance).item():
            return True
    return False


class GreedyCheckpoint:
    """Lazily retain a scalar-decoded cache independently of speculative KV.

    Catch-up checks previously emitted tokens rather than silently accepting a
    changed prefix. The checkpoint advances monotonically across recovery runs.
    Its grammar handles share the generation loop's final cleanup registry.
    """

    def __init__(self, model, prompt_ids, constraint, live_states, backend):
        self.model = model
        self.prompt_ids = prompt_ids
        self.constraint = constraint
        self.live_states = live_states
        self.backend = backend
        self.cache = None
        self.logits = None
        self.state = None
        self.replays = 0
        self.replayed_tokens = 0
        self.target_is_canonical = True
        self.scalar_logits = None

    def before_forward(self, target_cache, batch, generated):
        if batch.shape[1] > 1:
            if self.target_is_canonical and self.cache is not None and self.cache.length < target_cache.length:
                # Target-only steps since the last repair already used clean
                # scalar KV. Adopt that work before speculation can mutate it.
                consumed = self.cache.length - len(self.prompt_ids)
                end = target_cache.length - len(self.prompt_ids)
                for token in generated[consumed:end]:
                    self._advance_state(token)
                self.cache = copy.deepcopy(target_cache)
                self.logits = self.scalar_logits.clone()
            self.target_is_canonical = False

    def after_scalar(self, logits):
        if self.target_is_canonical:
            self.scalar_logits = logits[:, -1, :]

    def _select(self):
        if self.constraint is None:
            return self.logits.argmax(-1).item()
        return grammar_argmax(self.logits, self.constraint.get_valid_token_ids(self.state),
                              backend=self.backend).item()

    def _advance_state(self, token):
        if self.constraint is not None:
            previous = self.state
            self.state = self.constraint.advance_state(previous, token)
            self.live_states.add(self.state)
            self.constraint.release_states([previous])
            self.live_states.remove(previous)

    def replay(self, target_cache, batch, generated):
        if self.cache is None:
            initial = prefill(self.model, self.prompt_ids)
            self.cache = CacheState.from_prefill(initial.past_key_values, initial.logits.device)
            self.logits = initial.logits
            if self.constraint is not None:
                self.state = self.constraint.init_state()
                self.live_states.add(self.state)
        consumed = self.cache.length - len(self.prompt_ids)
        if not 0 <= consumed <= len(generated) - 1:
            raise RuntimeError("Canonical cache is not a prefix of generated tokens")
        with torch.inference_mode():
            for token in generated[consumed:-1]:
                if self._select() != token:
                    raise RuntimeError("Greedy numerical replay found a noncanonical emitted prefix")
                self.logits = self.cache.extend(self.model, torch.tensor(
                    [[token]], device=batch.device))[:, -1, :]
                self._advance_state(token)
                self.replayed_tokens += 1
            if self._select() != generated[-1]:
                raise RuntimeError("Greedy numerical replay found a noncanonical current token")
            logits = torch.cat([self.cache.extend(self.model, batch[:, i:i + 1])
                                for i in range(batch.shape[1])], dim=1)
        self.replayed_tokens += batch.shape[1]
        self.replays += 1
        target_cache.past_key_values = self.cache.past_key_values
        target_cache.attention_mask = self.cache.attention_mask
        target_cache.cache_position = self.cache.cache_position
        return logits

    def commit(self, target_cache, batch, accepted, logits):
        # Clone after rollback; the next speculative extension must not mutate
        # this canonical checkpoint. Keep only the next-position logits.
        self.cache = copy.deepcopy(target_cache)
        self.logits = logits[:, accepted, :].clone()
        self.scalar_logits = self.logits
        self.target_is_canonical = True
        for token in batch[0, :accepted + 1].tolist():
            self._advance_state(token)

    def report(self):
        return {"verification_replays": self.replays,
                "canonical_replay_tokens": self.replayed_tokens}
