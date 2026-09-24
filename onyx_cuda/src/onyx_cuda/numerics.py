"""Canonical cache replay for ambiguous low-precision greedy verification."""

import time

import torch

from onyx_cuda.cache import CacheState, snapshot_cache
from onyx_cuda.generation import _grammar_choices
from onyx_cuda.masking import grammar_argmax
from onyx_cuda.prefill import prefill


def ambiguous_logits(logits, token_count, choices=None):
    """Check the rounding scale of the two strongest eligible token scores.

    choices lists the token IDs selectable at each position; None allows all.
    This is a numerical guard, not a proof of an error bound through every
    transformer layer. Exact output comparison remains a required release gate.
    """
    if logits.dtype not in (torch.float16, torch.bfloat16):
        return False
    if choices is None:
        values = logits[0, :token_count].topk(2).values.float()
        tolerance = torch.finfo(logits.dtype).eps * values.abs().amax(-1).clamp_min(1)
        return bool(torch.any(values[:, 0] - values[:, 1] <= tolerance).item())
    for position, valid in enumerate(choices[:token_count]):
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
    """Retain clean scalar-decoded checkpoints independently of speculative KV.

    Catch-up checks previously emitted tokens rather than silently accepting a
    changed prefix. The checkpoint advances monotonically across recovery runs.
    Its grammar handles share the generation loop's final cleanup registry.
    """

    def __init__(self, model, prompt_ids, constraint, live_states, backend, *,
                 eos_token_ids=(), measure=False):
        self.model = model
        self.prompt_ids = prompt_ids
        self.constraint = constraint
        self.live_states = live_states
        self.backend = backend
        self.eos_token_ids = eos_token_ids
        self.cache = None
        self.logits = None
        self.state = None
        self.replays = 0
        self.replayed_tokens = 0
        self.target_is_canonical = True
        self.scalar_logits = None
        self.measure = measure
        self.profile = {name: 0.0 for name in (
            "snapshot_seconds", "prefill_seconds", "history_seconds", "proposal_seconds")}
        self.history_tokens = 0
        self.proposal_tokens = 0
        self.skipped_proposal_tokens = 0
        self.chunked_history_tokens = 0
        self.graph_replay_fallbacks = 0
        self.replay_backend = getattr(model, "_onyx_replay_backend", None)

    def _start(self):
        if not self.measure:
            return None
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _finish(self, name, started):
        if started is not None:
            device = next(self.model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.profile[name] += time.perf_counter() - started

    def seed(self, cache, logits):
        """Preserve the already computed clean prompt before any speculation."""
        started = self._start()
        self.cache = snapshot_cache(cache)
        self.logits = logits.clone()
        self.scalar_logits = self.logits
        if self.constraint is not None:
            self.state = self.constraint.init_state()
            self.live_states.add(self.state)
        self._finish("snapshot_seconds", started)

    def before_forward(self, target_cache, batch, generated):
        if batch.shape[1] > 1:
            if self.target_is_canonical and self.cache is not None and self.cache.length < target_cache.length:
                # Target-only steps since the last repair already used clean
                # scalar KV. Adopt that work before speculation can mutate it.
                consumed = self.cache.length - len(self.prompt_ids)
                end = target_cache.length - len(self.prompt_ids)
                for token in generated[consumed:end]:
                    self._advance_state(token)
                started = self._start()
                self.cache = snapshot_cache(target_cache)
                self.logits = self.scalar_logits.clone()
                self._finish("snapshot_seconds", started)
            self.target_is_canonical = False

    def after_scalar(self, logits):
        if self.target_is_canonical:
            self.scalar_logits = logits[:, -1, :]

    def _select(self):
        if self.constraint is None:
            return self.logits.argmax(-1).item()
        choices = _grammar_choices(self.constraint, self.state, self.eos_token_ids)
        return grammar_argmax(self.logits, choices, backend=self.backend).item()

    def _advance_state(self, token):
        # EOS ends generation and has no grammar bytes to consume.
        if self.constraint is not None and token not in self.eos_token_ids:
            previous = self.state
            self.state = self.constraint.advance_state(previous, token)
            self.live_states.add(self.state)
            self.constraint.release_states([previous])
            self.live_states.remove(previous)

    def replay(self, target_cache, batch, generated):
        if self.cache is None:
            started = self._start()
            initial = prefill(self.model, self.prompt_ids)
            self.cache = CacheState.from_prefill(initial.past_key_values, initial.logits.device)
            self.logits = initial.logits
            if self.constraint is not None:
                self.state = self.constraint.init_state()
                self.live_states.add(self.state)
            self._finish("prefill_seconds", started)
        consumed = self.cache.length - len(self.prompt_ids)
        if not 0 <= consumed <= len(generated) - 1:
            raise RuntimeError("Canonical cache is not a prefix of generated tokens")
        with torch.inference_mode():
            started = self._start()
            while consumed < len(generated) - 1:
                token = generated[consumed]
                if self._select() != token:
                    raise RuntimeError("Greedy numerical replay found a noncanonical emitted prefix")
                width = min(3, len(generated) - 1 - consumed)
                if self.constraint is None and self.replay_backend is not None and width > 1:
                    # Commit only a checked chunk. A numerical mismatch leaves
                    # the clean checkpoint intact for the original scalar path.
                    trial = snapshot_cache(self.cache)
                    ids = torch.tensor([generated[consumed:consumed + width]], device=batch.device)
                    chunk = self.replay_backend.extend(trial, ids)
                    if chunk is not None and chunk.argmax(-1).flatten().tolist() == generated[consumed + 1:consumed + width + 1]:
                        self.cache = trial
                        self.logits = chunk[:, -1, :].clone()
                        self.replayed_tokens += width
                        self.history_tokens += width
                        self.chunked_history_tokens += width
                        consumed += width
                        del trial, chunk
                        continue
                    self.graph_replay_fallbacks += 1
                    self.replay_backend = None
                    del trial, chunk
                self.logits = self.cache.extend(self.model, torch.tensor(
                    [[token]], device=batch.device))[:, -1, :]
                self._advance_state(token)
                self.replayed_tokens += 1
                self.history_tokens += 1
                consumed += 1
            if self._select() != generated[-1]:
                raise RuntimeError("Greedy numerical replay found a noncanonical current token")
            self._finish("history_seconds", started)
            started = self._start()
            rows = []
            state = self.state
            owned_state = None
            try:
                for i in range(batch.shape[1]):
                    row = self.cache.extend(self.model, batch[:, i:i + 1])
                    rows.append(row)
                    # The verifier accepts at most through EOS; later logits are unused.
                    if batch[0, i].item() in self.eos_token_ids:
                        break
                    if self.constraint is not None:
                        next_state = self.constraint.advance_state(state, batch[0, i].item())
                        self.live_states.add(next_state)
                        if owned_state is not None:
                            self.constraint.release_states([owned_state])
                            self.live_states.remove(owned_state)
                        state = owned_state = next_state
                        # The verifier stops when a proposed token completes a
                        # match that cannot grow; no following logits are used.
                        choices = _grammar_choices(self.constraint, state, self.eos_token_ids)
                        if not choices:
                            break
                        token = grammar_argmax(row[:, -1, :], choices, backend=self.backend).item()
                    else:
                        token = row[:, -1, :].argmax(-1).item()
                    if i + 1 < batch.shape[1] and token != batch[0, i + 1].item():
                        break
            finally:
                if owned_state is not None:
                    self.constraint.release_states([owned_state])
                    self.live_states.remove(owned_state)
            logits = torch.cat(rows, dim=1)
            self._finish("proposal_seconds", started)
        self.replayed_tokens += len(rows)
        self.proposal_tokens += len(rows)
        self.skipped_proposal_tokens += batch.shape[1] - len(rows)
        self.replays += 1
        target_cache.past_key_values = self.cache.past_key_values
        target_cache.attention_mask = self.cache.attention_mask
        target_cache.cache_position = self.cache.cache_position
        return logits

    def commit(self, target_cache, batch, accepted, logits):
        # Clone after rollback; the next speculative extension must not mutate
        # this canonical checkpoint. Keep only the next-position logits.
        started = self._start()
        self.cache = snapshot_cache(target_cache)
        self.logits = logits[:, accepted, :].clone()
        self.scalar_logits = self.logits
        self.target_is_canonical = True
        for token in batch[0, :accepted + 1].tolist():
            self._advance_state(token)
        self._finish("snapshot_seconds", started)

    def report(self):
        return {"verification_replays": self.replays,
                "canonical_replay_tokens": self.replayed_tokens,
                "history_replay_tokens": self.history_tokens,
                "proposal_replay_tokens": self.proposal_tokens,
                "skipped_proposal_tokens": self.skipped_proposal_tokens,
                "chunked_history_tokens": self.chunked_history_tokens,
                "graph_replay_fallbacks": self.graph_replay_fallbacks,
                "stage_seconds": dict(self.profile) if self.measure else None}
