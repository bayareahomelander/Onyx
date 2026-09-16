"""Request-local speculative scheduling using measured cost per committed token."""

from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class AdaptiveController:
    """Bounded trials, smoothed costs, and token-based recovery; no prompt features."""

    gamma: int = 2
    target_cost: float | None = None
    costs: dict[int, float] = field(default_factory=dict)
    rounds: int = 0
    streak: int = 0
    cooldown: int = 0
    retry_interval: int = 8
    retrying: bool = False
    histogram: dict[int, int] = field(default_factory=dict)
    transitions: list[tuple[int, int]] = field(default_factory=list)
    retries: int = 0
    recoveries: int = 0
    catchup_seconds: float = 0.0
    samples: dict[int, deque] = field(default_factory=dict)
    probe_seconds: float = 0.0
    probe_tokens: int = 0
    probe_rounds: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    execution_seconds: float = 0.0
    acceptance: deque = field(default_factory=lambda: deque(maxlen=16))
    draft_costs: deque = field(default_factory=lambda: deque(maxlen=16))

    def _shorter_or_pause(self, gamma: int) -> None:
        choices = [(self.costs[g], g) for g in (1, 2, 4)
                   if g < gamma and g in self.costs and
                   self.costs[g] < self.target_cost * 0.95]
        self._switch(min(choices)[1] if choices else 0)
        self.cooldown = self.retry_interval

    def _expansion_pays(self, gamma: int) -> bool:
        if not self.draft_costs:
            return True
        accepted = sum(a for a, _ in self.acceptance)
        tested = sum(n for _, n in self.acceptance)
        probability = accepted / tested if tested else 0.0
        draft_cost = sum(s for s, _ in self.draft_costs) / sum(n for _, n in self.draft_costs)
        def estimated_cost(length):
            emitted = 1 + sum(probability ** i for i in range(1, length + 1))
            # Full acceptance leaves one draft KV position to catch up before
            # the next proposal. Omitting it disproportionately favors gamma 1.
            draft_steps = length + probability ** length
            return (self.target_cost + draft_steps * draft_cost) / emitted
        return estimated_cost(gamma * 2) < estimated_cost(gamma) * 0.95

    def choose(self, remaining: int) -> int:
        if remaining <= 1:
            return 0
        if self.rounds == 1 and self.target_cost is None:
            return 0  # One real decode step calibrates the target cost.
        if self.gamma == 0 and self.cooldown <= 0:
            self.retrying = True
            self.probe_seconds = 0.0
            self.probe_tokens = 0
            self.probe_rounds = 0
            self.retries += 1
            self._switch(2)
        return min(self.gamma, remaining - 1)

    def _switch(self, gamma: int) -> None:
        if gamma != self.gamma:
            if len(self.transitions) < 128:
                self.transitions.append((self.gamma, gamma))
            self.gamma = gamma
            self.streak = 0

    def observe(self, gamma: int, proposed: int, accepted: int, tokens: int,
                seconds: float, *, catchup_seconds: float = 0.0,
                proposal_seconds: float | None = None) -> None:
        self.histogram[gamma] = self.histogram.get(gamma, 0) + 1
        self.rounds += 1
        self.proposed_tokens += proposed
        self.accepted_tokens += accepted
        self.catchup_seconds += catchup_seconds
        if tokens <= 0 or not math.isfinite(seconds) or seconds <= 0:
            return
        self.execution_seconds += seconds
        if proposed:
            # Only positions through the first rejection were actually tested.
            self.acceptance.append((accepted, accepted + int(accepted < proposed)))
            if proposal_seconds is not None and proposal_seconds > 0:
                self.draft_costs.append((proposal_seconds, proposed))
        samples = self.samples.setdefault(gamma, deque(maxlen=8))
        samples.append((seconds, tokens))
        # Ratio of totals, not an average of per-round ratios: a rejection
        # produces one token while a fully accepted round produces several.
        cost = sum(s for s, _ in samples) / sum(n for _, n in samples)
        self.costs[gamma] = cost
        if gamma == 0:
            self.target_cost = self.costs[0]
            self.cooldown -= tokens
            return
        if self.target_cost is None:
            return
        if seconds > 2 * tokens * self.target_cost:
            # A large measured loss (including numerical replay or catch-up)
            # does not need three more paid trials to justify target-only work.
            if self.retrying:
                self.retry_interval = min(self.retry_interval * 2, 64)
                self.retrying = False
            self._shorter_or_pause(gamma)
            return
        # A bounded three-round recovery trial must repay its catch-up cost.
        # One rejected position cannot by itself condemn the entire trial.
        profitable = cost < self.target_cost * 0.95
        if self.retrying:
            self.probe_seconds += seconds
            self.probe_tokens += tokens
            self.probe_rounds += 1
            if self.probe_rounds < 3:
                return
            self.retrying = False
            if self.probe_seconds < self.probe_tokens * self.target_cost * 0.95:
                self.recoveries += 1
                self.retry_interval = 8
                self.samples[gamma] = deque(list(samples)[-3:], maxlen=8)
                self.costs[gamma] = self.probe_seconds / self.probe_tokens
                # Past losing expansions should not prevent recovery after
                # the observed acceptance pattern has changed.
                for candidate in (4, 8):
                    self.costs.pop(candidate, None)
                    self.samples.pop(candidate, None)
                return
            else:
                self.retry_interval = min(self.retry_interval * 2, 64)
                self._switch(0)
                self.cooldown = self.retry_interval
                return
        if len(samples) >= 3 and cost > self.target_cost * 1.05:
            self._switch(gamma // 2 if gamma > 1 else 0)
            self.cooldown = self.retry_interval
            return
        # Beating target-only is insufficient if a shorter measured proposal
        # already delivers those tokens more cheaply. Reject costly expansions.
        alternatives = [(self.costs[g], g) for g in (1, 2, 4, 8)
                        if g != gamma and len(self.samples.get(g, ())) >= 2]
        if len(samples) >= 2 and alternatives:
            best_cost, best_gamma = min(alternatives)
            if cost > best_cost * 1.05:
                self._switch(best_gamma)
                return
        if profitable and proposed == gamma and accepted == proposed:
            self.streak += 1
            if self.streak >= 2 and gamma < 8:
                candidate = gamma * 2
                # Do not continually repeat a known losing expansion.
                if (self.costs.get(candidate, 0) < self.target_cost * 0.95
                        and self._expansion_pays(gamma)):
                    self._switch(candidate)
        else:
            self.streak = 0

    def report(self) -> dict:
        return {"gamma_histogram": dict(self.histogram), "transitions": list(self.transitions),
                "retry_count": self.retries, "recovery_count": self.recoveries,
                "catchup_seconds": self.catchup_seconds, "cost_per_token": dict(self.costs),
                "proposed_token_count": self.proposed_tokens,
                "accepted_proposal_count": self.accepted_tokens,
                "execution_seconds": self.execution_seconds}
