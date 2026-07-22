"""Pure-Python adaptive gamma controller for speculative decoding."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AdaptiveGammaConfig:
    min_gamma: int = 1
    max_gamma: int = 8
    initial_gamma: int = 4
    window_size: int = 4
    high_acceptance: float = 0.80
    low_acceptance: float = 0.45
    verify_dominance: float = 1.10
    high_mask_fraction: float = 0.25


@dataclass
class AdaptiveGammaController:
    """Simple bounded controller for speculative draft length."""

    config: AdaptiveGammaConfig = field(default_factory=AdaptiveGammaConfig)

    def __post_init__(self) -> None:
        if self.config.min_gamma < 1:
            raise ValueError("min_gamma must be at least 1")
        if self.config.max_gamma < self.config.min_gamma:
            raise ValueError("max_gamma must be >= min_gamma")

        self.current_gamma = min(
            self.config.max_gamma,
            max(self.config.min_gamma, self.config.initial_gamma),
        )
        self.window: List[Dict[str, float]] = []
        self.adjustments = 0

    def observe(
        self,
        *,
        proposed: int,
        accepted: int,
        draft_time: float,
        verify_time: float,
        mask_time: float,
    ) -> int:
        acceptance = accepted / proposed if proposed > 0 else 0.0
        total_time = max(draft_time + verify_time + mask_time, 1e-12)

        self.window.append(
            {
                "acceptance": acceptance,
                "draft_time": draft_time,
                "verify_time": verify_time,
                "mask_fraction": mask_time / total_time,
            }
        )
        if len(self.window) > self.config.window_size:
            self.window.pop(0)

        avg_acceptance = sum(x["acceptance"] for x in self.window) / len(self.window)
        avg_draft_time = sum(x["draft_time"] for x in self.window) / len(self.window)
        avg_verify_time = sum(x["verify_time"] for x in self.window) / len(self.window)
        avg_mask_fraction = sum(x["mask_fraction"] for x in self.window) / len(self.window)

        next_gamma = self.current_gamma
        high_mask_overhead = avg_mask_fraction >= self.config.high_mask_fraction
        verify_dominates = avg_verify_time >= avg_draft_time * self.config.verify_dominance

        if proposed == 0 or avg_acceptance <= self.config.low_acceptance or high_mask_overhead:
            next_gamma = max(self.config.min_gamma, self.current_gamma - 1)
        elif avg_acceptance >= self.config.high_acceptance and verify_dominates:
            next_gamma = min(self.config.max_gamma, self.current_gamma + 1)

        if next_gamma != self.current_gamma:
            self.adjustments += 1
            self.current_gamma = next_gamma

        return self.current_gamma


__all__ = ["AdaptiveGammaConfig", "AdaptiveGammaController"]
