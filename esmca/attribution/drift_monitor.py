"""Explanation Drift Monitor (EDM) -- paper Sec 3.2.4, Eq. 4.

    D_t = KL( phi'(x_old) || Phi_t ),           t < T+1
    L_attrib = sum_{t<T+1} lambda * ||phi_{T+1}(x_t) - Phi_t||^2   (only if D_t > delta)

If D_t exceeds a tunable threshold `delta`, forgetting is flagged
*before* accuracy degrades, triggering lightweight consolidation.
Averaging D_t across tasks gives the Attribution Drift Score (ADS),
the paper's novel proactive forgetting metric (Sec 6.2).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _to_distribution(attr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Attribution scores can be negative; convert to a proper probability
    distribution over token positions via abs-normalization."""
    abs_attr = attr.abs()
    return abs_attr / (abs_attr.sum() + eps)


class ExplanationDriftMonitor:
    def __init__(self, delta: float = 0.05, attrib_lambda: float = 1.0):
        self.delta = delta
        self.attrib_lambda = attrib_lambda

    def drift(self, current_attr: torch.Tensor, prototype: torch.Tensor, eps: float = 1e-8) -> float:
        p = _to_distribution(current_attr, eps)
        q = _to_distribution(prototype, eps)
        kl = F.kl_div((q + eps).log(), p, reduction="sum")
        return float(kl)

    def check_all(
        self, current_attrs: Dict[str, torch.Tensor], prototypes: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute D_t for every past task t whose prototype and a fresh
        attribution sample (on that task's held-out data) are provided."""
        return {t: self.drift(current_attrs[t], prototypes[t]) for t in prototypes if t in current_attrs}

    def tasks_needing_consolidation(self, drift_scores: Dict[str, float]) -> List[str]:
        return [t for t, d in drift_scores.items() if d > self.delta]

    def attribution_loss(
        self, current_attr: torch.Tensor, prototype: torch.Tensor
    ) -> torch.Tensor:
        return self.attrib_lambda * torch.sum((current_attr - prototype) ** 2)

    @staticmethod
    def ads(drift_scores: Dict[str, float]) -> float:
        """Attribution Drift Score: mean D_t across all past tasks (Sec 6.2)."""
        if not drift_scores:
            return 0.0
        return sum(drift_scores.values()) / len(drift_scores)
