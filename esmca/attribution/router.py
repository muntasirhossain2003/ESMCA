"""Attribution-Based Router (ABR) -- paper Sec 3.2.3, Eq. 1-3.

    s_t = cos(phi(x), Phi_t),           t = 1..T
    w_t = softmax(s / tau),             tau = 0.1
    delta_W_comp = sum_t w_t * B_t A_t

No task labels are required at inference; routing is fully
explainability-driven.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class AttributionBasedRouter:
    def __init__(self, tau: float = 0.1):
        self.tau = tau

    def route(self, phi_x: torch.Tensor, prototypes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """phi_x: [seq_len] (or [B, seq_len] for a single example's mean).
        prototypes: {task_name: Phi_t [seq_len]}.
        Returns per-task softmax routing weights.
        """
        if phi_x.dim() > 1:
            phi_x = phi_x.mean(dim=0)
        task_names = list(prototypes.keys())
        sims = torch.stack(
            [F.cosine_similarity(phi_x.unsqueeze(0), prototypes[t].unsqueeze(0)).squeeze(0) for t in task_names]
        )
        weights = F.softmax(sims / self.tau, dim=0)
        return {t: float(w) for t, w in zip(task_names, weights)}
