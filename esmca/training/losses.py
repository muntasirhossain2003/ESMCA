"""Loss terms for the ESMCA training protocol (Sec 3.3):

    L_total = L_task + alpha * L_attrib
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)
