"""Standard continual learning metrics (paper Sec 6.1): ACC, BWT, FWT.

R[i][j] = accuracy on task j's test set after the model has finished
training on tasks 1..i (i, j are 0-indexed positions in the task order).
"""
from __future__ import annotations

from typing import Callable, List

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], loader: DataLoader) -> float:
    correct, total = 0, 0
    for batch in loader:
        logits = predict_fn(batch["input_ids"], batch["attention_mask"])
        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return correct / total if total else 0.0


def compute_acc(R: List[List[float]]) -> float:
    """Average accuracy across all tasks after full sequential training (last row of R)."""
    T = len(R)
    final_row = R[T - 1]
    return sum(final_row) / len(final_row)


def compute_bwt(R: List[List[float]]) -> float:
    """BWT = 1/(T-1) * sum_{t=1}^{T-1} (R_{T,t} - R_{t,t})."""
    T = len(R)
    if T < 2:
        return 0.0
    total = sum(R[T - 1][t] - R[t][t] for t in range(T - 1))
    return total / (T - 1)


def compute_fwt(R: List[List[float]]) -> float:
    """FWT: zero-shot accuracy on task i using the model as it stood right
    before training on task i (i.e. after task i-1), averaged over i=1..T-1.
    Approximates "forward transfer" per Sec 6.1 in the absence of a
    fully-specified reference formula in the proposal."""
    T = len(R)
    if T < 2:
        return 0.0
    total = sum(R[i - 1][i] for i in range(1, T))
    return total / (T - 1)
