"""Task-specific LoRA adapter bank (paper Sec 3.2.1).

    delta_W_t = B_t A_t,  B_t in R^(d x r),  A_t in R^(r x k),  r in {4, 8, 16}

All previous adapters are frozen once a new task arrives; only WQ/WV
projections are adapted. At inference the Attribution-Based Router (see
`esmca.attribution.router`) supplies per-task weights w_t so several
task adapters can be composed:

    delta_W_comp = sum_t w_t * B_t A_t
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRAPair(nn.Module):
    """A single task's (A, B) low-rank pair for one linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int, dropout: float = 0.0):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)  # zero-init B so the adapter starts as a no-op
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x).to(self.A.dtype)
        return (x @ self.A.t()) @ self.B.t()


class LoRAInjectedLinear(nn.Module):
    """Wraps a frozen `nn.Linear` and adds a bank of per-task LoRA deltas.

    Two forward modes:
      - training mode: only the currently `active_task`'s adapter fires
        (all previously-trained adapters stay frozen, paper Sec 3.3 step 2).
      - composed/inference mode: a routing-weight dict (from the ABR)
        blends every stored task adapter's contribution (Eq. 3).
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.dropout = dropout
        self.adapters = nn.ModuleDict()
        self.active_task: Optional[str] = None
        self.routing_weights: Optional[Dict[str, float]] = None

    def add_task(self, task_name: str) -> None:
        pair = LoRAPair(self.in_features, self.out_features, self.rank, dropout=self.dropout)
        # Ensure adapter is on the same device as the base weights
        device = next(self.base.parameters()).device
        pair = pair.to(device)
        self.adapters[task_name] = pair

    def freeze_task(self, task_name: str) -> None:
        for p in self.adapters[task_name].parameters():
            p.requires_grad = False

    def set_active_task(self, task_name: Optional[str]) -> None:
        self.active_task = task_name
        self.routing_weights = None

    def set_routing_weights(self, weights: Optional[Dict[str, float]]) -> None:
        self.routing_weights = weights
        self.active_task = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # LoRA deltas are float32 (trainable); the frozen backbone may be fp16,
        # so cast the delta back to the base output's dtype before adding.
        if self.active_task is not None:
            delta = self.scaling * self.adapters[self.active_task](x)
            out = out + delta.to(out.dtype)
        elif self.routing_weights:
            for task_name, weight in self.routing_weights.items():
                if weight == 0.0 or task_name not in self.adapters:
                    continue
                delta = weight * self.scaling * self.adapters[task_name](x)
                out = out + delta.to(out.dtype)
        return out


class AdapterBank:
    """Manages every `LoRAInjectedLinear` module injected into the backbone."""

    def __init__(self, injected_linears: List[Tuple[str, LoRAInjectedLinear]]):
        self.injected_linears = injected_linears
        self.task_order: List[str] = []

    def add_task(self, task_name: str) -> None:
        self.task_order.append(task_name)
        for _, module in self.injected_linears:
            module.add_task(task_name)

    def freeze_task(self, task_name: str) -> None:
        for _, module in self.injected_linears:
            module.freeze_task(task_name)

    def set_active_task(self, task_name: Optional[str]) -> None:
        for _, module in self.injected_linears:
            module.set_active_task(task_name)

    def set_routing_weights(self, weights: Optional[Dict[str, float]]) -> None:
        for _, module in self.injected_linears:
            module.set_routing_weights(weights)

    def trainable_parameters(self, task_name: str):
        for _, module in self.injected_linears:
            yield from module.adapters[task_name].parameters()


def inject_lora(targets: List[Tuple[str, nn.Module, str]], rank: int, alpha: int, dropout: float = 0.0) -> AdapterBank:
    """Replace each targeted `nn.Linear` in-place with a `LoRAInjectedLinear`.

    `targets` are `(name, parent_module, attr_name)` triples from
    `esmca.models.backbone.find_lora_targets` (q/k/v/out across both
    RoBERTa and DeBERTa backbones).
    """
    injected: List[Tuple[str, LoRAInjectedLinear]] = []
    for name, parent, attr in targets:
        base_linear = getattr(parent, attr)
        wrapped = LoRAInjectedLinear(base_linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        injected.append((name, wrapped))
    return AdapterBank(injected)
