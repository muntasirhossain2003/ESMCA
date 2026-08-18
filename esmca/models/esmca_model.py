"""ESMCAModel: wires the frozen backbone, LoRA adapter bank, and shared
classification head into the two-pass forward described in Fig. 1:

  1. A frozen-backbone pass produces the representation used by the
     Attribution Map Extractor (AME) to compute phi(x) -- no adapters
     are active for this pass.
  2. The Attribution-Based Router (ABR) turns phi(x) into per-task
     weights w_t by comparing against stored prototypes.
  3. A second pass runs with those routing weights active, blending
     every task adapter's contribution into WQ/WV (Eq. 3) to produce
     the final task output.

During training, pass 2 is skipped -- only the current task's own
adapter is active (`AdapterBank.set_active_task`), matching the
training protocol in Sec 3.3.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from esmca.models.backbone import ClassificationHead, NUM_CHOICES, find_lora_targets, load_backbone
from esmca.models.lora import AdapterBank, inject_lora


class ESMCAModel(nn.Module):
    def __init__(self, model_name: str, lora_rank: int, lora_alpha: int, lora_targets=("q", "v"), lora_dropout: float = 0.0):
        super().__init__()
        self.backbone, self.tokenizer = load_backbone(model_name)
        targets = find_lora_targets(self.backbone, target_types=tuple(lora_targets))
        self.adapter_bank: AdapterBank = inject_lora(targets, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self.head = ClassificationHead(self.backbone.config.hidden_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids is [B, num_choices, L] (choice-scoring); flatten choices so
        # the whole batch runs through the encoder in one pass.
        batch_size = input_ids.shape[0]
        flat_ids = input_ids.reshape(batch_size * NUM_CHOICES, -1)
        flat_mask = attention_mask.reshape(batch_size * NUM_CHOICES, -1)
        outputs = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]  # [B * num_choices, H] [CLS]
        # Backbone runs fp16; up-cast to the head's fp32 so addmm isn't mixed-dtype.
        return cls_hidden.to(self.head.dense.weight.dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_hidden = self.encode(input_ids, attention_mask)
        scores = self.head(cls_hidden)  # [B * num_choices, 1]
        return scores.view(input_ids.shape[0], NUM_CHOICES)

    def forward_frozen(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pass with no adapters active -- used as the AME's attribution target."""
        self.adapter_bank.set_active_task(None)
        return self.forward(input_ids, attention_mask)

    def forward_composed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, routing_weights: Dict[str, float]
    ) -> torch.Tensor:
        """Pass with the ABR's blended task adapters active (Eq. 3)."""
        self.adapter_bank.set_routing_weights(routing_weights)
        logits = self.forward(input_ids, attention_mask)
        self.adapter_bank.set_routing_weights(None)
        return logits

    def forward_task(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task_name: str) -> torch.Tensor:
        """Pass with a single task's own adapter active -- used during training."""
        self.adapter_bank.set_active_task(task_name)
        logits = self.forward(input_ids, attention_mask)
        return logits

    def start_new_task(self, task_name: str) -> None:
        self.adapter_bank.add_task(task_name)
        self.adapter_bank.set_active_task(task_name)

    def freeze_task(self, task_name: str) -> None:
        self.adapter_bank.freeze_task(task_name)

    def task_trainable_parameters(self, task_name: str):
        yield from self.adapter_bank.trainable_parameters(task_name)
        yield from self.head.parameters()
