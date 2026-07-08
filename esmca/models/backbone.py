"""Frozen pretrained backbone (paper Sec 3.1: RoBERTa / ViT / LLaMA).

We use `roberta-base` for MMLU text classification. The backbone stays
frozen; only LoRA adapters (injected into WQ, WV) and the shared
classification head are trainable.
"""
from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

NUM_CHOICES = 4


def load_backbone(model_name: str = "roberta-base"):
    """Load and freeze the pretrained backbone + its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone, tokenizer


def find_query_value_linears(backbone) -> List[Tuple[str, nn.Linear]]:
    """Locate WQ/WV projections in every transformer layer (paper Sec 3.2.1:
    "Only WQ and WV projections are adapted, following empirical best
    practice.").
    """
    targets = []
    for layer_idx, layer in enumerate(backbone.encoder.layer):
        attn = layer.attention.self
        targets.append((f"layer{layer_idx}.query", attn.query))
        targets.append((f"layer{layer_idx}.value", attn.value))
    return targets


class ClassificationHead(nn.Module):
    """Shared 4-way classification head over the [CLS] representation.

    MMLU subjects all share the same 4-choice answer format, so a single
    task-agnostic head (trained jointly across tasks) is sufficient here;
    task-specific behavior is captured entirely by the LoRA adapters.
    """

    def __init__(self, hidden_size: int, num_choices: int = NUM_CHOICES):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_choices)
        self.activation = nn.Tanh()

    def forward(self, cls_hidden):
        x = self.activation(self.dense(cls_hidden))
        return self.out_proj(x)
