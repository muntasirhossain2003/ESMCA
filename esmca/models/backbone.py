"""Frozen pretrained backbone (paper Sec 3.1: RoBERTa / DeBERTa / ViT / LLaMA).

We use `microsoft/deberta-v3-large` for MMLU text classification. The
backbone stays frozen; only LoRA adapters (injected into the configured
projections) and the shared classification head are trainable.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

NUM_CHOICES = 4


def load_backbone(model_name: str = "microsoft/deberta-v3-large"):
    """Load and freeze the pretrained backbone + its tokenizer.

    The backbone is kept in float16 (the checkpoint ships in fp16 and a 6 GB
    GPU cannot hold DeBERTa-v3-large in fp32). Trainable components (LoRA
    adapters, classification head) stay float32 for optimizer stability; the
    model casts at the fp16/fp32 boundary (see `ESMCAModel.encode` and
    `LoRAInjectedLinear.forward`).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.float16)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone, tokenizer


def _projection_names(attn) -> Tuple[str, str, str]:
    """Return (query, key, value) attribute names for an attention module.

    RoBERTa-style self-attention uses `query`/`key`/`value`, while
    DeBERTa-v2/v3 `DisentangledSelfAttention` uses `query_proj`/
    `key_proj`/`value_proj`.
    """
    if hasattr(attn, "query_proj"):
        return "query_proj", "key_proj", "value_proj"
    if hasattr(attn, "query"):
        return "query", "key", "value"
    raise ValueError(f"Unsupported attention module type: {type(attn).__name__}")


def find_lora_targets(backbone, target_types: Tuple[str, ...] = ("q", "v")) -> List[Tuple[str, nn.Module, str]]:
    """Locate the Linear projections to adapt in every transformer layer.

    `target_types` is a subset of {"q", "k", "v", "out"} ("out" = the
    attention output dense projection). Returns `(name, parent_module,
    attr_name)` tuples so the injector can wrap + replace each `nn.Linear`
    generically across RoBERTa and DeBERTa backbones.
    """
    targets: List[Tuple[str, nn.Module, str]] = []
    for layer_idx, layer in enumerate(backbone.encoder.layer):
        attn = layer.attention.self
        q, k, v = _projection_names(attn)
        projection_attrs = {"q": q, "k": k, "v": v}
        for t in target_types:
            if t == "out":
                parent = layer.attention.output
                targets.append((f"layer{layer_idx}.out.dense", parent, "dense"))
            elif t in projection_attrs:
                attr = projection_attrs[t]
                targets.append((f"layer{layer_idx}.{t}.{attr}", attn, attr))
            else:
                raise ValueError(f"Unknown LoRA target type: {t!r} (expected q/k/v/out)")
    return targets


class ClassificationHead(nn.Module):
    """Shared scoring head over the [CLS] representation.

    MMLU is framed as choice-scoring: each of the 4 options is encoded
    separately (`question + choice`) and this head maps that encoding's
    [CLS] vector to a scalar score; the highest-scoring option wins.
    MMLU subjects all share the same 4-choice answer format, so a single
    task-agnostic head (trained jointly across tasks) is sufficient here;
    task-specific behavior is captured entirely by the LoRA adapters.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, cls_hidden):
        x = self.activation(self.dense(cls_hidden))
        return self.out_proj(x)
