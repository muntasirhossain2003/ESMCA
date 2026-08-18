"""Config loading for ESMCA experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class Config:
    backbone: str = "microsoft/deberta-v3-large"
    max_seq_len: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: list = field(default_factory=lambda: ["q", "v"])
    lora_dropout: float = 0.0
    # Fraction of each subject's "test" split used for training (the MMLU
    # "validation" split only holds 9-16 examples). None keeps legacy behavior.
    train_frac: Optional[float] = None
    subjects: list = field(default_factory=lambda: [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "college_biology",
    ])
    batch_size: int = 8
    epochs_per_task: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.0
    n_prototype_samples: int = 256
    ig_steps: int = 8
    router_tau: float = 0.1
    drift_delta: float = 0.05
    attrib_loss_weight: float = 1.0
    consolidation_lambda: float = 1.0
    ewc_lambda: float = 1000.0
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return Config(**raw)
