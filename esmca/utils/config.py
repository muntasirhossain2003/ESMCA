"""Config loading for ESMCA experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import yaml


@dataclass
class Config:
    backbone: str = "roberta-base"
    max_seq_len: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16
    subjects: list = field(default_factory=lambda: [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "college_biology",
    ])
    batch_size: int = 8
    epochs_per_task: int = 3
    lr: float = 1e-4
    n_prototype_samples: int = 256
    ig_steps: int = 32
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
