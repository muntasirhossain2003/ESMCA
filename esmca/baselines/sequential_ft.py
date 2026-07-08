"""Sequential Fine-tuning baseline (paper Table 3: "Lower bound, full
catastrophic forgetting"). The whole backbone + head is fine-tuned on
each task in turn with no regularization and no task-specific modules,
so later tasks are free to overwrite what earlier tasks learned.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from esmca.data.mmlu import Task
from esmca.evaluation.metrics import evaluate_accuracy
from esmca.models.backbone import ClassificationHead
from esmca.utils.logging import get_logger

logger = get_logger(__name__)


class SequentialFTModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)  # fully trainable, unlike ESMCA's frozen backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.head = ClassificationHead(self.backbone.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls_hidden)


def run_sequential_finetuning(
    model: SequentialFTModel,
    tasks: List[Task],
    device: torch.device,
    lr: float,
    epochs_per_task: int,
) -> List[List[float]]:
    """Trains sequentially over `tasks`, returning the accuracy matrix R
    (R[i][j] = accuracy on task j after finishing training on task i)."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    R: List[List[float]] = []

    for i, task in enumerate(tasks):
        model.train()
        for epoch in range(epochs_per_task):
            for batch in tqdm(task.train_loader, desc=f"[seq-ft:{task.name}] epoch {epoch + 1}/{epochs_per_task}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
                loss.backward()
                optimizer.step()

        model.eval()
        row = []
        for j in range(len(tasks)):
            def predict_fn(input_ids, attention_mask, _m=model, _d=device):
                return _m(input_ids.to(_d), attention_mask.to(_d))

            acc = evaluate_accuracy(predict_fn, tasks[j].test_loader)
            row.append(acc)
        R.append(row)
        logger.info(f"[seq-ft] finished '{task.name}', row={row}")
    return R
