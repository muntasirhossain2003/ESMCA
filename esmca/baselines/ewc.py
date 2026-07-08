"""Elastic Weight Consolidation baseline (paper Table 3: "Regularization-based CL").

Classic EWC (Kirkpatrick et al., 2017): after each task, estimate a
diagonal Fisher-information matrix over the trainable parameters and
anchor future training with a quadratic penalty toward that task's
optimal weights, weighted by their estimated importance.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from esmca.baselines.sequential_ft import SequentialFTModel
from esmca.data.mmlu import Task
from esmca.evaluation.metrics import evaluate_accuracy
from esmca.utils.logging import get_logger

logger = get_logger(__name__)


def _estimate_fisher(model: SequentialFTModel, loader, device: torch.device, n_batches: int = 30) -> Dict[str, torch.Tensor]:
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    seen = 0
    for batch in loader:
        if seen >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.nll_loss(log_probs, batch["labels"])
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        seen += 1
    for n in fisher:
        fisher[n] /= max(seen, 1)
    return fisher


def run_ewc(
    model: SequentialFTModel,
    tasks: List[Task],
    device: torch.device,
    lr: float,
    epochs_per_task: int,
    ewc_lambda: float = 1000.0,
) -> List[List[float]]:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    R: List[List[float]] = []

    fisher_history: List[Dict[str, torch.Tensor]] = []
    optimal_params_history: List[Dict[str, torch.Tensor]] = []

    for i, task in enumerate(tasks):
        model.train()
        for epoch in range(epochs_per_task):
            for batch in tqdm(task.train_loader, desc=f"[ewc:{task.name}] epoch {epoch + 1}/{epochs_per_task}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = torch.nn.functional.cross_entropy(logits, batch["labels"])

                for fisher, opt_params in zip(fisher_history, optimal_params_history):
                    for n, p in model.named_parameters():
                        if n in fisher:
                            loss = loss + (ewc_lambda / 2) * (fisher[n] * (p - opt_params[n]) ** 2).sum()

                loss.backward()
                optimizer.step()

        fisher_history.append(_estimate_fisher(model, task.val_loader, device))
        optimal_params_history.append({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})

        model.eval()
        row = []
        for j in range(len(tasks)):
            def predict_fn(input_ids, attention_mask, _m=model, _d=device):
                return _m(input_ids.to(_d), attention_mask.to(_d))

            acc = evaluate_accuracy(predict_fn, tasks[j].test_loader)
            row.append(acc)
        R.append(row)
        logger.info(f"[ewc] finished '{task.name}', row={row}")
    return R
