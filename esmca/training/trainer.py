"""ContinualTrainer: implements the ESMCA training protocol (Sec 3.3).

For each new task T+1:
  1. Initialize a fresh LoRA adapter (A_{T+1}, B_{T+1}).
  2. Freeze all previous adapters A_1..A_T (done when the *previous*
     task finished, see `_finish_task`).
  3. Train on D_{T+1} with L_total = L_task + alpha * L_attrib, where
     L_attrib only fires for past tasks the EDM flagged as drifting
     *before* this task's training began.
  4. Compute prototype Phi_{T+1} on a validation sample.
  5. Run EDM: compute D_t for all past tasks against the now-updated model.
  6. (handled inside step 3) attribution-anchored regularization.
  7. Store (A_{T+1}, B_{T+1}, Phi_{T+1}) in the bank.

Implementation note: for computational tractability, the drift check
that decides which past tasks get `L_attrib` regularization is run once
per task (pre-training) rather than every batch, and consolidation is
applied once per epoch on a small cached sample from each flagged task.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from esmca.attribution.ame import AttributionMapExtractor
from esmca.attribution.drift_monitor import ExplanationDriftMonitor
from esmca.attribution.router import AttributionBasedRouter
from esmca.data.mmlu import Task
from esmca.models.esmca_model import ESMCAModel
from esmca.training.losses import task_loss
from esmca.utils.logging import get_logger

try:  # torch >= 2.x unified AMP API
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover - older torch
    from torch.cuda.amp import GradScaler

logger = get_logger(__name__)


class ContinualTrainer:
    def __init__(
        self,
        model: ESMCAModel,
        device: torch.device,
        lr: float = 1e-4,
        epochs_per_task: int = 3,
        n_prototype_samples: int = 256,
        drift_delta: float = 0.05,
        attrib_loss_weight: float = 1.0,
        ig_steps: int = 32,
        weight_decay: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs_per_task = epochs_per_task
        self.n_prototype_samples = n_prototype_samples
        self.attrib_loss_weight = attrib_loss_weight

        self.ame = AttributionMapExtractor(model, n_steps=ig_steps)
        self.router = AttributionBasedRouter()
        self.edm = ExplanationDriftMonitor(delta=drift_delta, attrib_lambda=1.0)

        self.prototypes: Dict[str, torch.Tensor] = {}
        self.tasks: Dict[str, Task] = {}
        self.drift_history: Dict[str, Dict[str, float]] = {}

    def _batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _pre_training_drift_check(self, task_name: str) -> List[str]:
        """Step 5-ish, but run *before* training the new task: measure drift
        of past tasks against the model as it stands right now, so we know
        which ones to anchor with L_attrib while training the new task."""
        if not self.prototypes:
            return []
        current_attrs = {}
        for past_name, past_task in self.tasks.items():
            batch = next(iter(past_task.val_loader))
            batch = self._batch_to_device(batch)
            attrs = self.ame.attribute(batch["input_ids"], batch["attention_mask"], target=batch["labels"])
            current_attrs[past_name] = attrs.mean(dim=0)
        drift_scores = self.edm.check_all(current_attrs, self.prototypes)
        self.drift_history[task_name] = drift_scores
        flagged = self.edm.tasks_needing_consolidation(drift_scores)
        if flagged:
            logger.info(f"EDM flagged drift on tasks {flagged} before training '{task_name}'")
        return flagged

    def _consolidation_step(self, optimizer: AdamW, flagged_tasks: List[str], task_name: str) -> None:
        if not flagged_tasks:
            return
        # Accumulate gradients one task at a time to avoid OOM from
        # storing multiple autograd graphs simultaneously.
        grad_accum: Dict[torch.Tensor, torch.Tensor] = {}
        for past_name in flagged_tasks:
            optimizer.zero_grad()
            batch = next(iter(self.tasks[past_name].val_loader))
            batch = self._batch_to_device(batch)
            attrs = self.ame.attribute_differentiable(
                batch["input_ids"], batch["attention_mask"], target=batch["labels"]
            )
            current_attr = attrs.mean(dim=0)
            loss = self.attrib_loss_weight * self.edm.attribution_loss(
                current_attr, self.prototypes[past_name]
            )
            loss.backward()
            for p in optimizer.param_groups[0]["params"]:
                if p.grad is not None:
                    if p in grad_accum:
                        grad_accum[p] = grad_accum[p] + p.grad.clone()
                    else:
                        grad_accum[p] = p.grad.clone()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        # Apply accumulated gradients and step once
        for p, g in grad_accum.items():
            p.grad = g
        optimizer.step()

    def train_task(self, task: Task, compute_prototype: bool = True) -> None:
        task_name = task.name
        self.tasks[task_name] = task
        logger.info(f"=== Starting task '{task_name}' ({len(self.prototypes)} previous tasks) ===")
        flagged_tasks = self._pre_training_drift_check(task_name)
        if flagged_tasks:
            logger.info(f"  Consolidation will regularize: {flagged_tasks}")

        self.model.start_new_task(task_name)
        params = list(self.model.task_trainable_parameters(task_name))
        n_trainable = sum(p.numel() for p in params)
        logger.info(f"  Trainable parameters: {n_trainable:,}")
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        # The frozen backbone runs fp16 (6 GB GPU), so gradients flow through
        # 24 layers of fp16 activations and underflow without loss scaling.
        scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        for epoch in range(self.epochs_per_task):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in tqdm(task.train_loader, desc=f"[{task_name}] epoch {epoch + 1}/{self.epochs_per_task}"):
                batch = self._batch_to_device(batch)
                optimizer.zero_grad()
                logits = self.model.forward_task(batch["input_ids"], batch["attention_mask"], task_name)
                loss = task_loss(logits, batch["labels"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"  epoch {epoch + 1}/{self.epochs_per_task} — avg_loss={avg_loss:.4f}")
            self._consolidation_step(optimizer, flagged_tasks, task_name)

        self.model.eval()
        if compute_prototype:
            logger.info(f"  Computing prototype ({self.n_prototype_samples} samples)...")
            prototype = self.ame.compute_prototype(task.val_loader, n_samples=self.n_prototype_samples)
            self.prototypes[task_name] = prototype.detach()
        self.model.freeze_task(task_name)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"Finished task '{task_name}'. Bank now holds {len(self.prototypes)} task(s).")

    def predict_routed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Task-label-free inference: attribute against the frozen backbone,
        route via cosine similarity to stored prototypes, then compose."""
        self.model.eval()
        with torch.no_grad():
            phi_x = self.ame.attribute(input_ids, attention_mask)
            weights = self.router.route(phi_x, self.prototypes)
        return self.model.forward_composed(input_ids, attention_mask, weights)
