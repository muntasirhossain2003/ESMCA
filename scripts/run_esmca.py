#!/usr/bin/env python
"""Entrypoint: train and evaluate ESMCA on a sequence of MMLU subjects.

Usage:
    python scripts/run_esmca.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esmca.data.mmlu import load_mmlu_tasks
from esmca.evaluation.metrics import compute_acc, compute_bwt, compute_fwt, evaluate_accuracy
from esmca.evaluation.xai_metrics import attribution_drift_score, attribution_fidelity, routing_interpretability_score
from esmca.models.esmca_model import ESMCAModel
from esmca.training.trainer import ContinualTrainer
from esmca.utils.config import Config
from esmca.utils.logging import get_logger
from esmca.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = ESMCAModel(cfg.backbone, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha)
    tasks = load_mmlu_tasks(cfg.subjects, model.tokenizer, cfg.max_seq_len, cfg.batch_size)

    trainer = ContinualTrainer(
        model,
        device=device,
        lr=cfg.lr,
        epochs_per_task=cfg.epochs_per_task,
        n_prototype_samples=cfg.n_prototype_samples,
        drift_delta=cfg.drift_delta,
        attrib_loss_weight=cfg.attrib_loss_weight,
        ig_steps=cfg.ig_steps,
    )

    R = []
    for i, task in enumerate(tasks):
        trainer.train_task(task)

        # Evaluate on every task's test set (not just tasks seen so far): tasks
        # not yet trained have no adapter/prototype of their own, so routing
        # falls back to whatever adapters exist so far -- this zero-shot
        # generalization case is what FWT measures (Sec 6.1).
        row = []
        for j in range(len(tasks)):
            def predict_fn(input_ids, attention_mask, _t=trainer, _d=device):
                return _t.predict_routed(input_ids.to(_d), attention_mask.to(_d))

            acc = evaluate_accuracy(predict_fn, tasks[j].test_loader)
            row.append(acc)
        R.append(row)
        logger.info(f"[esmca] finished '{task.name}', row={row}")

    acc = compute_acc(R)
    bwt = compute_bwt(R)
    fwt = compute_fwt(R)
    ads = attribution_drift_score(trainer.drift_history)

    true_idx, routed_idx = [], []
    task_order = trainer.model.adapter_bank.task_order
    for j, task in enumerate(tasks):
        for batch in task.test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            phi_x = trainer.ame.attribute(batch["input_ids"], batch["attention_mask"])
            weights = trainer.router.route(phi_x, trainer.prototypes)
            routed_idx.append(task_order.index(max(weights, key=weights.get)))
            true_idx.append(j)
    ris = routing_interpretability_score(true_idx, routed_idx)
    fidelity = attribution_fidelity(trainer.prototypes)

    results = {
        "R": R,
        "ACC": acc,
        "BWT": bwt,
        "FWT": fwt,
        "ADS": ads,
        "RIS": ris,
        "attribution_fidelity_silhouette": fidelity["silhouette"],
        "task_order": task_order,
    }
    out_path = os.path.join(cfg.output_dir, "esmca_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"ACC={acc:.4f} BWT={bwt:.4f} FWT={fwt:.4f} ADS={ads:.4f} RIS={ris:.4f}")
    logger.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
