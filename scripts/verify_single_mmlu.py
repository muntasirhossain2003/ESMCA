#!/usr/bin/env python
"""Verify the ESMCA pipeline can solve a single MMLU subject on DeBERTa-v3-large.

Trains the full pipeline (frozen backbone + LoRA adapter bank + shared
classification head + attribution prototype + router) on ONE MMLU subject and
reports held-out accuracy against the 25% random baseline.

Because a subject's `test` split is only ~100 examples, a single 80/20 holdout
is too noisy for a verdict. This script averages over `--folds` seeded 80/20
splits (fresh model + split per fold) and reports mean +/- std.

Usage:
    python scripts/verify_single_mmlu.py --config configs/deberta_v3_large.yaml [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esmca.data.mmlu import load_mmlu_tasks
from esmca.evaluation.metrics import evaluate_accuracy
from esmca.models.esmca_model import ESMCAModel
from esmca.training.trainer import ContinualTrainer
from esmca.utils.config import Config
from esmca.utils.logging import get_logger
from esmca.utils.seed import set_seed

logger = get_logger(__name__)

RANDOM_CHANCE = 0.25
SOLVED_THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/deberta_v3_large.yaml")
    parser.add_argument("--folds", type=int, default=5, help="Number of seeded 80/20 splits to average over.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info(
        f"Backbone={cfg.backbone} lora_targets={cfg.lora_targets} rank={cfg.lora_rank} alpha={cfg.lora_alpha} "
        f"dropout={cfg.lora_dropout} lr={cfg.lr} wd={cfg.weight_decay} epochs={cfg.epochs_per_task}"
    )
    logger.info(f"Subjects={cfg.subjects} train_frac={cfg.train_frac} folds={args.folds} device={device}")

    per_fold: dict[str, dict[str, float]] = {}
    for fold in range(args.folds):
        seed = cfg.seed + fold
        set_seed(seed)
        logger.info(f"=== Fold {fold + 1}/{args.folds} (seed={seed}) ===")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = ESMCAModel(
            cfg.backbone,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_targets=cfg.lora_targets,
            lora_dropout=cfg.lora_dropout,
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        tasks = load_mmlu_tasks(
            cfg.subjects, model.tokenizer, cfg.max_seq_len, cfg.batch_size, train_frac=cfg.train_frac, seed=seed
        )

        trainer = ContinualTrainer(
            model,
            device=device,
            lr=cfg.lr,
            epochs_per_task=cfg.epochs_per_task,
            n_prototype_samples=cfg.n_prototype_samples,
            drift_delta=cfg.drift_delta,
            attrib_loss_weight=cfg.attrib_loss_weight,
            ig_steps=cfg.ig_steps,
            weight_decay=cfg.weight_decay,
        )

        for i, task in enumerate(tasks):
            # compute_prototype=False: IG prototypes only serve the multi-task
            # router, and with a single subject routing is the trivial weight-1
            # identity; skipping them keeps memory (and this laptop GPU) sane.
            trainer.train_task(task, compute_prototype=False)

            def task_fn(input_ids, attention_mask, _model=model, _task=task, _device=device):
                return _model.forward_task(input_ids.to(_device), attention_mask.to(_device), _task.name)

            train_acc = evaluate_accuracy(task_fn, task.train_loader)
            eval_acc = evaluate_accuracy(task_fn, task.test_loader)
            per_fold[f"{fold + 1}-{task.name}"] = {
                "seed": seed,
                "n_train": len(task.train_loader.dataset),
                "n_eval": len(task.test_loader.dataset),
                "train_acc": train_acc,
                "eval_acc": eval_acc,
                "eval_acc_task": eval_acc,
            }
            logger.info(f"  fold {fold + 1} '{task.name}' ({n_trainable:,} trainable): "
                        f"train={train_acc:.3f} eval={eval_acc:.3f}")

        del model, trainer, tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_accs = [d["eval_acc"] for d in per_fold.values()]
    mean_acc = statistics.mean(eval_accs)
    std_acc = statistics.stdev(eval_accs) if len(eval_accs) > 1 else 0.0
    solved = mean_acc >= SOLVED_THRESHOLD

    logger.info("=== Cross-validation verification results ===")
    logger.info(f"  per-fold eval accuracies: {[f'{a:.3f}' for a in eval_accs]}")
    logger.info(f"  mean eval acc = {mean_acc:.4f} +/- {std_acc:.4f}  (random chance = {RANDOM_CHANCE})")
    logger.info(f"  VERDICT: {'SOLVED' if solved else 'NOT SOLVED'} "
                f"(threshold={SOLVED_THRESHOLD})")
    if not solved:
        logger.info("  Hint: raise lora_rank / epochs_per_task, or lower lr in the config, then re-run.")

    results = {
        "backbone": cfg.backbone,
        "lora_targets": cfg.lora_targets,
        "subjects": cfg.subjects,
        "train_frac": cfg.train_frac,
        "folds": args.folds,
        "mean_eval_acc": mean_acc,
        "std_eval_acc": std_acc,
        "solved": solved,
        "solved_threshold": SOLVED_THRESHOLD,
        "per_fold": per_fold,
    }
    out_path = os.path.join(cfg.output_dir, "verify_single_mmlu_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()