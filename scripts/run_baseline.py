#!/usr/bin/env python
"""Entrypoint: train and evaluate a baseline continual learning method on MMLU.

Usage:
    python scripts/run_baseline.py --method sequential_ft --config configs/default.yaml
    python scripts/run_baseline.py --method ewc --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esmca.baselines.ewc import run_ewc
from esmca.baselines.sequential_ft import SequentialFTModel, run_sequential_finetuning
from esmca.data.mmlu import load_mmlu_tasks
from esmca.evaluation.metrics import compute_acc, compute_bwt, compute_fwt
from esmca.utils.config import Config
from esmca.utils.logging import get_logger
from esmca.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sequential_ft", "ewc"], required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = SequentialFTModel(cfg.backbone)
    tasks = load_mmlu_tasks(cfg.subjects, model.tokenizer, cfg.max_seq_len, cfg.batch_size)

    if args.method == "sequential_ft":
        R = run_sequential_finetuning(model, tasks, device, cfg.lr, cfg.epochs_per_task)
    else:
        R = run_ewc(model, tasks, device, cfg.lr, cfg.epochs_per_task, cfg.ewc_lambda)

    results = {"R": R, "ACC": compute_acc(R), "BWT": compute_bwt(R), "FWT": compute_fwt(R)}
    out_path = os.path.join(cfg.output_dir, f"{args.method}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"[{args.method}] ACC={results['ACC']:.4f} BWT={results['BWT']:.4f} FWT={results['FWT']:.4f}")
    logger.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
