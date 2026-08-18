"""MMLU dataset loading and task construction for continual learning.

Each MMLU subject becomes one task in the task-incremental sequence
(paper Table 2: "MMLU, 57 subjects (sequential), Large-scale
task-incremental stress test"). A 4-choice question is framed as a
choice-scoring problem: each candidate answer is encoded separately as
`question + choice`, and the model scores the four encodings, picking
the highest-scoring option (the standard MMLU framing for encoder
models).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

NUM_CHOICES = 4


@dataclass
class Task:
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class MMLUClassificationDataset(Dataset):
    """Wraps one MMLU subject split into choice-scoring examples.

    Each example is encoded as 4 separate sequences -- `question + " " +
    choice` for each of the 4 options (the standard MMLU framing for
    encoder models). The model scores each option and picks the highest;
    the label is the correct choice index.
    """

    def __init__(self, hf_split, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.examples = hf_split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        question = ex["question"]
        encs = []
        for choice in ex["choices"]:
            enc = self.tokenizer(
                question + " " + choice,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            encs.append(enc)
        input_ids = torch.cat([enc["input_ids"] for enc in encs], dim=0)  # [4, L]
        attention_mask = torch.cat([enc["attention_mask"] for enc in encs], dim=0)  # [4, L]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(ex["answer"], dtype=torch.long),
        }


def load_mmlu_tasks(
    subjects: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch_size: int,
    train_frac: Optional[float] = None,
    seed: int = 42,
) -> List[Task]:
    """Download (via Hugging Face `datasets`) and build one Task per MMLU subject.

    The `cais/mmlu` "validation" split only holds 9-16 examples per subject,
    far too few to fine-tune a LoRA adapter. When `train_frac` is set
    (0 < train_frac < 1), we train on a seeded `train_frac` slice of each
    subject's "test" split and hold the remainder out for evaluation; the
    training slice also serves as the validation/prototype source. When
    `train_frac` is None, the legacy behavior is kept (train/val on
    "validation", eval on "test").
    """
    tasks: List[Task] = []
    for subject in subjects:
        raw = load_dataset("cais/mmlu", subject)
        if train_frac is not None and 0.0 < train_frac < 1.0:
            test_raw = raw["test"]
            n_train = int(train_frac * len(test_raw))
            rng = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(test_raw), generator=rng).tolist()
            train_ds = MMLUClassificationDataset(test_raw.select(order[:n_train]), tokenizer, max_seq_len)
            val_ds = train_ds
            test_ds = MMLUClassificationDataset(test_raw.select(order[n_train:]), tokenizer, max_seq_len)
        else:
            # MMLU's "dev" split only holds 5 few-shot exemplars, too small to
            # adapt on. We train the per-task LoRA adapter on "validation" and
            # reserve "test" for evaluation and prototype computation.
            train_ds = MMLUClassificationDataset(raw["validation"], tokenizer, max_seq_len)
            val_ds = MMLUClassificationDataset(raw["validation"], tokenizer, max_seq_len)
            test_ds = MMLUClassificationDataset(raw["test"], tokenizer, max_seq_len)
        tasks.append(
            Task(
                name=subject,
                train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
                val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
                test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
            )
        )
    return tasks
