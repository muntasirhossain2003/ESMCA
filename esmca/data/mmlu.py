"""MMLU dataset loading and task construction for continual learning.

Each MMLU subject becomes one task in the task-incremental sequence
(paper Table 2: "MMLU, 57 subjects (sequential), Large-scale
task-incremental stress test"). A 4-choice question is framed as a
4-way classification problem: input = question + "[SEP]" + choice,
repeated per choice, with the correct choice index as the label
encoded via a per-question softmax over 4 concatenated encodings.

To keep this a plain single-label classification task (simpler LoRA /
attribution pipeline), we instead concatenate all four choices into a
single input sequence and classify which of the 4 positions is
correct - i.e. a 4-class classification head per example.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    """Wraps one MMLU subject split into tokenized 4-way classification examples."""

    def __init__(self, hf_split, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.examples = hf_split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        question = ex["question"]
        choices = ex["choices"]
        text = question + " " + " ".join(f"({i}) {c}" for i, c in enumerate(choices))
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["answer"], dtype=torch.long),
        }


def load_mmlu_tasks(
    subjects: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch_size: int,
) -> List[Task]:
    """Download (via Hugging Face `datasets`) and build one Task per MMLU subject."""
    tasks: List[Task] = []
    for subject in subjects:
        raw = load_dataset("cais/mmlu", subject)
        # MMLU's "dev" split only holds 5 few-shot exemplars, too small to
        # adapt on. We train the per-task LoRA adapter on "validation" and
        # reserve "test" for evaluation and prototype computation.
        train_ds = MMLUClassificationDataset(raw["validation"], tokenizer, max_seq_len)
        val_ds = MMLUClassificationDataset(raw["validation"], tokenizer, max_seq_len)
        test_ds = MMLUClassificationDataset(raw["test"], tokenizer, max_seq_len)
        tasks.append(
            Task(
                name=subject,
                train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
                test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            )
        )
    return tasks
