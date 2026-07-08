# ESMCA: Explanation-Steered Modular Continual Adapters

Implementation of the ESMCA research proposal (`esmca (1).pdf`): a
continual-learning system that replaces the learned, black-box task
router found in prior LoRA-based CL methods with **attribution maps**
(Integrated Gradients). The same attribution mechanism serves two
purposes:

1. **Routing** — at inference, an input's attribution map is compared
   (cosine similarity) against stored per-task attribution prototypes
   to produce softmax blending weights over the task adapter bank.
   No task label is ever required.
2. **Forgetting detection** — after each new task, attribution drift
   (KL divergence) between fresh attributions on old tasks' data and
   their stored prototypes flags forgetting *before* accuracy drops,
   triggering a lightweight consolidation loss.

Evaluated on the **MMLU** dataset (57 subjects), downloaded via the
Hugging Face `datasets` library (`cais/mmlu`), treating each subject as
one task in a task-incremental sequence.

## Architecture

```
Input x -> Frozen Backbone (RoBERTa) -+-> Attribution Map Extractor (AME, Integrated Gradients)
                                       |         |
                                       |         v
                                       |   Attribution-Based Router (ABR): cosine sim -> softmax(s/tau)
                                       |         |
                                       |         v
                                       +-> LoRA Adapter Bank {(A_t, B_t)} --composed--> Final Output
                                                 ^
                                                 |
                                     Explanation Drift Monitor (EDM): KL(phi'(x_old) || Phi_t)
                                     -> triggers attribution-anchored consolidation if drift > delta
```

| Component | File | Paper section |
|---|---|---|
| Task-Specific LoRA Adapter Bank | `esmca/models/lora.py` | 3.2.1 |
| Attribution Map Extractor (AME) | `esmca/attribution/ame.py` | 3.2.2 |
| Attribution-Based Router (ABR) | `esmca/attribution/router.py` | 3.2.3 |
| Explanation Drift Monitor (EDM) | `esmca/attribution/drift_monitor.py` | 3.2.4 |
| Training protocol | `esmca/training/trainer.py` | 3.3 |
| Standard CL metrics (ACC/BWT/FWT) | `esmca/evaluation/metrics.py` | 6.1 |
| XAI metrics (ADS/RIS/Fidelity) | `esmca/evaluation/xai_metrics.py` | 6.2 |
| Baselines (Sequential FT, EWC) | `esmca/baselines/` | Table 3 |

## Project layout

```
esmca/
  configs/                 YAML experiment configs
  esmca/
    data/mmlu.py            MMLU loading + tokenization (Hugging Face `datasets`)
    models/                 backbone, LoRA adapter bank, ESMCAModel
    attribution/            AME, ABR, EDM
    training/                training loop, losses
    evaluation/               ACC/BWT/FWT + ADS/RIS/Attribution Fidelity
    baselines/                Sequential Fine-tuning, EWC
    utils/                    config loading, seeding, logging
  scripts/
    run_esmca.py             train + evaluate ESMCA on a task sequence
    run_baseline.py           train + evaluate a baseline (sequential_ft | ewc)
  tests/                     unit tests (adapters, router, drift monitor, metrics)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a machine with no GPU (or to avoid pulling large CUDA wheels), install
the CPU-only PyTorch build first, then the rest of the requirements:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Requires a Hugging Face account is **not** needed — `cais/mmlu` is a
public dataset and downloads automatically on first use via
`datasets.load_dataset`.

## Running

Small dev run (5 MMLU subjects, `roberta-base`, LoRA rank 8):

```bash
python scripts/run_esmca.py --config configs/default.yaml
```

Full 57-subject MMLU sequence:

```bash
python scripts/run_esmca.py --config configs/full_mmlu.yaml
```

Baselines, using the same config/task sequence for a fair comparison:

```bash
python scripts/run_baseline.py --method sequential_ft --config configs/default.yaml
python scripts/run_baseline.py --method ewc --config configs/default.yaml
```

Each run writes a JSON results file (accuracy matrix `R`, ACC, BWT, FWT,
and for ESMCA also ADS/RIS/Attribution Fidelity) to `output_dir` in the
config (default: `outputs/<config_name>/`).

## Tests

```bash
pytest tests/
```

## Design decisions / simplifications vs. the proposal

The proposal is a research plan, not a fully-specified spec; a few
concrete choices were made to produce a runnable system:

- **MMLU as uniform 4-way classification.** Every MMLU question is
  4-choice, so all "tasks" (subjects) share one classification head
  format. This sidesteps the need for task-specific output heads: the
  classification head is shared and jointly trained, while all
  task-specific behavior lives in the LoRA adapters, matching the
  paper's emphasis on adapters as the unit of task specialization.
- **Backbone is fully frozen** (including for attribution), so the AME
  always reads off `ESMCAModel.forward_frozen` — matching Fig. 1, where
  the AME branches directly off the frozen backbone before any adapter
  is applied. Since the classification head still updates across
  tasks, this preserves a real (if smaller) drift signal for the EDM.
- **Per-batch routing weights.** The ABR computes one routing-weight
  vector from the batch-mean attribution rather than a fresh IG pass
  per individual example, trading a small amount of routing precision
  for tractable runtime (IG is $O(\text{n\_steps})$ forward/backward
  passes per call).
- **Drift-triggered consolidation is checked once per task** (before
  training starts on the new task) rather than every batch, and the
  consolidation gradient step runs once per epoch on a cached
  held-out sample from each flagged task — full per-batch IG-based
  regularization would be prohibitively slow for a research prototype.
- **FWT** has no fully-specified formula in the proposal; it's computed
  here as the average zero-shot accuracy on task $i$ using the model
  state right before training on task $i$ (standard CL convention).
- **Baselines** (Sequential Fine-tuning, EWC) fine-tune the *entire*
  backbone (unfrozen), consistent with their role as classical
  lower-bound / regularization-based CL comparisons — ESMCA is the only
  method that keeps the backbone frozen throughout.

## Datasets

Only **MMLU** (`cais/mmlu` on Hugging Face) is wired up, per the current
scope. The proposal's Table 2 also lists SuperGLUE, CLINC150, Split
CIFAR-100, DomainNet, CORe50, and VQA-v2 for a full multi-domain
evaluation — `esmca/data/mmlu.py` is intentionally isolated behind the
same `Task`/`DataLoader` interface so additional dataset loaders can be
dropped in alongside it without touching the model, attribution, or
training code.
