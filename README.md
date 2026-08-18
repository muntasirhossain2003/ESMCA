# ESMCA: Explanation-Steered Modular Continual Adapters

Implementation of the ESMCA research proposal: a continual-learning system
that replaces the learned, black-box task router found in prior LoRA-based
CL methods with **attribution maps** (Integrated Gradients). The same
attribution mechanism serves two purposes:

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

---

## Architecture

```
Input x → Frozen Backbone (RoBERTa) ─┬→ Attribution Map Extractor (AME, Integrated Gradients)
                                      │         │
                                      │         ↓
                                      │   Attribution-Based Router (ABR): cosine sim → softmax(s/τ)
                                      │         │
                                      │         ↓
                                      └→ LoRA Adapter Bank {(A_t, B_t)} ─composed──→ Final Output
                                                ↑
                                                │
                                    Explanation Drift Monitor (EDM): KL(φ'(x_old) || Φ_t)
                                    → triggers attribution-anchored consolidation if drift > δ
```

---

## Complete Module Reference

### `esmca/` — Core Package

#### `esmca/models/backbone.py` — Frozen Pretrained Backbone

| Symbol | Type | Description |
|---|---|---|
| `NUM_CHOICES = 4` | constant | MMLU is 4-way classification |
| `load_backbone(model_name)` | function | Loads + freezes `AutoModel` + `AutoTokenizer` (default `roberta-base`). All backbone params set to `requires_grad=False`. |
| `find_query_value_linears(backbone)` | function | Returns `List[Tuple[str, nn.Linear]]` — WQ/WV projections from every encoder layer for LoRA injection |
| `ClassificationHead` | class | Shared 4-way head: `Linear(hidden_size, hidden_size) → Tanh → Linear(hidden_size, 4)` |

#### `esmca/models/lora.py` — Task-Specific LoRA Adapter Bank (Sec 3.2.1)

Equation: `ΔW_t = B_t A_t`, `B_t ∈ R^(d×r)`, `A_t ∈ R^(r×k)`, `r ∈ {4, 8, 16}`

| Symbol | Type | Description |
|---|---|---|
| `LoRAPair` | class | Single task's (A, B) low-rank pair. A: Kaiming-uniform init; B: zero-init (no-op start). `forward(x) = (x @ Aᵀ) @ Bᵀ` |
| `LoRAInjectedLinear` | class | Wraps frozen `nn.Linear` with per-task LoRA delta bank via `nn.ModuleDict`. Two modes: **training** (single active adapter) and **composed/inference** (blended routing weights). Key methods: `add_task`, `freeze_task`, `set_active_task`, `set_routing_weights` |
| `AdapterBank` | class | Manages all `LoRAInjectedLinear` modules across backbone layers. Tracks `task_order`. Methods: `add_task`, `freeze_task`, `set_active_task`, `set_routing_weights`, `trainable_parameters` |
| `inject_lora(backbone, targets, rank, alpha)` | function | Replaces each WQ/WV `nn.Linear` in-place with `LoRAInjectedLinear`. Returns `AdapterBank` |

#### `esmca/models/esmca_model.py` — ESMCAModel (Central Model)

Wires frozen backbone + LoRA adapter bank + shared classification head.

| Method | Description |
|---|---|
| `encode(input_ids, attention_mask)` | Returns [CLS] hidden state from backbone |
| `forward(input_ids, attention_mask)` | Standard forward through backbone + head |
| `forward_frozen(input_ids, attention_mask)` | Forward with NO adapters active (AME attribution target) |
| `forward_composed(input_ids, attention_mask, routing_weights)` | Forward with ABR-blended adapters (inference) |
| `forward_task(input_ids, attention_mask, task_name)` | Forward with a single task's adapter active (training) |
| `start_new_task(task_name)` | Adds new LoRA adapter + sets it active |
| `freeze_task(task_name)` | Freezes completed task's adapter |
| `task_trainable_parameters(task_name)` | Yields adapter + head parameters |

#### `esmca/attribution/ame.py` — Attribution Map Extractor (Sec 3.2.2)

Uses Integrated Gradients (via `captum`) attributing frozen-backbone logits to input token embeddings with zero/padding baseline:

```math
φ_i(x) = (x_i - x'_i) ∫₀¹ ∂F(x' + α(x - x')) / ∂x_i dα
```

| Method | Description |
|---|---|
| `attribute(input_ids, attention_mask, target)` | Returns φ(x): one attribution scalar per token, shape `[B, seq_len]`. Non-differentiable, multi-step IG, detached. |
| `attribute_differentiable(input_ids, attention_mask, target)` | Grad-times-input attribution staying in autograd graph (for consolidation loss). Does NOT force frozen-only path. |
| `compute_prototype(loader, n_samples)` | Averages attributions over up to `n_samples` validation examples → per-task prototype Φ_t |

#### `esmca/attribution/router.py` — Attribution-Based Router (Sec 3.2.3, Eq. 1-3)

```math
s_t = cos(φ(x), Φ_t), \quad w_t = softmax(s_t / τ), \quad ΔW_comp = Σ_t w_t · B_t A_t
```

| Method | Description |
|---|---|
| `route(phi_x, prototypes)` | Cosine similarity → softmax(·/τ) → per-task weights dict |

#### `esmca/attribution/drift_monitor.py` — Explanation Drift Monitor (Sec 3.2.4, Eq. 4)

```math
D_t = KL(φ'(x_old) || Φ_t), \quad L_attrib = Σ_t λ · ||φ_{T+1}(x_t) - Φ_t||² \quad (\text{if } D_t > δ)
```

| Method | Description |
|---|---|
| `drift(current_attr, prototype, eps)` | KL divergence D_t between current attribution and stored prototype |
| `check_all(current_attrs, prototypes)` | Computes D_t for every past task |
| `tasks_needing_consolidation(drift_scores)` | Returns tasks whose drift exceeds δ |
| `attribution_loss(current_attr, prototype)` | MSE loss `‖φ - Φ_t‖²` scaled by λ |
| `ads(drift_scores)` (static) | Mean drift across all tasks = Attribution Drift Score |

Helper: `_to_distribution(attr, eps)` — abs-normalization for proper probability distribution.

#### `esmca/training/losses.py` — Loss Terms (Sec 3.3)

```math
L_total = L_task + α · L_attrib
```

| Symbol | Description |
|---|---|
| `task_loss(logits, labels)` | Cross-entropy loss |

#### `esmca/training/trainer.py` — ContinualTrainer (Sec 3.3)

Training protocol per new task T+1:
1. Initialize fresh LoRA adapter (A_{T+1}, B_{T+1})
2. All previous adapters already frozen
3. EDM pre-training drift check on past tasks
4. Train on D_{T+1} with L_total = L_task + α · L_attrib (L_attrib only for EDM-flagged tasks)
5. Compute prototype Φ_{T+1} on validation sample
6. Store (A_{T+1}, B_{T+1}, Φ_{T+1}) in bank

| Method | Description |
|---|---|
| `train_task(task)` | Full training cycle for one task |
| `_pre_training_drift_check(task_name)` | Runs EDM drift check before training new task; returns flagged tasks |
| `_consolidation_step(optimizer, flagged_tasks, task_name)` | Accumulates gradients from attribution-anchored consolidation loss for flagged past tasks. One-at-a-time to avoid OOM. |
| `predict_routed(input_ids, attention_mask)` | Task-label-free inference: attribute → route → compose |

State: `self.prototypes: Dict[str, Tensor]`, `self.tasks: Dict[str, Task]`, `self.drift_history: Dict[str, Dict[str, float]]`.

#### `esmca/data/mmlu.py` — MMLU Dataset Loading

| Symbol | Type | Description |
|---|---|---|
| `NUM_CHOICES = 4` | constant | |
| `Task` | dataclass | Holds `name`, `train_loader`, `val_loader`, `test_loader` for one MMLU subject |
| `MMLUClassificationDataset` | class | Wraps HF split. Formats: `"question (0) choice0 (1) choice1 (2) choice2 (3) choice3"`, tokenizes with padding/truncation |
| `load_mmlu_tasks(subjects, tokenizer, max_seq_len, batch_size)` | function | Downloads `cais/mmlu` per subject, creates Train/Val/Test DataLoaders. Train on `validation` split; test on `test` split. |

#### `esmca/evaluation/metrics.py` — Standard CL Metrics (Sec 6.1)

R[i][j] = accuracy on task j after training on tasks 1..i.

| Function | Formula | Description |
|---|---|---|
| `evaluate_accuracy(predict_fn, loader)` | — | Accuracy over a DataLoader |
| `compute_acc(R)` | `mean(R[T-1])` | Average accuracy after all tasks |
| `compute_bwt(R)` | `1/(T-1) · Σ_{t=1}^{T-1} (R_{T,t} - R_{t,t})` | Backward Transfer (negative = forgetting) |
| `compute_fwt(R)` | `1/(T-1) · Σ_{i=1}^{T-1} R_{i-1,i}` | Forward Transfer (zero-shot before training) |

#### `esmca/evaluation/xai_metrics.py` — XAI Metrics (Sec 6.2)

| Function | Description |
|---|---|
| `attribution_drift_score(drift_history)` | **ADS** — mean of all recorded D_t across all checkpoints. Lower = less forgetting in attribution space. |
| `routing_interpretability_score(true_idx, routed_idx)` | **RIS** — Spearman correlation between router's argmax choice and ground-truth task label. Never exposes label to router. |
| `attribution_fidelity(prototypes)` | t-SNE 2D embedding of task prototypes + silhouette score measuring cluster separability |

#### `esmca/utils/config.py` — Configuration

```python
@dataclass
class Config:
    backbone: str = "roberta-base"
    max_seq_len: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16
    subjects: list  # default 5 MMLU subjects
    batch_size: int = 8
    epochs_per_task: int = 3
    lr: float = 1e-4
    n_prototype_samples: int = 256
    ig_steps: int = 8
    router_tau: float = 0.1
    drift_delta: float = 0.05
    attrib_loss_weight: float = 1.0
    consolidation_lambda: float = 1.0
    ewc_lambda: float = 1000.0
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"

    @staticmethod
    def from_yaml(path: str) -> "Config"
```

#### `esmca/utils/logging.py` — Logging

| Symbol | Description |
|---|---|
| `_LOG_DIR` | `<project_root>/logs/` |
| `get_logger(name)` | Creates logger writing to both stdout and `logs/esmca.log` with `[timestamp] name - LEVEL - message` format |

#### `esmca/utils/seed.py` — Seeding

| Function | Description |
|---|---|
| `set_seed(seed)` | Sets `random`, `numpy`, `torch` (CPU + CUDA) seeds |

---

### Baselines

#### `esmca/baselines/sequential_ft.py` — Sequential Fine-Tuning

| Symbol | Description |
|---|---|
| `SequentialFTModel(nn.Module)` | Fully trainable backbone + head (unfrozen, unlike ESMCA). Wraps `AutoModel` + `ClassificationHead`. |
| `run_sequential_finetuning(model, tasks, device, lr, epochs_per_task)` | Trains sequentially with no regularization. Returns accuracy matrix R. |

Lower bound: full catastrophic forgetting baseline (Table 3).

#### `esmca/baselines/ewc.py` — Elastic Weight Consolidation

| Symbol | Description |
|---|---|
| `_estimate_fisher(model, loader, device, n_batches)` | Estimates diagonal Fisher information matrix over trainable params via gradient-squared on validation samples |
| `run_ewc(model, tasks, device, lr, epochs_per_task, ewc_lambda)` | Full EWC training loop. After each task: saves Fisher + optimal params. During new tasks: quadratic penalty Σ (λ/2) · F_n · (θ - θ*_n)² |

Regularization-based CL baseline (Kirkpatrick et al., 2017).

---

### Scripts (Entry Points)

#### `scripts/run_esmca.py` — Main ESMCA Experiment

```
python scripts/run_esmca.py --config configs/default.yaml
```

Workflow:
1. Load config + set seed + create model
2. Create MMLU tasks from config subjects
3. Create `ContinualTrainer`
4. For each task sequentially: train → evaluate on ALL tasks (zero-shot for future tasks)
5. Compute metrics: ACC, BWT, FWT, ADS, RIS, t-SNE silhouette
6. Write `esmca_results.json` to `<output_dir>/`

#### `scripts/run_baseline.py` — Baseline Experiments

```
python scripts/run_baseline.py --method sequential_ft --config configs/default.yaml
python scripts/run_baseline.py --method ewc --config configs/default.yaml
```

Workflow:
1. Load config + create `SequentialFTModel` + tasks
2. Run selected baseline method
3. Compute ACC/BWT/FWT
4. Write `<method>_results.json` to `<output_dir>/`

---

### Configuration Files

| Config | Subjects | Batch | Epochs | LR | LoRA | Prototype | IG Steps | Device |
|---|---|---|---|---|---|---|---|---|
| `configs/default.yaml` | 5 | 4 | 3 | 1e-4 | r=8, α=16 | 128 | 8 | CUDA |
| `configs/full_mmlu.yaml` | 57 (all) | 8 | 3 | 1e-4 | r=8, α=16 | 256 | 8 | CUDA |
| `configs/smoke.yaml` | 2 | 4 | 1 | 1e-4 | r=4, α=8 | 8 | 4 | CPU |

---

### Tests

Run: `pytest tests/`

| File | Tests | What it validates |
|---|---|---|
| `tests/test_lora.py` | 3 | (1) New adapter is no-op (B zero-init). (2) Frozen adapters stop gradients. (3) Routing weights blend multiple adapters (symmetric cancellation). |
| `tests/test_router.py` | 2 | (1) Weights sum to 1; closest prototype gets highest weight. (2) Batched φ is averaged before routing. |
| `tests/test_drift_monitor.py` | 4 | (1) Zero drift when attr == prototype. (2) Large drift flags consolidation. (3) ADS is mean of scores. (4) Empty history ADS = 0. |
| `tests/test_metrics.py` | 6 | (1) ACC uses last row. (2) BWT=0 when no forgetting. (3) BWT<0 when forgetting. (4) FWT uses pre-training diagonal. (5) ADS matches manual average. (6) RIS=1 for perfect alignment. |

#### `conftest.py`

Inserts project root into `sys.path` so `import esmca` works from pytest.

---

### Complete Workflow

```
                    ┌──────────────────────────────────────────────────┐
                    │            run_esmca.py / run_baseline.py        │
                    │  Load YAML config → set_seed → create model      │
                    └──────────┬───────────────────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────────────────┐
                    │   load_mmlu_tasks(subjects, tokenizer, ...)      │
                    │   → List[Task] (one per MMLU subject)            │
                    └──────────┬───────────────────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────────────────┐
                    │            ContinualTrainer                       │
                    │   for each task in sequence:                      │
                    │                                                   │
                    │   1. _pre_training_drift_check                    │
                    │      → AME.attribute on past tasks' val data      │
                    │      → EDM.check_all vs stored prototypes         │
                    │      → EDM.tasks_needing_consolidation            │
                    │                                                   │
                    │   2. model.start_new_task + optimizer              │
                    │                                                   │
                    │   3. for epoch in 1..epochs_per_task:             │
                    │      a. Train: forward_task → task_loss → step    │
                    │      b. _consolidation_step (if drift flagged):    │
                    │         → AME.attribute_differentiable            │
                    │         → EDM.attribution_loss → backward         │
                    │         → gradient accumulation → step            │
                    │                                                   │
                    │   4. AME.compute_prototype (n_samples) → Φ_t      │
                    │   5. model.freeze_task                            │
                    │                                                   │
                    │   6. Evaluate on ALL tasks:                       │
                    │      predict_routed (AME→ABR→forward_composed)    │
                    │      → evaluate_accuracy → matrix R[i][j]         │
                    └──────────┬───────────────────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────────────────┐
                    │           Final Metrics (ESMCA)                   │
                    │   ┌─────────────────────────────────────────┐    │
                    │   │ ACC  = compute_acc(R)                   │    │
                    │   │ BWT  = compute_bwt(R)                   │    │
                    │   │ FWT  = compute_fwt(R)                   │    │
                    │   │ ADS  = attribution_drift_score(hist)    │    │
                    │   │ RIS  = routing_interpretability_score   │    │
                    │   │ Sil  = attribution_fidelity(protos)     │    │
                    │   └─────────────────────────────────────────┘    │
                    │   → write esmca_results.json                     │
                    └──────────────────────────────────────────────────┘
```

---

### All Metrics Summary

| Metric | Symbol | Type | Range | Interpretation |
|---|---|---|---|---|
| **Accuracy** | ACC | Standard CL | [0, 1] | Higher = better overall performance |
| **Backward Transfer** | BWT | Standard CL | (-∞, +∞) | Negative = forgetting; higher = better retention |
| **Forward Transfer** | FWT | Standard CL | (-∞, +∞) | Higher = better zero-shot generalization |
| **Attribution Drift Score** | ADS | XAI/Novel | [0, +∞) | Lower = less forgetting in attribution space |
| **Routing Interpretability Score** | RIS | XAI/Novel | [-1, +1] | Higher = router aligns with ground-truth task (without labels) |
| **Attribution Fidelity** | Silhouette | XAI/Novel | [-1, +1] | Higher = task prototypes form distinct clusters |

---

### Dependencies

```
torch>=2.1.0          # Deep learning framework
transformers>=4.40.0  # HuggingFace models (roberta-base)
datasets>=2.19.0      # MMLU dataset loading (cais/mmlu)
captum>=0.7.0         # Integrated Gradients (AME)
scikit-learn>=1.4.0   # t-SNE + silhouette score
scipy>=1.11.0         # Spearman correlation (RIS)
numpy>=1.26.0         # Numerical ops
pyyaml>=6.0           # Config loading
tqdm>=4.66.0          # Progress bars
pytest>=8.0.0         # Testing
```

---

### Design Decisions / Simplifications vs. Proposal

- **MMLU as uniform 4-way classification** — all tasks share one classification head; task-specific behavior lives in LoRA adapters.
- **Backbone is fully frozen** (including for attribution) — AME always reads off `forward_frozen`.
- **Per-batch routing weights** — ABR computes one routing-weight vector from batch-mean attribution (not per-example IG), trading precision for tractable runtime.
- **Drift-triggered consolidation checked once per task** (pre-training) rather than every batch. Consolidation gradient step runs once per epoch on cached held-out samples.
- **FWT** — computed as average zero-shot accuracy on task i using model state before training on task i.
- **Baselines** (SeqFT, EWC) fine-tune the entire backbone (unfrozen), consistent with classical CL comparisons. Only ESMCA keeps the backbone frozen.

---

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

No Hugging Face login needed — `cais/mmlu` is public.

---

### Running

```bash
# Small dev run (5 subjects, roberta-base, LoRA rank 8)
python scripts/run_esmca.py --config configs/default.yaml

# Full 57-subject MMLU sequence
python scripts/run_esmca.py --config configs/full_mmlu.yaml

# Smoke test (2 subjects, CPU, minimal settings)
python scripts/run_esmca.py --config configs/smoke.yaml

# Baselines
python scripts/run_baseline.py --method sequential_ft --config configs/default.yaml
python scripts/run_baseline.py --method ewc --config configs/default.yaml
```

Results written to `<output_dir>/esmca_results.json` (or `<method>_results.json`).

---

### Tests

```bash
pytest tests/ -v
```

---

### Output Format

```json
{
  "R": [[0.34, 0.28, ...], ...],
  "ACC": 0.2498,
  "BWT": -0.0498,
  "FWT": 0.2257,
  "ADS": 0.5393,
  "RIS": -0.1691,
  "attribution_fidelity_silhouette": 0.0,
  "task_order": ["abstract_algebra", "anatomy", ...]
}
```

---

### Project File Inventory

**32 tracked files** (excluding `__pycache__/`, `.git/`, `.venv/`, etc.):

| Path | Purpose |
|---|---|
| `esmca/__init__.py` | Package marker |
| `esmca/models/backbone.py` | Frozen backbone + classification head |
| `esmca/models/lora.py` | LoRA adapter bank (LoRAPair, LoRAInjectedLinear, AdapterBank) |
| `esmca/models/esmca_model.py` | ESMCAModel (wires backbone + adapters + head) |
| `esmca/attribution/ame.py` | Attribution Map Extractor (IG) |
| `esmca/attribution/router.py` | Attribution-Based Router |
| `esmca/attribution/drift_monitor.py` | Explanation Drift Monitor |
| `esmca/training/losses.py` | Task loss (cross-entropy) |
| `esmca/training/trainer.py` | ContinualTrainer (training protocol) |
| `esmca/data/mmlu.py` | MMLU dataset + Task dataclass |
| `esmca/evaluation/metrics.py` | ACC/BWT/FWT |
| `esmca/evaluation/xai_metrics.py` | ADS/RIS/Attribution Fidelity |
| `esmca/baselines/sequential_ft.py` | Sequential fine-tuning baseline |
| `esmca/baselines/ewc.py` | EWC baseline |
| `esmca/utils/config.py` | Config dataclass + YAML loading |
| `esmca/utils/logging.py` | Shared stdout+file logger |
| `esmca/utils/seed.py` | Deterministic seeding |
| `configs/default.yaml` | Dev config (5 subjects) |
| `configs/full_mmlu.yaml` | Full experiment (57 subjects) |
| `configs/smoke.yaml` | Smoke test (2 subjects, CPU) |
| `scripts/run_esmca.py` | Main entry point |
| `scripts/run_baseline.py` | Baseline entry point |
| `tests/test_lora.py` | LoRA adapter tests |
| `tests/test_router.py` | ABR tests |
| `tests/test_drift_monitor.py` | EDM tests |
| `tests/test_metrics.py` | Metrics tests |
| `conftest.py` | Pytest path setup |
| `requirements.txt` | Dependencies |
| `.gitignore` | Git ignore rules |
| `README.md` | This file |
| `esmca (1).pdf` | Research proposal |
