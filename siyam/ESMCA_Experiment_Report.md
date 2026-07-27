# ESMCA Experiments — What I Did and What Happened

**Project:** Explanation-Steered Modular Continual Adapters  
**Goal:** Use attribution maps (IG) to route task-specific LoRA adapters without task labels, and detect forgetting — one signal, two jobs.  
**Period:** June–July 2026  
**Hardware:** Kaggle free T4 GPU (15.6GB VRAM, 12hr session limit)

---

## The Core Idea

In continual learning, you train a model on tasks one by one without forgetting old ones. Usually you need to tell the model which task it's on at inference — that's cheating in real deployment. The idea here: use **Integrated Gradients (IG)** attribution maps to figure out which task a new input belongs to, route it to the right LoRA adapter, and simultaneously detect if the model is forgetting old tasks. Same signal, two jobs.

---

## v1–v2: Pipeline Setup

Got the basic LoRA injection working, fixed device bugs (new adapters created on CPU after model moved to GPU), fixed KL divergence argument order. Mostly plumbing. No meaningful results yet.

---

## v3: First Real Run — RoBERTa + MMLU

**Config:** RoBERTa-base (125M), MMLU 5 subjects, 3 epochs, lr=1e-4  
**Result:** Every task predicted class 2. Oracle ~0.25 (random chance). Adapters barely moved logits (max Δ=0.03).

**What went wrong:** MMLU asks encyclopedic knowledge questions like "which theorem applies here?" RoBERTa doesn't know the answers — it just guesses the same class every time. 70 training examples can't teach it medical licensing content. Wrong dataset entirely for this model size.

---

## v4: RoBERTa + CLINC150 (10 domains)

**Config:** RoBERTa-base, CLINC150 10 intent domains, shared classification head, lr=1e-4, 3 epochs  
**Result:**
- Oracle started strong (0.942 for banking right after training) but collapsed to 0.167 by the end
- ESMCA = uniform = oracle (all identical) — routing completely broken
- BWT = -0.51 (severe forgetting)

**What went wrong:** Two bugs found via diagnosis:
1. Shared head kept getting retrained by every new task → earlier adapters' outputs became meaningless (head drift, dense layer norm-diff = 43.6 from fresh init)
2. Attribution prototypes had off-diagonal cosine ~0.96 — all domains looked identical to the router
3. IG target mismatch: prototypes used true labels, queries used model's own prediction → own-task cosine similarity was **-0.897** (pointing away from correct task)

---

## v5: Fixed Head Drift — Per-Task Heads

**Config:** RoBERTa-base, CLINC150 10 domains, per-task heads (frozen with adapter), lr=1e-3, 10 epochs  
**Result:**
- Oracle held at 0.929–0.996 across all 10 domains ✓ (head drift fixed)
- ESMCA = uniform = oracle again — routing still broken
- Proto off-diagonal cosine: mean=0.964 (still near-identical)
- Sample attribution silhouette: -0.041 (negative — clusters overlapping)

**What went wrong:** IG through frozen RoBERTa highlights function words ("to", "from", "can") equally across all domains. Pooling these into word embeddings produces identical vectors regardless of domain. The backbone is too small and too generic.

**Diagnostic finding:** Token scores showed "to" getting +0.0369 attribution for banking, while "money" and "account" got near-zero. IG was answering the wrong question.

---

## v6: Contrastive Attribution Routing

**Config:** RoBERTa-base, CLINC150 10 domains, per-task heads, contrastive IG (subtract mean of other tasks' scores), lr=1e-3, 10 epochs  
**Result:**
- Oracle: 0.951 ✓
- ESMCA: **0.579** ← first time beating uniform
- Uniform: 0.381
- BWT: 0.0 ✓
- Silhouette: **+0.139** ← first positive
- RIS: 0.622
- Balanced routing: 0.068 (below chance due to banking recall=0.0)
- Proto off-diagonal: mean=0.865 (improved from 0.964)

**What worked:** Contrastive reweighting (subtracting shared function word signal) helped separate domain prototypes. ESMCA beat uniform by 20 points — first real routing signal.

**What went wrong:** Banking prototype polarity flipped (own-task cosine was negative). One bad prototype dragged balanced routing below chance even though 8/10 tasks routed correctly on the confusion matrix diagonal.

---

## v7: Two-Stage Adapted Routing + Polarity Fix + Tau Calibration

**Config:** Same as v6 + polarity correction + val-set tau calibration + two-stage inference (frozen shortlist → adapted IG refine)  
**Result:**
- Oracle: 0.951
- ESMCA v7: **still ~0.416** (two-stage adapted routing didn't help much)
- Balanced routing: 0.085
- Proto off-diagonal: mean=0.865

**What went wrong:** The two-stage approach ran adapted IG at inference but still computed prototypes with frozen IG — mixing spaces. Also at this point the complexity of the system was far from the original clean idea. Decided to go back to basics with a better backbone.

---

## v8: LLaMA-3.2 3B + MultiNLI (The Right Backbone)

**Config:** LLaMA-3.2 3B, 4-bit NF4 quantization, MultiNLI 5 matched genres (fiction/government/slate/telephone/travel), 200 examples/genre, 5 epochs, lr=2e-4, ig_steps=8, batch=4

**Training result:** Completed in 98.7 min. All 5 genres trained successfully.

**Critical finding:** Proto off-diagonal cosine = **0.994** — same problem as RoBERTa. Bigger backbone didn't fix it. The importance-weighted word embedding pooling collapses to near-identical vectors regardless of model size because frozen IG scores are still dominated by generic high-frequency tokens.

**Runtime disaster:** R-matrix eval started. Each genre eval took ~6,794 seconds (113 minutes) due to IG steps through LLaMA's 28 layers. One row of 5 evals = ~9.5 hours. Kaggle killed session after 12 hours having completed only 1 of 25 eval passes.

**Section 14 never ran** — no oracle/ESMCA/uniform comparison obtained.

---

## Summary Table

| Version | Backbone | Dataset | Oracle | ESMCA | Uniform | BWT | Key Issue |
|---|---|---|---|---|---|---|---|
| v3 | RoBERTa-base | MMLU | 0.25 | 0.25 | 0.25 | 0.0 | Wrong dataset, adapters inert |
| v4 | RoBERTa-base | CLINC150 | 0.167* | 0.167 | 0.167 | -0.51 | Head drift, IG target mismatch |
| v5 | RoBERTa-base | CLINC150 | 0.951 | 0.313 | 0.313 | 0.0 | Proto cosine 0.964, function word dominance |
| v6 | RoBERTa-base | CLINC150 | 0.951 | **0.579** | 0.381 | 0.0 | Banking polarity bug |
| v7 | RoBERTa-base | CLINC150 | 0.951 | 0.416 | 0.363 | 0.0 | Mixed IG spaces, too complex |
| v8 | LLaMA-3.2 3B | MultiNLI | N/A | N/A | N/A | — | Proto cosine 0.994, eval timeout |

*v4 oracle collapsed due to head drift — not a true oracle measurement

---

## What Was Actually Proven

**Works:**
- Per-task LoRA adapters + frozen heads → BWT=0.0, no forgetting (v5 onward)
- Contrastive attribution beats uniform blending by 20 points on CLINC150 (v6)
- ADS (Attribution Drift Score) stays near 0 with stable per-task heads — drift detection works
- RIS=0.622 shows routing has real structure, not random noise (v6)
- Silhouette turned positive (+0.139) with contrastive approach (v6)

**Not yet proven:**
- Plain frozen IG routing working at any model scale
- Routing working at all with LLaMA (eval never completed)
- Adapted IG prototypes separating genres (diagnosed as the fix, not yet run)

---

## Root Cause Analysis

The core problem across all versions: **frozen backbone IG highlights gradient-sensitive tokens, not domain-identifying tokens.** "To", "from", "can", "please" appear in every domain and get high IG scores because they're syntactically critical to the model's predictions. Domain-specific words ("balance", "recipe", "flight") get lower scores because they're more predictable given context.

Contrastive attribution (v6) partially fixes this by subtracting the shared signal. Adapted IG (running through the trained adapter) was diagnosed as the complete fix — the v5 diagnostic showed banking vs travel adapted IG correlation = -0.40 vs frozen IG correlation = +0.96. But this was never successfully implemented with consistent prototype/query spaces.

---

## What's Next

The most principled remaining experiment: compute prototypes with adapted IG (through each task's trained adapter), and use two-stage inference (cheap frozen shortlist → 2 adapted IG passes). This preserves the original "one signal, two jobs" claim. The diagnostic needs to run on the trained v8 model to verify adapted IG prototypes actually separate MultiNLI genres before committing another full run.

---

## Datasets Tried

- **MMLU** — 57 subject knowledge QA. Wrong for small encoders. Needs GPT-scale models.
- **CLI15NC0** — 10 intent domains, short utterances. Adapters learn well (oracle 0.95+) but 8-word sentences give IG too little to work with.
- **MultiNLI** — 5 NLI genres, longer premise+hypothesis pairs. Right difficulty for LLaMA. Routing untested due to timeout.

## Models Tried

- **RoBERTa-base (125M)** — Fast, cheap, but frozen IG is too generic. Can't learn MMLU at all. CLINC150 adapters learn well.
- **LLaMA-3.2 3B 4-bit NF4** — Rich representations, learns NLI well, but IG eval too slow for Kaggle's 12hr limit. Proto separation still failed (0.994).
