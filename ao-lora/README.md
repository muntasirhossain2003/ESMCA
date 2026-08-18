# AO-LoRA — Attribution-Orthogonal LoRA

A new experimental branch, separate from `siyam/`'s v1–v8 iterations. Does not modify or
depend on anything in `siyam/` — reuses the same backbone/dataset choices (LLaMA-3.2 3B,
MultiNLI 5 genres) purely so results are comparable side by side.

## Why this branch exists

`siyam/ESMCA_Experiment_Report.md` documents 8 versions all hitting the same wall: attribution
maps computed through the **frozen** backbone are nearly identical across tasks (prototype
cosine similarity 0.96–0.99, every version, RoBERTa or LLaMA) because Integrated Gradients
keeps highlighting generic function words instead of task-specific content. Two clues in that
report point at the fix:

- v5's diagnostic: banking-vs-travel attribution correlation is **+0.96 through the frozen
  backbone** but **-0.40 through the trained adapter** — adapters do carve out distinguishable
  explanation spaces once trained; frozen attribution just never uses that.
- v6: contrastive attribution reweighting got real routing signal for the first time
  (ESMCA=0.579 vs uniform=0.381 on CLINC150) — proof the underlying idea works when the
  attribution signal is actually discriminative.
- v7 tried adapted attribution but mixed frozen-IG prototypes with adapted-IG queries
  inconsistently and regressed. Never cleanly re-run.

## What AO-LoRA changes

1. **Consistent adapted attribution everywhere** — training, prototypes, and inference
   queries all compute attribution through the relevant task's own adapter, never the frozen
   backbone. No frozen/adapted mixing possible by construction.
2. **Gradient×Input instead of multi-step Integrated Gradients** — one forward + one backward
   pass per (batch, adapter), instead of the `ig_steps`-fold blow-up that made v8's evaluation
   infeasible (single IG calls were taking ~113 minutes each on a T4).
3. **Orthogonality as a training objective** — while training task T+1's adapter, a
   contrastive hinge loss directly penalizes its attribution vector for being too similar to
   any earlier task's stored prototype. Extends O-LoRA's weight-space orthogonality into
   attribution space, instead of hoping separation emerges and discovering post-hoc that it
   didn't (which is what happened in v1–v8).
4. **No task-label routing, no shortlist stage needed** — since attribution is now cheap,
   full T-way comparison per batch is affordable without a two-stage shortlist.

See `ao-lora-v1-llama-3-2-3b-multinli.ipynb` for the full implementation and inline design
rationale (each section has a markdown cell explaining the choice made and which prior
version's finding motivated it).

## Status

Not yet run. Next step: run on Kaggle (T4), same as `siyam/`'s workflow, and compare the
`attribution_separation_score` / routing accuracy / runtime against the v8 numbers.
