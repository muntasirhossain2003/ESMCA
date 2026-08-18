"""Novel XAI-specific metrics (paper Sec 6.2, "Contributions"):
Attribution Drift Score (ADS), Routing Interpretability Score (RIS),
and Attribution Fidelity.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def attribution_drift_score(drift_history: Dict[str, Dict[str, float]]) -> float:
    """ADS = mean of every D_t ever recorded across all training checkpoints
    (Sec 6.2: "lower values indicate less forgetting in attribution space")."""
    all_scores = [d for per_task in drift_history.values() for d in per_task.values()]
    return float(np.mean(all_scores)) if all_scores else 0.0


def routing_interpretability_score(true_task_indices: List[int], routed_task_indices: List[int]) -> float:
    """RIS: Spearman correlation between the router's chosen task (argmax
    routing weight) and the ground-truth task label, computed without ever
    exposing that label to the router itself."""
    if len(true_task_indices) < 2:
        return 0.0
    corr, _ = spearmanr(true_task_indices, routed_task_indices)
    return float(corr) if corr == corr else 0.0  # guard against NaN when constant input


def attribution_fidelity(prototypes: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """t-SNE cluster separability of task attribution prototypes + silhouette score."""
    task_names = list(prototypes.keys())
    if len(task_names) < 2:
        return {"embedding": None, "silhouette": 0.0, "task_names": task_names}
    matrix = torch.stack([prototypes[t] for t in task_names]).cpu().numpy()
    perplexity = max(1, min(30, len(task_names) - 1))
    embedding = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(matrix)
    labels = np.arange(len(task_names))
    # silhouette_score requires n_labels < n_samples; with one prototype per
    # task they're equal, so it's undefined — fall back to 0.0.
    if len(task_names) > 2 and len(task_names) < len(matrix):
        silhouette = silhouette_score(matrix, labels)
    else:
        silhouette = 0.0
    return {"embedding": embedding, "silhouette": float(silhouette), "task_names": task_names}
