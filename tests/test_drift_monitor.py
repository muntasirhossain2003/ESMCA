import torch

from esmca.attribution.drift_monitor import ExplanationDriftMonitor


def test_zero_drift_when_attribution_matches_prototype():
    edm = ExplanationDriftMonitor(delta=0.05)
    attr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = edm.drift(attr, attr.clone())
    assert d < 1e-6


def test_large_drift_flags_consolidation():
    edm = ExplanationDriftMonitor(delta=0.05)
    prototype = torch.tensor([1.0, 0.0, 0.0, 0.0])
    drifted = torch.tensor([0.0, 0.0, 0.0, 1.0])
    scores = {"task_a": edm.drift(drifted, prototype)}
    flagged = edm.tasks_needing_consolidation(scores)
    assert "task_a" in flagged


def test_ads_is_mean_of_drift_scores():
    scores = {"task_a": 0.1, "task_b": 0.3}
    assert abs(ExplanationDriftMonitor.ads(scores) - 0.2) < 1e-6


def test_ads_empty_history_is_zero():
    assert ExplanationDriftMonitor.ads({}) == 0.0
