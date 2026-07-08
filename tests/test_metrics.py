from esmca.evaluation.metrics import compute_acc, compute_bwt, compute_fwt
from esmca.evaluation.xai_metrics import attribution_drift_score, routing_interpretability_score


def test_compute_acc_uses_last_row():
    R = [[0.9, 0.4, 0.3], [0.7, 0.8, 0.35], [0.6, 0.7, 0.85]]
    assert abs(compute_acc(R) - sum(R[-1]) / len(R[-1])) < 1e-9


def test_compute_bwt_zero_when_no_forgetting():
    # R[t][t] == R[T-1][t] for all t -> perfect retention -> BWT == 0
    R = [[0.8, 0.2, 0.1], [0.75, 0.9, 0.2], [0.8, 0.9, 0.85]]
    assert abs(compute_bwt(R)) < 1e-9


def test_compute_bwt_negative_when_forgetting():
    R = [[0.9, 0.3, 0.2], [0.85, 0.8, 0.3], [0.5, 0.6, 0.85]]
    assert compute_bwt(R) < 0


def test_compute_fwt_uses_pre_training_diagonal():
    R = [[0.5, 0.3, 0.4], [0.35, 0.8, 0.35], [0.4, 0.35, 0.9]]
    expected = (R[0][1] + R[1][2]) / 2
    assert abs(compute_fwt(R) - expected) < 1e-9


def test_ads_matches_manual_average():
    drift_history = {"task_b": {"task_a": 0.2}, "task_c": {"task_a": 0.4, "task_b": 0.2}}
    assert abs(attribution_drift_score(drift_history) - (0.2 + 0.4 + 0.2) / 3) < 1e-9


def test_ris_perfect_alignment_gives_correlation_one():
    true_idx = [0, 1, 2, 3]
    routed_idx = [0, 1, 2, 3]
    assert abs(routing_interpretability_score(true_idx, routed_idx) - 1.0) < 1e-9
