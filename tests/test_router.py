import torch

from esmca.attribution.router import AttributionBasedRouter


def test_routing_weights_sum_to_one_and_are_label_free():
    router = AttributionBasedRouter(tau=0.1)
    phi_x = torch.tensor([1.0, 0.0, 0.0, 0.0])
    prototypes = {
        "task_a": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "task_b": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "task_c": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    weights = router.route(phi_x, prototypes)
    assert set(weights.keys()) == set(prototypes.keys())
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    # phi_x is most similar to task_a's prototype -> should get the highest weight.
    assert weights["task_a"] > weights["task_b"]
    assert weights["task_a"] > weights["task_c"]


def test_router_handles_batched_phi_by_averaging():
    router = AttributionBasedRouter(tau=0.1)
    phi_x_batch = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])])
    prototypes = {"task_a": torch.tensor([1.0, 0.0]), "task_b": torch.tensor([0.0, 1.0])}
    weights = router.route(phi_x_batch, prototypes)
    assert weights["task_a"] > weights["task_b"]
