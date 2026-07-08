import torch
import torch.nn as nn

from esmca.models.lora import AdapterBank, LoRAInjectedLinear


def test_lora_injected_linear_is_noop_when_freshly_added():
    base = nn.Linear(8, 8)
    injected = LoRAInjectedLinear(base, rank=4, alpha=8)
    injected.add_task("task_a")
    injected.set_active_task("task_a")

    x = torch.randn(2, 8)
    out_with_adapter = injected(x)
    out_base_only = base(x)
    # B is zero-initialized, so the adapter starts as a mathematical no-op.
    assert torch.allclose(out_with_adapter, out_base_only, atol=1e-6)


def test_freezing_previous_task_adapter_stops_gradients():
    base = nn.Linear(8, 8)
    injected = LoRAInjectedLinear(base, rank=4, alpha=8)
    injected.add_task("task_a")
    injected.freeze_task("task_a")
    for p in injected.adapters["task_a"].parameters():
        assert p.requires_grad is False


def test_adapter_bank_routing_weights_blend_multiple_tasks():
    base = nn.Linear(4, 4)
    injected = LoRAInjectedLinear(base, rank=2, alpha=4)
    bank = AdapterBank([("layer0.query", injected)])
    bank.add_task("t1")
    bank.add_task("t2")

    with torch.no_grad():
        # give both tasks the same A so only B's sign differs -> contributions cancel
        injected.adapters["t2"].A.copy_(injected.adapters["t1"].A)
        injected.adapters["t1"].B.fill_(0.5)
        injected.adapters["t2"].B.fill_(-0.5)

    bank.set_routing_weights({"t1": 0.5, "t2": 0.5})
    x = torch.randn(3, 4)
    out = injected(x)
    # symmetric contributions should cancel out, leaving just the base linear.
    assert torch.allclose(out, base(x), atol=1e-5)
