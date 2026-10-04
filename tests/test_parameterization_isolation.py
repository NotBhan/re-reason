"""Tests for the 7-way parameterization-isolation ablation suite."""

import pytest
import torch
from re_reason.core.ablation_models import create_parameterization_isolation_suite


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm unavailable on this environment")
    return torch.device(request.param)


def test_parameterization_isolation_suite_parameter_parity():
    suite = create_parameterization_isolation_suite()
    assert len(suite) == 7

    target_params = 226_152  # Model D parameter count
    for name, model in suite.items():
        params = model.count_parameters()
        diff = abs(params - target_params)
        rel_diff = diff / target_params
        assert rel_diff <= 0.015, (
            f"Model {name} parameter count {params} deviates from target {target_params} "
            f"by {rel_diff:.2%} (must be <= 1.5%)"
        )


def test_parameterization_isolation_forward_and_shapes(device):
    suite = create_parameterization_isolation_suite()
    batch_size = 4
    seq_len = 10
    input_dim = 21

    inputs = torch.randn(batch_size, seq_len, input_dim, device=device)

    for name, model in suite.items():
        model = model.to(device)
        out = model(inputs)

        assert out.logits.shape == (batch_size, 4, 10), (
            f"Model {name} produced wrong logits shape: {out.logits.shape}"
        )
        assert out.final_state.shape[0] == batch_size, (
            f"Model {name} produced wrong final state shape: {out.final_state.shape}"
        )

        diag = model.forward_with_states(inputs, return_components=True)
        assert len(diag.states) == seq_len, (
            f"Model {name} states count {len(diag.states)} != {seq_len}"
        )
        assert diag.candidates is not None and len(diag.candidates) == seq_len
        assert diag.gates is not None and len(diag.gates) == seq_len


def test_parameterization_isolation_gradients(device):
    suite = create_parameterization_isolation_suite()
    batch_size = 2
    seq_len = 5
    input_dim = 21

    inputs = torch.randn(batch_size, seq_len, input_dim, device=device)
    target = torch.randint(0, 10, (batch_size, 4), device=device)
    criterion = torch.nn.CrossEntropyLoss()

    for name, model in suite.items():
        model = model.to(device)
        model.zero_grad()
        out = model(inputs)

        loss = sum(
            criterion(out.logits[:, r, :], target[:, r])
            for r in range(4)
        ) / 4.0

        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.norm().item() > 0:
                has_grad = True
                break
        assert has_grad, f"Model {name} failed to propagate gradients."
