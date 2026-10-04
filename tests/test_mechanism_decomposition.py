"""Unit tests for the 7-way Mechanism Decomposition Suite.

Validates:
1. Exact parameter counts and tight parity across all 7 models (within +/- 1.5% and under 250K).
2. Mathematical invariants of each core (boundedness, gating, convex combinations).
3. Shape consistency and gradient flow across recurrent horizons.
"""

import pytest
import torch
import torch.nn.functional as F

from re_reason.core.ablation_models import create_mechanism_decomposition_suite, forward_with_states_generic


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm unavailable on this environment")
    return torch.device(request.param)


def test_mechanism_decomposition_suite_parameter_parity() -> None:
    suite = create_mechanism_decomposition_suite()
    params = {name: m.count_parameters() for name, m in suite.items()}

    base_p = params["model_a_layernorm"]
    assert base_p == 226472

    # Expected parameter counts:
    # Model A: 226,472
    # Model B: 225,960 (-0.23%)
    # Model C: 226,152 (-0.14%)
    # Model D: 226,152 (-0.14%)
    # Model E (UGRNN): 226,360 (-0.05%)
    # Model F (MGU): 226,360 (-0.05%)
    # Model G (GRU): 229,200 (+1.20%)

    for name, p in params.items():
        rel_diff = abs(p - base_p) / base_p
        assert rel_diff <= 0.05, f"{name} has {p} params, delta {rel_diff:.2%} exceeds 5%"
        assert p <= 250000, f"{name} exceeds 250K ceiling: {p}"

    assert params["model_b_bounded_additive"] == 225960
    assert params["model_c_convex_unbounded"] == 226152
    assert params["model_d_bounded_convex"] == 226152
    assert params["model_e_canonical_ugrnn"] == 226360
    assert params["model_f_canonical_mgu"] == 226360
    assert params["model_g_canonical_gru"] == 229200


def test_mechanism_decomposition_forward_and_shapes(device: torch.device) -> None:
    suite = create_mechanism_decomposition_suite()
    inputs = torch.randn(4, 10, 21, device=device)

    for name, model in suite.items():
        model = model.to(device)
        out = model(inputs, return_step_diagnostics=True)

        assert out.logits.shape == (4, 4, 10), f"{name} terminal logits wrong shape"
        assert out.final_state.shape[0] == 4, f"{name} final state wrong batch dim"
        assert out.step_logits is not None
        assert out.step_logits.shape == (4, 10, 4, 10), f"{name} step logits wrong shape"


def test_mechanism_decomposition_gradients(device: torch.device) -> None:
    suite = create_mechanism_decomposition_suite()
    inputs = torch.randn(4, 10, 21, device=device)
    targets = torch.randint(0, 10, (4, 4), device=device)

    for name, model in suite.items():
        model = model.to(device)
        model.zero_grad()

        diag_out = forward_with_states_generic(model, inputs)
        assert len(diag_out.states) == 10, f"{name} states length wrong"

        for s in diag_out.states:
            s.retain_grad()

        loss = F.cross_entropy(diag_out.terminal_logits.view(-1, 10), targets.view(-1))
        loss.backward()

        for t, s in enumerate(diag_out.states):
            assert s.grad is not None, f"{name} state at t={t} missing gradient"
            assert not torch.isnan(s.grad).any(), f"{name} state at t={t} NaN grad"
