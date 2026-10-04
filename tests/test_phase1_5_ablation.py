"""Unit tests for Phase 1.5 Causal Ablation Models and Cores."""

import pytest
import torch
import torch.nn.functional as F

from re_reason.cognition.ablation_cores import (
    BoundedConvexCore,
    GatedResidualCore,
    ResidualNoLNCore,
)
from re_reason.core.ablation_models import (
    ReReasonBoundedConvexModel,
    ReReasonGatedModel,
    ReReasonNoLNModel,
    create_ablation_model_suite,
    create_models_a_e_d_suite,
    forward_with_states_generic,
)
from re_reason.core.model import ReReasonPhase1Model


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm unavailable on this environment")
    return torch.device(request.param)


def test_residual_no_ln_core_forward(device: torch.device) -> None:
    core = ResidualNoLNCore(state_dim=256, obs_dim=256, hidden_dim=256).to(device)
    state = torch.randn(8, 256, device=device)
    obs = torch.randn(8, 256, device=device)

    out = core(state, obs)
    assert out.shape == (8, 256)
    # Check that out is state + delta (not identical to state)
    assert not torch.allclose(out, state)


def test_gated_residual_core_forward(device: torch.device) -> None:
    core = GatedResidualCore(state_dim=256, obs_dim=256, hidden_dim=192).to(device)
    state = torch.randn(8, 256, device=device)
    obs = torch.randn(8, 256, device=device)

    out = core(state, obs)
    assert out.shape == (8, 256)
    assert not torch.allclose(out, state)


def test_ablation_model_suite_parameter_parity() -> None:
    suite = create_ablation_model_suite()
    params = {name: m.count_parameters() for name, m in suite.items()}

    base_p = params["model_a_layernorm"]
    assert base_p == 226472

    for name, p in params.items():
        rel_diff = abs(p - base_p) / base_p
        # All models must be within +/- 5% parameter parity
        assert rel_diff <= 0.05, f"Model {name} has {p} params, delta {rel_diff:.2%} exceeds 5%"
        # Must be under the 250K Phase 1 ceiling
        assert p <= 250000, f"Model {name} exceeds 250K ceiling: {p}"


def test_ablation_models_forward_and_shapes(device: torch.device) -> None:
    suite = create_ablation_model_suite()
    inputs = torch.randn(4, 10, 21, device=device)

    for name, model in suite.items():
        model = model.to(device)
        out = model(inputs, return_step_diagnostics=True)

        assert out.logits.shape == (4, 4, 10), f"{name} terminal logits wrong shape"
        assert out.final_state.shape[0] == 4, f"{name} final state wrong batch dim"
        assert out.step_logits is not None
        assert out.step_logits.shape == (4, 10, 4, 10), f"{name} step logits wrong shape"


def test_forward_with_states_and_gradients(device: torch.device) -> None:
    suite = create_ablation_model_suite()
    inputs = torch.randn(4, 10, 21, device=device)
    targets = torch.randint(0, 10, (4, 4), device=device)

    for name, model in suite.items():
        model = model.to(device)
        model.zero_grad()

        diag_out = forward_with_states_generic(model, inputs)
        assert len(diag_out.states) == 10, f"{name} should have 10 intermediate states"

        # Check gradient retention on intermediate states
        for s in diag_out.states:
            s.retain_grad()

        loss = F.cross_entropy(diag_out.terminal_logits.view(-1, 10), targets.view(-1))
        loss.backward()

        for t, s in enumerate(diag_out.states):
            assert s.grad is not None, f"{name} state at t={t} missing gradient"
            assert not torch.isnan(s.grad).any(), f"{name} state at t={t} has NaN gradients"


def test_bounded_convex_core_properties(device: torch.device) -> None:
    """Verifies that BoundedConvexCore satisfies all mathematical guarantees:

    1. Candidate state is strictly bounded in (-1, 1).
    2. Gate z is strictly in (0, 1).
    3. Update is a strict coordinate-wise convex interpolation.
    4. State norm is bounded: if S_0 in [-1, 1], then S_t in [-1, 1] for all t.
    """
    core = BoundedConvexCore(state_dim=256, obs_dim=256, hidden_dim=192).to(device)

    # Test with random states in [-1, 1] and random observations
    state = (torch.rand(16, 256, device=device) * 2.0) - 1.0
    obs = torch.randn(16, 256, device=device)

    new_state, candidate, gate = core(state, obs, return_components=True)

    # 1. Candidate is bounded by tanh in (-1, 1)
    assert torch.all(candidate >= -1.0) and torch.all(candidate <= 1.0)
    assert torch.all(torch.abs(candidate) < 1.0)

    # 2. Gate is strictly in (0, 1)
    assert torch.all(gate > 0.0) and torch.all(gate < 1.0)

    # 3. Element-wise convex interpolation: S_t = (1 - z) * S_(t-1) + z * candidate
    expected_new_state = (1.0 - gate) * state + gate * candidate
    assert torch.allclose(new_state, expected_new_state, atol=1e-6)

    # S_t,i must lie between min(S_(t-1), candidate) and max(S_(t-1), candidate)
    elem_min = torch.minimum(state, candidate)
    elem_max = torch.maximum(state, candidate)
    assert torch.all(new_state >= elem_min - 1e-6)
    assert torch.all(new_state <= elem_max + 1e-6)

    # 4. If previous state is in [-1, 1], new state MUST remain in [-1, 1]
    assert torch.all(new_state >= -1.0 - 1e-6)
    assert torch.all(new_state <= 1.0 + 1e-6)

    # Maximum possible L2 norm is sqrt(256) = 16.0
    state_norms = torch.norm(new_state, p=2, dim=-1)
    assert torch.all(state_norms <= 16.0001)


def test_model_e_parameter_count_and_parity() -> None:
    """Verifies that Model E matches Model A within +/- 5% and under 250K."""
    model_e = ReReasonBoundedConvexModel(
        input_dim=21,
        state_dim=256,
        encoder_hidden_dim=64,
        core_hidden_dim=192,
        num_registers=4,
        num_values=10,
    )
    p_e = model_e.count_parameters()
    assert p_e == 226152

    suite = create_models_a_e_d_suite()
    p_a = suite["model_a_layernorm"].count_parameters()
    p_d = suite["model_d_gru"].count_parameters()

    assert p_a == 226472
    assert p_d == 229200

    delta_e_vs_a = abs(p_e - p_a) / p_a
    assert delta_e_vs_a <= 0.05, f"Delta {delta_e_vs_a:.4f} exceeds 5%"
    assert p_e <= 250000, f"Model E exceeds 250K ceiling: {p_e}"


def test_model_e_forward_with_components(device: torch.device) -> None:
    """Verifies Model E forward_with_states returns valid candidates and gates."""
    model_e = ReReasonBoundedConvexModel().to(device)
    inputs = torch.randn(4, 10, 21, device=device)

    diag_out = model_e.forward_with_states(inputs, return_components=True)

    assert diag_out.candidates is not None
    assert diag_out.gates is not None
    assert len(diag_out.states) == 10
    assert len(diag_out.candidates) == 10
    assert len(diag_out.gates) == 10

    for t in range(10):
        cand = diag_out.candidates[t]
        gate = diag_out.gates[t]
        st = diag_out.states[t]
        assert cand.shape == (4, 256)
        assert gate.shape == (4, 256)
        assert st.shape == (4, 256)
        assert torch.all(cand >= -1.0) and torch.all(cand <= 1.0)
        assert torch.all((gate >= 0.0) & (gate <= 1.0))
