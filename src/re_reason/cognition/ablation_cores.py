"""Phase 1.5 Ablation Cognition Cores.

Implements architectural variants to isolate the causal drivers of performance:
1. ResidualNoLNCore: S_t = S_(t-1) + f_theta(S_(t-1), e_t)
   (Additive residual without LayerNorm)
2. GatedResidualCore: S_t = S_(t-1) + g_theta(S_(t-1), e_t) ⊙ f_theta(S_(t-1), e_t)
   (Coordinate-wise gated additive residual)
"""

import torch
import torch.nn as nn

from re_reason.cognition.base import BaseCognitionCore


class ResidualNoLNCore(BaseCognitionCore):
    """Residual update core without LayerNorm.

    Formulation:
        S_t = S_(t-1) + f_theta(S_(t-1), e_t)

    Tests whether LayerNorm's global state normalization causes representation
    interference or gradient attenuation across sequence steps.
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Executes un-normalized additive residual state update."""
        combined = torch.cat([state, observation], dim=-1)
        delta = self.net(combined)
        return state + delta


class GatedResidualCore(BaseCognitionCore):
    """Gated residual update core with coordinate-wise gating.

    Formulation:
        delta_t = f_theta(S_(t-1), e_t)
        gate_t = sigma(g_theta(S_(t-1), e_t))
        S_t = S_(t-1) + gate_t ⊙ delta_t

    Tests whether coordinate-wise gating (enabling dimensions to remain untouched
    when unrelated operations occur) resolves representation interference and
    stabilizes long-horizon credit assignment.
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 192,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Shared trunk to preserve tight parameter parity
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
        )
        # Combined projection for candidate delta and coordinate-wise gate
        self.proj = nn.Linear(hidden_dim, 2 * state_dim)

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.proj.bias[state_dim:].fill_(gate_bias_init)

    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Executes gated additive residual state update."""
        combined = torch.cat([state, observation], dim=-1)
        h = self.trunk(combined)
        candidate, gate_logits = torch.chunk(self.proj(h), chunks=2, dim=-1)
        gate = torch.sigmoid(gate_logits)
        return state + gate * candidate


class BoundedConvexCore(BaseCognitionCore):
    """Bounded Convex Residual Update Core (Model E).

    Formulation:
        z_t = sigmoid(g_theta(S_(t-1), e_t))
        candidate_t = tanh(f_theta(S_(t-1), e_t))
        S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ candidate_t

    Properties:
        - candidate_t in (-1, 1)^d (strictly bounded by tanh)
        - z_t in (0, 1)^d (coordinate-wise interpolation factor)
        - Strict element-wise convex interpolation:
          S_t,i = (1 - z_t,i) * S_(t-1),i + z_t,i * candidate_t,i
        - No GRU reset gate
        - R=1, K=1 fixed
        - Preserves state boundedness: if S_0 in [-1, 1], then S_t in [-1, 1] for all t
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 192,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Shared trunk for parameter parity with Model A and Model C
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
        )
        # Combined projection for candidate logits and gate logits
        self.proj = nn.Linear(hidden_dim, 2 * state_dim)

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.proj.bias[state_dim:].fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        """Executes bounded convex residual state update.

        Args:
            state: Previous state S_(t-1) in R^(B x state_dim).
            observation: Encoded observation e_t in R^(B x obs_dim).
            return_components: If True, returns (new_state, candidate, gate).
        """
        combined = torch.cat([state, observation], dim=-1)
        h = self.trunk(combined)
        candidate_logits, gate_logits = torch.chunk(self.proj(h), chunks=2, dim=-1)
        candidate = torch.tanh(candidate_logits)
        gate = torch.sigmoid(gate_logits)
        new_state = (1.0 - gate) * state + gate * candidate

        if return_components:
            return new_state, candidate, gate
        return new_state


class BoundedAdditiveCore(BaseCognitionCore):
    """Bounded Additive Update Core (Variant B).

    Formulation:
        delta_t = tanh(f_theta(S_(t-1), e_t))
        S_t = S_(t-1) + delta_t

    Isolates whether bounding the candidate update with tanh alone is sufficient to
    prevent representation drift in an additive residual, without convex decay (1 - z).
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        combined = torch.cat([state, observation], dim=-1)
        candidate = torch.tanh(self.net(combined))
        new_state = state + candidate
        if return_components:
            return new_state, candidate, None
        return new_state


class ConvexUnboundedCore(BaseCognitionCore):
    """Convex Unbounded Update Core (Variant C).

    Formulation:
        z_t = sigmoid(g_theta(S_(t-1), e_t))
        candidate_t = f_theta(S_(t-1), e_t)   (unbounded)
        S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ candidate_t

    Isolates whether coordinate-wise convex decay (1 - z) alone is sufficient when
    the candidate update is unconstrained (not bounded by tanh).
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 192,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, 2 * state_dim)

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.proj.bias[state_dim:].fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        combined = torch.cat([state, observation], dim=-1)
        h = self.trunk(combined)
        candidate, gate_logits = torch.chunk(self.proj(h), chunks=2, dim=-1)
        gate = torch.sigmoid(gate_logits)
        new_state = (1.0 - gate) * state + gate * candidate
        if return_components:
            return new_state, candidate, gate
        return new_state


class CanonicalUGRNNCore(BaseCognitionCore):
    """Canonical Update Gate RNN Core (Collins et al., 2017).

    Formulation:
        z_t = sigmoid(W_z e_t + U_z S_(t-1) + b_z)
        candidate_t = tanh(W_c e_t + U_c S_(t-1) + b_c)
        S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ candidate_t

    Canonical single-gate recurrent architecture with separate affine maps (no MLP trunk).
    """

    def __init__(
        self,
        state_dim: int = 212,
        obs_dim: int = 256,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.w_ih = nn.Linear(obs_dim, 2 * state_dim)
        self.w_hh = nn.Linear(state_dim, 2 * state_dim)

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.w_ih.bias[:state_dim].fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        gates = self.w_ih(observation) + self.w_hh(state)
        gate_logits, candidate_logits = torch.chunk(gates, chunks=2, dim=-1)
        gate = torch.sigmoid(gate_logits)
        candidate = torch.tanh(candidate_logits)
        new_state = (1.0 - gate) * state + gate * candidate
        if return_components:
            return new_state, candidate, gate
        return new_state


class CanonicalMGUCore(BaseCognitionCore):
    """Canonical Minimal Gated Unit Core (Zhou et al., 2016).

    Formulation:
        f_t = sigmoid(W_f e_t + U_f S_(t-1) + b_f)
        candidate_t = tanh(W_c e_t + U_c (f_t ⊙ S_(t-1)) + b_c)
        S_t = (1 - f_t) ⊙ S_(t-1) + f_t ⊙ candidate_t

    Canonical single-gate unit where the forget gate f_t is also used to pre-reset
    the previous state entering the candidate generator.
    """

    def __init__(
        self,
        state_dim: int = 212,
        obs_dim: int = 256,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.w_f_ih = nn.Linear(obs_dim, state_dim)
        self.w_f_hh = nn.Linear(state_dim, state_dim)
        self.w_c_ih = nn.Linear(obs_dim, state_dim)
        self.w_c_hh = nn.Linear(state_dim, state_dim)

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.w_f_ih.bias.fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        f_t = torch.sigmoid(self.w_f_ih(observation) + self.w_f_hh(state))
        h_reset = f_t * state
        candidate = torch.tanh(self.w_c_ih(observation) + self.w_c_hh(h_reset))
        new_state = (1.0 - f_t) * state + f_t * candidate
        if return_components:
            return new_state, candidate, f_t
        return new_state


class IndependentMLPUGRNNCore(BaseCognitionCore):
    """UGRNN with Independent Nonlinear MLPs Core.

    Formulation:
        z_t = sigmoid(MLP_gate([S_(t-1), e_t]))
        candidate_t = tanh(MLP_cand([S_(t-1), e_t]))
        S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ candidate_t

    Isolates trunk sharing: evaluates whether nonlinear feature mixing alone is sufficient,
    or whether sharing representations between gate and candidate is critical.
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 128,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.mlp_cand = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.mlp_gate = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.mlp_gate[-1].bias.fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        combined = torch.cat([state, observation], dim=-1)
        candidate = torch.tanh(self.mlp_cand(combined))
        gate = torch.sigmoid(self.mlp_gate(combined))
        new_state = (1.0 - gate) * state + gate * candidate
        if return_components:
            return new_state, candidate, gate
        return new_state


class SharedDeepMLPTrunkCore(BaseCognitionCore):
    """Shared Deep Nonlinear MLP Trunk Core.

    Formulation:
        h_t = GELU(MLP([S_(t-1), e_t]))  (2-layer MLP trunk)
        candidate_logits = W_c h_t + b_c
        gate_logits = W_z h_t + b_z
        candidate_t = tanh(candidate_logits)
        z_t = sigmoid(gate_logits)
        S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ candidate_t

    Isolates trunk depth: tests whether deepening the shared trunk (2-layer MLP vs 1-layer in Model D)
    improves feature synthesis or causes optimization degradation.
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 164,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, 2 * state_dim)

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.proj.bias[state_dim:].fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        combined = torch.cat([state, observation], dim=-1)
        h = self.trunk(combined)
        candidate_logits, gate_logits = torch.chunk(self.proj(h), chunks=2, dim=-1)
        candidate = torch.tanh(candidate_logits)
        gate = torch.sigmoid(gate_logits)
        new_state = (1.0 - gate) * state + gate * candidate
        if return_components:
            return new_state, candidate, gate
        return new_state


class SharedLinearTrunkCore(BaseCognitionCore):
    """Shared Trunk Without GELU (Linear Bottleneck) Core.

    Formulation:
        h_t = W_trunk [S_(t-1), e_t] + b_trunk   (strictly linear / affine, NO GELU)
        candidate_logits, gate_logits = chunk(W_proj h_t + b_proj)
        candidate_t = tanh(candidate_logits)
        z_t = sigmoid(gate_logits)
        S_t = (1 - z_t) ⊙ S_(t-1) + z_t ⊙ candidate_t

    Direct causal ablation of nonlinearity: perfectly parameter-matched to Model D (192-dim bottleneck),
    isolating whether the GELU activation is causal for performance gains.
    """

    def __init__(
        self,
        state_dim: int = 256,
        obs_dim: int = 256,
        hidden_dim: int = 192,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Pure linear projection without activation
        self.trunk = nn.Linear(state_dim + obs_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, 2 * state_dim)

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.proj.bias[state_dim:].fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        combined = torch.cat([state, observation], dim=-1)
        h = self.trunk(combined)  # No non-linear activation
        candidate_logits, gate_logits = torch.chunk(self.proj(h), chunks=2, dim=-1)
        candidate = torch.tanh(candidate_logits)
        gate = torch.sigmoid(gate_logits)
        new_state = (1.0 - gate) * state + gate * candidate
        if return_components:
            return new_state, candidate, gate
        return new_state


class NonlinearCandidateMGUCore(BaseCognitionCore):
    """MGU with Nonlinear Candidate Processing Core.

    Formulation:
        f_t = sigmoid(W_f e_t + U_f S_(t-1) + b_f)
        h_reset = f_t ⊙ S_(t-1)
        candidate_t = tanh(MLP_cand([h_reset, e_t]))
        S_t = (1 - f_t) ⊙ S_(t-1) + f_t ⊙ candidate_t

    Isolates whether adding nonlinear feature mixing to MGU's candidate state generator
    bridges the remaining performance gap between MGU and Model D.
    """

    def __init__(
        self,
        state_dim: int = 212,
        obs_dim: int = 256,
        hidden_dim: int = 146,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.w_f_ih = nn.Linear(obs_dim, state_dim)
        self.w_f_hh = nn.Linear(state_dim, state_dim)

        self.cand_mlp = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.w_f_ih.bias.fill_(gate_bias_init)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        return_components: bool = False,
    ):
        f_t = torch.sigmoid(self.w_f_ih(observation) + self.w_f_hh(state))
        h_reset = f_t * state
        combined = torch.cat([h_reset, observation], dim=-1)
        candidate = torch.tanh(self.cand_mlp(combined))
        new_state = (1.0 - f_t) * state + f_t * candidate
        if return_components:
            return new_state, candidate, f_t
        return new_state

