"""Pathfinding-specialized neural hypothesis branching for Milestone 3A.

Implements the frozen Milestone 3A architecture for the 12-node constrained
pathfinding task:

1. :class:`ObservationEncoder` — ``220 -> 128 -> 256`` + LayerNorm (61,824
   trainable parameters).
2. :class:`ActionConditionedBrancher` — one shared ``12 x 32`` node embedding
   table (384 parameters, shared by current-node, goal-node, and candidate-action
   lookups), the observation encoder (61,824), a linear policy head
   ``512 -> 12`` (6,156), and one shared action-conditioned residual branch core
   ``544 -> 128 -> 256`` + LayerNorm (103,296). Total: exactly **171,660**
   trainable parameters.

The observation representation is the public pathfinding observation surface
only: current-node embedding (32), goal-node embedding (32), current-node
allowed-move mask (12), and the flattened effective adjacency matrix (144), for
a total of 220 features. The brancher never reads raw topology edges or blockade
identities from the ground-truth episode; it uses only
``PathfindingObservation.current``, ``goal``, ``allowed_moves``, and
``effective_adjacency``.

Candidate generation is deterministic: policy logits are restricted to the
current node's legal moves (minus any caller-supplied attempted-action mask), and
the top ``K`` actions are selected by descending logit value with ties broken by
the smaller node id. Candidate ids are always distinct. No stochastic sampling,
learned diversity objective, beam-search frontier, or heuristic ranking is used.

Candidate successor tensors follow the approved residual formulation::

    s^(k) = LayerNorm(S_t + f([S_t || e_t || E_node[a^(k)]]))

where the shared ``f`` is ``Linear(544 -> 128) -> GELU -> Linear(128 -> 256)`` and
the LayerNorm is part of the shared branch core.

Nominal forward-pass accounting convention for three candidate evaluations:
122,880 encoder FLOPs + 12,312 policy FLOPs + 3 x 205,824 branch-core FLOPs =
**752,664 FLOPs** per decision. This is an accounting convention, not a measured
runtime or a claim of computational advantage.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from re_reason.benchmarks.pathfinding_benchmark import PathfindingObservation


class ObservationEncoder(nn.Module):
    """Two-layer pathfinding observation encoder with GELU and LayerNorm.

    Frozen dimensions: ``Linear(220 -> 128)`` + GELU + ``Linear(128 -> 256)`` +
    ``LayerNorm(256)``, totaling exactly 61,824 trainable parameters.
    """

    def __init__(
        self,
        input_dim: int = 220,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim), ("output_dim", output_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def count_parameters(self) -> int:
        """Returns the total trainable parameter count (expected: 61,824)."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, observation_features: torch.Tensor) -> torch.Tensor:
        """Encodes observation features of shape ``(..., input_dim)`` to ``(..., output_dim)``.

        Single observations ``(220,)`` and batched observations ``(B, 220)`` are
        both supported; the leading dimensions are preserved.
        """
        if not isinstance(observation_features, torch.Tensor):
            raise TypeError(
                f"observation_features must be a torch.Tensor, got {type(observation_features).__name__}"
            )
        if observation_features.ndim == 0:
            raise ValueError("observation_features must have at least one dimension")
        if observation_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"observation_features last dimension {observation_features.shape[-1]} does not "
                f"match the encoder input dimension {self.input_dim}"
            )
        return self.net(observation_features)


@dataclass(frozen=True)
class CandidateProposal:
    """One typed candidate: action identity, ranking score, and successor state.

    Attributes:
        action_id: Destination-node id proposed for this candidate.
        score: Policy logit ordering key (finite; not a calibrated probability).
        state: Candidate successor latent tensor of shape ``(B, state_dim)``.
    """

    action_id: int
    score: float
    state: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.action_id, bool) or not isinstance(self.action_id, int) or self.action_id < 0:
            raise ValueError(f"action_id must be a non-negative integer, got {self.action_id!r}")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be a finite real number, got {self.score!r}")
        if not math.isfinite(float(self.score)):
            raise ValueError(f"score must be finite, got {self.score!r}")
        if not isinstance(self.state, torch.Tensor) or self.state.ndim != 2:
            raise ValueError("candidate state must be a 2-D tensor of shape (B, state_dim)")


@dataclass(frozen=True)
class BranchProposal:
    """Typed deterministic candidate proposal for one decision point.

    Candidates are ordered by descending policy priority. The collection may be
    empty when no eligible action remains. Every candidate carries its own action
    identity, score, and successor state, so the association between them is
    explicit rather than positional.
    """

    candidates: Tuple[CandidateProposal, ...]

    def __post_init__(self) -> None:
        action_ids = [candidate.action_id for candidate in self.candidates]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("candidate action ids must be distinct")
        if self.candidates:
            reference = self.candidates[0].state
            for candidate in self.candidates[1:]:
                if candidate.state.shape != reference.shape:
                    raise ValueError("candidate state shapes must match")
                if candidate.state.dtype != reference.dtype:
                    raise ValueError("candidate state dtypes must match")
                if candidate.state.device != reference.device:
                    raise ValueError("candidate state devices must match")

    @property
    def candidate_action_ids(self) -> Tuple[int, ...]:
        """Action ids in descending policy priority order."""
        return tuple(candidate.action_id for candidate in self.candidates)

    @property
    def candidate_scores(self) -> Tuple[float, ...]:
        """Ranking scores in descending policy priority order."""
        return tuple(float(candidate.score) for candidate in self.candidates)

    def stacked_states(self) -> torch.Tensor:
        """Stacks candidate states into a ``(B, K, state_dim)`` tensor."""
        if not self.candidates:
            raise ValueError("cannot stack states for an empty proposal")
        return torch.stack([candidate.state for candidate in self.candidates], dim=1)


class ActionConditionedBrancher(nn.Module):
    """Deterministic action-conditioned candidate generator for pathfinding.

    The module owns exactly one node embedding table, one observation encoder,
    one policy head, and one branch core; all candidate actions and all candidate
    states are evaluated through those shared modules, so no per-candidate copies
    of parameters exist.
    """

    STATE_DIM = 256
    NUM_NODES = 12
    NODE_EMBEDDING_DIM = 32
    OBSERVATION_INPUT_DIM = 220
    ENCODER_HIDDEN_DIM = 128
    BRANCH_HIDDEN_DIM = 128
    DEFAULT_CANDIDATE_BUDGET = 3

    def __init__(
        self,
        num_nodes: int = NUM_NODES,
        node_embedding_dim: int = NODE_EMBEDDING_DIM,
        state_dim: int = STATE_DIM,
        observation_input_dim: int = OBSERVATION_INPUT_DIM,
        encoder_hidden_dim: int = ENCODER_HIDDEN_DIM,
        branch_hidden_dim: int = BRANCH_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embedding_dim = node_embedding_dim
        self.state_dim = state_dim
        self.observation_input_dim = observation_input_dim

        expected_observation_input = 2 * node_embedding_dim + num_nodes + num_nodes * num_nodes
        if observation_input_dim != expected_observation_input:
            raise ValueError(
                f"observation_input_dim {observation_input_dim} is inconsistent with the "
                f"current/goal embedding and adjacency layout (expected {expected_observation_input})"
            )

        self.node_embeddings = nn.Embedding(num_nodes, node_embedding_dim)
        self.observation_encoder = ObservationEncoder(
            input_dim=observation_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=state_dim,
        )
        self.policy_head = nn.Linear(2 * state_dim, num_nodes)
        self.branch_input = nn.Linear(
            2 * state_dim + node_embedding_dim, branch_hidden_dim
        )
        self.branch_output = nn.Linear(branch_hidden_dim, state_dim)
        self.branch_norm = nn.LayerNorm(state_dim)

    def count_parameters(self) -> int:
        """Returns the total trainable parameter count (expected: 171,660)."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def observation_features(self, observation: PathfindingObservation) -> torch.Tensor:
        """Builds the 220-dimensional public observation representation.

        Layout: ``[E(current) (32) || E(goal) (32) || allowed-move mask (12) ||
        effective adjacency (144)]``. Only public observation fields are used.
        """
        if not isinstance(observation, PathfindingObservation):
            raise TypeError(
                f"observation must be a PathfindingObservation, got {type(observation).__name__}"
            )
        if len(observation.effective_adjacency) != self.num_nodes:
            raise ValueError(
                f"observation has {len(observation.effective_adjacency)} nodes; the model expects "
                f"{self.num_nodes}"
            )
        for node in observation.allowed_moves:
            if isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < self.num_nodes:
                raise ValueError(f"observation contains an out-of-range move id: {node!r}")
        for node, neighbors in enumerate(observation.effective_adjacency):
            for neighbor in neighbors:
                if (
                    isinstance(neighbor, bool)
                    or not isinstance(neighbor, int)
                    or not 0 <= neighbor < self.num_nodes
                ):
                    raise ValueError(
                        f"observation adjacency row {node} contains an out-of-range node id: "
                        f"{neighbor!r}"
                    )

        reference = self.node_embeddings.weight
        device, dtype = reference.device, reference.dtype
        current_embedding = self.node_embeddings(
            torch.tensor([observation.current], dtype=torch.long, device=device)
        ).squeeze(0)
        goal_embedding = self.node_embeddings(
            torch.tensor([observation.goal], dtype=torch.long, device=device)
        ).squeeze(0)

        move_mask = torch.zeros(self.num_nodes, dtype=dtype, device=device)
        for node in observation.allowed_moves:
            move_mask[node] = 1.0

        adjacency = torch.zeros((self.num_nodes, self.num_nodes), dtype=dtype, device=device)
        for node, neighbors in enumerate(observation.effective_adjacency):
            for neighbor in neighbors:
                adjacency[node, neighbor] = 1.0

        return torch.cat(
            [current_embedding, goal_embedding, move_mask, adjacency.reshape(-1)], dim=-1
        )

    def encode_observation(self, observation: PathfindingObservation) -> torch.Tensor:
        """Encodes one typed observation into a 256-dimensional embedding ``(256,)``."""
        return self.observation_encoder(self.observation_features(observation))

    def policy_logits(
        self, committed_state: torch.Tensor, encoded_observation: torch.Tensor
    ) -> torch.Tensor:
        """Computes destination logits from ``[S_t || e_t]`` for each batch row."""
        committed_state, encoded_observation = self._validate_state_pair(
            committed_state, encoded_observation
        )
        return self.policy_head(torch.cat([committed_state, encoded_observation], dim=-1))

    def evaluate_candidates(
        self,
        committed_state: torch.Tensor,
        encoded_observation: torch.Tensor,
        action_ids: Iterable[int],
    ) -> torch.Tensor:
        """Evaluates the shared branch core for every action id.

        Returns candidate successor states of shape ``(B, K, state_dim)``. The
        mathematical function per candidate is identical to evaluating each
        candidate through the same shared core; batching only changes scheduling.
        """
        committed_state, encoded_observation = self._validate_state_pair(
            committed_state, encoded_observation
        )
        action_ids = self._validate_action_ids(action_ids)
        batch_size = committed_state.shape[0]
        reference = self.node_embeddings.weight

        if not action_ids:
            return torch.empty(
                (batch_size, 0, self.state_dim), dtype=committed_state.dtype, device=committed_state.device
            )

        action_tensor = torch.tensor(action_ids, dtype=torch.long, device=reference.device)
        action_embeddings = self.node_embeddings(action_tensor)  # (K, node_embedding_dim)
        branch_features = torch.cat(
            [
                committed_state.unsqueeze(1).expand(-1, len(action_ids), -1),
                encoded_observation.unsqueeze(1).expand(-1, len(action_ids), -1),
                action_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
            ],
            dim=-1,
        )  # (B, K, 544)
        hidden = F.gelu(self.branch_input(branch_features))  # (B, K, 128)
        delta = self.branch_output(hidden)  # (B, K, 256)
        return self.branch_norm(committed_state.unsqueeze(1) + delta)  # (B, K, 256)

    def propose_candidates(
        self,
        committed_state: torch.Tensor,
        observation: PathfindingObservation,
        num_candidates: int = DEFAULT_CANDIDATE_BUDGET,
        masked_actions: Iterable[int] = (),
    ) -> BranchProposal:
        """Produces a deterministic top-K candidate proposal for one decision point.

        Eligibility is the observation's legal moves minus the attempted-action
        mask. Candidates are ordered by descending policy logit with ties broken
        by smaller node id. Fewer than ``num_candidates`` (possibly zero)
        candidates are returned when fewer eligible actions remain.

        Args:
            committed_state: Single committed latent state of shape ``(1, state_dim)``.
            observation: Public pathfinding observation at the decision point.
            num_candidates: Candidate budget ``K`` (>= 1).
            masked_actions: Actions attempted (and rejected) in earlier proposal
                rounds at the same decision point.
        """
        if not isinstance(committed_state, torch.Tensor):
            raise TypeError(f"committed_state must be a torch.Tensor, got {type(committed_state).__name__}")
        if (
            committed_state.ndim != 2
            or committed_state.shape[0] != 1
            or committed_state.shape[1] != self.state_dim
        ):
            raise ValueError(
                "propose_candidates requires a single committed state of shape "
                f"(1, {self.state_dim}), got {tuple(committed_state.shape)}"
            )
        reference = self.node_embeddings.weight
        if committed_state.dtype != reference.dtype or committed_state.device != reference.device:
            raise ValueError(
                f"committed_state dtype/device ({committed_state.dtype}, {committed_state.device}) "
                f"must match the model ({reference.dtype}, {reference.device})"
            )
        if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates < 1:
            raise ValueError(f"num_candidates must be an integer >= 1, got {num_candidates!r}")
        masked = set(self._validate_action_ids(masked_actions))

        features = self.observation_features(observation).unsqueeze(0)  # (1, 220)
        encoded_observation = self.observation_encoder(features)  # (1, 256)
        logits = self.policy_logits(committed_state, encoded_observation)  # (1, num_nodes)

        eligible = [action for action in observation.allowed_moves if action not in masked]
        ordered = sorted(eligible, key=lambda action: (-float(logits[0, action].item()), action))
        chosen = tuple(ordered[:num_candidates])
        candidate_states = self.evaluate_candidates(committed_state, encoded_observation, chosen)
        candidates = tuple(
            CandidateProposal(
                action_id=action,
                score=float(logits[0, action].item()),
                state=candidate_states[:, index, :],
            )
            for index, action in enumerate(chosen)
        )
        return BranchProposal(candidates=candidates)

    def _validate_state_pair(
        self, committed_state: torch.Tensor, encoded_observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for name, tensor in (
            ("committed_state", committed_state),
            ("encoded_observation", encoded_observation),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if tensor.ndim != 2 or tensor.shape[1] != self.state_dim:
                raise ValueError(
                    f"{name} must have shape (B, {self.state_dim}), got {tuple(tensor.shape)}"
                )
        if committed_state.shape[0] != encoded_observation.shape[0]:
            raise ValueError("committed_state and encoded_observation batch sizes must match")
        if committed_state.dtype != encoded_observation.dtype:
            raise ValueError("committed_state and encoded_observation dtypes must match")
        reference = self.node_embeddings.weight
        for name, tensor in (
            ("committed_state", committed_state),
            ("encoded_observation", encoded_observation),
        ):
            if tensor.dtype != reference.dtype or tensor.device != reference.device:
                raise ValueError(
                    f"{name} dtype/device ({tensor.dtype}, {tensor.device}) must match the model "
                    f"({reference.dtype}, {reference.device})"
                )
        return committed_state, encoded_observation

    def _validate_action_ids(self, action_ids: Iterable[int]) -> Tuple[int, ...]:
        validated = []
        for action in action_ids:
            if isinstance(action, bool) or not isinstance(action, int):
                raise ValueError(f"action ids must be integers, got {action!r}")
            if not 0 <= action < self.num_nodes:
                raise ValueError(f"action id {action} out of range for {self.num_nodes} nodes")
            validated.append(action)
        if len(set(validated)) != len(validated):
            raise ValueError("action ids must be distinct")
        return tuple(validated)
