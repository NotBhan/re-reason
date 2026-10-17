"""Milestone 3B comparator baselines over the shared pathfinding recurrent architecture.

All three approved comparators reuse exactly the same trainable modules as
RE:Reason — the shared node embedding table, observation encoder, policy head, and
residual MLP transition core — totaling exactly **171,660 trainable parameters**.
They differ only in inference-time action selection and state-commit behavior:

* :class:`UnGatedRecurrentBaseline` — single trajectory, policy argmax over all
  nodes, unconditional latent-state commitment, with no candidate branching,
  verification, or transactional rollback. This is a recurrent residual baseline,
  not a GRU.
* :class:`ActionMaskingRecurrentBaseline` — single trajectory, public move mask
  applied to the policy logits before argmax, unconditional latent commitment.
* :class:`TwoRolloutRecurrentBaseline` — two independent stochastic trajectories
  at fixed temperature ``tau = 1.0`` with static horizons ``T1 = 26`` and
  ``T2 = 27``; trajectory selection uses only public verification (goal arrival)
  and executed length.

The algorithmically redundant sequential "Beam-3" comparator is deliberately not
implemented. Inference starts from a zero-initialized latent state, matching the
existing RE:Reason model convention. No training or optimizer logic lives here.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from re_reason.benchmarks.pathfinding_benchmark import (
    PathfindingAction,
    PathfindingEpisode,
    PathfindingObservation,
    PathfindingState,
)
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher
from re_reason.utils.rng_namespaces import make_rollout_generator, rollout_seed
from re_reason.verification.pathfinding_verifier import PathfindingVerifier


SINGLE_TRAJECTORY_TRANSITIONS = 24
TWO_ROLLOUT_HORIZONS = (26, 27)
ROLLOUT_TEMPERATURE = 1.0


@dataclass(frozen=True)
class TrajectoryResult:
    """Outcome of one single-trajectory rollout.

    ``actions`` records every attempted action in order; ``transitions_used`` is
    the number of attempted decisions (equal to ``len(actions)``). For the
    un-gated baseline, ``invalid_attempts`` counts attempted illegal transitions:
    the environment does not move for those steps, but the latent state has
    already advanced unconditionally.
    """

    reached_goal: bool
    actions: Tuple[int, ...]
    final_environment_state: PathfindingState
    invalid_attempts: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple) or any(
            isinstance(action, bool) or not isinstance(action, int) for action in self.actions
        ):
            raise TypeError("actions must be a tuple of integer action ids")
        if not isinstance(self.final_environment_state, PathfindingState):
            raise TypeError("final_environment_state must be a PathfindingState")
        if (
            isinstance(self.invalid_attempts, bool)
            or not isinstance(self.invalid_attempts, int)
            or self.invalid_attempts < 0
        ):
            raise ValueError("invalid_attempts must be a non-negative integer")

    @property
    def transitions_used(self) -> int:
        """Number of attempted decision transitions."""
        return len(self.actions)


def select_trajectory(
    trajectory_1: TrajectoryResult, trajectory_2: TrajectoryResult
) -> Optional[int]:
    """Applies the protocol's public trajectory-selection rule.

    Prefer a goal-reaching trajectory; if both succeed, select the shorter one;
    break equal-length ties in favor of trajectory 1; if neither succeeds, return
    ``None``. No oracle or shortest-path information is used.
    """
    if not isinstance(trajectory_1, TrajectoryResult) or not isinstance(
        trajectory_2, TrajectoryResult
    ):
        raise TypeError("select_trajectory requires two TrajectoryResult instances")
    if trajectory_1.reached_goal and trajectory_2.reached_goal:
        return 1 if trajectory_1.transitions_used <= trajectory_2.transitions_used else 2
    if trajectory_1.reached_goal:
        return 1
    if trajectory_2.reached_goal:
        return 2
    return None


@dataclass(frozen=True)
class TwoTrajectoryResult:
    """Outcome of the episode-budget-matched two-rollout comparator."""

    trajectory_1: TrajectoryResult
    trajectory_2: TrajectoryResult
    selected_trajectory: Optional[int]

    def __post_init__(self) -> None:
        if self.selected_trajectory not in (None, 1, 2):
            raise ValueError(
                f"selected_trajectory must be None, 1, or 2, got {self.selected_trajectory!r}"
            )
        expected = select_trajectory(self.trajectory_1, self.trajectory_2)
        if expected != self.selected_trajectory:
            raise ValueError(
                f"selected_trajectory {self.selected_trajectory!r} is inconsistent with the "
                f"protocol selection rule (expected {expected!r})"
            )

    @property
    def reached_goal(self) -> bool:
        """True when at least one trajectory reached the goal."""
        return self.selected_trajectory is not None

    @property
    def selected_result(self) -> Optional[TrajectoryResult]:
        """The selected trajectory result, or ``None`` when both failed."""
        if self.selected_trajectory == 1:
            return self.trajectory_1
        if self.selected_trajectory == 2:
            return self.trajectory_2
        return None

    @property
    def selected_transitions(self) -> int:
        """Executed transitions of the selected trajectory (0 on failure)."""
        result = self.selected_result
        return 0 if result is None else result.transitions_used


class PathfindingRecurrentBaseline(nn.Module):
    """Shared 171,660-parameter recurrent architecture for the comparators."""

    def __init__(self, brancher: Optional[ActionConditionedBrancher] = None) -> None:
        super().__init__()
        if brancher is not None and not isinstance(brancher, ActionConditionedBrancher):
            raise TypeError(
                f"brancher must be an ActionConditionedBrancher, got {type(brancher).__name__}"
            )
        self.brancher = brancher if brancher is not None else ActionConditionedBrancher()
        self.state_dim = self.brancher.state_dim
        self.num_nodes = self.brancher.num_nodes

    def count_parameters(self) -> int:
        """Returns the total trainable parameter count (expected: 171,660)."""
        return self.brancher.count_parameters()

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Returns a zero-initialized latent state of shape ``(batch_size, state_dim)``."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}")
        reference = self.brancher.node_embeddings.weight
        return torch.zeros(
            (batch_size, self.state_dim), dtype=reference.dtype, device=reference.device
        )

    def encode_observation(self, observation: PathfindingObservation) -> torch.Tensor:
        """Encodes one public observation to a 256-dimensional embedding."""
        return self.brancher.encode_observation(observation)

    def policy_logits(
        self, committed_state: torch.Tensor, encoded_observation: torch.Tensor
    ) -> torch.Tensor:
        """Returns destination logits for ``[S_t || e_t]``."""
        return self.brancher.policy_logits(committed_state, encoded_observation)

    def advance_state(
        self,
        committed_state: torch.Tensor,
        encoded_observation: torch.Tensor,
        action_id: int,
    ) -> torch.Tensor:
        """Advances the shared residual core by one action; returns ``(B, state_dim)``."""
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError(f"action_id must be an integer, got {action_id!r}")
        if not 0 <= action_id < self.num_nodes:
            raise ValueError(f"action_id {action_id} out of range for {self.num_nodes} nodes")
        if not isinstance(committed_state, torch.Tensor) or committed_state.ndim != 2:
            raise ValueError("committed_state must be a 2-D tensor of shape (B, state_dim)")
        action_ids = torch.full(
            (committed_state.shape[0],),
            action_id,
            dtype=torch.long,
            device=committed_state.device,
        )
        return self.brancher.advance_states(committed_state, encoded_observation, action_ids)


class UnGatedRecurrentBaseline(PathfindingRecurrentBaseline):
    """Single-trajectory, un-gated recurrent residual baseline (policy argmax)."""

    def select_action(self, logits: torch.Tensor) -> int:
        """Selects the argmax destination over all nodes (no legality masking)."""
        if not isinstance(logits, torch.Tensor) or logits.numel() != self.num_nodes:
            raise ValueError(
                f"logits must be a tensor with {self.num_nodes} entries, got {logits!r}"
            )
        return int(torch.argmax(logits.reshape(-1)).item())


class ActionMaskingRecurrentBaseline(PathfindingRecurrentBaseline):
    """Single-trajectory baseline with the public move mask applied before argmax."""

    def select_action(
        self, logits: torch.Tensor, observation: PathfindingObservation
    ) -> Optional[int]:
        """Selects the highest-logit legal move, or ``None`` when none exists."""
        if not isinstance(logits, torch.Tensor) or logits.numel() != self.num_nodes:
            raise ValueError(
                f"logits must be a tensor with {self.num_nodes} entries, got {logits!r}"
            )
        if not isinstance(observation, PathfindingObservation):
            raise TypeError(
                f"observation must be a PathfindingObservation, got {type(observation).__name__}"
            )
        legal_moves = observation.allowed_moves
        if not legal_moves:
            return None
        flat_logits = logits.reshape(-1)
        legal_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=flat_logits.device)
        legal_mask[list(legal_moves)] = True
        masked_logits = flat_logits.masked_fill(~legal_mask, float("-inf"))
        return int(torch.argmax(masked_logits).item())


class TwoRolloutRecurrentBaseline(PathfindingRecurrentBaseline):
    """Episode-budget-matched two-rollout baseline with fixed-temperature sampling."""

    def sample_action(
        self,
        logits: torch.Tensor,
        observation: PathfindingObservation,
        generator: torch.Generator,
    ) -> Optional[int]:
        """Samples a legal action from ``Softmax(logits / tau)`` with ``tau = 1.0``.

        Sampling is performed on the CPU with the supplied isolated generator so it
        is reproducible and never perturbs the global training stream.
        """
        if not isinstance(logits, torch.Tensor) or logits.numel() != self.num_nodes:
            raise ValueError(
                f"logits must be a tensor with {self.num_nodes} entries, got {logits!r}"
            )
        if not isinstance(observation, PathfindingObservation):
            raise TypeError(
                f"observation must be a PathfindingObservation, got {type(observation).__name__}"
            )
        if not isinstance(generator, torch.Generator):
            raise TypeError(f"generator must be a torch.Generator, got {type(generator).__name__}")
        legal_moves = observation.allowed_moves
        if not legal_moves:
            return None
        flat_logits = logits.reshape(-1).detach()
        legal_logits = flat_logits[list(legal_moves)] / ROLLOUT_TEMPERATURE
        probabilities = torch.softmax(legal_logits.to("cpu"), dim=-1)
        sampled_index = int(
            torch.multinomial(probabilities, num_samples=1, generator=generator).item()
        )
        return int(legal_moves[sampled_index])


def _validate_single_trajectory_rollout(
    model: PathfindingRecurrentBaseline,
    episode: PathfindingEpisode,
    max_transitions: int,
    expected_type: type,
) -> None:
    if not isinstance(model, expected_type):
        raise TypeError(
            f"model must be a {expected_type.__name__}, got {type(model).__name__}"
        )
    if not isinstance(episode, PathfindingEpisode):
        raise TypeError(f"episode must be a PathfindingEpisode, got {type(episode).__name__}")
    if (
        isinstance(max_transitions, bool)
        or not isinstance(max_transitions, int)
        or max_transitions < 1
    ):
        raise ValueError(f"max_transitions must be an integer >= 1, got {max_transitions!r}")


def rollout_un_gated(
    model: UnGatedRecurrentBaseline,
    episode: PathfindingEpisode,
    max_transitions: int = SINGLE_TRAJECTORY_TRANSITIONS,
) -> TrajectoryResult:
    """Executes one un-gated single trajectory.

    The policy argmax is committed to the latent state unconditionally. If the
    selected action is illegal, the environment does not move but the latent state
    still advances on the hallucinated action; the attempt is counted in
    ``invalid_attempts``.
    """
    _validate_single_trajectory_rollout(
        model, episode, max_transitions, UnGatedRecurrentBaseline
    )
    with torch.no_grad():
        latent_state = model.initial_state(1)
        environment_state = episode.initial_state()
        actions = []
        invalid_attempts = 0
        for _ in range(max_transitions):
            if episode.is_goal_reached(environment_state):
                break
            observation = episode.observe(environment_state)
            encoded = model.encode_observation(observation).unsqueeze(0)
            logits = model.policy_logits(latent_state, encoded)
            action = model.select_action(logits)
            actions.append(action)
            latent_state = model.advance_state(latent_state, encoded, action)
            if episode.is_allowed_move(environment_state.current, action):
                environment_state = episode.apply_transition(
                    environment_state,
                    PathfindingAction(current=environment_state.current, target=action),
                )
            else:
                invalid_attempts += 1
        return TrajectoryResult(
            reached_goal=episode.is_goal_reached(environment_state),
            actions=tuple(actions),
            final_environment_state=environment_state,
            invalid_attempts=invalid_attempts,
        )


def rollout_action_masked(
    model: ActionMaskingRecurrentBaseline,
    episode: PathfindingEpisode,
    max_transitions: int = SINGLE_TRAJECTORY_TRANSITIONS,
) -> TrajectoryResult:
    """Executes one action-masked single trajectory over legal moves only.

    The environment stops early when no legal move remains (dead end / stuck)
    without reaching the goal; the selected legal action is committed
    unconditionally to the latent state.
    """
    _validate_single_trajectory_rollout(
        model, episode, max_transitions, ActionMaskingRecurrentBaseline
    )
    with torch.no_grad():
        latent_state = model.initial_state(1)
        environment_state = episode.initial_state()
        actions = []
        for _ in range(max_transitions):
            if episode.is_goal_reached(environment_state):
                break
            observation = episode.observe(environment_state)
            encoded = model.encode_observation(observation).unsqueeze(0)
            logits = model.policy_logits(latent_state, encoded)
            action = model.select_action(logits, observation)
            if action is None:
                break
            actions.append(action)
            latent_state = model.advance_state(latent_state, encoded, action)
            environment_state = episode.apply_transition(
                environment_state,
                PathfindingAction(current=environment_state.current, target=action),
            )
        return TrajectoryResult(
            reached_goal=episode.is_goal_reached(environment_state),
            actions=tuple(actions),
            final_environment_state=environment_state,
            invalid_attempts=0,
        )


def _rollout_stochastic(
    model: TwoRolloutRecurrentBaseline,
    episode: PathfindingEpisode,
    horizon: int,
    generator: torch.Generator,
) -> TrajectoryResult:
    verifier = PathfindingVerifier(episode)
    with torch.no_grad():
        latent_state = model.initial_state(1)
        environment_state = episode.initial_state()
        actions = []
        for _ in range(horizon):
            if episode.is_goal_reached(environment_state):
                break
            observation = episode.observe(environment_state)
            encoded = model.encode_observation(observation).unsqueeze(0)
            logits = model.policy_logits(latent_state, encoded)
            action = model.sample_action(logits, observation, generator)
            if action is None:
                break
            verification = verifier.verify_transition(
                environment_state,
                PathfindingAction(current=environment_state.current, target=action),
            )
            if not verification.valid:
                break
            actions.append(action)
            latent_state = model.advance_state(latent_state, encoded, action)
            environment_state = episode.apply_transition(
                environment_state,
                PathfindingAction(current=environment_state.current, target=action),
            )
        return TrajectoryResult(
            reached_goal=episode.is_goal_reached(environment_state),
            actions=tuple(actions),
            final_environment_state=environment_state,
            invalid_attempts=0,
        )


def rollout_two_trajectory(
    model: TwoRolloutRecurrentBaseline,
    episode: PathfindingEpisode,
    model_seed: int,
    episode_index: int,
    partition: str,
) -> TwoTrajectoryResult:
    """Executes the two independent stochastic rollouts and applies the selection rule.

    Trajectory horizons are static (``T1 = 26``, ``T2 = 27``); unused steps are not
    dynamically banked. Sampling generators are isolated and seeded by the
    protocol's partition-scoped rollout formula. Selection uses only public goal
    arrival and executed length; no oracle information is consulted.
    """
    if not isinstance(model, TwoRolloutRecurrentBaseline):
        raise TypeError(
            f"model must be a TwoRolloutRecurrentBaseline, got {type(model).__name__}"
        )
    if not isinstance(episode, PathfindingEpisode):
        raise TypeError(f"episode must be a PathfindingEpisode, got {type(episode).__name__}")

    trajectory_1 = _rollout_stochastic(
        model,
        episode,
        horizon=TWO_ROLLOUT_HORIZONS[0],
        generator=make_rollout_generator(
            rollout_seed(model_seed, episode_index, 1, partition)
        ),
    )
    trajectory_2 = _rollout_stochastic(
        model,
        episode,
        horizon=TWO_ROLLOUT_HORIZONS[1],
        generator=make_rollout_generator(
            rollout_seed(model_seed, episode_index, 2, partition)
        ),
    )
    return TwoTrajectoryResult(
        trajectory_1=trajectory_1,
        trajectory_2=trajectory_2,
        selected_trajectory=select_trajectory(trajectory_1, trajectory_2),
    )
