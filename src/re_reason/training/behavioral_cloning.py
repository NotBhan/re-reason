"""BFS shortest-path behavioral cloning for the Milestone 3B comparators.

Implements the pre-3B protocol's teacher-forced sequence-unrolling contract:

* Demonstrations are BFS shortest paths produced by the benchmark's reference
  oracle. Unsolved episodes (and degenerate zero-transition solved episodes) are
  skipped defensively without raising and without touching protected partitions.
* At each step the public observation is encoded, policy logits are produced from
  ``[S_t || e_t]``, and full-support cross-entropy is computed against the
  demonstration action ``a*_t`` (all 12 classes participate; the demonstration
  action does not need to rank in the inference-time top-K).
* The latent state advances through the shared branch core on the demonstration
  action: ``S_{t+1} = BranchCore(S_t, e_t, E_node[a*_t])``. Downstream policy
  losses therefore backpropagate into the branch core and upstream state.
* The batch loss is the transition-weighted mean over valid transitions,
  ``sum(M[b, t] * CE[b, t]) / sum(M)``. Trailing padded steps are zero-masked and
  contribute zero loss and zero gradient.
* The final demonstration transition is evaluated (its successor state is
  produced) but has no downstream policy loss, so the branch core at that step
  receives zero backpropagated policy loss (documented terminal-step boundary).

No auxiliary losses, scheduled sampling, on-policy rollouts, transition-prediction
objectives, loss reweighting, or hyperparameter sweeps are applied. Training
latent states start at zero, matching the existing RE:Reason model convention.
Training code is restricted to the ``train`` partition by construction.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from re_reason.benchmarks.pathfinding_benchmark import (
    PathfindingEpisode,
    PathfindingGeneratorConfig,
    PathfindingObservation,
    PathfindingState,
    generate_pathfinding_partition,
    solve_shortest_path,
)
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher
from re_reason.utils.rng_namespaces import shuffled_batch_indices


@dataclass(frozen=True)
class Demonstration:
    """One teacher-forced BFS demonstration: observations paired with actions.

    ``observation[t]`` is the public observation at path node ``v_t`` and
    ``action[t]`` is the demonstration successor ``v_{t+1}``.
    """

    observations: Tuple[PathfindingObservation, ...]
    actions: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.observations) != len(self.actions):
            raise ValueError("observations and actions must have the same length")
        if len(self.actions) == 0:
            raise ValueError("a demonstration must contain at least one transition")
        for observation in self.observations:
            if not isinstance(observation, PathfindingObservation):
                raise TypeError(
                    f"observations must be PathfindingObservation instances, got "
                    f"{type(observation).__name__}"
                )
        num_nodes = len(self.observations[0].effective_adjacency)
        for action in self.actions:
            if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < num_nodes:
                raise ValueError(f"demonstration action {action!r} is not a valid node id")

    @property
    def length(self) -> int:
        """Number of supervised demonstration transitions ``L``."""
        return len(self.actions)


@dataclass(frozen=True)
class BehavioralCloningLoss:
    """Result of one transition-weighted teacher-forced batch loss."""

    loss: torch.Tensor
    total_transitions: int
    batch_size: int
    max_length: int


def build_demonstration(episode: PathfindingEpisode) -> Optional[Demonstration]:
    """Builds a teacher-forced demonstration from the BFS shortest path.

    Returns ``None`` for unsolved episodes and for degenerate solved episodes with
    zero transitions (start equals goal), which carry no supervision signal.
    """
    if not isinstance(episode, PathfindingEpisode):
        raise TypeError(f"episode must be a PathfindingEpisode, got {type(episode).__name__}")
    solution = solve_shortest_path(episode)
    if not solution.solved or solution.num_moves == 0:
        return None
    path = solution.path
    observations = tuple(
        episode.observe(PathfindingState(current=path[index], transitions=index))
        for index in range(solution.num_moves)
    )
    actions = tuple(path[index + 1] for index in range(solution.num_moves))
    return Demonstration(observations=observations, actions=actions)


def build_demonstrations(
    episodes: Sequence[PathfindingEpisode],
) -> Tuple[Demonstration, ...]:
    """Builds demonstrations for all solvable episodes, skipping unsolved ones."""
    demonstrations: List[Demonstration] = []
    for episode in episodes:
        demonstration = build_demonstration(episode)
        if demonstration is not None:
            demonstrations.append(demonstration)
    return tuple(demonstrations)


def teacher_forced_batch_loss(
    brancher: ActionConditionedBrancher,
    demonstrations: Sequence[Demonstration],
) -> BehavioralCloningLoss:
    """Computes the transition-weighted teacher-forced behavioral-cloning loss.

    Padded positions are zero-masked; the normalizer is the total number of valid
    demonstration transitions in the batch, exactly as specified by the protocol.
    """
    if not isinstance(brancher, ActionConditionedBrancher):
        raise TypeError(
            f"brancher must be an ActionConditionedBrancher, got {type(brancher).__name__}"
        )
    if isinstance(demonstrations, (str, bytes)) or not isinstance(demonstrations, Sequence):
        raise TypeError("demonstrations must be a sequence of Demonstration")
    if len(demonstrations) == 0:
        raise ValueError("at least one demonstration is required")
    for demonstration in demonstrations:
        if not isinstance(demonstration, Demonstration):
            raise TypeError(
                f"demonstrations must contain Demonstration instances, got "
                f"{type(demonstration).__name__}"
            )

    batch_size = len(demonstrations)
    max_length = max(demonstration.length for demonstration in demonstrations)
    reference = brancher.node_embeddings.weight
    device, dtype = reference.device, reference.dtype

    features = torch.zeros(
        (batch_size, max_length, brancher.observation_input_dim), dtype=dtype, device=device
    )
    actions = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, max_length), dtype=dtype, device=device)
    for row, demonstration in enumerate(demonstrations):
        for step in range(demonstration.length):
            features[row, step] = brancher.observation_features(demonstration.observations[step])
            actions[row, step] = demonstration.actions[step]
            mask[row, step] = 1.0
    total_transitions = int(mask.sum().item())

    encoded = brancher.observation_encoder(features)  # (B, T, state_dim)
    state = torch.zeros((batch_size, brancher.state_dim), dtype=dtype, device=device)
    step_logits: List[torch.Tensor] = []
    for step in range(max_length):
        step_logits.append(brancher.policy_logits(state, encoded[:, step, :]))
        state = brancher.advance_states(state, encoded[:, step, :], actions[:, step])

    logits = torch.stack(step_logits, dim=1)  # (B, T, num_nodes)
    per_step = F.cross_entropy(
        logits.reshape(batch_size * max_length, brancher.num_nodes),
        actions.reshape(batch_size * max_length),
        reduction="none",
    ).reshape(batch_size, max_length)
    loss = (per_step * mask).sum() / mask.sum()

    return BehavioralCloningLoss(
        loss=loss,
        total_transitions=total_transitions,
        batch_size=batch_size,
        max_length=max_length,
    )


def iter_training_batches(
    demonstrations: Sequence[Demonstration],
    batch_size: int,
    generator: torch.Generator,
    drop_last: bool = True,
) -> Iterator[Tuple[Demonstration, ...]]:
    """Yields deterministic shuffled batches of demonstrations for one epoch."""
    for indices in shuffled_batch_indices(
        len(demonstrations), batch_size, generator, drop_last=drop_last
    ):
        yield tuple(demonstrations[index] for index in indices)


def load_behavioral_cloning_demonstrations(
    config: Optional[PathfindingGeneratorConfig] = None,
    *,
    size: int = 200,
    base_seed: int = 42,
) -> Tuple[Demonstration, ...]:
    """Builds BFS demonstrations from the authorized ``train`` partition only.

    Hard-restricted to the training partition: training code must never
    materialize the validation or held-out test partitions. Graph generation uses
    the independent graph RNG namespace (``base_seed``); model-training and
    rollout-sampling RNG streams are separate.
    """
    resolved_config = PathfindingGeneratorConfig() if config is None else config
    episodes = generate_pathfinding_partition(resolved_config, "train", size, base_seed)
    return build_demonstrations(episodes)
