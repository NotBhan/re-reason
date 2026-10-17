"""Constrained graph pathfinding benchmark: procedural episodes, reference solver, and split handling.

Task definition
---------------
An episode is a directed graph over nodes ``0..num_nodes-1`` with a distinguished
``start`` and ``goal``:

* ``edges`` is the underlying topology: every transition that structurally exists.
* ``blocked_edges`` is a subset of ``edges`` whose transitions are currently
  impassable (dynamic blockades). Blocked transitions are illegal moves and are
  never treated as traversable.
* The *effective* transition graph is ``edges - blocked_edges``; a move is allowed
  exactly when it is an effective transition.

Two failure modes are deliberately distinct:

* **Blockades** are illegal transitions (rejected by transition verification).
* **Dead ends** are legal moves into regions from which the goal is unreachable.

Solvability convention: an episode is solvable when at least one sequence of
allowed moves connects ``start`` to ``goal``. The procedural generator produces
solvable instances only and validates every constructed instance against the
reference solver. Unsolvable graphs remain representable through
:class:`PathfindingEpisode` and are reported explicitly by
:func:`solve_shortest_path` (``solved=False``).

Visibility model (fully observable)
-----------------------------------
This initial benchmark is fully observable: the agent-facing
:class:`PathfindingObservation` exposes the complete effective adjacency, the
``current`` node, and the ``goal``. No task-relevant information is hidden and
blockades are public constraints. The raw topology/blockade split stays in the
ground-truth episode and is not exposed through the observation surface, because
the effective graph already determines the task.

Information boundaries
----------------------
* Ground truth: :class:`PathfindingEpisode` (topology, blockades, start, goal).
* Agent observation: :class:`PathfindingObservation` (current, goal, effective adjacency).
* Transition verification: ``re_reason.verification.pathfinding_verifier.PathfindingVerifier``.
* Oracle: :func:`solve_shortest_path` and :func:`goal_reachable_from` (full ground truth).

The generator uses the oracle only to verify constructed instances; the oracle
and the deterministic verifier remain separate components with separate purposes.

Scope note: the ``BaseTask``/``TaskBatch`` contract in ``re_reason.benchmarks.base``
is a tensor-batch interface aimed at neural models. This milestone deliberately
does not implement it: no neural integration is authorized yet, and freezing an
episode-to-tensor encoding now would lock in design decisions better made
together with the model-integration milestone. The structured episode,
observation, state, and action APIs above are the stable interface for this
benchmark.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Iterable, Optional, Sequence

import numpy as np


TRANSITION_OK = "ok"
TRANSITION_UNKNOWN_NODE = "unknown_node"
TRANSITION_NO_SUCH_EDGE = "no_such_edge"
TRANSITION_BLOCKED = "blocked_transition"
TRANSITION_STATE_MISMATCH = "state_mismatch"

PATH_EMPTY = "empty_path"
PATH_START_MISMATCH = "path_start_mismatch"
PATH_GOAL_MISMATCH = "path_goal_mismatch"

PARTITION_SEED_OFFSETS = {"train": 1000, "validation": 50000, "test": 90000}


def _require_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _coerce_edge_set(entries: Iterable[object], label: str) -> frozenset[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"{label} entries must be (u, v) pairs, got {entry!r}")
        u, v = entry
        for endpoint in (u, v):
            if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                raise ValueError(f"{label} endpoints must be integer node ids, got {endpoint!r}")
        if u == v:
            raise ValueError(f"{label} must not contain self-loops, got {entry!r}")
        edges.add((u, v))
    return frozenset(edges)


@dataclass(frozen=True)
class PathfindingGeneratorConfig:
    """Deterministic construction parameters for constrained pathfinding episodes.

    Construction scheme (all node selections derive from one seeded RNG stream):

    * ``spine_length`` (derived from ``num_nodes``) nodes form the guaranteed
      start-to-goal chain; consecutive spine edges are never blocked.
    * ``num_traps`` dead-end chains of ``trap_length`` nodes are attached to spine
      decision points (the first trap always attaches to the start node); any
      remaining nodes extend the last trap chain.
    * ``num_extra_edges`` directed shortcut edges are sampled between distinct
      spine nodes; ``num_blockades`` of those extra edges are then marked
      impassable. The guaranteed spine chain is never blocked.

    Instances are validated before being returned (solvability plus dead-end and
    alternative-route coverage where traps are requested). Generation is bounded
    by ``max_generation_attempts`` and raises :class:`RuntimeError` explicitly if
    no valid instance is produced within that bound.

    With ``num_traps=0`` every node must belong to the spine chain
    (``num_nodes == spine_length``), so no leftover nodes are left stranded.
    """

    num_nodes: int = 12
    num_extra_edges: int = 3
    num_traps: int = 1
    trap_length: int = 2
    num_blockades: int = 2
    max_generation_attempts: int = 32

    def __post_init__(self) -> None:
        _require_int("num_nodes", self.num_nodes, 3)
        _require_int("num_extra_edges", self.num_extra_edges, 0)
        _require_int("num_traps", self.num_traps, 0)
        _require_int("trap_length", self.trap_length, 1)
        _require_int("num_blockades", self.num_blockades, 0)
        _require_int("max_generation_attempts", self.max_generation_attempts, 1)

        spine = self.spine_length
        if spine + self.num_traps * self.trap_length > self.num_nodes:
            raise ValueError(
                f"node budget too small: spine_length {spine} + {self.num_traps} trap chain(s) "
                f"of {self.trap_length} node(s) exceeds num_nodes={self.num_nodes}"
            )
        if self.num_traps == 0 and spine != self.num_nodes:
            raise ValueError(
                f"with num_traps=0 every node must belong to the spine chain, but "
                f"spine_length {spine} != num_nodes {self.num_nodes}"
            )
        capacity = (spine - 1) ** 2
        if self.num_extra_edges > capacity:
            raise ValueError(
                f"num_extra_edges {self.num_extra_edges} exceeds the {capacity} distinct "
                f"spine-to-spine shortcut edges available for num_nodes={self.num_nodes}"
            )
        if self.num_blockades > self.num_extra_edges:
            raise ValueError(
                f"num_blockades {self.num_blockades} cannot exceed num_extra_edges "
                f"{self.num_extra_edges}"
            )

    @property
    def spine_length(self) -> int:
        """Length of the guaranteed start-to-goal chain derived from ``num_nodes``."""
        return max(3, self.num_nodes // 2)


@dataclass(frozen=True)
class PathfindingEpisode:
    """Immutable ground-truth description of one constrained pathfinding instance.

    Consistency is validated at construction: node ids must be in range, edges
    must not contain self-loops, and ``blocked_edges`` must be a subset of
    ``edges`` (a blocked transition cannot be absent from the topology). Input
    collections are defensively copied into frozensets, so later mutation of the
    caller's containers cannot alter the episode.
    """

    num_nodes: int
    start: int
    goal: int
    edges: frozenset[tuple[int, int]]
    blocked_edges: frozenset[tuple[int, int]] = frozenset()

    def __post_init__(self) -> None:
        _require_int("num_nodes", self.num_nodes, 1)
        for name, value in (("start", self.start), ("goal", self.goal)):
            _require_int(name, value, 0)
            if value >= self.num_nodes:
                raise ValueError(f"{name} {value} out of range for num_nodes={self.num_nodes}")

        edges = _coerce_edge_set(self.edges, "edges")
        blocked = _coerce_edge_set(self.blocked_edges, "blocked_edges")
        for label, edge_set in (("edges", edges), ("blocked_edges", blocked)):
            for u, v in edge_set:
                if u >= self.num_nodes or v >= self.num_nodes:
                    raise ValueError(
                        f"{label} entry {(u, v)} references a node id outside [0, {self.num_nodes})"
                    )
        if not blocked <= edges:
            raise ValueError(
                "blocked_edges must be a subset of edges; a blocked transition cannot be "
                "absent from the underlying topology"
            )

        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "blocked_edges", blocked)
        effective = edges - blocked
        object.__setattr__(self, "_effective_edges", effective)
        object.__setattr__(
            self,
            "_adjacency",
            tuple(
                tuple(sorted(v for (u, v) in effective if u == node))
                for node in range(self.num_nodes)
            ),
        )

    def has_transition(self, u: int, v: int) -> bool:
        """True when the topology contains the directed transition ``u -> v``."""
        return (u, v) in self.edges

    def is_blocked(self, u: int, v: int) -> bool:
        """True when the topology transition ``u -> v`` is currently impassable."""
        return (u, v) in self.blocked_edges

    def is_allowed_move(self, u: int, v: int) -> bool:
        """True when ``u -> v`` is an effective (non-blocked) transition."""
        return (u, v) in self.edges and (u, v) not in self.blocked_edges

    def effective_edges(self) -> frozenset[tuple[int, int]]:
        """The effective transition graph: topology edges minus blocked edges."""
        return self._effective_edges

    def allowed_moves(self, node: int) -> tuple[int, ...]:
        """Sorted effective moves available at ``node``; empty for unknown nodes."""
        if isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < self.num_nodes:
            return ()
        return self._adjacency[node]

    def transition_reason(self, u: int, v: int) -> str:
        """Authoritative classification of a proposed node-to-node transition.

        Returns one of ``TRANSITION_OK``, ``TRANSITION_UNKNOWN_NODE``,
        ``TRANSITION_NO_SUCH_EDGE``, or ``TRANSITION_BLOCKED``. This classifies
        the transition itself; state/action bookkeeping is the verifier's concern.
        """
        for value in (u, v):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.num_nodes:
                return TRANSITION_UNKNOWN_NODE
        if (u, v) not in self.edges:
            return TRANSITION_NO_SUCH_EDGE
        if (u, v) in self.blocked_edges:
            return TRANSITION_BLOCKED
        return TRANSITION_OK

    def digest(self) -> str:
        """Canonical SHA-256 content digest for duplicate and leakage checks."""
        payload = "|".join(
            [
                str(self.num_nodes),
                str(self.start),
                str(self.goal),
                ";".join(f"{u}>{v}" for u, v in sorted(self.edges)),
                ";".join(f"{u}>{v}" for u, v in sorted(self.blocked_edges)),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def initial_state(self) -> "PathfindingState":
        """The initial task state at the episode's start node."""
        return PathfindingState(current=self.start, transitions=0)

    def observe(self, state: "PathfindingState") -> "PathfindingObservation":
        """Builds the fully observable agent view at ``state``."""
        if not isinstance(state, PathfindingState):
            raise TypeError(f"state must be a PathfindingState, got {type(state).__name__}")
        if not 0 <= state.current < self.num_nodes:
            raise ValueError(
                f"state.current {state.current} out of range for num_nodes={self.num_nodes}"
            )
        return PathfindingObservation(
            current=state.current,
            goal=self.goal,
            effective_adjacency=self._adjacency,
        )

    def apply_transition(
        self, state: "PathfindingState", action: "PathfindingAction"
    ) -> "PathfindingState":
        """Pure commit of one transition; returns the successor state.

        Rejected transitions (state/action mismatch, unknown nodes, missing edges,
        or blockades) raise :class:`ValueError` and leave the given state and the
        episode untouched. A committed transition advances the position exactly
        once and increments the transition counter exactly once.
        """
        if not isinstance(state, PathfindingState):
            raise TypeError(f"state must be a PathfindingState, got {type(state).__name__}")
        if not isinstance(action, PathfindingAction):
            raise TypeError(f"action must be a PathfindingAction, got {type(action).__name__}")
        if action.current != state.current:
            raise ValueError(
                f"action starts at node {action.current} but the state is at node "
                f"{state.current} ({TRANSITION_STATE_MISMATCH})"
            )
        reason = self.transition_reason(state.current, action.target)
        if reason != TRANSITION_OK:
            raise ValueError(
                f"transition {state.current} -> {action.target} is not an allowed move ({reason})"
            )
        return PathfindingState(current=action.target, transitions=state.transitions + 1)

    def is_goal_reached(self, state: "PathfindingState") -> bool:
        """True when ``state`` stands at the goal node."""
        if not isinstance(state, PathfindingState):
            raise TypeError(f"state must be a PathfindingState, got {type(state).__name__}")
        return state.current == self.goal

    def is_stuck(self, state: "PathfindingState") -> bool:
        """True when ``state`` is not at the goal and has no effective moves left."""
        return not self.is_goal_reached(state) and len(self.allowed_moves(state.current)) == 0


@dataclass(frozen=True)
class PathfindingState:
    """Current task position within an episode."""

    current: int
    transitions: int = 0

    def __post_init__(self) -> None:
        _require_int("current", self.current, 0)
        _require_int("transitions", self.transitions, 0)


@dataclass(frozen=True)
class PathfindingAction:
    """A proposed movement: the node it starts from and the target node."""

    current: int
    target: int

    def __post_init__(self) -> None:
        _require_int("current", self.current, 0)
        _require_int("target", self.target, 0)


@dataclass(frozen=True)
class PathfindingObservation:
    """Agent-facing fully observable view of an episode at one decision point.

    Exposes exactly the task-relevant public surface: the current node, the goal,
    and the effective adjacency (allowed moves per node). The underlying
    topology/blockade split is ground truth and is not part of this surface.
    """

    current: int
    goal: int
    effective_adjacency: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _require_int("current", self.current, 0)
        _require_int("goal", self.goal, 0)
        if not isinstance(self.effective_adjacency, tuple) or len(self.effective_adjacency) == 0:
            raise ValueError("effective_adjacency must be a non-empty tuple of per-node move tuples")
        if self.current >= len(self.effective_adjacency) or self.goal >= len(self.effective_adjacency):
            raise ValueError("current/goal out of range for the effective adjacency")

    @property
    def allowed_moves(self) -> tuple[int, ...]:
        """Effective moves available at the current node."""
        return self.effective_adjacency[self.current]


@dataclass(frozen=True)
class PathfindingSolution:
    """Reference-solver result for one episode (or one explicit start/goal query).

    Path-length convention: a path containing ``k`` moves consists of ``k + 1``
    nodes, so ``num_moves == len(path) - 1`` when solved and ``0`` when unsolved.
    """

    solved: bool
    path: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.solved and len(self.path) == 0:
            raise ValueError("a solved solution must contain at least the start node")
        if not self.solved and len(self.path) != 0:
            raise ValueError("an unsolved solution must carry an empty path")

    @property
    def num_moves(self) -> int:
        """Number of transitions in the solution path (0 when unsolved)."""
        return len(self.path) - 1 if self.solved else 0


@dataclass(frozen=True)
class PathValidation:
    """Result of independent complete-path validation (never produced by the solver)."""

    valid: bool
    reason: str


def solve_shortest_path(
    episode: PathfindingEpisode,
    start: Optional[int] = None,
    goal: Optional[int] = None,
) -> PathfindingSolution:
    """Deterministic BFS reference solver over the effective transition graph.

    Unit edge weights make BFS the canonical shortest-path oracle: the returned
    path minimizes the number of moves. Ties are resolved deterministically by
    sorted neighbor order. Relies exclusively on the episode's authoritative
    rules; no learned scores or heuristics are involved.
    """
    if not isinstance(episode, PathfindingEpisode):
        raise TypeError(f"episode must be a PathfindingEpisode, got {type(episode).__name__}")
    source = episode.start if start is None else _require_int("start", start, 0)
    target = episode.goal if goal is None else _require_int("goal", goal, 0)
    for name, value in (("start", source), ("goal", target)):
        if value >= episode.num_nodes:
            raise ValueError(f"{name} {value} out of range for num_nodes={episode.num_nodes}")

    if source == target:
        return PathfindingSolution(solved=True, path=(source,))

    parents: dict[int, int] = {}
    visited = {source}
    frontier = deque([source])
    while frontier:
        node = frontier.popleft()
        for neighbor in episode.allowed_moves(node):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parents[neighbor] = node
            if neighbor == target:
                path = [target]
                while path[-1] != source:
                    path.append(parents[path[-1]])
                path.reverse()
                return PathfindingSolution(solved=True, path=tuple(path))
            frontier.append(neighbor)
    return PathfindingSolution(solved=False, path=())


def goal_reachable_from(episode: PathfindingEpisode, node: int, goal: Optional[int] = None) -> bool:
    """Reachability query sharing the solver's BFS core (single source of truth)."""
    node = _require_int("node", node, 0)
    return solve_shortest_path(episode, start=node, goal=goal).solved


def validate_path(episode: PathfindingEpisode, path: Sequence[int]) -> PathValidation:
    """Independently validates a complete node sequence as a start-to-goal walk.

    This is deliberately separate from the solver: it checks the path's endpoints,
    node validity, and every consecutive transition against the episode's
    authoritative rules, returning structured reasons instead of trusting the
    solver's own output.
    """
    if not isinstance(episode, PathfindingEpisode):
        raise TypeError(f"episode must be a PathfindingEpisode, got {type(episode).__name__}")
    nodes = tuple(path)
    if len(nodes) == 0:
        return PathValidation(False, PATH_EMPTY)
    if nodes[0] != episode.start:
        return PathValidation(False, PATH_START_MISMATCH)
    if nodes[-1] != episode.goal:
        return PathValidation(False, PATH_GOAL_MISMATCH)
    for u, v in zip(nodes, nodes[1:]):
        reason = episode.transition_reason(u, v)
        if reason != TRANSITION_OK:
            return PathValidation(False, reason)
    return PathValidation(True, TRANSITION_OK)


def generate_pathfinding_episode(config: PathfindingGeneratorConfig, seed: int) -> PathfindingEpisode:
    """Generates one validated, solvable episode deterministically from ``seed``.

    Construction is controlled (not rejection-sampling of arbitrary random
    graphs): a guaranteed spine chain plus controlled trap and shortcut placement
    yield the requested properties by design. Each constructed instance is still
    validated against the oracle, and generation is bounded by
    ``config.max_generation_attempts``; if no valid instance is produced within
    the bound, :class:`RuntimeError` is raised explicitly.
    """
    if not isinstance(config, PathfindingGeneratorConfig):
        raise TypeError(
            f"config must be a PathfindingGeneratorConfig, got {type(config).__name__}"
        )
    _require_int("seed", seed, 0)

    rng = np.random.default_rng(seed)
    failure: Optional[str] = None
    for _ in range(config.max_generation_attempts):
        episode = _construct_episode(config, rng)
        failure = _generated_instance_failure(episode, config)
        if failure is None:
            return episode
    raise RuntimeError(
        f"bounded generation failed after {config.max_generation_attempts} attempts "
        f"(seed {seed}): {failure}"
    )


def _sample_indices(rng: np.random.Generator, pool_size: int, count: int) -> list[int]:
    if count == 0:
        return []
    chosen = rng.choice(pool_size, size=count, replace=False)
    return [int(index) for index in np.atleast_1d(chosen)]


def _construct_episode(config: PathfindingGeneratorConfig, rng: np.random.Generator) -> PathfindingEpisode:
    order = [int(node) for node in rng.permutation(config.num_nodes)]
    spine_length = config.spine_length
    spine = order[:spine_length]
    remaining = order[spine_length:]

    edges: set[tuple[int, int]] = set()
    for index in range(spine_length - 1):
        edges.add((spine[index], spine[index + 1]))

    if config.num_traps > 0:
        chains: list[list[int]] = []
        cursor = 0
        for _ in range(config.num_traps):
            chains.append(remaining[cursor : cursor + config.trap_length])
            cursor += config.trap_length
        chains[-1].extend(remaining[cursor:])
        for chain_index, chain in enumerate(chains):
            attach = spine[chain_index % (spine_length - 1)]
            edges.add((attach, chain[0]))
            for index in range(len(chain) - 1):
                edges.add((chain[index], chain[index + 1]))

    spine_edges = {(spine[index], spine[index + 1]) for index in range(spine_length - 1)}
    candidates = [(u, v) for u in spine for v in spine if u != v and (u, v) not in spine_edges]
    extras = [candidates[index] for index in _sample_indices(rng, len(candidates), config.num_extra_edges)]
    edges.update(extras)

    blocked = {
        extras[index] for index in _sample_indices(rng, len(extras), config.num_blockades)
    }

    return PathfindingEpisode(
        num_nodes=config.num_nodes,
        start=spine[0],
        goal=spine[-1],
        edges=frozenset(edges),
        blocked_edges=frozenset(blocked),
    )


def _generated_instance_failure(
    episode: PathfindingEpisode, config: PathfindingGeneratorConfig
) -> Optional[str]:
    if len(episode.blocked_edges) != config.num_blockades:
        return (
            f"constructed instance has {len(episode.blocked_edges)} blockades; "
            f"config requests {config.num_blockades}"
        )
    if not goal_reachable_from(episode, episode.start):
        return "goal is not reachable from start over allowed moves"
    if config.num_traps > 0:
        moves = episode.allowed_moves(episode.start)
        if len(moves) < 2:
            return "start node does not branch into at least two effective moves"
        if not any(goal_reachable_from(episode, neighbor) for neighbor in moves):
            return "no effective move from start reaches the goal"
        if not any(not goal_reachable_from(episode, neighbor) for neighbor in moves):
            return "no effective move from start enters a dead-end region"
    return None


@dataclass(frozen=True)
class PathfindingSplitSet:
    """Reproducible train/validation/test partitions with disjoint instance identities.

    The generation configuration and base seed are recorded alongside the
    instances so every partition is traceable to its generation rule.
    """

    config: PathfindingGeneratorConfig
    base_seed: int
    train: tuple[PathfindingEpisode, ...]
    validation: tuple[PathfindingEpisode, ...]
    test: tuple[PathfindingEpisode, ...]


def generate_pathfinding_partition(
    config: PathfindingGeneratorConfig,
    partition: str,
    size: int,
    base_seed: int = 42,
) -> tuple[PathfindingEpisode, ...]:
    """Generates one named partition without materializing the other partitions.

    Partition seed spaces follow the repository convention (``base_seed + 1000``
    for ``train``, ``+50000`` for ``validation``, ``+90000`` for ``test``).
    Duplicate draws within the partition are skipped under a bounded attempt
    budget. Callers that are restricted to one partition (for example authorized
    training code restricted to ``train``) can generate it without constructing
    the held-out partitions.
    """
    if not isinstance(config, PathfindingGeneratorConfig):
        raise TypeError(
            f"config must be a PathfindingGeneratorConfig, got {type(config).__name__}"
        )
    if partition not in PARTITION_SEED_OFFSETS:
        raise ValueError(
            f"partition must be one of {sorted(PARTITION_SEED_OFFSETS)}, got {partition!r}"
        )
    _require_int("size", size, 1)
    _require_int("base_seed", base_seed, 0)
    return _generate_unique_partition(
        config, base_seed + PARTITION_SEED_OFFSETS[partition], size
    )


def generate_pathfinding_splits(
    train_size: int,
    validation_size: int,
    test_size: int,
    config: PathfindingGeneratorConfig,
    base_seed: int = 42,
) -> PathfindingSplitSet:
    """Generates deterministic, duplicate-free, mutually disjoint episode partitions.

    Partition seed spaces follow the repository convention: ``base_seed + 1000``
    (train), ``base_seed + 50000`` (validation), and ``base_seed + 90000`` (test).
    Instances are identified by their canonical content digest; duplicate draws
    within a partition are skipped under a bounded attempt budget, and overlap
    between partitions raises :class:`RuntimeError` explicitly. No dataset is
    materialized beyond the requested in-memory episodes.
    """
    if not isinstance(config, PathfindingGeneratorConfig):
        raise TypeError(
            f"config must be a PathfindingGeneratorConfig, got {type(config).__name__}"
        )
    _require_int("train_size", train_size, 1)
    _require_int("validation_size", validation_size, 1)
    _require_int("test_size", test_size, 1)
    _require_int("base_seed", base_seed, 0)

    partitions = {
        "train": generate_pathfinding_partition(config, "train", train_size, base_seed),
        "validation": generate_pathfinding_partition(
            config, "validation", validation_size, base_seed
        ),
        "test": generate_pathfinding_partition(config, "test", test_size, base_seed),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = {episode.digest() for episode in partitions[left]} & {
            episode.digest() for episode in partitions[right]
        }
        if overlap:
            raise RuntimeError(
                f"Data leakage detected! {len(overlap)} instances overlap between "
                f"{left} and {right}"
            )
    return PathfindingSplitSet(
        config=config,
        base_seed=base_seed,
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
    )


def _generate_unique_partition(
    config: PathfindingGeneratorConfig,
    seed_base: int,
    size: int,
    max_attempts_per_instance: int = 16,
) -> tuple[PathfindingEpisode, ...]:
    seen: set[str] = set()
    episodes: list[PathfindingEpisode] = []
    max_attempts = size * max_attempts_per_instance
    for attempt in range(max_attempts):
        if len(episodes) == size:
            break
        episode = generate_pathfinding_episode(config, seed_base + attempt)
        digest = episode.digest()
        if digest not in seen:
            seen.add(digest)
            episodes.append(episode)
    if len(episodes) != size:
        raise RuntimeError(
            f"exhausted {max_attempts} bounded attempts while generating {size} unique "
            f"instances for the partition seeded at {seed_base}"
        )
    return tuple(episodes)
