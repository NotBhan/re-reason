"""Unit tests for the constrained pathfinding benchmark: episodes, generator, splits, and reference oracle."""

import dataclasses
import gc
import weakref

import pytest

from re_reason.benchmarks.pathfinding_benchmark import (
    PATH_EMPTY,
    PATH_GOAL_MISMATCH,
    PATH_START_MISMATCH,
    TRANSITION_BLOCKED,
    TRANSITION_NO_SUCH_EDGE,
    TRANSITION_OK,
    TRANSITION_UNKNOWN_NODE,
    PathfindingAction,
    PathfindingEpisode,
    PathfindingGeneratorConfig,
    PathfindingObservation,
    PathfindingState,
    generate_pathfinding_episode,
    generate_pathfinding_splits,
    goal_reachable_from,
    solve_shortest_path,
    validate_path,
)


def _config(**overrides) -> PathfindingGeneratorConfig:
    defaults = dict(
        num_nodes=12,
        num_extra_edges=3,
        num_traps=1,
        trap_length=2,
        num_blockades=2,
    )
    return PathfindingGeneratorConfig(**{**defaults, **overrides})


def _episode(num_nodes, start, goal, edges, blocked=()) -> PathfindingEpisode:
    return PathfindingEpisode(
        num_nodes=num_nodes,
        start=start,
        goal=goal,
        edges=frozenset(edges),
        blocked_edges=frozenset(blocked),
    )


# ---------------------------------------------------------------------------
# Episode contract and immutability
# ---------------------------------------------------------------------------

def test_episode_queries_follow_the_effective_graph() -> None:
    episode = _episode(4, 0, 3, {(0, 1), (1, 3), (0, 2), (2, 1)}, blocked={(0, 1)})

    assert episode.has_transition(0, 1) and episode.is_blocked(0, 1)
    assert not episode.is_allowed_move(0, 1)
    assert episode.is_allowed_move(0, 2)
    assert episode.effective_edges() == frozenset({(1, 3), (0, 2), (2, 1)})
    assert episode.allowed_moves(0) == (2,)
    assert episode.allowed_moves(99) == ()
    assert episode.transition_reason(0, 2) == TRANSITION_OK
    assert episode.transition_reason(0, 1) == TRANSITION_BLOCKED
    assert episode.transition_reason(1, 0) == TRANSITION_NO_SUCH_EDGE
    assert episode.transition_reason(0, 9) == TRANSITION_UNKNOWN_NODE


def test_episode_construction_rejects_inconsistent_specifications() -> None:
    with pytest.raises(ValueError):
        PathfindingEpisode(num_nodes=0, start=0, goal=0, edges=set())
    with pytest.raises(ValueError):
        PathfindingEpisode(num_nodes=3, start=3, goal=2, edges=set())
    with pytest.raises(ValueError):
        PathfindingEpisode(num_nodes=3, start=-1, goal=2, edges=set())
    with pytest.raises(ValueError):
        PathfindingEpisode(num_nodes=3, start=0, goal=2, edges={(0, 0)})
    with pytest.raises(ValueError):
        PathfindingEpisode(num_nodes=3, start=0, goal=2, edges={(0, 1)}, blocked_edges={(0, 2)})
    with pytest.raises(ValueError):
        PathfindingEpisode(num_nodes=3, start=0, goal=2, edges={(0, 5)})
    with pytest.raises(ValueError):
        PathfindingEpisode(num_nodes=3, start=0, goal=2, edges={0})


def test_episode_is_immutable_and_defensively_copies_inputs() -> None:
    edges_source = {(0, 1), (1, 2)}
    blocked_source = {(1, 2)}
    episode = _episode(3, 0, 2, edges_source, blocked_source)

    edges_source.add((0, 2))
    blocked_source.clear()

    assert episode.edges == frozenset({(0, 1), (1, 2)})
    assert episode.blocked_edges == frozenset({(1, 2)})
    with pytest.raises(dataclasses.FrozenInstanceError):
        episode.start = 1


def test_episode_digest_is_stable_and_content_sensitive() -> None:
    base = _episode(4, 0, 3, {(0, 1), (1, 3)})
    same = _episode(4, 0, 3, {(0, 1), (1, 3)})
    extra_edge = _episode(4, 0, 3, {(0, 1), (1, 3), (0, 2)})
    blocked = _episode(4, 0, 3, {(0, 1), (1, 3)}, blocked={(0, 1)})

    assert base.digest() == same.digest()
    assert base.digest() != extra_edge.digest()
    assert base.digest() != blocked.digest()


# ---------------------------------------------------------------------------
# Generator configuration and deterministic generation
# ---------------------------------------------------------------------------

def test_generator_config_rejects_impossible_budgets() -> None:
    with pytest.raises(ValueError):
        PathfindingGeneratorConfig(num_nodes=12, num_extra_edges=2, num_blockades=5)
    with pytest.raises(ValueError):
        PathfindingGeneratorConfig(num_nodes=8, num_traps=3, trap_length=3)
    with pytest.raises(ValueError):
        PathfindingGeneratorConfig(num_nodes=10, num_traps=0)
    with pytest.raises(ValueError):
        PathfindingGeneratorConfig(num_nodes=12, num_extra_edges=26)
    with pytest.raises(ValueError):
        PathfindingGeneratorConfig(num_nodes=2)
    with pytest.raises(ValueError):
        PathfindingGeneratorConfig(max_generation_attempts=0)


def test_identical_seed_and_config_generate_identical_instances() -> None:
    config = _config()
    first = generate_pathfinding_episode(config, seed=7)
    second = generate_pathfinding_episode(config, seed=7)

    assert first == second
    assert first.digest() == second.digest()


def test_different_seeds_generate_distinct_valid_instances() -> None:
    config = _config()
    episodes = [generate_pathfinding_episode(config, seed=seed) for seed in range(10)]
    digests = {episode.digest() for episode in episodes}

    assert len(digests) >= 2
    for episode in episodes:
        assert episode.start != episode.goal
        assert len(episode.blocked_edges) == config.num_blockades


def test_generated_instances_are_internally_consistent() -> None:
    config = _config()
    for seed in (0, 1, 2, 42):
        episode = generate_pathfinding_episode(config, seed=seed)
        assert all(0 <= u < episode.num_nodes and 0 <= v < episode.num_nodes for u, v in episode.edges)
        assert all(u != v for u, v in episode.edges)
        assert episode.blocked_edges <= episode.edges
        assert len(episode.blocked_edges) == config.num_blockades
        for node in range(episode.num_nodes):
            derived = tuple(
                sorted(v for (u, v) in episode.edges if u == node and (u, v) not in episode.blocked_edges)
            )
            assert episode.allowed_moves(node) == derived


def test_generated_instances_are_solvable_with_valid_reference_path() -> None:
    episode_config = _config()
    for seed in range(5):
        episode = generate_pathfinding_episode(episode_config, seed=seed)
        solution = solve_shortest_path(episode)
        assert solution.solved
        assert solution.path[0] == episode.start
        assert solution.path[-1] == episode.goal
        assert validate_path(episode, solution.path).valid


def test_generated_start_branches_include_dead_end_and_alternate_route() -> None:
    episode = generate_pathfinding_episode(_config(), seed=11)
    moves = episode.allowed_moves(episode.start)

    assert len(moves) >= 2
    trapped = [node for node in moves if not goal_reachable_from(episode, node)]
    reachable = [node for node in moves if goal_reachable_from(episode, node)]
    assert trapped and reachable
    assert set(trapped).isdisjoint(reachable)


def test_walking_into_dead_end_gets_stuck_while_alternate_route_reaches_goal() -> None:
    episode = generate_pathfinding_episode(_config(), seed=11)
    state = episode.initial_state()
    trapped = [node for node in episode.allowed_moves(state.current) if not goal_reachable_from(episode, node)]
    assert trapped

    state = episode.apply_transition(state, PathfindingAction(current=state.current, target=trapped[0]))
    while not episode.is_stuck(state):
        moves = episode.allowed_moves(state.current)
        assert moves
        state = episode.apply_transition(state, PathfindingAction(current=state.current, target=moves[0]))

    assert episode.is_stuck(state)
    assert not episode.is_goal_reached(state)

    alternate = [node for node in episode.allowed_moves(episode.start) if goal_reachable_from(episode, node)]
    assert alternate
    assert solve_shortest_path(episode, start=alternate[0]).solved


def test_blockades_are_excluded_from_the_effective_graph() -> None:
    episode = generate_pathfinding_episode(_config(), seed=3)

    assert len(episode.blocked_edges) == 2
    for u, v in episode.blocked_edges:
        assert episode.has_transition(u, v)
        assert not episode.is_allowed_move(u, v)
        assert v not in episode.allowed_moves(u)


def test_generator_handles_multi_trap_and_full_blockade_configs() -> None:
    configurations = [
        PathfindingGeneratorConfig(num_nodes=12, num_extra_edges=1, num_traps=2, trap_length=2, num_blockades=1),
        PathfindingGeneratorConfig(num_nodes=9, num_extra_edges=2, num_traps=3, trap_length=1, num_blockades=2),
        PathfindingGeneratorConfig(num_nodes=6, num_extra_edges=4, num_traps=1, trap_length=3, num_blockades=4),
    ]
    for config in configurations:
        for seed in range(3):
            episode = generate_pathfinding_episode(config, seed=seed)
            solution = solve_shortest_path(episode)
            assert solution.solved
            assert validate_path(episode, solution.path).valid
            assert len(episode.blocked_edges) == config.num_blockades
            trapped = [
                node
                for node in episode.allowed_moves(episode.start)
                if not goal_reachable_from(episode, node)
            ]
            assert trapped


# ---------------------------------------------------------------------------
# Observation and state contracts
# ---------------------------------------------------------------------------

def test_observation_matches_the_effective_graph() -> None:
    episode = generate_pathfinding_episode(_config(), seed=5)
    observation = episode.observe(episode.initial_state())

    expected_adjacency = tuple(episode.allowed_moves(node) for node in range(episode.num_nodes))
    assert observation.effective_adjacency == expected_adjacency
    assert observation.allowed_moves == episode.allowed_moves(episode.start)
    assert observation.current == episode.start
    assert observation.goal == episode.goal
    assert isinstance(observation, PathfindingObservation)


def test_observation_reveals_only_the_public_surface_and_is_immutable() -> None:
    episode = generate_pathfinding_episode(_config(), seed=5)
    observation = episode.observe(episode.initial_state())

    field_names = tuple(field.name for field in dataclasses.fields(PathfindingObservation))
    assert field_names == ("current", "goal", "effective_adjacency")
    assert not hasattr(observation, "blocked_edges")
    assert not hasattr(observation, "edges")
    assert all(isinstance(row, tuple) for row in observation.effective_adjacency)
    with pytest.raises(TypeError):
        observation.effective_adjacency[0] = (99,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.current = 99


def test_state_helpers_and_terminal_conditions() -> None:
    episode = _episode(3, 0, 2, {(0, 1)})
    initial = episode.initial_state()
    assert initial == PathfindingState(current=0, transitions=0)

    stuck_state = PathfindingState(current=1)
    assert episode.is_stuck(stuck_state)
    assert not episode.is_goal_reached(stuck_state)

    assert not episode.is_stuck(initial)
    goal_state = PathfindingState(current=2)
    assert episode.is_goal_reached(goal_state)
    assert not episode.is_stuck(goal_state)


def test_apply_transition_advances_exactly_once() -> None:
    episode = _episode(5, 0, 4, {(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)})
    state = episode.initial_state()

    next_state = episode.apply_transition(state, PathfindingAction(current=0, target=1))
    assert next_state.current == 1 and next_state.transitions == 1
    assert state.current == 0 and state.transitions == 0

    final_state = episode.apply_transition(next_state, PathfindingAction(current=1, target=2))
    assert final_state.current == 2 and final_state.transitions == 2
    assert next_state.current == 1 and next_state.transitions == 1


def test_rejected_transitions_do_not_mutate_state_or_episode() -> None:
    episode = _episode(4, 0, 3, {(0, 1), (1, 3), (0, 2), (2, 1)}, blocked={(0, 1)})
    digest_before = episode.digest()
    state = PathfindingState(current=0)

    rejected = [
        PathfindingAction(current=2, target=1),
        PathfindingAction(current=0, target=9),
        PathfindingAction(current=0, target=3),
        PathfindingAction(current=0, target=1),
    ]
    for action in rejected:
        with pytest.raises(ValueError):
            episode.apply_transition(state, action)

    assert state == PathfindingState(current=0)
    assert episode.digest() == digest_before


# ---------------------------------------------------------------------------
# Reference oracle and independent path validation
# ---------------------------------------------------------------------------

def test_solver_finds_shortest_path_on_known_example() -> None:
    episode = _episode(5, 0, 4, {(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)})
    solution = solve_shortest_path(episode)

    assert solution.solved
    assert solution.path == (0, 3, 4)
    assert solution.num_moves == 2
    assert validate_path(episode, solution.path).valid


def test_solver_tie_break_is_deterministic() -> None:
    episode = _episode(5, 0, 4, {(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)})

    first = solve_shortest_path(episode)
    second = solve_shortest_path(episode)

    assert first.path == second.path == (0, 1, 3, 4)
    assert first.num_moves == 3


def test_solver_excludes_blocked_edges() -> None:
    detour = _episode(4, 0, 3, {(0, 1), (1, 3), (0, 2), (2, 1)}, blocked={(0, 1)})
    solution = solve_shortest_path(detour)
    assert solution.solved
    assert solution.path == (0, 2, 1, 3)
    assert validate_path(detour, solution.path).valid

    all_blocked = _episode(3, 0, 2, {(0, 1), (1, 2)}, blocked={(0, 1)})
    assert not solve_shortest_path(all_blocked).solved


def test_solver_handles_cycles_and_dead_ends() -> None:
    cyclic = _episode(4, 0, 3, {(0, 1), (1, 2), (2, 1), (2, 3)})
    solution = solve_shortest_path(cyclic)
    assert solution.solved and solution.num_moves == 3

    dead_end = _episode(4, 0, 3, {(0, 1), (0, 2), (2, 3)})
    assert solve_shortest_path(dead_end).path == (0, 2, 3)
    assert not goal_reachable_from(dead_end, 1)


def test_solver_reports_unreachable_goal_explicitly() -> None:
    episode = _episode(3, 0, 2, {(0, 1)})
    solution = solve_shortest_path(episode)

    assert solution.solved is False
    assert solution.path == ()
    assert solution.num_moves == 0


def test_solver_start_equals_goal_returns_zero_move_path() -> None:
    episode = _episode(2, 0, 0, set())
    solution = solve_shortest_path(episode)

    assert solution.solved
    assert solution.path == (0,)
    assert solution.num_moves == 0
    assert validate_path(episode, solution.path).valid


def test_solver_rejects_invalid_start_goal_arguments() -> None:
    episode = _episode(3, 0, 2, {(0, 1)})
    with pytest.raises(ValueError):
        solve_shortest_path(episode, start=99)
    with pytest.raises(ValueError):
        solve_shortest_path(episode, goal=-1)


def test_validate_path_rejects_broken_paths_with_reasons() -> None:
    episode = _episode(5, 0, 4, {(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)})
    blocked_episode = _episode(4, 0, 3, {(0, 1), (1, 3), (0, 2), (2, 1)}, blocked={(0, 1)})

    assert validate_path(episode, (0, 3, 4)).valid

    cases = [
        (episode, (), PATH_EMPTY),
        (episode, (3, 4), PATH_START_MISMATCH),
        (episode, (0, 3, 1), PATH_GOAL_MISMATCH),
        (episode, (0, 1, 4), TRANSITION_NO_SUCH_EDGE),
        (episode, (0, 99, 4), TRANSITION_UNKNOWN_NODE),
        (blocked_episode, (0, 1, 3), TRANSITION_BLOCKED),
    ]
    for target_episode, path, expected_reason in cases:
        validation = validate_path(target_episode, path)
        assert validation.valid is False
        assert validation.reason == expected_reason


def test_goal_reachable_from_agrees_with_solver() -> None:
    episode = generate_pathfinding_episode(_config(), seed=9)
    for node in range(episode.num_nodes):
        assert goal_reachable_from(episode, node) == solve_shortest_path(episode, start=node).solved


# ---------------------------------------------------------------------------
# Splits and resource behavior
# ---------------------------------------------------------------------------

def test_splits_are_reproducible_disjoint_and_well_sized() -> None:
    config = _config()
    splits = generate_pathfinding_splits(4, 3, 3, config, base_seed=123)

    assert len(splits.train) == 4
    assert len(splits.validation) == 3
    assert len(splits.test) == 3
    assert splits.config == config
    assert splits.base_seed == 123

    train_digests = [episode.digest() for episode in splits.train]
    validation_digests = [episode.digest() for episode in splits.validation]
    test_digests = [episode.digest() for episode in splits.test]
    assert len(set(train_digests)) == 4
    assert len(set(validation_digests)) == 3
    assert len(set(test_digests)) == 3
    assert set(train_digests).isdisjoint(validation_digests)
    assert set(train_digests).isdisjoint(test_digests)
    assert set(validation_digests).isdisjoint(test_digests)

    repeated = generate_pathfinding_splits(4, 3, 3, config, base_seed=123)
    assert repeated == splits


def test_split_generation_fails_explicitly_when_instance_space_is_too_small() -> None:
    tiny = PathfindingGeneratorConfig(
        num_nodes=3, num_extra_edges=0, num_traps=0, trap_length=1, num_blockades=0
    )
    with pytest.raises(RuntimeError, match="bounded attempts"):
        generate_pathfinding_splits(50, 1, 1, tiny, base_seed=1)


def test_repeated_construction_does_not_retain_references() -> None:
    config = _config()
    references: list[weakref.ReferenceType] = []

    for seed in range(300):
        episode = generate_pathfinding_episode(config, seed=seed)
        observation = episode.observe(episode.initial_state())
        references.append(weakref.ref(episode))
        references.append(weakref.ref(observation))

    del episode, observation
    gc.collect()

    alive = [reference for reference in references if reference() is not None]
    assert alive == [], f"{len(alive)} pathfinding objects outlived their generation scope"
    assert len(gc.garbage) == 0
