"""Tests for MCTS search engine (A1-A8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcts.search import (
    MCTSConfig,
    MCTSNode,
    MCTSSearch,
    _terminal_values,
    clone_game_state,
)
from monopoly_engine.actions import RollDice
from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_root(player_to_move: int = 0) -> MCTSNode:
    """Create a root node with a minimal state dict."""
    game = MonopolyGame(num_players=4, seed=42)
    return MCTSNode(
        state_dict=game.to_dict(),
        player_to_move=player_to_move,
    )


def _add_child(
    parent: MCTSNode,
    action: int,
    prior: float = 0.5,
    visit_count: int = 0,
    total_value: dict[int, float] | None = None,
) -> MCTSNode:
    """Add a child node with given stats."""
    child = MCTSNode(
        state_dict=parent.state_dict,
        parent=parent,
        action=action,
        prior=prior,
        player_to_move=0,
    )
    child.visit_count = visit_count
    if total_value is not None:
        child.total_value = total_value
    parent.children[action] = child
    return child


# ===========================================================================
# A1: MCTSNode tests
# ===========================================================================


class TestMCTSNodeCreation:
    """Test node initialization and basic properties."""

    def test_node_creation_defaults(self) -> None:
        """Verify initial state of a freshly created node."""
        game = MonopolyGame(num_players=4, seed=42)
        node = MCTSNode(state_dict=game.to_dict())

        assert node.parent is None
        assert node.action is None
        assert node.children == {}
        assert node.visit_count == 0
        assert node.total_value == {}
        assert node.prior == 1.0
        assert node.player_to_move == 0
        assert node.is_terminal is False

    def test_node_creation_with_parent(self) -> None:
        """Verify child node links to parent correctly."""
        root = _make_root()
        child = MCTSNode(
            state_dict=root.state_dict,
            parent=root,
            action=5,
            prior=0.3,
            player_to_move=1,
            is_terminal=False,
        )

        assert child.parent is root
        assert child.action == 5
        assert child.prior == 0.3
        assert child.player_to_move == 1

    def test_node_creation_terminal(self) -> None:
        """Terminal nodes should be marked as such."""
        node = MCTSNode(
            state_dict={},
            is_terminal=True,
        )
        assert node.is_terminal is True

    def test_node_uses_slots(self) -> None:
        """MCTSNode should use __slots__ for memory efficiency."""
        node = MCTSNode(state_dict={})
        assert hasattr(node, "__slots__")
        with pytest.raises(AttributeError):
            node.nonexistent_attr = 42  # type: ignore[attr-defined]


class TestMCTSNodeStructure:
    """Test is_leaf and is_root."""

    def test_is_root_true_for_root(self) -> None:
        root = _make_root()
        assert root.is_root() is True

    def test_is_root_false_for_child(self) -> None:
        root = _make_root()
        child = _add_child(root, action=0)
        assert child.is_root() is False

    def test_is_leaf_true_when_no_children(self) -> None:
        root = _make_root()
        assert root.is_leaf() is True

    def test_is_leaf_false_when_has_children(self) -> None:
        root = _make_root()
        _add_child(root, action=0)
        assert root.is_leaf() is False


class TestUCB1Score:
    """Test PUCT score computation."""

    def test_ucb1_unvisited_returns_infinity(self) -> None:
        """Unvisited nodes should return inf to ensure exploration."""
        root = _make_root()
        root.visit_count = 10
        child = _add_child(root, action=0, prior=0.5, visit_count=0)
        score = child.ucb1_score(exploration_constant=1.41, player_id=0)
        assert score == float("inf")

    def test_ucb1_visited_node_finite(self) -> None:
        """Visited nodes should return a finite score."""
        root = _make_root()
        root.visit_count = 10
        child = _add_child(
            root,
            action=0,
            prior=0.5,
            visit_count=5,
            total_value={0: 2.5},
        )
        score = child.ucb1_score(exploration_constant=1.41, player_id=0)
        assert math.isfinite(score)

    def test_ucb1_formula_correctness(self) -> None:
        """Verify the PUCT formula: Q + c * P * sqrt(N_parent) / (1 + N_child)."""
        root = _make_root()
        root.visit_count = 100

        child = _add_child(
            root,
            action=0,
            prior=0.25,
            visit_count=20,
            total_value={0: 10.0},
        )

        c = 1.41
        expected_q = 10.0 / 20  # = 0.5
        expected_explore = c * 0.25 * math.sqrt(100) / (1 + 20)
        expected = expected_q + expected_explore

        score = child.ucb1_score(exploration_constant=c, player_id=0)
        assert abs(score - expected) < 1e-10

    def test_ucb1_higher_exploration_favors_less_visited(self) -> None:
        """Higher exploration constant should favor less-visited nodes more."""
        root = _make_root()
        root.visit_count = 50

        # Frequently visited child with moderate value
        child_a = _add_child(
            root,
            action=0,
            prior=0.5,
            visit_count=30,
            total_value={0: 18.0},  # Q = 0.6
        )
        # Rarely visited child with lower value
        child_b = _add_child(
            root,
            action=1,
            prior=0.5,
            visit_count=3,
            total_value={0: 1.0},  # Q = 0.33
        )

        # With low exploration, high-value child should win
        score_a_low = child_a.ucb1_score(exploration_constant=0.1, player_id=0)
        score_b_low = child_b.ucb1_score(exploration_constant=0.1, player_id=0)

        # With high exploration, less-visited child should win
        score_a_high = child_a.ucb1_score(exploration_constant=5.0, player_id=0)
        score_b_high = child_b.ucb1_score(exploration_constant=5.0, player_id=0)

        # With low exploration: child_a (Q=0.6) > child_b (Q=0.33)
        assert score_a_low > score_b_low
        # With high exploration: child_b (less visited) should be preferred
        assert score_b_high > score_a_high

    def test_ucb1_missing_player_id_treated_as_zero(self) -> None:
        """If a player has no value entries, Q should be 0."""
        root = _make_root()
        root.visit_count = 10
        child = _add_child(
            root,
            action=0,
            prior=0.5,
            visit_count=5,
            total_value={1: 3.0},  # Only player 1 has value
        )
        score = child.ucb1_score(exploration_constant=1.41, player_id=0)
        # Q = 0 for player 0
        expected_explore = 1.41 * 0.5 * math.sqrt(10) / (1 + 5)
        assert abs(score - expected_explore) < 1e-10

    def test_ucb1_multi_player_uses_correct_perspective(self) -> None:
        """UCB1 should use the value for the specified player."""
        root = _make_root()
        root.visit_count = 20
        child = _add_child(
            root,
            action=0,
            prior=0.5,
            visit_count=10,
            total_value={0: 5.0, 1: -5.0, 2: 3.0, 3: -3.0},
        )

        score_p0 = child.ucb1_score(exploration_constant=1.41, player_id=0)
        score_p1 = child.ucb1_score(exploration_constant=1.41, player_id=1)

        # Player 0 has positive Q, player 1 has negative Q
        assert score_p0 > score_p1


class TestBestChild:
    """Test best_child selection."""

    def test_best_child_selects_highest_ucb(self) -> None:
        """best_child should pick the child with highest UCB score."""
        root = _make_root()
        root.visit_count = 30

        _add_child(root, action=0, prior=0.3, visit_count=15, total_value={0: 3.0})
        best = _add_child(root, action=1, prior=0.3, visit_count=5, total_value={0: 4.0})
        _add_child(root, action=2, prior=0.3, visit_count=10, total_value={0: 1.0})

        result = root.best_child(exploration_constant=1.41, player_id=0)
        assert result is best

    def test_best_child_prefers_unvisited(self) -> None:
        """Unvisited children (inf score) should always be selected."""
        root = _make_root()
        root.visit_count = 10

        _add_child(root, action=0, prior=0.5, visit_count=5, total_value={0: 3.0})
        unvisited = _add_child(root, action=1, prior=0.5, visit_count=0)

        result = root.best_child(exploration_constant=1.41, player_id=0)
        assert result is unvisited

    def test_best_child_no_children_raises(self) -> None:
        """Calling best_child on a leaf should raise ValueError."""
        root = _make_root()
        with pytest.raises(ValueError, match="no children"):
            root.best_child(exploration_constant=1.41, player_id=0)

    def test_best_child_single_child(self) -> None:
        """With one child, it must be selected."""
        root = _make_root()
        root.visit_count = 5
        only_child = _add_child(root, action=42, prior=1.0, visit_count=3)
        result = root.best_child(exploration_constant=1.41, player_id=0)
        assert result is only_child


# ===========================================================================
# A2: State cloning tests
# ===========================================================================


class TestCloneGameState:
    """Test game state cloning for MCTS simulations."""

    def test_clone_state_matches_original(self) -> None:
        """Cloned game should have identical state fields."""
        game = MonopolyGame(num_players=4, seed=42)
        clone = clone_game_state(game, new_seed=99)

        orig_dict = game.to_dict()
        clone_dict = clone.to_dict()

        # Core state fields should match
        assert clone_dict["current_player"] == orig_dict["current_player"]
        assert clone_dict["turn_number"] == orig_dict["turn_number"]
        assert clone_dict["game_over"] == orig_dict["game_over"]
        assert len(clone_dict["players"]) == len(orig_dict["players"])

        # Player states should match
        for i in range(len(orig_dict["players"])):
            assert clone_dict["players"][i]["money"] == orig_dict["players"][i]["money"]
            assert clone_dict["players"][i]["position"] == orig_dict["players"][i]["position"]

    def test_clone_rng_independent_with_seed(self) -> None:
        """Clones with different seeds should produce different dice rolls."""
        game = MonopolyGame(num_players=2, seed=42)
        clone_a = clone_game_state(game, new_seed=100)
        clone_b = clone_game_state(game, new_seed=200)

        rolls_a = [clone_a.rng.randint(1, 6) for _ in range(20)]
        rolls_b = [clone_b.rng.randint(1, 6) for _ in range(20)]

        # With different seeds, rolls should differ
        assert rolls_a != rolls_b

    def test_clone_rng_independent_no_seed(self) -> None:
        """Clones without a seed should get random (non-deterministic) RNG."""
        game = MonopolyGame(num_players=2, seed=42)
        clone = clone_game_state(game, new_seed=None)

        # Just verify the clone has an RNG that works
        roll = clone.rng.randint(1, 6)
        assert 1 <= roll <= 6

    def test_clone_does_not_consume_original_rng(self) -> None:
        """Cloning should not advance the original game's RNG state."""
        game = MonopolyGame(num_players=2, seed=42)

        # Record original RNG state
        expected_roll = game.rng.randint(1, 6)

        # Reset and clone
        game2 = MonopolyGame(num_players=2, seed=42)
        _ = clone_game_state(game2, new_seed=99)
        actual_roll = game2.rng.randint(1, 6)

        assert actual_roll == expected_roll

    def test_clone_mutation_independent(self) -> None:
        """Modifying clone state should not affect original."""
        game = MonopolyGame(num_players=4, seed=42)
        original_money = game.players[0].money

        clone = clone_game_state(game, new_seed=99)
        clone.players[0].money = 0

        # Original should be unaffected
        assert game.players[0].money == original_money

    def test_clone_preserves_game_over(self) -> None:
        """A game-over state should be preserved in the clone."""
        game = MonopolyGame(num_players=2, seed=42)
        # Force game over by bankrupting player 1
        game.state.game_over = True
        game.state.winner = 0

        clone = clone_game_state(game, new_seed=99)
        assert clone.game_over is True
        assert clone.winner == 0

    def test_clone_same_seed_produces_same_rolls(self) -> None:
        """Two clones with the same seed should produce identical rolls."""
        game = MonopolyGame(num_players=2, seed=42)
        clone_a = clone_game_state(game, new_seed=123)
        clone_b = clone_game_state(game, new_seed=123)

        rolls_a = [clone_a.rng.randint(1, 6) for _ in range(20)]
        rolls_b = [clone_b.rng.randint(1, 6) for _ in range(20)]

        assert rolls_a == rolls_b


# ===========================================================================
# A7: MCTSConfig tests
# ===========================================================================


class TestMCTSConfig:
    """Test MCTS configuration."""

    def test_config_defaults(self) -> None:
        """Verify all default values are set correctly."""
        config = MCTSConfig()
        assert config.num_simulations == 200
        assert config.exploration_constant == 1.41
        assert config.max_rollout_depth == 500
        assert config.use_value_network is False
        assert config.temperature == 0.0
        assert config.opponent_policy == "rule_based"
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25

    def test_config_custom_values(self) -> None:
        """Config should accept custom values."""
        config = MCTSConfig(
            num_simulations=500,
            exploration_constant=2.0,
            max_rollout_depth=300,
            use_value_network=True,
            temperature=1.0,
            opponent_policy="random",
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
        )
        assert config.num_simulations == 500
        assert config.exploration_constant == 2.0
        assert config.use_value_network is True
        assert config.opponent_policy == "random"

    def test_config_serialization_roundtrip(self) -> None:
        """Config should survive dict serialization and deserialization."""
        config = MCTSConfig(
            num_simulations=300,
            exploration_constant=2.5,
            temperature=0.5,
        )
        data = config.to_dict()
        restored = MCTSConfig.from_dict(data)

        assert restored.num_simulations == config.num_simulations
        assert restored.exploration_constant == config.exploration_constant
        assert restored.temperature == config.temperature
        assert restored.max_rollout_depth == config.max_rollout_depth
        assert restored.opponent_policy == config.opponent_policy

    def test_config_to_dict_keys(self) -> None:
        """to_dict should produce all expected keys."""
        config = MCTSConfig()
        data = config.to_dict()

        expected_keys = {
            "num_simulations",
            "exploration_constant",
            "max_rollout_depth",
            "use_value_network",
            "temperature",
            "opponent_policy",
            "dirichlet_alpha",
            "dirichlet_epsilon",
        }
        assert set(data.keys()) == expected_keys


# ===========================================================================
# A3: Expansion tests
# ===========================================================================


def _make_game_after_roll(seed: int = 42, num_players: int = 4) -> MonopolyGame:
    """Create a game where player 0 has already rolled dice.

    The action mask requires a player to have rolled before choosing
    post-roll actions (buy, end turn, etc.).
    """
    game = MonopolyGame(num_players=num_players, seed=seed)
    roll = RollDice(player_id=0)
    valid, _ = roll.validate(game)
    if valid:
        roll.execute(game)
    return game


class TestMCTSExpansion:
    """Test MCTSSearch.expand() method."""

    def test_expand_creates_children_for_valid_actions(self) -> None:
        """Expansion should create one child per valid action."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)
        encoder = ActionEncoder(enable_trades=False)

        mask = encoder.get_action_mask(game, game.current_player)
        num_valid = int(mask.sum())

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        assert len(node.children) == num_valid
        assert num_valid > 0  # At minimum EndTurn should be valid

    def test_expand_child_actions_match_valid_actions(self) -> None:
        """Child action indices should match the valid action mask."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)
        encoder = ActionEncoder(enable_trades=False)

        mask = encoder.get_action_mask(game, game.current_player)
        valid_actions = set(np.where(mask)[0].tolist())

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        assert set(node.children.keys()) == valid_actions

    def test_expand_child_states_differ(self) -> None:
        """Each child should have a different state (different action applied)."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        if len(node.children) >= 2:
            states = [child.state_dict for child in node.children.values()]
            # At least some states should differ
            all_same = all(s == states[0] for s in states[1:])
            assert not all_same

    def test_expand_uniform_priors(self) -> None:
        """Without policy network, priors should be uniform and sum to ~1.0."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        priors = [child.prior for child in node.children.values()]
        assert len(priors) > 0

        # All priors should be equal (uniform)
        assert all(abs(p - priors[0]) < 1e-10 for p in priors)

        # Priors should sum to ~1.0
        assert abs(sum(priors) - 1.0) < 1e-10

    def test_expand_with_policy_priors(self) -> None:
        """With policy priors, child priors should match the provided values."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)
        encoder = ActionEncoder(enable_trades=False)

        mask = encoder.get_action_mask(game, game.current_player)
        valid_actions = np.where(mask)[0]

        # Create fake policy priors
        policy = np.zeros(149, dtype=np.float32)
        for i, a in enumerate(valid_actions):
            policy[a] = i + 1.0
        policy /= policy.sum()  # normalize

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game, policy_priors=policy)

        for action_idx, child in node.children.items():
            assert abs(child.prior - float(policy[action_idx])) < 1e-6

    def test_expand_terminal_node_not_expanded(self) -> None:
        """Terminal nodes should not be expanded."""
        game = MonopolyGame(num_players=2, seed=42)
        game.state.game_over = True
        game.state.winner = 0

        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=0,
            is_terminal=True,
        )
        search.expand(node, game)

        assert len(node.children) == 0

    def test_expand_children_have_correct_parent(self) -> None:
        """All children should reference the expanded node as parent."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        for child in node.children.values():
            assert child.parent is node

    def test_expand_children_have_zero_visits(self) -> None:
        """Newly expanded children should have zero visits."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        for child in node.children.values():
            assert child.visit_count == 0
            assert child.total_value == {}

    def test_expand_node_no_longer_leaf(self) -> None:
        """After expansion, the node should no longer be a leaf."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        assert node.is_leaf()
        search.expand(node, game)
        assert not node.is_leaf()

    def test_expand_does_not_mutate_input_game(self) -> None:
        """Expansion should not modify the input game state."""
        game = _make_game_after_roll()
        original_dict = game.to_dict()
        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        # Original game should be unchanged
        assert game.to_dict() == original_dict

    def test_expand_end_turn_advances_player(self) -> None:
        """When EndTurn is a valid action, its child should have a different player."""
        game = _make_game_after_roll()
        config = MCTSConfig()
        search = MCTSSearch(config)

        node = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(node, game)

        # EndTurn is action index 146
        if 146 in node.children:
            end_turn_child = node.children[146]
            assert end_turn_child.player_to_move != node.player_to_move


# ===========================================================================
# A4: Simulation / Rollout tests
# ===========================================================================


class TestTerminalValues:
    """Test _terminal_values helper function."""

    def test_winner_gets_positive(self) -> None:
        """Winner should get +1.0, all others -1.0."""
        game = MonopolyGame(num_players=4, seed=42)
        game.state.game_over = True
        game.state.winner = 2

        values = _terminal_values(game)
        assert values[2] == 1.0
        assert values[0] == -1.0
        assert values[1] == -1.0
        assert values[3] == -1.0

    def test_truncated_uses_net_worth_ranking(self) -> None:
        """Without a winner, values should be based on net worth ranking."""
        game = MonopolyGame(num_players=4, seed=42)
        # Give players different amounts of money to create a ranking
        game.players[0].money = 100
        game.players[1].money = 500
        game.players[2].money = 300
        game.players[3].money = 1000

        values = _terminal_values(game)

        # Player 3 (most money) should be highest
        assert values[3] == 1.0
        # Player 0 (least money) should be lowest
        assert values[0] == -1.0
        # Middle players should be intermediate
        assert -1.0 < values[1] < 1.0
        assert -1.0 < values[2] < 1.0
        # Player 1 ($500) should be ranked above player 2 ($300)
        assert values[1] > values[2]

    def test_all_values_bounded(self) -> None:
        """All values should be in [-1, 1]."""
        game = MonopolyGame(num_players=4, seed=42)
        values = _terminal_values(game)

        for v in values.values():
            assert -1.0 <= v <= 1.0

    def test_all_players_have_values(self) -> None:
        """Every player should have a value entry."""
        game = MonopolyGame(num_players=4, seed=42)
        values = _terminal_values(game)
        assert set(values.keys()) == {0, 1, 2, 3}

    def test_two_player_game(self) -> None:
        """Terminal values should work for 2-player games."""
        game = MonopolyGame(num_players=2, seed=42)
        game.state.game_over = True
        game.state.winner = 0

        values = _terminal_values(game)
        assert values[0] == 1.0
        assert values[1] == -1.0


class TestRandomRollout:
    """Test MCTSSearch._random_rollout() method."""

    def test_random_rollout_terminates(self) -> None:
        """Rollout should terminate within max_depth actions."""
        game = MonopolyGame(num_players=2, seed=42)
        # Roll dice first so the game is in a valid action state
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(max_rollout_depth=50)
        search = MCTSSearch(config)

        clone = clone_game_state(game, new_seed=123)
        values = search._random_rollout(clone, max_depth=50)

        # Should return values for all players
        assert len(values) == 2
        assert 0 in values
        assert 1 in values

    def test_random_rollout_values_bounded(self) -> None:
        """All rollout values should be in [-1, 1]."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(max_rollout_depth=100)
        search = MCTSSearch(config)

        clone = clone_game_state(game, new_seed=456)
        values = search._random_rollout(clone, max_depth=100)

        for v in values.values():
            assert -1.0 <= v <= 1.0

    def test_random_rollout_different_seeds_different_results(self) -> None:
        """Rollouts with different RNG should produce different trajectories."""
        game = MonopolyGame(num_players=2, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(max_rollout_depth=100)
        search = MCTSSearch(config)

        results = []
        for seed in range(10):
            clone = clone_game_state(game, new_seed=seed)
            values = search._random_rollout(clone, max_depth=100)
            results.append(values[0])

        # Not all results should be identical (different RNG seeds)
        assert len(set(results)) > 1

    def test_random_rollout_completed_game_returns_winner(self) -> None:
        """If game completes naturally, winner should get +1."""
        game = MonopolyGame(num_players=2, seed=42)
        game.state.game_over = True
        game.state.winner = 1

        config = MCTSConfig()
        search = MCTSSearch(config)

        values = search._random_rollout(game, max_depth=100)
        assert values[1] == 1.0
        assert values[0] == -1.0


class TestSimulate:
    """Test MCTSSearch.simulate() dispatch method."""

    def test_simulate_uses_rollout_by_default(self) -> None:
        """Without value network, simulate should use random rollout."""
        game = MonopolyGame(num_players=2, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(use_value_network=False, max_rollout_depth=50)
        search = MCTSSearch(config)

        clone = clone_game_state(game, new_seed=789)
        values = search.simulate(clone, player_id=0)

        assert len(values) == 2
        for v in values.values():
            assert -1.0 <= v <= 1.0

    def test_simulate_returns_terminal_values_for_finished_game(self) -> None:
        """If game is already over, simulate returns terminal values directly."""
        game = MonopolyGame(num_players=4, seed=42)
        game.state.game_over = True
        game.state.winner = 0

        config = MCTSConfig()
        search = MCTSSearch(config)

        values = search.simulate(game, player_id=0)
        assert values[0] == 1.0
        assert values[1] == -1.0
        assert values[2] == -1.0
        assert values[3] == -1.0

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_simulate_with_value_network_flag_but_no_network(self) -> None:
        """With use_value_network=True but no network, should fall back to rollout."""
        game = MonopolyGame(num_players=2, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(use_value_network=True, max_rollout_depth=50)
        search = MCTSSearch(config, value_network=None)

        clone = clone_game_state(game, new_seed=111)
        values = search.simulate(clone, player_id=0)

        # Should fall back to rollout since network is None
        assert len(values) == 2

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_simulate_dispatches_to_value_network(self) -> None:
        """With use_value_network=True and a network, should use network eval."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(use_value_network=True)
        # Use a sentinel object as "network" to verify dispatch
        fake_network = object()
        search = MCTSSearch(config, value_network=fake_network)

        clone = clone_game_state(game, new_seed=222)
        values = search.simulate(clone, player_id=0)

        # _value_network_evaluate returns zeros for now
        assert len(values) == 4
        for v in values.values():
            assert v == 0.0


# ===========================================================================
# A5: Backpropagation tests
# ===========================================================================


class TestBackpropagate:
    """Test MCTSSearch.backpropagate() method."""

    def test_backprop_updates_visit_counts(self) -> None:
        """Visit counts should be incremented for all ancestors up to root."""
        root = _make_root()
        child = _add_child(root, action=0)
        grandchild = _add_child(child, action=1)

        config = MCTSConfig()
        search = MCTSSearch(config)

        search.backpropagate(grandchild, {0: 1.0, 1: -1.0})

        assert grandchild.visit_count == 1
        assert child.visit_count == 1
        assert root.visit_count == 1

    def test_backprop_accumulates_visit_counts(self) -> None:
        """Multiple backpropagations should accumulate visit counts."""
        root = _make_root()
        child = _add_child(root, action=0)

        config = MCTSConfig()
        search = MCTSSearch(config)

        search.backpropagate(child, {0: 1.0})
        search.backpropagate(child, {0: -1.0})
        search.backpropagate(child, {0: 0.5})

        assert child.visit_count == 3
        assert root.visit_count == 3

    def test_backprop_updates_values(self) -> None:
        """Values should be accumulated correctly at each node."""
        root = _make_root()
        child = _add_child(root, action=0)

        config = MCTSConfig()
        search = MCTSSearch(config)

        search.backpropagate(child, {0: 1.0, 1: -1.0})

        assert child.total_value[0] == 1.0
        assert child.total_value[1] == -1.0
        assert root.total_value[0] == 1.0
        assert root.total_value[1] == -1.0

    def test_backprop_accumulates_values(self) -> None:
        """Multiple backpropagations should sum values."""
        root = _make_root()
        child = _add_child(root, action=0)

        config = MCTSConfig()
        search = MCTSSearch(config)

        search.backpropagate(child, {0: 1.0, 1: -1.0})
        search.backpropagate(child, {0: 0.5, 1: -0.5})

        assert abs(child.total_value[0] - 1.5) < 1e-10
        assert abs(child.total_value[1] - (-1.5)) < 1e-10
        assert abs(root.total_value[0] - 1.5) < 1e-10
        assert abs(root.total_value[1] - (-1.5)) < 1e-10

    def test_backprop_multi_player_independent(self) -> None:
        """Per-player values should be tracked independently."""
        root = _make_root()
        child = _add_child(root, action=0)

        config = MCTSConfig()
        search = MCTSSearch(config)

        search.backpropagate(child, {0: 1.0, 1: -1.0, 2: 0.5, 3: -0.5})

        assert child.total_value[0] == 1.0
        assert child.total_value[1] == -1.0
        assert child.total_value[2] == 0.5
        assert child.total_value[3] == -0.5

    def test_backprop_deep_tree(self) -> None:
        """Backpropagation should work through a deep tree (5 levels)."""
        root = _make_root()
        nodes = [root]
        for i in range(4):
            child = _add_child(nodes[-1], action=i)
            nodes.append(child)

        config = MCTSConfig()
        search = MCTSSearch(config)

        leaf = nodes[-1]
        search.backpropagate(leaf, {0: 1.0})

        # All 5 nodes should have been updated
        for node in nodes:
            assert node.visit_count == 1
            assert node.total_value[0] == 1.0

    def test_backprop_sibling_independence(self) -> None:
        """Backpropagation through one child should not affect siblings."""
        root = _make_root()
        child_a = _add_child(root, action=0)
        child_b = _add_child(root, action=1)

        config = MCTSConfig()
        search = MCTSSearch(config)

        search.backpropagate(child_a, {0: 1.0})

        # child_b should be unaffected
        assert child_b.visit_count == 0
        assert child_b.total_value == {}

        # root should be updated (it's an ancestor of child_a)
        assert root.visit_count == 1
        assert root.total_value[0] == 1.0

    def test_backprop_root_only(self) -> None:
        """Backpropagating from root should update only the root."""
        root = _make_root()

        config = MCTSConfig()
        search = MCTSSearch(config)

        search.backpropagate(root, {0: 0.5, 1: -0.5})

        assert root.visit_count == 1
        assert root.total_value[0] == 0.5
        assert root.total_value[1] == -0.5


# ===========================================================================
# A6: Opponent Modeling tests
# ===========================================================================


class TestOpponentModeling:
    """Test MCTSSearch._get_opponent_action() method."""

    def test_opponent_random_returns_valid_action(self) -> None:
        """Random opponent policy should return a valid action index."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(opponent_policy="random")
        search = MCTSSearch(config)
        encoder = ActionEncoder(enable_trades=False)

        mask = encoder.get_action_mask(game, 0)
        action = search._get_opponent_action(game, 0)

        assert mask[action], f"Action {action} is not valid"

    def test_opponent_rule_based_returns_valid_action(self) -> None:
        """Rule-based opponent policy should return a valid action index."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(opponent_policy="rule_based")
        search = MCTSSearch(config)
        encoder = ActionEncoder(enable_trades=False)

        mask = encoder.get_action_mask(game, 0)
        action = search._get_opponent_action(game, 0)

        assert mask[action], f"Action {action} is not valid"

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_opponent_network_returns_valid_action(self) -> None:
        """Network opponent policy (placeholder) should return a valid action."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(opponent_policy="network")
        search = MCTSSearch(config)
        encoder = ActionEncoder(enable_trades=False)

        mask = encoder.get_action_mask(game, 0)
        action = search._get_opponent_action(game, 0)

        assert mask[action], f"Action {action} is not valid"

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_opponent_policy_config_respected(self) -> None:
        """Different policies should be usable via config."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        for policy in ("random", "rule_based", "network"):
            config = MCTSConfig(opponent_policy=policy)
            search = MCTSSearch(config)
            action = search._get_opponent_action(game, 0)
            # All should return a valid int action
            assert isinstance(action, int)
            assert 0 <= action < 158

    def test_opponent_rule_based_caches_agents(self) -> None:
        """Rule-based agents should be cached per player_id."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        config = MCTSConfig(opponent_policy="rule_based")
        search = MCTSSearch(config)

        search._get_opponent_action(game, 0)
        search._get_opponent_action(game, 0)

        # Should have created exactly one agent for player 0
        assert 0 in search._opponent_agents
        assert len(search._opponent_agents) == 1

    def test_rollout_uses_opponent_policy(self) -> None:
        """The rollout should use the configured opponent policy (smoke test)."""
        game = MonopolyGame(num_players=2, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        # Run rollout with rule_based policy - should complete without error
        config = MCTSConfig(opponent_policy="rule_based", max_rollout_depth=50)
        search = MCTSSearch(config)
        clone = clone_game_state(game, new_seed=42)
        values = search._random_rollout(clone, max_depth=50)

        assert len(values) == 2
        for v in values.values():
            assert -1.0 <= v <= 1.0


# ===========================================================================
# A8: Action Selection and Main Search Loop tests
# ===========================================================================


class TestSearch:
    """Test MCTSSearch.search() main entry point."""

    def test_search_returns_visit_counts(self) -> None:
        """search() should return a non-empty dict with valid action keys."""
        game = _make_game_after_roll()
        config = MCTSConfig(
            num_simulations=10,
            max_rollout_depth=20,
            opponent_policy="random",
            dirichlet_alpha=0.0,  # Disable noise for determinism
        )
        search = MCTSSearch(config)

        visit_counts = search.search(game, player_id=0)

        assert len(visit_counts) > 0
        # All keys should be valid action indices
        encoder = ActionEncoder(enable_trades=False)
        mask = encoder.get_action_mask(game, game.current_player)
        for action_idx in visit_counts:
            assert mask[action_idx], f"Action {action_idx} is not valid"

    def test_search_visit_counts_sum(self) -> None:
        """Total visit counts should equal num_simulations."""
        game = _make_game_after_roll()
        num_sims = 15
        config = MCTSConfig(
            num_simulations=num_sims,
            max_rollout_depth=20,
            opponent_policy="random",
            dirichlet_alpha=0.0,
        )
        search = MCTSSearch(config)

        visit_counts = search.search(game, player_id=0)
        total = sum(visit_counts.values())

        assert total == num_sims

    def test_search_does_not_mutate_input_game(self) -> None:
        """search() should not modify the input game state."""
        game = _make_game_after_roll()
        original_dict = game.to_dict()

        config = MCTSConfig(
            num_simulations=5,
            max_rollout_depth=20,
            opponent_policy="random",
            dirichlet_alpha=0.0,
        )
        search = MCTSSearch(config)
        search.search(game, player_id=0)

        assert game.to_dict() == original_dict

    def test_search_empty_for_terminal_game(self) -> None:
        """search() on a terminal game should return empty visit counts."""
        game = MonopolyGame(num_players=2, seed=42)
        game.state.game_over = True
        game.state.winner = 0

        config = MCTSConfig(num_simulations=10, dirichlet_alpha=0.0)
        search = MCTSSearch(config)

        # After expansion the root will have no children (terminal)
        visit_counts = search.search(game, player_id=0)
        assert visit_counts == {}

    def test_search_with_dirichlet_noise(self) -> None:
        """search() with Dirichlet noise should still complete correctly."""
        game = _make_game_after_roll()
        config = MCTSConfig(
            num_simulations=10,
            max_rollout_depth=20,
            opponent_policy="random",
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )
        search = MCTSSearch(config)

        visit_counts = search.search(game, player_id=0)
        assert len(visit_counts) > 0
        assert sum(visit_counts.values()) == 10


class TestSelect:
    """Test MCTSSearch.select() tree traversal."""

    def test_select_returns_leaf(self) -> None:
        """select() should return a leaf node."""
        game = _make_game_after_roll()
        config = MCTSConfig(dirichlet_alpha=0.0)
        search = MCTSSearch(config)

        root = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(root, game)

        # With all children unvisited (inf score), select should pick one
        leaf, leaf_game = search.select(root)
        assert leaf.is_leaf()
        assert leaf_game is not None

    def test_select_returns_root_if_leaf(self) -> None:
        """If root is a leaf, select() should return the root itself."""
        game = MonopolyGame(num_players=2, seed=42)
        config = MCTSConfig()
        search = MCTSSearch(config)

        root = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        # Don't expand -- root is a leaf

        leaf, leaf_game = search.select(root)
        assert leaf is root

    def test_select_traverses_to_depth(self) -> None:
        """After visiting children, select should go deeper into the tree."""
        game = _make_game_after_roll()
        config = MCTSConfig(
            num_simulations=5,
            max_rollout_depth=20,
            opponent_policy="random",
            dirichlet_alpha=0.0,
        )
        search = MCTSSearch(config)

        # Run a few simulations to build up the tree
        root = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(root, game)

        # Manually visit some children to test deeper selection
        for child in root.children.values():
            child_game = MonopolyGame.from_dict(child.state_dict)
            child_game.rng = __import__("random").Random()
            search.expand(child, child_game)
            child.visit_count = 1
            child.total_value = {0: 0.5}
            break

        root.visit_count = 1

        # Now select should go deeper (into grandchildren)
        leaf, _ = search.select(root)
        # Leaf should be a grandchild (depth 2)
        assert leaf.parent is not None
        # It should be a leaf (no children yet)
        assert leaf.is_leaf()


class TestSelectAction:
    """Test MCTSSearch.select_action() method."""

    def test_select_action_deterministic(self) -> None:
        """temperature=0 should always pick the most visited action."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        visit_counts = {5: 100, 10: 50, 15: 200, 20: 75}

        # Should always pick action 15 (200 visits)
        for _ in range(10):
            action = search.select_action(visit_counts, temperature=0)
            assert action == 15

    def test_select_action_stochastic(self) -> None:
        """temperature>0 should produce varied selections over many calls."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        visit_counts = {5: 40, 10: 30, 15: 20, 20: 10}

        actions_seen: set[int] = set()
        for _ in range(100):
            action = search.select_action(visit_counts, temperature=1.0)
            actions_seen.add(action)

        # With temperature=1 and 100 samples, should see multiple actions
        assert len(actions_seen) > 1

    def test_select_action_empty_raises(self) -> None:
        """Empty visit counts should raise ValueError."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        with pytest.raises(ValueError, match="empty"):
            search.select_action({}, temperature=0)

    def test_select_action_single_action(self) -> None:
        """With one action, it should always be selected."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        action = search.select_action({42: 10}, temperature=0)
        assert action == 42

        action = search.select_action({42: 10}, temperature=1.0)
        assert action == 42

    def test_select_action_high_temperature_more_uniform(self) -> None:
        """Very high temperature should make selection more uniform."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        visit_counts = {0: 100, 1: 1}

        # With very high temperature, even the low-count action should appear
        action_1_count = 0
        trials = 500
        for _ in range(trials):
            action = search.select_action(visit_counts, temperature=10.0)
            if action == 1:
                action_1_count += 1

        # Action 1 should appear at least a few times with high temperature
        assert action_1_count > 0


class TestPolicyDistribution:
    """Test MCTSSearch.get_policy_distribution() method."""

    def test_policy_distribution_sums_to_one(self) -> None:
        """Policy should be a valid probability distribution summing to 1."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        visit_counts = {5: 10, 10: 20, 15: 30}
        policy = search.get_policy_distribution(visit_counts)

        assert policy.shape == (158,)
        assert abs(float(policy.sum()) - 1.0) < 1e-6

    def test_policy_distribution_proportional(self) -> None:
        """Policy probabilities should be proportional to visit counts."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        visit_counts = {0: 10, 1: 30, 2: 60}
        policy = search.get_policy_distribution(visit_counts)

        assert abs(policy[0] - 0.1) < 1e-6
        assert abs(policy[1] - 0.3) < 1e-6
        assert abs(policy[2] - 0.6) < 1e-6

    def test_policy_distribution_zeros_for_unvisited(self) -> None:
        """Actions not in visit_counts should have zero probability."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        visit_counts = {5: 10}
        policy = search.get_policy_distribution(visit_counts)

        assert policy[5] > 0
        assert policy[0] == 0.0
        assert policy[148] == 0.0

    def test_policy_distribution_empty_visit_counts(self) -> None:
        """Empty visit counts should return all-zero policy."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        policy = search.get_policy_distribution({})
        assert policy.shape == (158,)
        assert float(policy.sum()) == 0.0

    def test_policy_distribution_shape(self) -> None:
        """Policy should be float32 array of shape (158,)."""
        config = MCTSConfig()
        search = MCTSSearch(config)

        visit_counts = {10: 5, 20: 15}
        policy = search.get_policy_distribution(visit_counts)

        assert policy.dtype == np.float32
        assert policy.shape == (158,)


class TestDirichletNoise:
    """Test MCTSSearch._add_dirichlet_noise() method."""

    def test_dirichlet_noise_modifies_priors(self) -> None:
        """After adding noise, priors should differ from their original values."""
        game = _make_game_after_roll()
        config = MCTSConfig(dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
        search = MCTSSearch(config)

        root = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(root, game)

        # Record original priors
        original_priors = {a: c.prior for a, c in root.children.items()}

        search._add_dirichlet_noise(root)

        # At least some priors should have changed
        changed = sum(
            1 for a, c in root.children.items() if abs(c.prior - original_priors[a]) > 1e-10
        )
        assert changed > 0

    def test_dirichlet_noise_priors_positive(self) -> None:
        """All priors should remain positive after adding noise."""
        game = _make_game_after_roll()
        config = MCTSConfig(dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
        search = MCTSSearch(config)

        root = MCTSNode(
            state_dict=game.to_dict(),
            player_to_move=game.current_player,
        )
        search.expand(root, game)
        search._add_dirichlet_noise(root)

        for child in root.children.values():
            assert child.prior > 0

    def test_dirichlet_noise_no_children_no_error(self) -> None:
        """Adding noise to a node with no children should not error."""
        config = MCTSConfig(dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
        search = MCTSSearch(config)

        root = MCTSNode(state_dict={}, is_terminal=True)
        search._add_dirichlet_noise(root)  # Should not raise


class TestFullSearchIntegration:
    """Integration tests for the complete MCTS search loop."""

    def test_full_search_returns_valid_action(self) -> None:
        """Full search on a real game should return a valid action."""
        game = _make_game_after_roll(seed=123)
        config = MCTSConfig(
            num_simulations=20,
            max_rollout_depth=30,
            opponent_policy="random",
            dirichlet_alpha=0.0,
        )
        search = MCTSSearch(config)

        visit_counts = search.search(game, player_id=0)
        action = search.select_action(visit_counts, temperature=0)

        encoder = ActionEncoder(enable_trades=False)
        mask = encoder.get_action_mask(game, game.current_player)
        assert mask[action], f"Selected action {action} is not valid"

    def test_full_search_policy_distribution(self) -> None:
        """search + get_policy_distribution should produce valid policy."""
        game = _make_game_after_roll(seed=456)
        config = MCTSConfig(
            num_simulations=15,
            max_rollout_depth=30,
            opponent_policy="random",
            dirichlet_alpha=0.0,
        )
        search = MCTSSearch(config)

        visit_counts = search.search(game, player_id=0)
        policy = search.get_policy_distribution(visit_counts)

        assert policy.shape == (158,)
        assert abs(float(policy.sum()) - 1.0) < 1e-6

    def test_full_search_with_rule_based_rollouts(self) -> None:
        """Search with rule-based opponent policy should complete."""
        game = _make_game_after_roll(seed=789)
        config = MCTSConfig(
            num_simulations=5,
            max_rollout_depth=30,
            opponent_policy="rule_based",
            dirichlet_alpha=0.0,
        )
        search = MCTSSearch(config)

        visit_counts = search.search(game, player_id=0)
        assert len(visit_counts) > 0
        assert sum(visit_counts.values()) == 5
