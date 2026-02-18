"""Tests for MCTS search engine (A1-A3, A7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from monopoly_engine.actions import RollDice
from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

from mcts.search import MCTSConfig, MCTSNode, MCTSSearch, clone_game_state


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
        best = _add_child(
            root, action=1, prior=0.3, visit_count=5, total_value={0: 4.0}
        )
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
            assert (
                clone_dict["players"][i]["money"]
                == orig_dict["players"][i]["money"]
            )
            assert (
                clone_dict["players"][i]["position"]
                == orig_dict["players"][i]["position"]
            )

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
            policy[a] = (i + 1.0)
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
