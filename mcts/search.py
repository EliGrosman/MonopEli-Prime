"""MCTS search engine for Monopoly.

Contains the core MCTS data structures, expansion, simulation, and state
cloning utilities. Backpropagation and the main search loop are added in
subsequent tasks (A5-A8).
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from monopoly_engine.actions import RollDice
from monopoly_engine.game import MonopolyGame
from monopoly_engine.rules import calculate_net_worth
from monopoly_gym.action_space import ActionEncoder


class MCTSNode:
    """A node in the MCTS search tree.

    Each node represents a game state reached by taking a specific action
    from the parent node. The root node has no parent and no action.

    Attributes:
        state_dict: Serialized game state (from game.to_dict()).
        parent: Parent node (None for root).
        action: Action index that led to this node (None for root).
        children: Dict mapping action_idx -> child MCTSNode.
        visit_count: Number of times this node has been visited.
        total_value: Sum of all backpropagated values for each player.
        prior: Prior probability from policy network (or uniform).
        player_to_move: Player ID whose turn it is at this node.
        is_terminal: Whether this state is a terminal game state.
    """

    __slots__ = (
        "state_dict",
        "parent",
        "action",
        "children",
        "visit_count",
        "total_value",
        "prior",
        "player_to_move",
        "is_terminal",
    )

    def __init__(
        self,
        state_dict: dict[str, Any],
        parent: MCTSNode | None = None,
        action: int | None = None,
        prior: float = 1.0,
        player_to_move: int = 0,
        is_terminal: bool = False,
    ) -> None:
        self.state_dict = state_dict
        self.parent = parent
        self.action = action
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: dict[int, float] = {}
        self.prior = prior
        self.player_to_move = player_to_move
        self.is_terminal = is_terminal

    def ucb1_score(self, exploration_constant: float, player_id: int) -> float:
        """Compute PUCT score for this node from the perspective of player_id.

        Uses the AlphaZero PUCT formula:
            Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)

        where:
            Q(s,a) = total_value[player_id] / visit_count  (average value)
            P(s,a) = prior probability
            N_parent = parent.visit_count
            N_child = visit_count

        Unvisited nodes return infinity to ensure they are explored first.

        Args:
            exploration_constant: Controls exploration vs exploitation.
            player_id: The player whose value we are maximizing.

        Returns:
            PUCT score (higher is better).
        """
        if self.visit_count == 0:
            return float("inf")

        q_value = self.total_value.get(player_id, 0.0) / self.visit_count

        parent_visits = self.parent.visit_count if self.parent is not None else 1
        exploration = (
            exploration_constant
            * self.prior
            * math.sqrt(parent_visits)
            / (1 + self.visit_count)
        )

        return q_value + exploration

    def best_child(self, exploration_constant: float, player_id: int) -> MCTSNode:
        """Select child with highest PUCT score.

        Args:
            exploration_constant: UCB exploration constant.
            player_id: The player whose value we are maximizing.

        Returns:
            The child node with the highest PUCT score.

        Raises:
            ValueError: If the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select best child from a node with no children.")

        return max(
            self.children.values(),
            key=lambda child: child.ucb1_score(exploration_constant, player_id),
        )

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Return True if this node has no parent."""
        return self.parent is None


@dataclass
class MCTSConfig:
    """Configuration for MCTS search.

    Attributes:
        num_simulations: Number of MCTS simulations per move. More = stronger
            but slower. 200 is a good balance for Monopoly.
        exploration_constant: UCB1/PUCT exploration constant. Higher values
            encourage exploring less-visited nodes. 1.41 (sqrt(2)) is standard.
        max_rollout_depth: Maximum number of actions in a random rollout
            before truncating. Prevents infinite games.
        use_value_network: If True and a network is provided, use it for
            leaf evaluation instead of random rollouts.
        temperature: Controls action selection from root visit counts.
            0 = deterministic (most visited), 1 = proportional to visits,
            >1 = more uniform. Use >0 during training data generation.
        opponent_policy: Policy for opponent players during search.
            Options: "rule_based", "random", "network".
        dirichlet_alpha: Dirichlet noise alpha for root exploration (AlphaZero).
            0.0 = no noise. 0.3 is typical for board games.
        dirichlet_epsilon: Fraction of root prior replaced with Dirichlet noise.
            0.25 is typical.
    """

    num_simulations: int = 200
    exploration_constant: float = 1.41
    max_rollout_depth: int = 500
    use_value_network: bool = False
    temperature: float = 0.0
    opponent_policy: str = "rule_based"
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCTSConfig:
        """Deserialize config from a dictionary."""
        return cls(**data)


def clone_game_state(
    game: MonopolyGame, new_seed: int | None = None
) -> MonopolyGame:
    """Clone a game state for MCTS simulation.

    Uses game.to_dict() / MonopolyGame.from_dict() for deep copy.
    Assigns a fresh RNG seed so dice rolls in the simulation are
    independent of other simulations and of the real game.

    Args:
        game: The game to clone.
        new_seed: Seed for the cloned game's RNG. If None, uses a
                  random seed (non-deterministic simulations).

    Returns:
        A new MonopolyGame instance with identical state but independent RNG.
    """
    state_dict = game.to_dict()
    clone = MonopolyGame.from_dict(state_dict)
    if new_seed is not None:
        clone.rng = random.Random(new_seed)
    else:
        clone.rng = random.Random()
    return clone


def _auto_roll_dice(game: MonopolyGame) -> None:
    """Automatically roll dice for the current player if needed.

    In Monopoly, each turn starts with a dice roll before the player
    chooses actions. The 149-dim action space only covers post-roll
    decisions. This function handles the automatic dice roll so that
    child game states are ready for action selection.

    Args:
        game: The game state (mutated in place).
    """
    if game.game_over:
        return
    player = game.players[game.current_player]
    if player.bankrupt:
        return
    roll_action = RollDice(player_id=game.current_player)
    is_valid, _ = roll_action.validate(game)
    if is_valid:
        roll_action.execute(game)


def _terminal_values(game: MonopolyGame) -> dict[int, float]:
    """Compute per-player values for a terminal or truncated game state.

    If the game has a winner: winner gets +1.0, all others get -1.0.
    If the game is ongoing (truncated): rank players by net worth.
    Best net worth gets +1.0, worst gets -1.0, intermediate ranks
    are linearly interpolated.

    Args:
        game: The game state to evaluate.

    Returns:
        Dict mapping player_id -> value in [-1, 1].
    """
    num_players = len(game.players)
    values: dict[int, float] = {}

    if game.game_over and game.winner is not None:
        for i in range(num_players):
            values[i] = 1.0 if i == game.winner else -1.0
        return values

    # Truncated or ongoing -- rank by net worth
    net_worths = [
        calculate_net_worth(game.players[i], game.property_manager)
        for i in range(num_players)
    ]
    ranked = list(np.argsort(net_worths))  # ascending: index 0 = worst
    for rank, pid in enumerate(ranked):
        if num_players > 1:
            values[pid] = 2.0 * rank / (num_players - 1) - 1.0
        else:
            values[pid] = 0.0

    return values


class MCTSSearch:
    """Orchestrates the MCTS search process.

    Handles tree expansion, and will later include simulation,
    backpropagation, and the full search loop (A4-A8).

    Attributes:
        config: Search configuration.
        value_network: Optional value network for leaf evaluation.
    """

    def __init__(
        self,
        config: MCTSConfig,
        value_network: Any | None = None,
    ) -> None:
        self.config = config
        self.value_network = value_network
        self._encoder = ActionEncoder(enable_trades=False)

    def expand(
        self,
        node: MCTSNode,
        game: MonopolyGame,
        policy_priors: NDArray[np.float32] | None = None,
    ) -> None:
        """Expand a leaf node by adding children for all valid actions.

        For each valid action, clones the game state, executes the action,
        and creates a child node. If policy_priors are provided (from a
        policy network), they are used as priors; otherwise uniform priors
        are used.

        After executing an action that ends the turn, automatically rolls
        dice for the next player so the child state is ready for action
        selection.

        Args:
            node: The leaf node to expand.
            game: The game state at this node (not mutated).
            policy_priors: Optional policy network output (149-dim probability
                          distribution). If None, uniform priors are used.
        """
        if node.is_terminal:
            return

        mask = self._encoder.get_action_mask(game, node.player_to_move)
        valid_actions = np.where(mask)[0]

        if len(valid_actions) == 0:
            return

        num_valid = len(valid_actions)

        for action_idx in valid_actions:
            child_game = clone_game_state(game)
            action = self._encoder.decode(
                int(action_idx), node.player_to_move, child_game
            )
            action.execute(child_game)

            # If the action changed the current player (e.g. EndTurn),
            # auto-roll dice for the new current player.
            if child_game.current_player != node.player_to_move:
                _auto_roll_dice(child_game)

            # Determine prior
            if policy_priors is not None:
                prior = float(policy_priors[action_idx])
            else:
                prior = 1.0 / num_valid

            child = MCTSNode(
                state_dict=child_game.to_dict(),
                parent=node,
                action=int(action_idx),
                prior=prior,
                player_to_move=child_game.current_player,
                is_terminal=child_game.game_over,
            )
            node.children[int(action_idx)] = child

    def simulate(
        self,
        game: MonopolyGame,
        player_id: int,
    ) -> dict[int, float]:
        """Evaluate a game state, returning estimated value per player.

        If a value network is available and config.use_value_network is True,
        uses the network for instant evaluation. Otherwise, performs a random
        rollout to terminal state.

        Args:
            game: The game state to evaluate (will be mutated during rollout).
            player_id: The player whose perspective we primarily care about
                       (used for feature encoding with value network).

        Returns:
            Dict mapping player_id -> estimated value in [-1, 1].
            +1 = win, -1 = loss, intermediate for ongoing games.
        """
        if game.game_over:
            return _terminal_values(game)

        if (
            self.config.use_value_network
            and self.value_network is not None
        ):
            return self._value_network_evaluate(game, player_id)

        return self._random_rollout(game, self.config.max_rollout_depth)

    def _random_rollout(
        self,
        game: MonopolyGame,
        max_depth: int,
    ) -> dict[int, float]:
        """Play game to completion using random actions.

        Each step: get the current player's valid actions, pick one
        uniformly at random, decode and execute it. If the action ends
        the turn, auto-roll dice for the next player.

        Args:
            game: Game state (will be mutated).
            max_depth: Maximum number of actions before truncating.

        Returns:
            Per-player values: +1 for winner, -1 for losers.
            If truncated, uses net worth ranking to assign fractional values.
        """
        rng = np.random.RandomState()  # noqa: NPY002

        for _ in range(max_depth):
            if game.game_over:
                break

            current = game.current_player
            player = game.players[current]
            if player.bankrupt:
                # Skip bankrupt players by ending their turn
                from monopoly_engine.actions import EndTurn

                end = EndTurn(player_id=current)
                valid, _ = end.validate(game)
                if valid:
                    end.execute(game)
                    _auto_roll_dice(game)
                continue

            mask = self._encoder.get_action_mask(game, current)
            valid_indices = np.where(mask)[0]

            if len(valid_indices) == 0:
                # No valid actions -- force end turn
                from monopoly_engine.actions import EndTurn

                end = EndTurn(player_id=current)
                valid, _ = end.validate(game)
                if valid:
                    end.execute(game)
                    _auto_roll_dice(game)
                continue

            chosen = int(rng.choice(valid_indices))
            action = self._encoder.decode(chosen, current, game)
            action.execute(game)

            # Auto-roll dice if turn changed
            if game.current_player != current:
                _auto_roll_dice(game)

        return _terminal_values(game)

    def _value_network_evaluate(
        self,
        game: MonopolyGame,
        player_id: int,
    ) -> dict[int, float]:
        """Evaluate game state using the value network.

        Extracts features, runs a forward pass, and returns per-player
        value estimates. This is a placeholder that will be fully wired
        in Workstream B (B1-B2).

        Args:
            game: Game state to evaluate.
            player_id: Player perspective for feature encoding.

        Returns:
            Per-player value estimates from the network.
        """
        # Value network integration will be completed in Workstream B.
        # For now, return zeros if somehow called without a proper network.
        num_players = len(game.players)
        return {i: 0.0 for i in range(num_players)}

    def backpropagate(self, node: MCTSNode, values: dict[int, float]) -> None:
        """Propagate simulation values from leaf to root.

        Updates visit_count and total_value for every node on the path
        from the given node back to the root. Each player's value is
        stored independently at every node.

        Args:
            node: The leaf node where simulation ended.
            values: Per-player value estimates from simulation.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visit_count += 1
            for player_id, value in values.items():
                current.total_value[player_id] = (
                    current.total_value.get(player_id, 0.0) + value
                )
            current = current.parent
