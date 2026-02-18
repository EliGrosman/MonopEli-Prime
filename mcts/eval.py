"""Evaluation harness for the MCTSAgent.

Provides play_evaluation_game() for single-game runs and
evaluate_mcts_agent() for multi-opponent benchmarking.
Results follow the same EvaluationResult pattern as training/evaluate.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

# Agent imports are deferred to function bodies to avoid a circular import:
#   agents/mcts_agent.py  →  mcts/__init__.py  →  mcts/eval.py
# TYPE_CHECKING-only imports satisfy mypy without triggering the cycle.
if TYPE_CHECKING:
    from agents.base import Agent
    from agents.mcts_agent import MCTSAgent


@dataclass
class MCTSEvalResult:
    """Results from evaluating MCTSAgent against one opponent type.

    Attributes:
        opponent_type: Name of the opponent ("random", "rule_based", etc.).
        num_games: Number of games played.
        wins: Games where MCTSAgent (player 0) won.
        losses: Games where MCTSAgent lost.
        draws: Games truncated at max_turns with no winner.
        win_rate: wins / num_games.
        avg_game_length: Average action-count per game.
        avg_time_sec: Average wall-clock seconds per game.
        total_time_sec: Total wall-clock seconds for all games.
        avg_mcts_time_per_move: Average seconds MCTSAgent spent per search.
    """

    opponent_type: str
    num_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_game_length: float
    avg_time_sec: float
    total_time_sec: float
    avg_mcts_time_per_move: float

    def __str__(self) -> str:
        return (
            f"vs {self.opponent_type}: "
            f"{self.win_rate:.1%} win rate "
            f"({self.wins}W/{self.losses}L/{self.draws}D), "
            f"avg game {self.avg_game_length:.0f} actions, "
            f"{self.avg_time_sec:.1f}s/game, "
            f"{self.avg_mcts_time_per_move*1000:.1f}ms/move"
        )

    def to_dict(self) -> dict[str, float]:
        """Return a flat dict of numeric metrics for logging/export."""
        return {
            "win_rate": self.win_rate,
            "avg_game_length": self.avg_game_length,
            "avg_time_sec": self.avg_time_sec,
            "avg_mcts_time_per_move": self.avg_mcts_time_per_move,
            "wins": float(self.wins),
            "losses": float(self.losses),
            "draws": float(self.draws),
            "num_games": float(self.num_games),
        }


def _make_opponent(opponent_type: str, player_id: int) -> Agent:
    """Instantiate an opponent agent by type string.

    Args:
        opponent_type: One of "random", "rule_based", "aggressive", "conservative".
        player_id: The player ID to assign.

    Returns:
        An Agent instance.

    Raises:
        ValueError: If the opponent type is unknown.
    """
    # Local imports to avoid circular dependency with agents package.
    from agents.random_agent import RandomAgent
    from agents.rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent

    if opponent_type == "random":
        return RandomAgent(player_id=player_id)
    if opponent_type == "rule_based":
        return RuleBasedAgent(player_id=player_id)
    if opponent_type == "aggressive":
        return AggressiveAgent(player_id=player_id)
    if opponent_type == "conservative":
        return ConservativeAgent(player_id=player_id)
    raise ValueError(
        f"Unknown opponent type {opponent_type!r}. "
        "Expected one of: random, rule_based, aggressive, conservative."
    )


def play_evaluation_game(
    mcts_agent: MCTSAgent,
    opponents: list[Agent],
    max_turns: int = 500,
    seed: int | None = None,
) -> tuple[int | None, int, float]:
    """Play a single evaluation game between MCTSAgent and opponents.

    MCTSAgent is always player 0. Opponents fill slots 1..N-1.
    All agents choose actions via choose_action(); dice rolling is handled
    as a normal action selected by the agent.

    Args:
        mcts_agent: The MCTSAgent under evaluation (player 0).
        opponents: List of opponent agents (players 1..N-1).
        max_turns: Maximum action steps before truncation.
        seed: Random seed for game creation.

    Returns:
        Tuple of (winner_id, action_count, elapsed_seconds).
        winner_id is None if the game was truncated.
    """
    num_players = 1 + len(opponents)
    game = MonopolyGame(num_players=num_players, seed=seed)
    encoder = ActionEncoder(enable_trades=False)

    # Reset all agents for a fresh game
    mcts_agent.reset()
    for opp in opponents:
        opp.reset()

    agents: dict[int, Agent] = {0: mcts_agent}
    for i, opp in enumerate(opponents):
        agents[i + 1] = opp

    action_count = 0
    t0 = time.monotonic()

    while not game.game_over and action_count < max_turns:
        pid = game.current_player
        agent = agents[pid]
        mask = encoder.get_action_mask(game, pid)
        obs: dict[str, Any] = {}

        action_idx = agent.choose_action(obs, mask, game)
        action = encoder.decode(action_idx, pid, game)
        valid, _ = action.validate(game)
        if valid:
            action.execute(game)

        action_count += 1

    elapsed = time.monotonic() - t0
    return game.winner, action_count, elapsed


def evaluate_mcts_agent(
    network_path: str | Path | None,
    num_simulations: int = 200,
    num_games: int = 50,
    opponents: list[str] | None = None,
    num_players: int = 4,
    max_turns: int = 500,
    seed: int | None = 42,
    verbose: bool = True,
    tensorboard_dir: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate MCTSAgent against various opponent types.

    Plays num_games against each opponent type and reports win rates,
    timing, and game length statistics. MCTSAgent is always player 0;
    all other player slots are filled with the same opponent type.

    Args:
        network_path: Path to ValueNetwork checkpoint. None = random rollouts.
        num_simulations: MCTS simulations per move.
        num_games: Games to play against each opponent type.
        opponents: List of opponent types to evaluate against.
            Default: ["random", "rule_based"].
        num_players: Total players per game (including MCTS as player 0).
        max_turns: Maximum action steps per game before truncation.
        seed: Base random seed (each game uses seed + game_index).
        verbose: Print per-game and summary results to stdout.
        tensorboard_dir: Optional directory for TensorBoard scalar logging.

    Returns:
        Dict mapping opponent_type -> flat metrics dict with keys:
            "win_rate", "avg_game_length", "avg_time_sec",
            "avg_mcts_time_per_move", "wins", "losses", "draws", "num_games".
    """
    if opponents is None:
        opponents = ["random", "rule_based"]

    num_opponents = num_players - 1

    # Local import to avoid circular dependency (agents → mcts → eval → agents).
    from agents.mcts_agent import MCTSAgent

    mcts_agent = MCTSAgent(
        player_id=0,
        num_simulations=num_simulations,
        temperature=0.0,
        network_path=network_path,
    )

    writer = None
    if tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    all_results: dict[str, MCTSEvalResult] = {}

    for opp_type in opponents:
        wins = 0
        losses = 0
        draws = 0
        total_actions = 0
        total_time = 0.0

        for game_idx in range(num_games):
            game_seed = (seed + game_idx) if seed is not None else None
            opp_agents = [
                _make_opponent(opp_type, player_id=i + 1)
                for i in range(num_opponents)
            ]

            winner, actions, elapsed = play_evaluation_game(
                mcts_agent,
                opp_agents,
                max_turns=max_turns,
                seed=game_seed,
            )

            if winner == 0:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1

            total_actions += actions
            total_time += elapsed

            if verbose:
                outcome = "W" if winner == 0 else ("D" if winner is None else "L")
                print(
                    f"  [{opp_type}] game {game_idx + 1}/{num_games}: "
                    f"{outcome} (winner={winner}, {actions} actions, {elapsed:.1f}s)"
                )

        result = MCTSEvalResult(
            opponent_type=opp_type,
            num_games=num_games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / num_games,
            avg_game_length=total_actions / num_games,
            avg_time_sec=total_time / num_games,
            total_time_sec=total_time,
            avg_mcts_time_per_move=mcts_agent.search_stats.avg_time_per_move,
        )
        all_results[opp_type] = result

        if verbose:
            print(f"  {result}")

        if writer is not None:
            writer.add_scalar(f"eval/win_rate_vs_{opp_type}", result.win_rate, 0)
            writer.add_scalar(
                f"eval/avg_time_vs_{opp_type}", result.avg_time_sec, 0
            )

    if writer is not None:
        writer.close()

    return {k: v.to_dict() for k, v in all_results.items()}
