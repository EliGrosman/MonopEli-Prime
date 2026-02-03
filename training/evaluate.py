"""Evaluation harness for trained Monopoly agents.

This module provides tools to evaluate trained agents against various baselines
and generate statistics on performance.

Usage:
    from training import evaluate_agent

    results = evaluate_agent(
        model_path="models/ppo_monopoly",
        num_games=100,
        opponents=["random", "rule_based"],
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationResult:
    """Results from evaluating an agent.

    Attributes:
        opponent_type: Type of opponent faced
        num_games: Number of games played
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws (truncated games)
        win_rate: Win percentage
        avg_game_length: Average turns per game
        avg_reward: Average total reward
        games_per_second: Throughput metric
    """
    opponent_type: str
    num_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_game_length: float
    avg_reward: float
    games_per_second: float

    def __str__(self) -> str:
        return (
            f"vs {self.opponent_type}: "
            f"{self.win_rate:.1%} win rate "
            f"({self.wins}W/{self.losses}L/{self.draws}D), "
            f"avg length: {self.avg_game_length:.1f} turns"
        )


@dataclass
class EvaluationSummary:
    """Summary of evaluation against multiple opponents.

    Attributes:
        model_path: Path to evaluated model
        results: Results per opponent type
        total_games: Total games played
        total_wins: Total wins
        overall_win_rate: Overall win percentage
    """
    model_path: str
    results: dict[str, EvaluationResult] = field(default_factory=dict)

    @property
    def total_games(self) -> int:
        return sum(r.num_games for r in self.results.values())

    @property
    def total_wins(self) -> int:
        return sum(r.wins for r in self.results.values())

    @property
    def overall_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games

    def __str__(self) -> str:
        lines = [f"Evaluation of {self.model_path}:", ""]
        for result in self.results.values():
            lines.append(f"  {result}")
        lines.append("")
        lines.append(f"Overall: {self.overall_win_rate:.1%} ({self.total_wins}/{self.total_games})")
        return "\n".join(lines)


def play_game(
    model: Any,
    opponent_type: str = "random",
    num_players: int = 2,
    max_turns: int = 1000,
    seed: int | None = None,
    deterministic: bool = True,
) -> tuple[bool, int, float]:
    """Play a single game and return result.

    Args:
        model: Trained model
        opponent_type: Type of opponent
        num_players: Number of players
        max_turns: Max turns before truncation
        seed: Random seed
        deterministic: Whether to use deterministic actions

    Returns:
        Tuple of (won, game_length, total_reward)
    """
    from monopoly_gym import SingleAgentMonopolyEnv

    env = SingleAgentMonopolyEnv(
        num_players=num_players,
        opponent_type=opponent_type,
        max_turns=max_turns,
        reward_type="sparse",
        seed=seed,
    )

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action_mask = info.get("action_mask")
        if action_mask is not None:
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=deterministic)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            break

    env.close()

    # Determine if we won (reward > 0 means win in sparse reward)
    won = float(reward) > 0
    # Note: draw = truncated and not terminated (unused but kept for clarity)

    return won, steps, total_reward


def evaluate_against_opponent(
    model: Any,
    opponent_type: str,
    num_games: int = 100,
    num_players: int = 2,
    max_turns: int = 1000,
    seed: int | None = None,
    deterministic: bool = True,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate model against a specific opponent type.

    Args:
        model: Trained model
        opponent_type: Type of opponent
        num_games: Number of games to play
        num_players: Number of players
        max_turns: Max turns per game
        seed: Base random seed
        deterministic: Whether to use deterministic actions
        verbose: Print progress

    Returns:
        Evaluation results
    """
    wins = 0
    losses = 0
    draws = 0
    total_length = 0
    total_reward = 0.0

    start_time = time.time()

    for i in range(num_games):
        game_seed = seed + i if seed is not None else None
        won, length, reward = play_game(
            model,
            opponent_type=opponent_type,
            num_players=num_players,
            max_turns=max_turns,
            seed=game_seed,
            deterministic=deterministic,
        )

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        total_length += length
        total_reward += reward

        if verbose and (i + 1) % 10 == 0:
            current_win_rate = wins / (i + 1)
            print(f"  Game {i + 1}/{num_games}: {current_win_rate:.1%} win rate")

    elapsed = time.time() - start_time

    return EvaluationResult(
        opponent_type=opponent_type,
        num_games=num_games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / num_games if num_games > 0 else 0.0,
        avg_game_length=total_length / num_games if num_games > 0 else 0.0,
        avg_reward=total_reward / num_games if num_games > 0 else 0.0,
        games_per_second=num_games / elapsed if elapsed > 0 else 0.0,
    )


def evaluate_agent(
    model_path: str | Path,
    num_games: int = 100,
    opponents: list[str] | None = None,
    num_players: int = 2,
    max_turns: int = 1000,
    seed: int | None = 42,
    deterministic: bool = True,
    verbose: bool = True,
) -> EvaluationSummary:
    """Evaluate a trained agent against multiple opponents.

    Args:
        model_path: Path to saved model
        num_games: Number of games per opponent
        opponents: List of opponent types (default: ["random", "rule_based"])
        num_players: Number of players per game
        max_turns: Max turns per game
        seed: Random seed
        deterministic: Whether to use deterministic actions
        verbose: Print progress

    Returns:
        Evaluation summary with results per opponent
    """
    from sb3_contrib import MaskablePPO

    if opponents is None:
        opponents = ["random", "rule_based"]

    model = MaskablePPO.load(str(model_path))

    summary = EvaluationSummary(model_path=str(model_path))

    for opponent in opponents:
        if verbose:
            print(f"\nEvaluating vs {opponent}...")

        result = evaluate_against_opponent(
            model,
            opponent_type=opponent,
            num_games=num_games,
            num_players=num_players,
            max_turns=max_turns,
            seed=seed,
            deterministic=deterministic,
            verbose=verbose,
        )

        summary.results[opponent] = result

        if verbose:
            print(f"  Result: {result}")

    return summary


def quick_evaluate(
    model: Any,
    num_games: int = 20,
    opponent_type: str = "random",
) -> float:
    """Quick evaluation for use during training.

    Args:
        model: Model to evaluate
        num_games: Number of games
        opponent_type: Opponent type

    Returns:
        Win rate
    """
    result = evaluate_against_opponent(
        model,
        opponent_type=opponent_type,
        num_games=num_games,
        deterministic=True,
        verbose=False,
    )
    return result.win_rate
