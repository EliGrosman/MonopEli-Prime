"""Generate expert demonstration dataset for behavioral cloning.

Usage:
    uv run python scripts/generate_expert_data.py \\
        --agent rule_based \\
        --num-games 10000 \\
        --num-players 4 \\
        --save-dir data/expert/rule_based_4p/ \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from monopoly_gym.env import MonopolyEnv


def create_agent(agent_type: str, player_id: int) -> Any:
    """Create agent instance.

    Args:
        agent_type: Type of agent ("random", "rule_based", "conservative", "aggressive")
        player_id: The player ID this agent controls

    Returns:
        Agent instance
    """
    # Import here to avoid circular dependencies
    if agent_type == "random":
        from agents import RandomAgent

        return RandomAgent(player_id=player_id, seed=42)
    elif agent_type == "rule_based":
        from agents import RuleBasedAgent

        return RuleBasedAgent(player_id=player_id)
    elif agent_type == "conservative":
        from agents import ConservativeAgent

        return ConservativeAgent(player_id=player_id)
    elif agent_type == "aggressive":
        from agents import AggressiveAgent

        return AggressiveAgent(player_id=player_id)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


MAX_ACTIONS = 20_000  # Safety limit on real actions per game
MAX_DEAD_STEPS = 50  # Max consecutive dead steps before forcing turn advance


def play_game(
    env: "MonopolyEnv", agents: list[Any], verbose: bool = False
) -> tuple[list[dict[str, Any]], bool]:
    """Play one game and collect (obs, action, mask, outcome) tuples.

    Handles a known env bug where game.current_player gets stuck on a bankrupt
    player, causing an infinite dead-agent loop. Detects this and forces the
    game to advance the turn.

    Args:
        env: The Monopoly environment
        agents: List of agent instances (one per player)
        verbose: Whether to print game progress

    Returns:
        Tuple of (experiences list, completed normally). The bool is False
        if the game was cut short due to hitting the action safety limit.
    """
    from monopoly_gym.observation import flatten_observation

    experiences: list[dict[str, Any]] = []
    env.reset()
    action_count = 0
    consecutive_dead = 0

    while action_count < MAX_ACTIONS:
        if env.game is None or env.game.game_over or env.game.turn_number >= env.max_turns:
            break

        agent_name = env.agent_selection
        player_id = env.agent_name_mapping[agent_name]

        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            consecutive_dead += 1
            if consecutive_dead > MAX_DEAD_STEPS:
                # Stuck: game.current_player is bankrupt but env keeps
                # cycling dead agent steps without advancing the turn.
                # Force the game to advance past the bankrupt player.
                if env.game.players[env.game.current_player].bankrupt:
                    env.game.end_turn()
                    new_agent = f"player_{env.game.current_player}"
                    env.agent_selection = new_agent
                    env._handle_turn_start()  # noqa: SLF001
                consecutive_dead = 0
                continue
            env.step(None)
            continue

        consecutive_dead = 0

        # Get action from agent
        agent = agents[player_id]
        action_mask = info["action_mask"]
        action = agent.choose_action(obs, action_mask, env.game)

        # Flatten observation for storage
        obs_flat = flatten_observation(obs)

        # Record experience
        experiences.append({
            "observation": obs_flat,
            "action": int(action),
            "action_mask": action_mask.astype(np.bool_),
            "player_id": player_id,
        })

        env.step(action)
        action_count += 1

    completed = action_count < MAX_ACTIONS

    # Get winner
    winner = None
    if env.game:
        if env.game.winner is not None:
            winner = env.game.winner
        else:
            alive_players = [i for i, p in enumerate(env.game.players) if not p.bankrupt]
            if len(alive_players) == 1:
                winner = alive_players[0]
            else:
                from monopoly_engine import calculate_net_worth

                max_worth = -1
                for i in alive_players:
                    worth = calculate_net_worth(
                        env.game.players[i], env.game.property_manager
                    )
                    if worth > max_worth:
                        max_worth = worth
                        winner = i

    for exp in experiences:
        exp["winner"] = winner

    if verbose:
        print(f"Game finished. Winner: Player {winner}, Steps: {len(experiences)}")

    return experiences, completed


def save_to_hdf5(experiences: list[dict[str, Any]], save_path: Path) -> None:
    """Save experiences to HDF5 file.

    HDF5 structure:
        /observation - shape [N, obs_dim] (flattened observations)
        /actions - shape [N]
        /action_masks - shape [N, action_dim]
        /player_ids - shape [N]
        /winners - shape [N]

    Where N = total number of steps across all games.

    Args:
        experiences: List of experience dicts
        save_path: Path to save HDF5 file
    """
    print(f"Saving {len(experiences)} experiences to {save_path}")

    # Extract arrays
    observations = np.array([exp["observation"] for exp in experiences], dtype=np.float32)
    actions = np.array([exp["action"] for exp in experiences], dtype=np.int16)
    action_masks = np.array([exp["action_mask"] for exp in experiences], dtype=np.bool_)
    player_ids = np.array([exp["player_id"] for exp in experiences], dtype=np.int8)
    winners = np.array([exp["winner"] if exp["winner"] is not None else -1
                       for exp in experiences], dtype=np.int8)

    # Save to HDF5
    with h5py.File(save_path, "w") as f:
        f.create_dataset("observation", data=observations, compression="gzip")
        f.create_dataset("actions", data=actions, compression="gzip")
        f.create_dataset("action_masks", data=action_masks, compression="gzip")
        f.create_dataset("player_ids", data=player_ids, compression="gzip")
        f.create_dataset("winners", data=winners, compression="gzip")

    print(f"Saved to {save_path}")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Actions: {len(actions)}")
    print(f"  Action mask shape: {action_masks.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate expert demonstrations for behavioral cloning"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rule_based",
        choices=["random", "rule_based", "conservative", "aggressive"],
        help="Type of agent to use for demonstrations",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10000,
        help="Number of games to play",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        help="Number of players per game",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save expert data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--enable-trades",
        action="store_true",
        help="Enable trade actions in environment",
    )
    args = parser.parse_args()

    # Import monopoly_gym components
    from monopoly_gym import MonopolyEnv

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = MonopolyEnv(
        num_players=args.num_players,
        enable_trades=args.enable_trades,
        max_turns=1000,
    )

    # Create agents (all same type for expert data)
    agents = [create_agent(args.agent, i) for i in range(args.num_players)]

    # Collect experiences
    all_experiences: list[dict[str, Any]] = []
    game_lengths: list[int] = []
    winners: list[int | None] = []
    skipped_games = 0

    np.random.seed(args.seed)

    print(f"Generating {args.num_games} games with {args.agent} agents...")
    print(f"  Safety limit: {MAX_ACTIONS} real actions per game")

    for game_idx in tqdm(range(args.num_games)):
        experiences, completed = play_game(
            env, agents, verbose=(game_idx % 1000 == 0 and game_idx > 0)
        )
        if not completed:
            # Game hit the safety limit - skip it to avoid bad data
            skipped_games += 1
            continue
        all_experiences.extend(experiences)
        game_lengths.append(len(experiences))
        if experiences:
            winners.append(experiences[0]["winner"])

    print(f"\nCollected {len(all_experiences)} total experiences")
    print(f"Completed games: {len(game_lengths)}/{args.num_games} "
          f"(skipped {skipped_games} due to step limit)")
    if game_lengths:
        print(f"Average game length: {np.mean(game_lengths):.1f} steps")

    if not all_experiences:
        print("ERROR: No experiences collected! All games were skipped.")
        return

    # Shuffle and split
    indices = np.random.permutation(len(all_experiences))
    split_idx = int(len(all_experiences) * (1 - args.val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_experiences = [all_experiences[i] for i in train_indices]
    val_experiences = [all_experiences[i] for i in val_indices]

    print(f"Train: {len(train_experiences)} experiences")
    print(f"Val: {len(val_experiences)} experiences")

    # Save to HDF5
    save_to_hdf5(train_experiences, save_dir / "train.h5")
    save_to_hdf5(val_experiences, save_dir / "val.h5")

    # Save statistics
    winners_arr = np.array([w for w in winners if w is not None])
    winner_distribution: dict[str, int] = {
        str(i): int(np.sum(winners_arr == i))
        for i in range(args.num_players)
    }

    stats: dict[str, Any] = {
        "num_games_requested": args.num_games,
        "num_games_completed": len(game_lengths),
        "num_games_skipped": skipped_games,
        "num_players": args.num_players,
        "agent_type": args.agent,
        "total_experiences": len(all_experiences),
        "train_experiences": len(train_experiences),
        "val_experiences": len(val_experiences),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "std_game_length": float(np.std(game_lengths)) if game_lengths else 0.0,
        "min_game_length": int(np.min(game_lengths)) if game_lengths else 0,
        "max_game_length": int(np.max(game_lengths)) if game_lengths else 0,
        "winner_distribution": winner_distribution,
        "enable_trades": args.enable_trades,
    }

    with open(save_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to {save_dir / 'stats.json'}")
    print(f"Winner distribution: {winner_distribution}")


if __name__ == "__main__":
    main()
