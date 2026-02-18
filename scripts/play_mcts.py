"""Interactive MCTS game demo.

Plays a game with MCTS agent (player 0) vs RuleBasedAgent opponents and
prints detailed search information after each MCTS move.

Usage:
    uv run python scripts/play_mcts.py
    uv run python scripts/play_mcts.py --network models/mcts_best/
    uv run python scripts/play_mcts.py --simulations 500 --verbose
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.mcts_agent import MCTSAgent
from agents.rule_based import RuleBasedAgent
from monopoly_engine.board import Board
from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_network_path(ckpt_path: str) -> Path:
    """Resolve checkpoint dir to network.pt, or return path directly."""
    p = Path(ckpt_path)
    return p / "network.pt" if p.is_dir() else p


def _space_name(position: int) -> str:
    """Return a short name for a board position."""
    try:
        return Board.get_space(position).name
    except (IndexError, AttributeError):
        return f"pos {position}"


def _top_actions(
    visit_counts: dict[int, int],
    encoder: ActionEncoder,
    total_visits: int,
    top_k: int = 5,
) -> list[tuple[str, int, float]]:
    """Return the top-k actions sorted by visit count.

    Returns:
        List of (action_name, visits, visit_pct).
    """
    sorted_actions = sorted(visit_counts.items(), key=lambda kv: kv[1], reverse=True)
    result = []
    for action_idx, visits in sorted_actions[:top_k]:
        name = encoder.get_action_name(action_idx)
        pct = visits / total_visits if total_visits > 0 else 0.0
        result.append((name, visits, pct))
    return result


# ---------------------------------------------------------------------------
# Core demo function
# ---------------------------------------------------------------------------

def play_demo_game(
    network_path: str | None = None,
    num_simulations: int = 200,
    num_players: int = 4,
    max_turns: int = 200,
    verbose: bool = True,
    seed: int | None = 42,
) -> None:
    """Play a demo game and print MCTS analysis.

    MCTSAgent is always player 0; all other slots are RuleBasedAgent.
    After each MCTS move the top-5 actions and timing are printed.

    Args:
        network_path: Path to ValueNetwork checkpoint dir or .pt file.
            None = pure MCTS with random rollouts.
        num_simulations: MCTS simulations per move.
        num_players: Total players (1 MCTS + rest rule-based).
        max_turns: Maximum action steps before truncation.
        verbose: If True, also print non-MCTS player moves.
        seed: Random seed for game reproducibility.
    """
    # Resolve network
    net_file: str | None = None
    if network_path is not None:
        p = _resolve_network_path(network_path)
        net_file = str(p) if p.exists() else None
        if net_file is None:
            print(f"[warn] network not found at {p}, using random rollouts")

    mcts_agent = MCTSAgent(
        player_id=0,
        num_simulations=num_simulations,
        temperature=0.0,
        network_path=net_file,
    )
    opponents: list[RuleBasedAgent] = [
        RuleBasedAgent(player_id=i + 1) for i in range(num_players - 1)
    ]

    game = MonopolyGame(num_players=num_players, seed=seed)
    encoder = ActionEncoder(enable_trades=False)

    agents: dict[int, Any] = {0: mcts_agent}
    for opp in opponents:
        agents[opp.player_id] = opp

    mode = "network" if net_file else "random rollouts"
    print(
        f"\nMCTS Game Demo — {num_players} players, "
        f"{num_simulations} simulations, {mode}"
    )
    print("=" * 60)

    action_count = 0
    mcts_moves = 0

    while not game.game_over and action_count < max_turns:
        pid = game.current_player
        agent = agents[pid]
        mask = encoder.get_action_mask(game, pid)
        obs: dict[str, Any] = {}

        if pid == 0:
            # ---- MCTS move ----
            player = game.players[pid]
            space_name = _space_name(player.position)
            mcts_moves += 1
            print(
                f"\nTurn {action_count + 1} — MCTS Agent (Player 0) "
                f"— ${player.money:,} — Position: {player.position} ({space_name})"
            )
            print(f"  Running MCTS search ({num_simulations} simulations)...")

            t0 = time.monotonic()
            visit_counts = mcts_agent._mcts.search(game, player_id=0)
            elapsed = time.monotonic() - t0

            print(f"  Search completed in {elapsed:.2f}s")

            if visit_counts:
                total_visits = sum(visit_counts.values())
                top = _top_actions(visit_counts, encoder, total_visits)

                print(f"\n  Top {len(top)} actions considered:")
                for rank, (action_name, visits, pct) in enumerate(top, 1):
                    bar = "#" * int(pct * 20)
                    print(
                        f"    {rank}. {action_name:<28} "
                        f"— {visits:4d} visits ({pct:5.1%}) "
                        f"[{bar:<20}]"
                    )

                action_idx = mcts_agent._mcts.select_action(visit_counts, temperature=0.0)
                selected_name = encoder.get_action_name(action_idx)
                print(f"\n  Selected: {selected_name}")
            else:
                # No valid actions from search — fall back to mask
                valid_actions = list(np.where(mask)[0])
                action_idx = int(valid_actions[0]) if valid_actions else 0
                print("  (no search results — using fallback action)")

            # Update agent stats manually so reset() still works
            mcts_agent.search_stats.total_searches += 1
            mcts_agent.search_stats.total_time_sec += elapsed
            mcts_agent.search_stats.total_simulations += num_simulations

        else:
            # ---- Opponent move ----
            action_idx = agent.choose_action(obs, mask, game)
            if verbose:
                action_name = encoder.get_action_name(action_idx)
                player = game.players[pid]
                print(
                    f"  [RuleBasedAgent P{pid}] "
                    f"${player.money:,} @ pos {player.position}: "
                    f"{action_name}"
                )

        # Execute action
        action = encoder.decode(action_idx, pid, game)
        valid, _ = action.validate(game)
        if valid:
            action.execute(game)

        action_count += 1

    # ---- Summary ----
    print("\n" + "=" * 60)
    if game.game_over and game.winner is not None:
        winner_id = game.winner
        agent_label = "MCTS Agent" if winner_id == 0 else f"RuleBasedAgent P{winner_id}"
        print(f"Game over — winner: {agent_label} (Player {winner_id})")
    else:
        print(f"Game truncated after {action_count} actions (no winner)")

    stats = mcts_agent.search_stats
    if mcts_moves > 0:
        print(
            f"\nMCTS stats over {mcts_moves} moves: "
            f"{stats.avg_time_per_move * 1000:.1f}ms/move avg, "
            f"{stats.total_simulations} total simulations"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCTS Game Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--network", type=str, default=None,
                        help="Path to checkpoint dir or network.pt file")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move")
    parser.add_argument("--num-players", type=int, default=4,
                        help="Total players (1 MCTS + rest rule-based)")
    parser.add_argument("--max-turns", type=int, default=200,
                        help="Max action steps before truncation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print opponent moves too")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    play_demo_game(
        network_path=args.network,
        num_simulations=args.simulations,
        num_players=args.num_players,
        max_turns=args.max_turns,
        verbose=args.verbose,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
