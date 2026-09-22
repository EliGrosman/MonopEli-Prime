#!/usr/bin/env python3
"""CLI runner for the Monopoly game engine.

This script provides multiple modes for testing and demonstrating the game:
- Demo mode: Watch an automated game with detailed output
- Interactive mode: Play manually
- Batch mode: Run multiple games for testing
- Export mode: Generate JSON game states
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monopoly_engine.actions import (
    BuyProperty,
    EndTurn,
    PayJailFine,
    RollDice,
    UseJailCard,
)
from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import SpaceType


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str) -> None:
    """Print a colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_player_status(game: MonopolyGame) -> None:
    """Print current status of all players."""
    print(f"{Colors.CYAN}{Colors.BOLD}Player Status:{Colors.ENDC}")
    for player in game.state.players:
        if player.bankrupt:
            status = f"{Colors.RED}[BANKRUPT]{Colors.ENDC}"
        elif player.in_jail:
            status = f"{Colors.YELLOW}[IN JAIL]{Colors.ENDC}"
        else:
            status = f"{Colors.GREEN}[ACTIVE]{Colors.ENDC}"

        position = game.board.get_space(player.position).name
        print(f"  {status} {Colors.BOLD}{player.name}{Colors.ENDC}: ${player.money} at {position}")
    print()


def print_property_status(game: MonopolyGame, player_id: int) -> None:
    """Print properties owned by a player."""
    properties = [
        (prop_id, prop)
        for prop_id, prop in game.state.property_manager.properties.items()
        if prop.owner == player_id
    ]

    if not properties:
        return

    player = game.state.players[player_id]
    print(f"{Colors.BLUE}Properties owned by {player.name}:{Colors.ENDC}")
    for prop_id, prop in properties:
        space = game.board.get_space(prop_id)
        details = []
        if prop.mortgaged:
            details.append(f"{Colors.YELLOW}MORTGAGED{Colors.ENDC}")
        if prop.houses > 0 and prop.houses < 5:
            details.append(f"{prop.houses}H")
        elif prop.houses == 5:
            details.append(f"{Colors.GREEN}HOTEL{Colors.ENDC}")

        detail_str = f" ({', '.join(details)})" if details else ""
        print(f"  - {space.name}{detail_str}")
    print()


def print_game_event(message: str) -> None:
    """Print a game event."""
    print(f"{Colors.CYAN}>>> {message}{Colors.ENDC}")


def run_demo_game(
    num_players: int = 4, seed: int | None = None, max_turns: int = 100, verbose: bool = True
) -> MonopolyGame:
    """Run a demonstration game with automated play.

    Args:
        num_players: Number of players
        seed: Random seed for deterministic gameplay
        max_turns: Maximum number of turns before stopping
        verbose: Whether to print detailed output

    Returns:
        The completed game instance
    """
    if verbose:
        print_header(f"Starting Demo Game ({num_players} players)")

    game = MonopolyGame(num_players=num_players, seed=seed)

    if verbose:
        print_player_status(game)

    turn_count = 0
    while not game.state.game_over and turn_count < max_turns:
        current_player = game.state.current_player
        player = game.state.players[current_player]

        if player.bankrupt:
            # Skip bankrupt players
            EndTurn(player_id=current_player).execute(game)
            continue

        if verbose:
            print(f"\n{Colors.BOLD}Turn {turn_count + 1}: {player.name}'s turn{Colors.ENDC}")

        # Handle jail
        if player.in_jail:
            if verbose:
                print_game_event(f"{player.name} is in jail (turn {player.jail_turns + 1}/3)")

            # Try to use jail card first
            if player.jail_cards > 0:
                action = UseJailCard(player_id=current_player)
                valid, msg = action.validate(game)
                if valid:
                    if verbose:
                        print_game_event(f"{player.name} uses Get Out of Jail Free card")
                    action.execute(game)
            # Otherwise pay fine if we have money
            elif player.money >= 50:
                action = PayJailFine(player_id=current_player)
                valid, msg = action.validate(game)
                if valid:
                    if verbose:
                        print_game_event(f"{player.name} pays $50 jail fine")
                    action.execute(game)

        # Roll dice
        action = RollDice(player_id=current_player)
        valid, msg = action.validate(game)
        if valid:
            action.execute(game)
            if game.state.last_roll and verbose:
                d1, d2 = game.state.last_roll
                doubles = "DOUBLES!" if d1 == d2 else ""
                print_game_event(f"{player.name} rolled {d1} + {d2} = {d1 + d2} {doubles}")
                new_pos = game.board.get_space(player.position).name
                print_game_event(f"{player.name} landed on {new_pos}")

            # Try to buy property if landed on one
            if not player.in_jail and not player.bankrupt:
                space = game.board.get_space(player.position)
                if space.space_type in (SpaceType.PROPERTY, SpaceType.RAILROAD, SpaceType.UTILITY):
                    prop = game.state.property_manager.properties.get(player.position)
                    if prop and prop.owner is None:
                        # Simple strategy: buy if we have enough money (keep $200 reserve)
                        price = getattr(space, "cost", 0)
                        if price > 0 and player.money >= price + 200:
                            buy_action = BuyProperty(
                                player_id=current_player, property_id=player.position
                            )
                            buy_valid, _ = buy_action.validate(game)
                            if buy_valid:
                                if verbose:
                                    print_game_event(
                                        f"{player.name} buys {space.name} for ${price}"
                                    )
                                buy_action.execute(game)

        # End turn
        EndTurn(player_id=current_player).execute(game)
        turn_count += 1

        if verbose and turn_count % 10 == 0:
            print_player_status(game)

    if verbose:
        print_header("Game Complete!")
        print_player_status(game)
        if game.state.winner is not None:
            winner = game.state.players[game.state.winner]
            msg = f"Winner: {winner.name} with ${winner.money}!"
            print(f"{Colors.GREEN}{Colors.BOLD}{msg}{Colors.ENDC}\n")
        else:
            print(f"{Colors.YELLOW}Game ended after {max_turns} turns (no winner){Colors.ENDC}\n")

        # Show final property ownership
        for player_id, player in enumerate(game.state.players):
            if not player.bankrupt:
                print_property_status(game, player_id)

    return game


def run_batch_games(
    num_games: int = 10, num_players: int = 4, max_turns: int = 200, start_seed: int = 0
) -> dict[str, Any]:
    """Run multiple games and collect statistics.

    Args:
        num_games: Number of games to run
        num_players: Number of players per game
        max_turns: Maximum turns per game
        start_seed: Starting seed value

    Returns:
        Dictionary with statistics
    """
    print_header(f"Running {num_games} Games")

    stats: dict[str, Any] = {
        "total_games": num_games,
        "completed_games": 0,
        "incomplete_games": 0,
        "total_turns": 0,
        "bankruptcies": 0,
        "winners": {},
    }

    for i in range(num_games):
        seed = start_seed + i
        game = run_demo_game(num_players=num_players, seed=seed, max_turns=max_turns, verbose=False)

        if game.state.game_over:
            stats["completed_games"] += 1
            if game.state.winner is not None:
                winner_name = game.state.players[game.state.winner].name
                stats["winners"][winner_name] = stats["winners"].get(winner_name, 0) + 1
        else:
            stats["incomplete_games"] += 1

        stats["total_turns"] += game.state.turn_number
        stats["bankruptcies"] += sum(1 for p in game.state.players if p.bankrupt)

        # Progress indicator
        if (i + 1) % max(1, num_games // 10) == 0:
            print(f"Progress: {i + 1}/{num_games} games completed")

    # Calculate averages
    if stats["total_games"] > 0:
        stats["avg_turns_per_game"] = stats["total_turns"] / stats["total_games"]
        stats["completion_rate"] = stats["completed_games"] / stats["total_games"]

    return stats


def export_game_state(game: MonopolyGame, output_file: str) -> None:
    """Export game state to JSON file.

    Args:
        game: Game instance to export
        output_file: Path to output JSON file
    """
    state_dict = game.to_dict()
    with open(output_file, "w") as f:
        json.dump(state_dict, f, indent=2)
    print(f"{Colors.GREEN}Game state exported to {output_file}{Colors.ENDC}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Monopoly Game Engine CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a demo game
  python scripts/cli_runner.py demo

  # Run with specific seed
  python scripts/cli_runner.py demo --seed 42 --players 4

  # Run batch tests
  python scripts/cli_runner.py batch --games 100 --max-turns 300

  # Export game state
  python scripts/cli_runner.py demo --export game_state.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demonstration game")
    demo_parser.add_argument(
        "--players", type=int, default=4, help="Number of players (default: 4)"
    )
    demo_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    demo_parser.add_argument(
        "--max-turns", type=int, default=100, help="Maximum turns (default: 100)"
    )
    demo_parser.add_argument("--quiet", action="store_true", help="Minimal output")
    demo_parser.add_argument("--export", type=str, help="Export final state to JSON file")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run multiple games")
    batch_parser.add_argument(
        "--games", type=int, default=10, help="Number of games to run (default: 10)"
    )
    batch_parser.add_argument(
        "--players", type=int, default=4, help="Number of players (default: 4)"
    )
    batch_parser.add_argument(
        "--max-turns", type=int, default=200, help="Maximum turns per game (default: 200)"
    )
    batch_parser.add_argument(
        "--start-seed", type=int, default=0, help="Starting seed value (default: 0)"
    )
    batch_parser.add_argument("--export", type=str, help="Export statistics to JSON file")

    args = parser.parse_args()

    if args.command == "demo":
        game = run_demo_game(
            num_players=args.players,
            seed=args.seed,
            max_turns=args.max_turns,
            verbose=not args.quiet,
        )
        if args.export:
            export_game_state(game, args.export)

    elif args.command == "batch":
        stats = run_batch_games(
            num_games=args.games,
            num_players=args.players,
            max_turns=args.max_turns,
            start_seed=args.start_seed,
        )

        print_header("Batch Results")
        print(f"Total games: {stats['total_games']}")
        print(f"Completed: {stats['completed_games']} ({stats['completion_rate']:.1%})")
        print(f"Incomplete: {stats['incomplete_games']}")
        print(f"Average turns/game: {stats['avg_turns_per_game']:.1f}")
        print(f"Total bankruptcies: {stats['bankruptcies']}")
        print("\nWinner distribution:")
        for winner, count in sorted(stats["winners"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {winner}: {count} wins")

        if args.export:
            with open(args.export, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n{Colors.GREEN}Statistics exported to {args.export}{Colors.ENDC}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
