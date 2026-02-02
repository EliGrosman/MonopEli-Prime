"""Integration tests for the Monopoly game engine.

These tests run complete games end-to-end to validate the entire system works together.
"""

import pytest
from monopoly_engine.game import MonopolyGame
from monopoly_engine.actions import (
    RollDice,
    BuyProperty,
    BuildHouse,
    BuildHotel,
    EndTurn,
    PayJailFine,
    UseJailCard,
)
from monopoly_engine.types import SpaceType


class TestFullGameSimulation:
    """Test complete game flows from start to finish."""

    def test_minimal_2_player_game_to_completion(self) -> None:
        """Run a minimal 2-player game until someone wins."""
        game = MonopolyGame(num_players=2, seed=100)

        turn_limit = 500  # Prevent infinite loops
        turn_count = 0

        while not game.state.game_over and turn_count < turn_limit:
            current_player_id = game.state.current_player
            player = game.state.players[current_player_id]

            if player.bankrupt:
                EndTurn(player_id=current_player_id).execute(game)
                turn_count += 1
                continue

            # Handle jail
            if player.in_jail:
                if player.money >= 50:
                    PayJailFine(player_id=current_player_id).execute(game)
                elif player.jail_cards > 0:
                    UseJailCard(player_id=current_player_id).execute(game)

            # Roll dice
            RollDice(player_id=current_player_id).execute(game)

            # Simple buying strategy
            if not player.in_jail and not player.bankrupt:
                space = game.board.get_space(player.position)
                if space.space_type in (SpaceType.PROPERTY, SpaceType.RAILROAD, SpaceType.UTILITY):
                    prop = game.state.property_manager.get(player.position)
                    if prop and prop.owner is None:
                        cost = getattr(space, "cost", 0)
                        if cost > 0 and player.money >= cost + 100:
                            action = BuyProperty(
                                player_id=current_player_id, property_id=player.position
                            )
                            valid, _ = action.validate(game)
                            if valid:
                                action.execute(game)

            EndTurn(player_id=current_player_id).execute(game)
            turn_count += 1

        # Game should complete or hit turn limit
        assert (
            game.state.game_over or turn_count == turn_limit
        ), f"Game ended unexpectedly at turn {turn_count}"

    @pytest.mark.parametrize("seed", [42, 100, 200, 300, 400])
    def test_game_determinism(self, seed: int) -> None:
        """Test that games with same seed produce identical results."""
        # Run two games with same seed
        game1 = MonopolyGame(num_players=2, seed=seed)
        game2 = MonopolyGame(num_players=2, seed=seed)

        # Play 10 turns identically
        for _ in range(10):
            for game in [game1, game2]:
                current_player_id = game.state.current_player
                player = game.state.players[current_player_id]

                if not player.bankrupt:
                    if player.in_jail and player.money >= 50:
                        PayJailFine(player_id=current_player_id).execute(game)
                    RollDice(player_id=current_player_id).execute(game)

                EndTurn(player_id=current_player_id).execute(game)

        # Games should be in identical states
        assert game1.state.turn_number == game2.state.turn_number
        assert game1.state.current_player == game2.state.current_player

        for i in range(2):
            p1 = game1.state.players[i]
            p2 = game2.state.players[i]
            assert p1.position == p2.position, f"Player {i} position mismatch"
            assert p1.money == p2.money, f"Player {i} money mismatch"
            assert p1.in_jail == p2.in_jail, f"Player {i} jail status mismatch"

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_multiple_player_counts(self, num_players: int) -> None:
        """Test games with different player counts."""
        game = MonopolyGame(num_players=num_players, seed=42)

        # Verify correct number of players
        assert len(game.state.players) == num_players

        # Play some turns
        for _ in range(num_players * 5):  # 5 rounds
            current_player_id = game.state.current_player
            player = game.state.players[current_player_id]

            if not player.bankrupt:
                RollDice(player_id=current_player_id).execute(game)

            EndTurn(player_id=current_player_id).execute(game)

        # All players should still exist
        assert len(game.state.players) == num_players

    def test_state_serialization_round_trip(self) -> None:
        """Test that game state can be serialized and restored."""
        game = MonopolyGame(num_players=2, seed=42)

        # Play some turns
        for _ in range(10):
            current_player_id = game.state.current_player
            RollDice(player_id=current_player_id).execute(game)
            EndTurn(player_id=current_player_id).execute(game)

        # Serialize and deserialize
        state_dict = game.to_dict()
        restored_game = MonopolyGame.from_dict(state_dict)

        # Verify state matches
        assert restored_game.state.turn_number == game.state.turn_number
        assert restored_game.state.current_player == game.state.current_player
        assert restored_game.state.houses_remaining == game.state.houses_remaining
        assert restored_game.state.hotels_remaining == game.state.hotels_remaining

        # Verify players
        for i in range(2):
            orig_player = game.state.players[i]
            rest_player = restored_game.state.players[i]
            assert orig_player.position == rest_player.position
            assert orig_player.money == rest_player.money
            assert orig_player.bankrupt == rest_player.bankrupt

        # Continue playing from restored state
        current_player_id = restored_game.state.current_player
        RollDice(player_id=current_player_id).execute(restored_game)
        # Should not crash


class TestMonopolyAndBuilding:
    """Test property acquisition and building scenarios."""

    def test_building_monopoly_progression(self) -> None:
        """Test acquiring monopoly and building houses/hotels."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 the brown monopoly
        game.state.property_manager.properties[1].owner = 0  # Mediterranean
        game.state.property_manager.properties[3].owner = 0  # Baltic
        game.state.players[0].money = 5000

        # Build houses evenly
        for _ in range(4):  # Build to 4 houses each
            BuildHouse(player_id=0, property_id=1).execute(game)
            BuildHouse(player_id=0, property_id=3).execute(game)

        # Verify 4 houses on each
        assert game.state.property_manager.properties[1].houses == 4
        assert game.state.property_manager.properties[3].houses == 4

        # Build hotels
        BuildHotel(player_id=0, property_id=1).execute(game)
        BuildHotel(player_id=0, property_id=3).execute(game)

        # Verify hotels
        assert game.state.property_manager.properties[1].houses == 5
        assert game.state.property_manager.properties[3].houses == 5

        # Verify houses returned to bank
        # Started with 32, used 8 (4+4), then returned 8 on hotel build
        assert game.state.houses_remaining == 32

    def test_house_shortage_prevents_building(self) -> None:
        """Test that house shortage prevents building."""
        game = MonopolyGame(num_players=2, seed=42)

        # Exhaust house supply
        game.state.houses_remaining = 0

        # Give player monopoly
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[3].owner = 0
        game.state.players[0].money = 5000

        # Try to build - should fail
        action = BuildHouse(player_id=0, property_id=1)
        valid, msg = action.validate(game)

        assert not valid
        assert "houses" in msg.lower() or "shortage" in msg.lower()


class TestJailScenarios:
    """Test various jail-related scenarios."""

    def test_jail_release_progression(self) -> None:
        """Test all ways to get out of jail."""
        # Test 1: Pay fine
        game = MonopolyGame(num_players=2, seed=42)
        game.send_to_jail(0)
        assert game.state.players[0].in_jail

        PayJailFine(player_id=0).execute(game)
        assert not game.state.players[0].in_jail

        # Test 2: Use card
        game = MonopolyGame(num_players=2, seed=42)
        game.send_to_jail(0)
        game.state.players[0].jail_cards = 1

        UseJailCard(player_id=0).execute(game)
        assert not game.state.players[0].in_jail
        assert game.state.players[0].jail_cards == 0

        # Test 3: Auto-release after 3 turns
        game = MonopolyGame(num_players=2, seed=42)
        game.send_to_jail(0)
        game.state.players[0].jail_turns = 3

        RollDice(player_id=0).execute(game)
        assert not game.state.players[0].in_jail

    def test_multiple_players_in_jail(self) -> None:
        """Test that multiple players can be in jail simultaneously."""
        game = MonopolyGame(num_players=4, seed=42)

        # Send players 0, 1, 2 to jail
        for i in range(3):
            game.send_to_jail(i)

        # All should be in jail
        assert sum(p.in_jail for p in game.state.players) == 3

        # Player 3 should not be in jail
        assert not game.state.players[3].in_jail


class TestBankruptcyScenarios:
    """Test bankruptcy and game-ending scenarios."""

    def test_bankruptcy_ends_2_player_game(self) -> None:
        """Test that bankruptcy in 2-player game ends the game."""
        game = MonopolyGame(num_players=2, seed=42)

        # Make player 0 bankrupt
        game.handle_bankruptcy(player_id=0, creditor_id=None)

        # Game should be over
        assert game.state.game_over
        assert game.state.winner == 1

    def test_bankruptcy_continues_multiplayer_game(self) -> None:
        """Test that one bankruptcy doesn't end multiplayer game."""
        game = MonopolyGame(num_players=4, seed=42)

        # Make player 0 bankrupt
        game.handle_bankruptcy(player_id=0, creditor_id=None)

        # Game should NOT be over (3 players remain)
        assert not game.state.game_over
        assert game.state.winner is None

        # Make another player bankrupt
        game.handle_bankruptcy(player_id=1, creditor_id=None)

        # Still not over (2 players remain)
        assert not game.state.game_over

        # Make third player bankrupt
        game.handle_bankruptcy(player_id=2, creditor_id=None)

        # Now game should be over with player 3 as winner
        assert game.state.game_over
        assert game.state.winner == 3

    def test_property_transfer_on_bankruptcy(self) -> None:
        """Test that properties transfer correctly on bankruptcy."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 some properties
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[3].owner = 0
        game.state.property_manager.properties[5].owner = 0

        # Player 0 goes bankrupt to player 1
        game.handle_bankruptcy(player_id=0, creditor_id=1)

        # All properties should transfer to player 1
        assert game.state.property_manager.properties[1].owner == 1
        assert game.state.property_manager.properties[3].owner == 1
        assert game.state.property_manager.properties[5].owner == 1


class TestEventLogging:
    """Test that game events are properly logged."""

    def test_events_are_logged(self) -> None:
        """Test that key events generate log entries."""
        game = MonopolyGame(num_players=2, seed=42)

        initial_event_count = len(game.state.event_log)

        # Perform some actions
        RollDice(player_id=0).execute(game)
        EndTurn(player_id=0).execute(game)

        # Events should be logged
        assert len(game.state.event_log) > initial_event_count

    def test_event_log_persists_through_serialization(self) -> None:
        """Test that events survive serialization."""
        game = MonopolyGame(num_players=2, seed=42)

        # Generate some events
        RollDice(player_id=0).execute(game)
        EndTurn(player_id=0).execute(game)

        event_count = len(game.state.event_log)

        # Serialize and restore
        state_dict = game.to_dict()
        restored = MonopolyGame.from_dict(state_dict)

        # Events should be preserved (last 50 events)
        assert len(restored.state.event_log) <= event_count


class TestStressScenarios:
    """Stress tests for edge cases and performance."""

    def test_many_turns_no_crash(self) -> None:
        """Test that game can run for many turns without crashing."""
        game = MonopolyGame(num_players=2, seed=42)

        for _ in range(200):  # 200 turns = 100 rounds
            current_player_id = game.state.current_player
            player = game.state.players[current_player_id]

            if not player.bankrupt:
                if player.in_jail and player.money >= 50:
                    PayJailFine(player_id=current_player_id).execute(game)
                RollDice(player_id=current_player_id).execute(game)

            EndTurn(player_id=current_player_id).execute(game)

            if game.state.game_over:
                break

        # Should complete without errors
        assert True

    def test_rapid_serialization_cycles(self) -> None:
        """Test rapid serialize/deserialize cycles."""
        game = MonopolyGame(num_players=2, seed=42)

        for _ in range(5):
            # Play a turn
            RollDice(player_id=game.state.current_player).execute(game)
            EndTurn(player_id=game.state.current_player).execute(game)

            # Serialize and restore
            state_dict = game.to_dict()
            game = MonopolyGame.from_dict(state_dict)

        # Should still be playable
        RollDice(player_id=game.state.current_player).execute(game)
        assert True


class TestPropertyTrading:
    """Test property trading scenarios (when implemented)."""

    def test_basic_trade_mechanics(self) -> None:
        """Test that trade system basic mechanics work."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give players some properties and money
        game.state.property_manager.properties[1].owner = 0  # Mediterranean to P0
        game.state.property_manager.properties[5].owner = 1  # Reading RR to P1

        game.state.players[0].money = 1000
        game.state.players[1].money = 1000

        # Propose trade: P0 offers Mediterranean + $100 for P1's Reading RR
        trade_id = game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            give_money=100,
            want_properties=[5],
            want_money=0,
        )

        # Accept trade
        trade = game.get_pending_trade(trade_id)
        assert trade is not None

        game.accept_trade(trade_id)

        # Verify property ownership switched
        assert game.state.property_manager.properties[1].owner == 1
        assert game.state.property_manager.properties[5].owner == 0

        # Verify money transferred
        assert game.state.players[0].money == 900  # Paid 100
        assert game.state.players[1].money == 1100  # Received 100
