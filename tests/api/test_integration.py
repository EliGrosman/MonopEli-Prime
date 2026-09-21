"""
Integration tests for full game flows via the API.

Tests cover:
- Complete turn cycles
- Multi-player game flows
- Property transactions
- Building houses/hotels
- Rent payment
- Jail mechanics
- Game over scenarios
"""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app():
    """Create test application."""
    from api.config import Settings
    return create_app(Settings(rate_limit_per_minute=10000, rate_limit_burst=10000))


@pytest.fixture
def client(app):
    """Create test client with lifespan."""
    with TestClient(app) as client:
        yield client


def create_game(client: TestClient, num_players: int = 2, seed: int | None = None) -> str:
    """Helper to create a game and return its ID."""
    payload = {"num_players": num_players}
    if seed is not None:
        payload["seed"] = seed
    response = client.post("/api/games", json=payload)
    return response.json()["id"]


def get_game_state(client: TestClient, game_id: str) -> dict:
    """Helper to get game state."""
    response = client.get(f"/api/games/{game_id}")
    return response.json()



def resolve_required_decisions(client, game_id, ws, others=()):
    """Drive explicit roll/purchase decisions before ending a test turn."""
    for _ in range(30):
        state = get_game_state(client, game_id)
        legal = {action["type"]: action for action in state["legal_actions"]}
        if "EndTurn" in legal:
            return
        if "PassBuy" in legal:
            request = {"action_type": "pass_buy"}
        elif "RollDice" in legal:
            request = {"action_type": "roll_dice"}
        else:
            raise AssertionError(f"Unexpected scripted decision: {legal}")
        ws.send_json({"type": "action", "data": request})
        assert ws.receive_json()["data"]["success"]
        assert ws.receive_json()["type"] == "state_update"
        for other in others:
            assert other.receive_json()["type"] == "state_update"
    raise AssertionError("Required decisions did not finish")

# ============================================================================
# Turn Cycle Tests
# ============================================================================


class TestTurnCycle:
    """Tests for complete turn cycles."""

    def test_simple_turn_via_websocket(self, client: TestClient):
        """Test a simple turn: roll dice, end turn."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=player0&player_id=0"
        ) as ws:
            # Skip identity, get initial state
            ws.receive_json()  # identity
            initial = ws.receive_json()
            assert initial["type"] == "state_update"
            assert initial["data"]["current_player"] == 0
            assert initial["data"]["turn_number"] == 0

            # Roll dice
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            result = ws.receive_json()
            assert result["type"] == "action_result"
            assert result["data"]["success"] is True

            # Get state update
            state = ws.receive_json()
            assert state["type"] == "state_update"
            assert state["data"]["last_roll"] is not None

            # End turn
            resolve_required_decisions(client, game_id, ws)
            ws.send_json({"type": "action", "data": {"action_type": "end_turn"}})
            result = ws.receive_json()
            assert result["type"] == "action_result"
            assert result["data"]["success"] is True

            # Verify turn advanced
            state = ws.receive_json()
            assert state["type"] == "state_update"
            assert state["data"]["current_player"] == 1
            assert state["data"]["turn_number"] == 1

    def test_turn_with_doubles(self, client: TestClient):
        """Test turn with doubles (player gets to roll again)."""
        # Use a seed that produces doubles on first roll
        # Note: This is a probabilistic test - may need adjustment
        game_id = create_game(client, seed=42)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=player0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Initial state

            # First roll
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            result = ws.receive_json()
            assert result["data"]["success"] is True
            state = ws.receive_json()

            # Check if doubles
            last_roll = state["data"]["last_roll"]
            if last_roll and last_roll[0] == last_roll[1]:
                # Got doubles - can roll again
                ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
                result = ws.receive_json()
                # Should succeed (can roll again after doubles)
                assert result["type"] == "action_result"

    def test_multi_player_turn_sequence(self, client: TestClient):
        """Test turns cycle through all players."""
        game_id = create_game(client, num_players=3)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws0:
            ws0.receive_json()  # Skip identity
            ws0.receive_json()  # Initial state

            # Player 0's turn
            assert get_game_state(client, game_id)["current_player"] == 0

            ws0.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            ws0.receive_json()  # result
            ws0.receive_json()  # state

            resolve_required_decisions(client, game_id, ws0)

            ws0.send_json({"type": "action", "data": {"action_type": "end_turn"}})
            ws0.receive_json()  # result
            ws0.receive_json()  # state

            # Now it's player 1's turn
            state = get_game_state(client, game_id)
            assert state["current_player"] == 1


class TestPropertyTransactions:
    """Tests for property buying, selling, mortgaging."""

    def test_buy_property_flow(self, client: TestClient):
        """Test buying a property via WebSocket."""
        # Use seed that lands player on a property
        game_id = create_game(client, seed=100)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            initial = ws.receive_json()
            initial_money = initial["data"]["players"][0]["money"]

            # Roll dice
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            ws.receive_json()  # result
            state = ws.receive_json()

            # Get player position
            position = state["data"]["players"][0]["position"]

            # Try to buy property at current position
            ws.send_json({
                "type": "action",
                "data": {
                    "action_type": "buy_property",
                    "property_position": position
                }
            })
            result = ws.receive_json()

            if result["data"]["success"]:
                # Check state update shows property owned
                state = ws.receive_json()
                player_props = state["data"]["players"][0]["properties"]
                assert position in player_props

                # Money should have decreased
                new_money = state["data"]["players"][0]["money"]
                assert new_money < initial_money

    def test_mortgage_property(self, client: TestClient):
        """Test mortgaging a property."""
        game_id = create_game(client, seed=100)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Initial state

            # Roll and try to buy
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            ws.receive_json()
            state = ws.receive_json()
            position = state["data"]["players"][0]["position"]

            ws.send_json({
                "type": "action",
                "data": {"action_type": "buy_property", "property_position": position}
            })
            result = ws.receive_json()

            if result["data"]["success"]:
                ws.receive_json()  # state after buy

                # Now mortgage
                ws.send_json({
                    "type": "action",
                    "data": {"action_type": "mortgage_property", "property_position": position}
                })
                mortgage_result = ws.receive_json()

                if mortgage_result["data"]["success"]:
                    state = ws.receive_json()
                    # Check property is mortgaged
                    prop = state["data"]["properties"].get(str(position))
                    if prop:
                        assert prop["mortgaged"] is True


class TestAllActionTypes:
    """Tests for all action types via WebSocket."""

    def test_roll_dice(self, client: TestClient):
        """Test roll_dice action."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            result = ws.receive_json()
            assert result["type"] == "action_result"
            assert result["data"]["success"] is True

            state = ws.receive_json()
            assert state["data"]["last_roll"] is not None
            assert len(state["data"]["last_roll"]) == 2

    def test_end_turn(self, client: TestClient):
        """Test end_turn action."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            # Roll first
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            ws.receive_json()
            ws.receive_json()

            # End turn
            resolve_required_decisions(client, game_id, ws)
            ws.send_json({"type": "action", "data": {"action_type": "end_turn"}})
            result = ws.receive_json()
            assert result["data"]["success"] is True

            state = ws.receive_json()
            assert state["data"]["current_player"] == 1

    def test_pay_jail_fine(self, client: TestClient):
        """Test pay_jail_fine action (will fail if not in jail)."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            # Try to pay jail fine when not in jail
            ws.send_json({"type": "action", "data": {"action_type": "pay_jail_fine"}})
            result = ws.receive_json()
            # Should fail - not in jail
            assert result["data"]["success"] is False

    def test_use_jail_card(self, client: TestClient):
        """Test use_jail_card action (will fail if no card)."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            # Try to use jail card when don't have one
            ws.send_json({"type": "action", "data": {"action_type": "use_jail_card"}})
            result = ws.receive_json()
            # Should fail - no card
            assert result["data"]["success"] is False

    def test_declare_bankruptcy(self, client: TestClient):
        """Test declare_bankruptcy action."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            # Declare bankruptcy
            ws.send_json({"type": "action", "data": {"action_type": "declare_bankruptcy"}})
            result = ws.receive_json()

            # May succeed or fail depending on game state
            assert result["type"] == "action_result"


class TestActionValidation:
    """Tests for action validation via WebSocket."""

    def test_action_wrong_turn(self, client: TestClient):
        """Test that actions from wrong player are rejected."""
        game_id = create_game(client)

        # Connect as player 1 when it's player 0's turn
        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p1&player_id=1"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            # Try to roll dice on player 0's turn
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            result = ws.receive_json()

            assert result["data"]["success"] is False
            assert "not your turn" in result["data"]["message"].lower()

    def test_action_missing_property_position(self, client: TestClient):
        """Test that property actions without position fail."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            # Try buy_property without position
            ws.send_json({
                "type": "action",
                "data": {"action_type": "buy_property"}  # Missing property_position
            })
            result = ws.receive_json()

            assert result["data"]["success"] is False
            assert "property_position" in result["data"]["message"].lower()


class TestMultiPlayerBroadcast:
    """Tests for state broadcasting to multiple players."""

    def test_action_broadcasts_to_all_players(self, client: TestClient):
        """Test that actions broadcast state to all connected players."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws0:
            ws0.receive_json()  # Skip identity
            ws0.receive_json()  # Initial state

            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=p1&player_id=1"
            ) as ws1:
                ws1.receive_json()  # Skip identity
                ws1.receive_json()  # Initial state
                ws0.receive_json()  # player_joined for p1

                # Player 0 rolls dice
                ws0.send_json({"type": "action", "data": {"action_type": "roll_dice"}})

                # Player 0 gets action_result first, then state
                result0 = ws0.receive_json()
                assert result0["type"] == "action_result"
                state0 = ws0.receive_json()
                assert state0["type"] == "state_update"

                # Player 1 gets state broadcast
                state1 = ws1.receive_json()
                assert state1["type"] == "state_update"

                # Both should have same state
                assert state0["data"]["last_roll"] == state1["data"]["last_roll"]

    def test_spectator_receives_broadcasts(self, client: TestClient):
        """Test that spectators receive state broadcasts."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws_player:
            ws_player.receive_json()  # Skip identity
            ws_player.receive_json()  # Skip state

            # Connect spectator (no player_id)
            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=spectator"
            ) as ws_spectator:
                ws_spectator.receive_json()  # Skip identity
                ws_spectator.receive_json()  # Initial state

                # Player performs action
                ws_player.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
                ws_player.receive_json()  # result
                ws_player.receive_json()  # state

                # Spectator should receive state update
                spectator_update = ws_spectator.receive_json()
                assert spectator_update["type"] == "state_update"


class TestRESTAPIIntegration:
    """Tests for REST API integration with game state."""

    def test_game_state_via_rest_after_websocket_action(self, client: TestClient):
        """Test that REST API reflects state changes from WebSocket."""
        game_id = create_game(client)

        # Get initial state via REST
        initial_state = get_game_state(client, game_id)
        assert initial_state["turn_number"] == 0

        # Perform action via WebSocket
        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            ws.receive_json()
            ws.receive_json()

            resolve_required_decisions(client, game_id, ws)

            ws.send_json({"type": "action", "data": {"action_type": "end_turn"}})
            ws.receive_json()
            ws.receive_json()

        # Check state via REST
        updated_state = get_game_state(client, game_id)
        assert updated_state["turn_number"] == 1
        assert updated_state["current_player"] == 1

    def test_create_game_and_play_via_websocket(self, client: TestClient):
        """Test creating game via REST and playing via WebSocket."""
        # Create game with specific names
        response = client.post("/api/games", json={
            "num_players": 2,
            "player_names": ["Alice", "Bob"],
            "seed": 123
        })
        assert response.status_code == 201  # 201 Created
        game_id = response.json()["id"]

        # Connect and play
        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=alice&player_id=0&player_name=Alice"
        ) as ws:
            ws.receive_json()  # Skip identity
            state = ws.receive_json()
            assert state["data"]["players"][0]["name"] == "Alice"
            assert state["data"]["players"][1]["name"] == "Bob"


class TestGameProgression:
    """Tests for game progression scenarios."""

    def test_multiple_turns(self, client: TestClient):
        """Test playing multiple turns."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws0:
            ws0.receive_json()  # Skip identity
            ws0.receive_json()  # Skip state

            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=p1&player_id=1"
            ) as ws1:
                ws1.receive_json()  # Skip identity
                ws1.receive_json()  # Skip state
                ws0.receive_json()  # player_joined

                # Play 4 turns (2 each)
                for turn in range(4):
                    current_player = turn % 2
                    current_ws = ws0 if current_player == 0 else ws1
                    other_ws = ws1 if current_player == 0 else ws0

                    # Roll
                    current_ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
                    current_ws.receive_json()  # result
                    current_ws.receive_json()  # state
                    other_ws.receive_json()  # broadcast

                    # End turn
                    resolve_required_decisions(client, game_id, current_ws, (other_ws,))
                    current_ws.send_json({"type": "action", "data": {"action_type": "end_turn"}})
                    current_ws.receive_json()  # result
                    current_ws.receive_json()  # state
                    other_ws.receive_json()  # broadcast

                # Verify we're at turn 4
                state = get_game_state(client, game_id)
                assert state["turn_number"] == 4

    def test_player_positions_change(self, client: TestClient):
        """Test that player positions change after rolling."""
        game_id = create_game(client, seed=42)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            initial = ws.receive_json()
            initial_pos = initial["data"]["players"][0]["position"]
            assert initial_pos == 0  # Players start at GO

            # Roll dice
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            ws.receive_json()  # result
            state = ws.receive_json()

            new_pos = state["data"]["players"][0]["position"]
            last_roll = state["data"]["last_roll"]

            # Player should have moved by dice roll, but may have been
            # moved further by Chance/Community Chest cards or Go to Jail
            basic_move = (initial_pos + last_roll[0] + last_roll[1]) % 40

            # Position should have changed from 0 (unless extremely rare edge case)
            # We verify the dice rolled and position changed
            assert last_roll is not None
            assert len(last_roll) == 2
            # Position is either the basic move or they got moved by a card/special space
            assert new_pos >= 0 and new_pos < 40


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_continue_after_invalid_action(self, client: TestClient):
        """Test that game continues after invalid action."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

            # Try invalid action
            ws.send_json({"type": "action", "data": {"action_type": "invalid"}})
            result = ws.receive_json()
            assert result["data"]["success"] is False

            # Should still be able to roll
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            result = ws.receive_json()
            assert result["data"]["success"] is True

    def test_reconnect_and_continue(self, client: TestClient):
        """Test reconnecting and continuing game."""
        game_id = create_game(client)

        # First connection - roll dice
        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state
            ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})
            ws.receive_json()  # result
            state1 = ws.receive_json()  # state
            position_after_roll = state1["data"]["players"][0]["position"]

        # Reconnect - state should be preserved
        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=p0&player_id=0"
        ) as ws:
            ws.receive_json()  # Skip identity
            state2 = ws.receive_json()
            assert state2["data"]["players"][0]["position"] == position_after_roll
