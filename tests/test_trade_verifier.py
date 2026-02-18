"""Tests for MCTS trade verifier."""

from __future__ import annotations

from typing import Any

from mcts.search import MCTSConfig
from mcts.trade_verifier import MCTSTradeVerifier, TradeEvaluation
from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import TradeOfferData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game() -> MonopolyGame:
    """Create a game with property ownership for trade evaluation tests."""
    game = MonopolyGame(num_players=3, seed=42)
    pm = game.property_manager

    # Player 0: owns Mediterranean [1], Baltic [3] (Brown monopoly)
    pm.properties[1].owner = 0
    pm.properties[3].owner = 0

    # Player 1: owns Oriental [6], Vermont [8] (Light Blue, missing CT [9])
    pm.properties[6].owner = 1
    pm.properties[8].owner = 1

    # Player 0 owns Connecticut [9] (blocks Player 1's light blue monopoly)
    pm.properties[9].owner = 0

    # Player 2: owns St. Charles [11]
    pm.properties[11].owner = 2

    return game


def _make_trade() -> TradeOfferData:
    """Trade: Player 0 gives Connecticut [9] to Player 1 for $200."""
    return TradeOfferData(
        from_player=0,
        to_player=1,
        give_properties=[9],
        give_money=0,
        want_properties=[],
        want_money=200,
    )


def _make_bad_trade() -> TradeOfferData:
    """Trade: Player 0 gives Brown monopoly for nothing."""
    return TradeOfferData(
        from_player=0,
        to_player=1,
        give_properties=[1, 3],
        give_money=100,
        want_properties=[],
        want_money=0,
    )


# ---------------------------------------------------------------------------
# TradeEvaluation dataclass
# ---------------------------------------------------------------------------

class TestTradeEvaluation:
    def test_fields(self) -> None:
        trade = _make_trade()
        ev = TradeEvaluation(
            trade=trade,
            value_before=0.3,
            value_after=0.5,
            value_delta=0.2,
            recommended=True,
            simulations_used=50,
        )
        assert ev.value_before == 0.3
        assert ev.value_after == 0.5
        assert ev.value_delta == 0.2
        assert ev.recommended is True
        assert ev.simulations_used == 50
        assert ev.trade is trade


# ---------------------------------------------------------------------------
# MCTSTradeVerifier.__init__
# ---------------------------------------------------------------------------

class TestVerifierInit:
    def test_default_config(self) -> None:
        verifier = MCTSTradeVerifier()
        assert verifier.simulations_per_eval == 50
        assert verifier.acceptance_threshold == 0.02

    def test_custom_config(self) -> None:
        verifier = MCTSTradeVerifier(
            simulations_per_eval=100,
            acceptance_threshold=0.05,
        )
        assert verifier.simulations_per_eval == 100
        assert verifier.acceptance_threshold == 0.05

    def test_custom_mcts_config(self) -> None:
        config = MCTSConfig(num_simulations=10)
        verifier = MCTSTradeVerifier(mcts_config=config)
        assert verifier._config is config


# ---------------------------------------------------------------------------
# _apply_trade_to_clone
# ---------------------------------------------------------------------------

class TestApplyTradeToClone:
    def test_does_not_mutate_original(self) -> None:
        game = _make_game()
        original_money_0 = game.players[0].money
        original_money_1 = game.players[1].money
        original_owner_9 = game.property_manager.properties[9].owner

        trade = _make_trade()
        clone = MCTSTradeVerifier._apply_trade_to_clone(game, trade)

        # Original unchanged
        assert game.players[0].money == original_money_0
        assert game.players[1].money == original_money_1
        assert game.property_manager.properties[9].owner == original_owner_9

        # Clone has trade applied
        assert clone.property_manager.properties[9].owner == 1
        assert clone.players[0].money == original_money_0 + 200  # want_money
        assert clone.players[1].money == original_money_1 - 200

    def test_property_swap(self) -> None:
        game = _make_game()
        trade = TradeOfferData(
            from_player=0,
            to_player=1,
            give_properties=[9],
            give_money=0,
            want_properties=[6],
            want_money=0,
        )
        clone = MCTSTradeVerifier._apply_trade_to_clone(game, trade)

        assert clone.property_manager.properties[9].owner == 1
        assert clone.property_manager.properties[6].owner == 0

    def test_cash_transfer(self) -> None:
        game = _make_game()
        p0_money = game.players[0].money
        p1_money = game.players[1].money

        trade = TradeOfferData(
            from_player=0,
            to_player=1,
            give_properties=[],
            give_money=150,
            want_properties=[],
            want_money=50,
        )
        clone = MCTSTradeVerifier._apply_trade_to_clone(game, trade)

        # P0: -give(150) + want(50) = -100
        assert clone.players[0].money == p0_money - 100
        # P1: +give(150) - want(50) = +100
        assert clone.players[1].money == p1_money + 100


# ---------------------------------------------------------------------------
# evaluate_trade (with mocked MCTS)
# ---------------------------------------------------------------------------

class TestEvaluateTrade:
    def test_recommends_good_trade(self) -> None:
        """Mock simulate to return better value after trade."""
        verifier = MCTSTradeVerifier(
            simulations_per_eval=0,  # use simulate() directly
            acceptance_threshold=0.02,
        )
        game = _make_game()
        trade = _make_trade()

        call_count = 0
        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Before trade
                return {0: 0.3, 1: 0.3, 2: 0.4}
            else:
                # After trade - player 0 gets cash, improves
                return {0: 0.4, 1: 0.35, 2: 0.25}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        result = verifier.evaluate_trade(game, player_id=0, trade=trade)
        assert result.value_before == 0.3
        assert result.value_after == 0.4
        assert abs(result.value_delta - 0.1) < 1e-9
        assert result.recommended is True
        assert result.simulations_used == 1

    def test_rejects_bad_trade(self) -> None:
        """Mock simulate to return worse value after trade."""
        verifier = MCTSTradeVerifier(
            simulations_per_eval=0,
            acceptance_threshold=0.02,
        )
        game = _make_game()
        trade = _make_bad_trade()

        call_count = 0
        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {0: 0.4, 1: 0.3, 2: 0.3}
            else:
                return {0: 0.2, 1: 0.5, 2: 0.3}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        result = verifier.evaluate_trade(game, player_id=0, trade=trade)
        assert result.value_delta < 0
        assert result.recommended is False

    def test_threshold_boundary(self) -> None:
        """Delta exactly at threshold is recommended."""
        verifier = MCTSTradeVerifier(
            simulations_per_eval=0,
            acceptance_threshold=0.10,
        )
        game = _make_game()
        trade = _make_trade()

        call_count = 0
        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {0: 0.30, 1: 0.35, 2: 0.35}
            else:
                return {0: 0.40, 1: 0.30, 2: 0.30}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        result = verifier.evaluate_trade(game, player_id=0, trade=trade)
        assert result.recommended is True  # delta=0.10 >= threshold=0.10

    def test_threshold_just_below(self) -> None:
        """Delta just below threshold is not recommended."""
        verifier = MCTSTradeVerifier(
            simulations_per_eval=0,
            acceptance_threshold=0.10,
        )
        game = _make_game()
        trade = _make_trade()

        call_count = 0
        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {0: 0.30, 1: 0.35, 2: 0.35}
            else:
                return {0: 0.39, 1: 0.30, 2: 0.31}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        result = verifier.evaluate_trade(game, player_id=0, trade=trade)
        assert result.recommended is False  # delta=0.09 < 0.10

    def test_preserves_original_game(self) -> None:
        """Verify the original game is not mutated during evaluation."""
        verifier = MCTSTradeVerifier(simulations_per_eval=0)
        game = _make_game()
        trade = _make_trade()

        original_money = game.players[0].money
        original_owner = game.property_manager.properties[9].owner

        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            return {0: 0.3, 1: 0.3, 2: 0.4}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        verifier.evaluate_trade(game, player_id=0, trade=trade)

        assert game.players[0].money == original_money
        assert game.property_manager.properties[9].owner == original_owner


# ---------------------------------------------------------------------------
# rank_trades
# ---------------------------------------------------------------------------

class TestRankTrades:
    def test_ranks_by_value_delta(self) -> None:
        verifier = MCTSTradeVerifier(
            simulations_per_eval=0,
            acceptance_threshold=0.02,
        )
        game = _make_game()

        trade_good = _make_trade()  # Give CT[9] for $200
        trade_bad = _make_bad_trade()  # Give Brown monopoly for nothing

        call_count = 0
        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Before (shared for both trades)
                return {0: 0.3, 1: 0.3, 2: 0.4}
            elif call_count == 2:
                # After first candidate (trade_bad)
                return {0: 0.15, 1: 0.55, 2: 0.3}
            else:
                # After second candidate (trade_good)
                return {0: 0.45, 1: 0.25, 2: 0.3}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        results = verifier.rank_trades(
            game, player_id=0, candidates=[trade_bad, trade_good],
        )

        assert len(results) == 2
        # Good trade should be ranked first (higher delta)
        assert results[0].trade is trade_good
        assert results[0].recommended is True
        assert results[1].trade is trade_bad
        assert results[1].recommended is False

    def test_empty_candidates(self) -> None:
        verifier = MCTSTradeVerifier(simulations_per_eval=0)
        game = _make_game()

        results = verifier.rank_trades(game, player_id=0, candidates=[])
        assert results == []

    def test_single_candidate(self) -> None:
        verifier = MCTSTradeVerifier(
            simulations_per_eval=0,
            acceptance_threshold=0.0,
        )
        game = _make_game()
        trade = _make_trade()

        call_count = 0
        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {0: 0.3, 1: 0.3, 2: 0.4}
            else:
                return {0: 0.35, 1: 0.30, 2: 0.35}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        results = verifier.rank_trades(
            game, player_id=0, candidates=[trade],
        )
        assert len(results) == 1
        assert results[0].recommended is True

    def test_reuses_before_value(self) -> None:
        """Verify before value is evaluated once and reused."""
        verifier = MCTSTradeVerifier(simulations_per_eval=0)
        game = _make_game()

        trade1 = _make_trade()
        trade2 = _make_bad_trade()

        call_count = 0
        def fake_simulate(
            sim_game: Any, player_id: int,
        ) -> dict[int, float]:
            nonlocal call_count
            call_count += 1
            return {0: 0.3, 1: 0.3, 2: 0.4}

        verifier._searcher.simulate = fake_simulate  # type: ignore[assignment]

        results = verifier.rank_trades(
            game, player_id=0, candidates=[trade1, trade2],
        )

        # 1 before eval + 2 after evals = 3 total calls
        assert call_count == 3
        # Both should have same before value
        assert results[0].value_before == results[1].value_before


# ---------------------------------------------------------------------------
# Full search mode (simulations_per_eval > 0)
# ---------------------------------------------------------------------------

class TestFullSearchMode:
    def test_uses_search_then_simulate(self) -> None:
        """With simulations > 0, search() is called then simulate() for value."""
        verifier = MCTSTradeVerifier(
            simulations_per_eval=10,
            acceptance_threshold=0.0,
        )
        game = _make_game()
        trade = _make_trade()

        search_calls = 0
        simulate_calls = 0

        def mock_search(g: Any, pid: int) -> dict[int, int]:
            nonlocal search_calls
            search_calls += 1
            return {0: 5, 1: 3}  # dummy visit counts

        def mock_simulate(g: Any, pid: int) -> dict[int, float]:
            nonlocal simulate_calls
            simulate_calls += 1
            return {0: 0.3, 1: 0.3, 2: 0.4}

        verifier._searcher.search = mock_search  # type: ignore[assignment]
        verifier._searcher.simulate = mock_simulate  # type: ignore[assignment]

        result = verifier.evaluate_trade(game, player_id=0, trade=trade)

        # search is called, then simulate for value extraction
        assert search_calls == 2  # before + after
        assert simulate_calls == 2  # before + after (value extraction)
        assert result.simulations_used == 10

    def test_fallback_on_empty_search(self) -> None:
        """If search returns empty visit counts, falls back to simulate."""
        verifier = MCTSTradeVerifier(
            simulations_per_eval=10,
            acceptance_threshold=0.0,
        )
        game = _make_game()
        trade = _make_trade()

        def mock_search(g: Any, pid: int) -> dict[int, int]:
            return {}  # no valid actions

        simulate_calls = 0
        def mock_simulate(g: Any, pid: int) -> dict[int, float]:
            nonlocal simulate_calls
            simulate_calls += 1
            return {0: 0.3, 1: 0.3, 2: 0.4}

        verifier._searcher.search = mock_search  # type: ignore[assignment]
        verifier._searcher.simulate = mock_simulate  # type: ignore[assignment]

        verifier.evaluate_trade(game, player_id=0, trade=trade)
        assert simulate_calls == 2  # fallback for both before and after
