"""Tests for LLM response caching and budget management."""

from __future__ import annotations

from mcts.llm.budget import ResponseCache, TokenBudget
from monopoly_engine.game import MonopolyGame

# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_defaults(self) -> None:
        budget = TokenBudget()
        assert budget.max_calls_per_game == 20
        assert budget.max_tokens_per_game == 50_000
        assert budget.calls_used == 0
        assert budget.tokens_used == 0

    def test_budget_remaining_initially_true(self) -> None:
        budget = TokenBudget()
        assert budget.budget_remaining is True

    def test_record_usage(self) -> None:
        budget = TokenBudget()
        budget.record_usage(tokens=500)
        assert budget.calls_used == 1
        assert budget.tokens_used == 500

    def test_record_usage_no_tokens(self) -> None:
        budget = TokenBudget()
        budget.record_usage()
        assert budget.calls_used == 1
        assert budget.tokens_used == 0

    def test_budget_exhausted_by_calls(self) -> None:
        budget = TokenBudget(max_calls_per_game=3)
        for _ in range(3):
            budget.record_usage(tokens=10)
        assert budget.budget_remaining is False

    def test_budget_exhausted_by_tokens(self) -> None:
        budget = TokenBudget(max_tokens_per_game=1000)
        budget.record_usage(tokens=1000)
        assert budget.budget_remaining is False

    def test_budget_remaining_just_under_limit(self) -> None:
        budget = TokenBudget(max_calls_per_game=5, max_tokens_per_game=1000)
        for _ in range(4):
            budget.record_usage(tokens=200)
        assert budget.budget_remaining is True

    def test_reset(self) -> None:
        budget = TokenBudget()
        budget.record_usage(tokens=500)
        budget.record_usage(tokens=300)
        budget.reset()
        assert budget.calls_used == 0
        assert budget.tokens_used == 0
        assert budget.budget_remaining is True

    def test_custom_limits(self) -> None:
        budget = TokenBudget(max_calls_per_game=5, max_tokens_per_game=2000)
        assert budget.max_calls_per_game == 5
        assert budget.max_tokens_per_game == 2000

    def test_cumulative_tokens(self) -> None:
        budget = TokenBudget()
        budget.record_usage(tokens=100)
        budget.record_usage(tokens=200)
        budget.record_usage(tokens=300)
        assert budget.calls_used == 3
        assert budget.tokens_used == 600


# ---------------------------------------------------------------------------
# ResponseCache - basic operations
# ---------------------------------------------------------------------------


class TestResponseCache:
    def test_put_and_get(self) -> None:
        cache = ResponseCache()
        cache.put("state1", "prompt1", {"decision": "accept"})
        result = cache.get("state1", "prompt1")
        assert result == {"decision": "accept"}

    def test_miss_returns_none(self) -> None:
        cache = ResponseCache()
        assert cache.get("missing", "missing") is None

    def test_different_keys_independent(self) -> None:
        cache = ResponseCache()
        cache.put("s1", "p1", {"a": 1})
        cache.put("s1", "p2", {"a": 2})
        cache.put("s2", "p1", {"a": 3})

        assert cache.get("s1", "p1") == {"a": 1}
        assert cache.get("s1", "p2") == {"a": 2}
        assert cache.get("s2", "p1") == {"a": 3}

    def test_overwrite_existing_key(self) -> None:
        cache = ResponseCache()
        cache.put("s1", "p1", {"v": 1})
        cache.put("s1", "p1", {"v": 2})
        assert cache.get("s1", "p1") == {"v": 2}
        assert cache.size == 1

    def test_clear(self) -> None:
        cache = ResponseCache()
        cache.put("s1", "p1", {"a": 1})
        cache.put("s2", "p2", {"a": 2})
        cache.clear()
        assert cache.size == 0
        assert cache.get("s1", "p1") is None

    def test_size(self) -> None:
        cache = ResponseCache()
        assert cache.size == 0
        cache.put("s1", "p1", {"a": 1})
        assert cache.size == 1
        cache.put("s2", "p2", {"a": 2})
        assert cache.size == 2


# ---------------------------------------------------------------------------
# ResponseCache - eviction
# ---------------------------------------------------------------------------


class TestResponseCacheEviction:
    def test_evicts_oldest_when_full(self) -> None:
        cache = ResponseCache(max_size=3)
        cache.put("s1", "p1", {"a": 1})
        cache.put("s2", "p2", {"a": 2})
        cache.put("s3", "p3", {"a": 3})
        # This should evict s1:p1
        cache.put("s4", "p4", {"a": 4})

        assert cache.size == 3
        assert cache.get("s1", "p1") is None  # evicted
        assert cache.get("s2", "p2") == {"a": 2}
        assert cache.get("s4", "p4") == {"a": 4}

    def test_access_refreshes_lru(self) -> None:
        cache = ResponseCache(max_size=3)
        cache.put("s1", "p1", {"a": 1})
        cache.put("s2", "p2", {"a": 2})
        cache.put("s3", "p3", {"a": 3})

        # Access s1 to make it most recently used
        cache.get("s1", "p1")

        # Now s2 is the oldest — should be evicted
        cache.put("s4", "p4", {"a": 4})
        assert cache.get("s1", "p1") == {"a": 1}  # kept (refreshed)
        assert cache.get("s2", "p2") is None  # evicted
        assert cache.get("s3", "p3") == {"a": 3}
        assert cache.get("s4", "p4") == {"a": 4}

    def test_max_size_one(self) -> None:
        cache = ResponseCache(max_size=1)
        cache.put("s1", "p1", {"a": 1})
        cache.put("s2", "p2", {"a": 2})
        assert cache.size == 1
        assert cache.get("s1", "p1") is None
        assert cache.get("s2", "p2") == {"a": 2}


# ---------------------------------------------------------------------------
# ResponseCache - hashing
# ---------------------------------------------------------------------------


class TestResponseCacheHashing:
    def test_hash_game_state_deterministic(self) -> None:
        game = MonopolyGame(num_players=3, seed=42)
        h1 = ResponseCache.hash_game_state(game, player_id=0)
        h2 = ResponseCache.hash_game_state(game, player_id=0)
        assert h1 == h2

    def test_hash_game_state_differs_by_player(self) -> None:
        game = MonopolyGame(num_players=3, seed=42)
        h0 = ResponseCache.hash_game_state(game, player_id=0)
        h1 = ResponseCache.hash_game_state(game, player_id=1)
        assert h0 != h1

    def test_hash_game_state_differs_after_change(self) -> None:
        game = MonopolyGame(num_players=3, seed=42)
        h_before = ResponseCache.hash_game_state(game, player_id=0)

        # Change game state
        game.players[0].money -= 100
        h_after = ResponseCache.hash_game_state(game, player_id=0)
        assert h_before != h_after

    def test_hash_game_state_differs_by_ownership(self) -> None:
        game = MonopolyGame(num_players=3, seed=42)
        h_before = ResponseCache.hash_game_state(game, player_id=0)

        game.property_manager.properties[1].owner = 0
        h_after = ResponseCache.hash_game_state(game, player_id=0)
        assert h_before != h_after

    def test_hash_game_state_is_hex_string(self) -> None:
        game = MonopolyGame(num_players=2, seed=1)
        h = ResponseCache.hash_game_state(game, player_id=0)
        assert isinstance(h, str)
        assert len(h) == 16
        int(h, 16)  # Should not raise

    def test_hash_prompt_deterministic(self) -> None:
        h1 = ResponseCache.hash_prompt("test prompt")
        h2 = ResponseCache.hash_prompt("test prompt")
        assert h1 == h2

    def test_hash_prompt_differs(self) -> None:
        h1 = ResponseCache.hash_prompt("prompt A")
        h2 = ResponseCache.hash_prompt("prompt B")
        assert h1 != h2

    def test_hash_prompt_is_hex_string(self) -> None:
        h = ResponseCache.hash_prompt("some prompt")
        assert isinstance(h, str)
        assert len(h) == 16
        int(h, 16)  # Should not raise
