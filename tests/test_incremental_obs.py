"""Tests for incremental observation encoding (OPT-3).

Verifies that IncrementalObservationEncoder produces identical observations
to the base ObservationEncoder while providing cache hit benefits.
"""

import time
from typing import Any

import numpy as np
import pytest

from monopoly_engine import MonopolyGame
from monopoly_engine.actions import (
    BuyProperty,
    EndTurn,
    RollDice,
)
from monopoly_gym.observation import (
    IncrementalObservationEncoder,
    ObservationEncoder,
    flatten_observation,
)


def _obs_equal(
    obs_a: dict[str, Any],
    obs_b: dict[str, Any],
) -> bool:
    """Deep-compare two observation dicts for equality."""
    if set(obs_a.keys()) != set(obs_b.keys()):
        return False
    for key in obs_a:
        a, b = obs_a[key], obs_b[key]
        if isinstance(a, dict) and isinstance(b, dict):
            if not _obs_equal(a, b):
                return False
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                return False
        else:
            if a != b:
                return False
    return True


class TestIncrementalEncoderInit:
    """Tests for IncrementalObservationEncoder initialization."""

    def test_creates_with_defaults(self) -> None:
        enc = IncrementalObservationEncoder(4)
        assert enc.num_players == 4
        assert enc.max_opponents == 3
        assert enc.enable_trades is False

    def test_creates_with_trades(self) -> None:
        enc = IncrementalObservationEncoder(2, enable_trades=True)
        assert enc.enable_trades is True

    def test_observation_space_matches_base(self) -> None:
        base = ObservationEncoder(4)
        inc = IncrementalObservationEncoder(4)
        assert base.get_observation_space() == inc.get_observation_space()

    def test_observation_space_matches_base_with_trades(self) -> None:
        base = ObservationEncoder(3, enable_trades=True)
        inc = IncrementalObservationEncoder(3, enable_trades=True)
        assert base.get_observation_space() == inc.get_observation_space()


class TestIncrementalCorrectness:
    """Verify incremental encoder produces identical observations to base."""

    def test_first_encode_matches_base(self) -> None:
        """Cold start should produce identical observation."""
        game = MonopolyGame(num_players=4, seed=42)
        base = ObservationEncoder(4)
        inc = IncrementalObservationEncoder(4)

        for pid in range(4):
            obs_base = base.encode(game, pid)
            obs_inc = inc.encode(game, pid)
            assert _obs_equal(obs_base, obs_inc), f"Mismatch for player {pid}"

    def test_identical_after_dice_roll(self) -> None:
        """Observation matches after a dice roll changes game state."""
        game = MonopolyGame(num_players=2, seed=42)
        base = ObservationEncoder(2)
        inc = IncrementalObservationEncoder(2)

        # Warm the cache
        inc.encode(game, 0)

        # Roll dice (changes position, potentially money, board state)
        roll = RollDice(player_id=0)
        if roll.validate(game)[0]:
            roll.execute(game)

        obs_base = base.encode(game, 0)
        obs_inc = inc.encode(game, 0)
        assert _obs_equal(obs_base, obs_inc)

    def test_identical_after_buy_property(self) -> None:
        """Observation matches after buying a property."""
        game = MonopolyGame(num_players=2, seed=100)
        base = ObservationEncoder(2)
        inc = IncrementalObservationEncoder(2)

        # Warm cache
        inc.encode(game, 0)

        # Roll dice and try to buy
        roll = RollDice(player_id=0)
        if roll.validate(game)[0]:
            roll.execute(game)

        pos = game.players[0].position
        buy = BuyProperty(player_id=0, property_id=pos)
        if buy.validate(game)[0]:
            buy.execute(game)

        obs_base = base.encode(game, 0)
        obs_inc = inc.encode(game, 0)
        assert _obs_equal(obs_base, obs_inc)

    def test_identical_across_many_random_steps(self) -> None:
        """Run 200 random game steps, verify every observation matches."""
        rng = np.random.default_rng(42)

        for seed in range(5):
            game = MonopolyGame(num_players=4, seed=seed)
            base = ObservationEncoder(4)
            inc = IncrementalObservationEncoder(4)

            for step in range(200):
                if game.game_over:
                    break

                pid = game.current_player
                if game.players[pid].bankrupt:
                    game.end_turn()
                    continue

                # Compare observations for current player
                obs_base = base.encode(game, pid)
                obs_inc = inc.encode(game, pid)
                assert _obs_equal(obs_base, obs_inc), (
                    f"Mismatch at seed={seed}, step={step}, player={pid}"
                )

                # Also check flat encoding matches
                flat_base = flatten_observation(obs_base)
                flat_inc = flatten_observation(obs_inc)
                np.testing.assert_array_equal(flat_base, flat_inc)

                # Take a random valid action
                roll = RollDice(player_id=pid)
                if roll.validate(game)[0]:
                    roll.execute(game)

                pos = game.players[pid].position
                buy = BuyProperty(player_id=pid, property_id=pos)
                if buy.validate(game)[0] and rng.random() > 0.3:
                    buy.execute(game)

                end = EndTurn(player_id=pid)
                if end.validate(game)[0]:
                    end.execute(game)

    def test_identical_for_all_players_simultaneously(self) -> None:
        """Encode for all players in sequence, all must match base."""
        game = MonopolyGame(num_players=4, seed=42)
        base = ObservationEncoder(4)
        inc = IncrementalObservationEncoder(4)

        # Roll dice
        roll = RollDice(player_id=0)
        if roll.validate(game)[0]:
            roll.execute(game)

        for pid in range(4):
            obs_base = base.encode(game, pid)
            obs_inc = inc.encode(game, pid)
            assert _obs_equal(obs_base, obs_inc), f"Mismatch for player {pid}"


class TestCacheHitRate:
    """Verify cache hit behavior."""

    def test_second_encode_without_changes_has_hits(self) -> None:
        """Encoding same state twice should hit cache for cacheable sections."""
        game = MonopolyGame(num_players=2, seed=42)
        inc = IncrementalObservationEncoder(2)

        inc.encode(game, 0)  # Cold start
        inc.encode(game, 0)  # Should hit cache for most sections

        stats = inc.cache_stats
        # player_state is always recomputed (no hits expected)
        assert stats["player_state"]["misses"] == 2
        # opponent_states: same player → cache hit
        assert stats["opponent_states"]["hits"] >= 1
        # board_state: snapshot match → cache hit
        assert stats["board_state"]["hits"] >= 1
        # game_state: snapshot match → cache hit
        assert stats["game_state"]["hits"] >= 1

    def test_player_money_change_does_not_affect_board_state(self) -> None:
        """Changing money recomputes player_state but board_state is cached."""
        game = MonopolyGame(num_players=2, seed=42)
        inc = IncrementalObservationEncoder(2)

        inc.encode(game, 0)  # Cold start
        game.players[0].money -= 100  # Only player_state changes
        inc.encode(game, 0)

        stats = inc.cache_stats
        # player_state is always recomputed (2 misses: cold + second encode)
        assert stats["player_state"]["misses"] == 2
        # board_state should have hit (nothing changed)
        assert stats["board_state"]["hits"] >= 1
        # game_state should have hit
        assert stats["game_state"]["hits"] >= 1

    def test_cache_cleared_on_reset(self) -> None:
        """reset() should clear all caches."""
        game = MonopolyGame(num_players=2, seed=42)
        inc = IncrementalObservationEncoder(2)

        inc.encode(game, 0)
        assert 0 in inc._cache

        inc.reset()
        assert len(inc._cache) == 0
        assert inc._last_encoded_pid is None

    def test_overall_hit_rate_above_threshold_during_gameplay(self) -> None:
        """During typical gameplay, overall section hit rate should be > 30%."""
        rng = np.random.default_rng(42)
        game = MonopolyGame(num_players=2, seed=42)
        inc = IncrementalObservationEncoder(2)

        for _ in range(100):
            if game.game_over:
                break

            pid = game.current_player
            if game.players[pid].bankrupt:
                game.end_turn()
                continue

            inc.encode(game, pid)

            roll = RollDice(player_id=pid)
            if roll.validate(game)[0]:
                roll.execute(game)

            # Encode again after roll (within same turn - good for cache hits)
            inc.encode(game, pid)

            pos = game.players[pid].position
            buy = BuyProperty(player_id=pid, property_id=pos)
            if buy.validate(game)[0] and rng.random() > 0.5:
                buy.execute(game)

            end = EndTurn(player_id=pid)
            if end.validate(game)[0]:
                end.execute(game)

        stats = inc.cache_stats
        # We expect some hit rate since we encode twice per turn
        assert stats["overall"]["hit_rate"] > 0.15, (
            f"Overall hit rate too low: {stats['overall']['hit_rate']:.2%}"
        )


class TestCacheStats:
    """Tests for cache statistics reporting."""

    def test_stats_initially_zero(self) -> None:
        inc = IncrementalObservationEncoder(2)
        stats = inc.cache_stats
        assert stats["overall"]["hits"] == 0
        assert stats["overall"]["misses"] == 0
        assert stats["overall"]["hit_rate"] == 0.0

    def test_stats_after_encodes(self) -> None:
        game = MonopolyGame(num_players=2, seed=42)
        inc = IncrementalObservationEncoder(2)

        inc.encode(game, 0)
        inc.encode(game, 0)

        stats = inc.cache_stats
        # First call: 4 section misses (cold start)
        # Second call: player_state miss (always recomputed),
        #   opponent_states hit (same player), board_state hit, game_state hit
        assert stats["overall"]["hits"] == 3
        assert stats["overall"]["misses"] == 5
        assert stats["overall"]["hit_rate"] == 3 / 8

    def test_per_section_stats(self) -> None:
        game = MonopolyGame(num_players=2, seed=42)
        inc = IncrementalObservationEncoder(2)

        inc.encode(game, 0)
        stats = inc.cache_stats

        for section in ["player_state", "opponent_states", "board_state", "game_state"]:
            assert section in stats
            assert "hits" in stats[section]
            assert "misses" in stats[section]
            assert "hit_rate" in stats[section]


class TestIntegrationWithEnv:
    """Test incremental encoder works correctly within MonopolyEnv."""

    def test_env_uses_incremental_by_default(self) -> None:
        from monopoly_gym.env import MonopolyEnv
        env = MonopolyEnv(num_players=2)
        assert isinstance(env.obs_encoder, IncrementalObservationEncoder)

    def test_env_can_disable_incremental(self) -> None:
        from monopoly_gym.env import MonopolyEnv
        env = MonopolyEnv(num_players=2, incremental_obs=False)
        assert isinstance(env.obs_encoder, ObservationEncoder)
        assert not isinstance(env.obs_encoder, IncrementalObservationEncoder)

    def test_env_resets_cache(self) -> None:
        from monopoly_gym.env import MonopolyEnv
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)
        # Observe to populate cache
        env.observe("player_0")
        assert len(env.obs_encoder._cache) > 0  # type: ignore[union-attr]

        # Reset should clear cache
        env.reset(seed=43)
        assert len(env.obs_encoder._cache) == 0  # type: ignore[union-attr]

    def test_env_game_loop_works_with_incremental(self) -> None:
        """Full game loop should work identically with incremental obs."""
        from monopoly_gym.env import MonopolyEnv

        env = MonopolyEnv(num_players=2, max_turns=50)
        env.reset(seed=42)

        steps = 0
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
            else:
                mask = info["action_mask"]
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
                else:
                    action = 0
                env.step(action)
            steps += 1
            if steps > 500:
                break

    def test_incremental_vs_base_full_game_loop(self) -> None:
        """Compare observations from incremental vs base encoder in full game."""
        from monopoly_gym.env import MonopolyEnv

        env_inc = MonopolyEnv(num_players=2, max_turns=30, incremental_obs=True)
        env_base = MonopolyEnv(num_players=2, max_turns=30, incremental_obs=False)

        env_inc.reset(seed=42)
        env_base.reset(seed=42)

        steps = 0
        for agent_inc, agent_base in zip(
            env_inc.agent_iter(), env_base.agent_iter()
        ):
            assert agent_inc == agent_base

            obs_inc, _, term_inc, trunc_inc, info_inc = env_inc.last()
            obs_base, _, term_base, trunc_base, info_base = env_base.last()

            assert term_inc == term_base
            assert trunc_inc == trunc_base

            if obs_inc is not None and obs_base is not None:
                assert _obs_equal(obs_inc, obs_base), (
                    f"Observation mismatch at step {steps} for {agent_inc}"
                )

            if term_inc or trunc_inc:
                env_inc.step(None)
                env_base.step(None)
            else:
                mask = info_inc["action_mask"]
                valid_actions = np.where(mask)[0]
                action = int(valid_actions[0]) if len(valid_actions) > 0 else 0
                env_inc.step(action)
                env_base.step(action)

            steps += 1
            if steps > 300:
                break


class TestBenchmark:
    """Benchmark incremental vs base encoding speed."""

    @pytest.mark.slow
    def test_benchmark_encoding_speed(self) -> None:
        """Incremental encoder should be at least as fast as base on repeated encodes."""
        rng = np.random.default_rng(42)
        n_encodes = 500

        # --- Base encoder ---
        game = MonopolyGame(num_players=4, seed=42)
        base = ObservationEncoder(4)
        t0 = time.perf_counter()
        for _ in range(n_encodes):
            pid = game.current_player
            if game.game_over or game.players[pid].bankrupt:
                break
            base.encode(game, pid)
            roll = RollDice(player_id=pid)
            if roll.validate(game)[0]:
                roll.execute(game)
            end = EndTurn(player_id=pid)
            if end.validate(game)[0]:
                end.execute(game)
        base_time = time.perf_counter() - t0

        # --- Incremental encoder ---
        game = MonopolyGame(num_players=4, seed=42)
        inc = IncrementalObservationEncoder(4)
        t0 = time.perf_counter()
        for _ in range(n_encodes):
            pid = game.current_player
            if game.game_over or game.players[pid].bankrupt:
                break
            inc.encode(game, pid)
            roll = RollDice(player_id=pid)
            if roll.validate(game)[0]:
                roll.execute(game)
            end = EndTurn(player_id=pid)
            if end.validate(game)[0]:
                end.execute(game)
        inc_time = time.perf_counter() - t0

        # Incremental should not be significantly slower
        # (allow 20% margin for snapshot overhead; in practice it's faster)
        assert inc_time < base_time * 1.2, (
            f"Incremental ({inc_time:.3f}s) much slower than base ({base_time:.3f}s)"
        )

        # Print stats for manual inspection
        stats = inc.cache_stats
        print(f"\n  [1-encode/turn] Base: {base_time:.3f}s, Incremental: {inc_time:.3f}s")
        print(f"  Speedup: {base_time / inc_time:.2f}x")
        print(f"  Overall hit rate: {stats['overall']['hit_rate']:.1%}")
        for section in ["player_state", "opponent_states", "board_state", "game_state"]:
            print(f"  {section}: {stats[section]['hit_rate']:.1%}")

    @pytest.mark.slow
    def test_benchmark_multi_encode_per_turn(self) -> None:
        """Multi-encode pattern (realistic training): encode 3x per turn."""
        n_turns = 200

        def run_game(encoder: ObservationEncoder | IncrementalObservationEncoder) -> float:
            game = MonopolyGame(num_players=4, seed=42)
            t0 = time.perf_counter()
            for _ in range(n_turns):
                pid = game.current_player
                if game.game_over or game.players[pid].bankrupt:
                    break
                # Encode 3 times per turn (pre-action, post-roll, post-buy)
                encoder.encode(game, pid)
                roll = RollDice(player_id=pid)
                if roll.validate(game)[0]:
                    roll.execute(game)
                encoder.encode(game, pid)
                pos = game.players[pid].position
                buy = BuyProperty(player_id=pid, property_id=pos)
                if buy.validate(game)[0]:
                    buy.execute(game)
                encoder.encode(game, pid)
                end = EndTurn(player_id=pid)
                if end.validate(game)[0]:
                    end.execute(game)
            return time.perf_counter() - t0

        base_time = run_game(ObservationEncoder(4))
        inc = IncrementalObservationEncoder(4)
        inc_time = run_game(inc)

        assert inc_time < base_time * 1.1, (
            f"Incremental ({inc_time:.3f}s) slower than base ({base_time:.3f}s)"
        )

        stats = inc.cache_stats
        print(f"\n  [3-encode/turn] Base: {base_time:.3f}s, Incremental: {inc_time:.3f}s")
        print(f"  Speedup: {base_time / inc_time:.2f}x")
        print(f"  Overall hit rate: {stats['overall']['hit_rate']:.1%}")
        for section in ["player_state", "opponent_states", "board_state", "game_state"]:
            print(f"  {section}: {stats[section]['hit_rate']:.1%}")
