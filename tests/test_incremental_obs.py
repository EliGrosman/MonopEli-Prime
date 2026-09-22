"""Tests for incremental observation encoding (OPT-3).

Verifies that IncrementalObservationEncoder produces identical observations
to the base ObservationEncoder while providing cache hit benefits.
"""

from typing import Any

import numpy as np

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
        enc = IncrementalObservationEncoder(2, rules_id="foundation-trade-v1")
        assert enc.enable_trades is True

    def test_observation_space_matches_base(self) -> None:
        base = ObservationEncoder(4)
        inc = IncrementalObservationEncoder(4)
        assert base.get_observation_space() == inc.get_observation_space()

    def test_observation_space_matches_base_with_trades(self) -> None:
        base = ObservationEncoder(3, rules_id="foundation-trade-v1")
        inc = IncrementalObservationEncoder(3, rules_id="foundation-trade-v1")
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


class TestIntegrationWithEnv:
    """Test incremental encoder works correctly within MonopolyEnv."""

    def test_env_uses_fresh_encoding_by_default(self) -> None:
        from monopoly_gym.env import MonopolyEnv

        env = MonopolyEnv(num_players=2)
        assert type(env.obs_encoder) is ObservationEncoder

    def test_env_can_disable_incremental(self) -> None:
        from monopoly_gym.env import MonopolyEnv

        env = MonopolyEnv(num_players=2, incremental_obs=False)
        assert isinstance(env.obs_encoder, ObservationEncoder)
        assert not isinstance(env.obs_encoder, IncrementalObservationEncoder)

    def test_env_resets_cache(self) -> None:
        from monopoly_gym.env import MonopolyEnv

        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)
        old = env.observe("player_0")
        env.game.players[1].money = 0
        assert not _obs_equal(old, env.observe("player_0"))
        env.reset(seed=42)
        assert _obs_equal(old, env.observe("player_0"))

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
        for agent_inc, agent_base in zip(env_inc.agent_iter(), env_base.agent_iter()):
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
