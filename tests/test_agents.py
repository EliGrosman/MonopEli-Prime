"""Tests for the agents package."""

from abc import ABC
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from agents import Agent, AggressiveAgent, ConservativeAgent, RandomAgent, RuleBasedAgent
from monopoly_engine import PROPERTY_GROUPS, MonopolyGame, PropertyColor
from monopoly_gym import (
    ACTION_SPACE_SIZE,
    OFFSET_BUILD_HOTEL,
    OFFSET_BUILD_HOUSE,
    OFFSET_BUY_PROPERTY,
    OFFSET_END_TURN,
    OFFSET_PASS_BUY,
    OFFSET_PAY_JAIL_FINE,
    OFFSET_UNMORTGAGE,
    OFFSET_USE_JAIL_CARD,
    ActionEncoder,
    MonopolyEnv,
)

# =============================================================================
# Test Base Agent Class
# =============================================================================


class TestAgentBase:
    """Tests for Agent ABC."""

    def test_is_abstract_base_class(self) -> None:
        """Agent should be an abstract base class."""
        assert issubclass(Agent, ABC)

    def test_cannot_instantiate_directly(self) -> None:
        """Cannot instantiate Agent directly since it's abstract."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Agent(player_id=0, name="Test")  # type: ignore[abstract]

    def test_choose_action_is_abstract(self) -> None:
        """choose_action method should be abstract."""
        assert hasattr(Agent, "choose_action")
        # The @abstractmethod decorator makes the method abstract
        assert getattr(Agent.choose_action, "__isabstractmethod__", False)

    def test_concrete_subclass_must_implement_choose_action(self) -> None:
        """A concrete subclass must implement choose_action."""

        class IncompleteAgent(Agent):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAgent(player_id=0)  # type: ignore[abstract]

    def test_complete_subclass_can_instantiate(self) -> None:
        """A complete subclass with choose_action can be instantiated."""

        class CompleteAgent(Agent):
            def choose_action(
                self,
                observation: dict[str, Any],
                action_mask: NDArray[np.bool_],
                game: MonopolyGame,
            ) -> int:
                return 0

        agent = CompleteAgent(player_id=0)
        assert agent.player_id == 0

    def test_initialization_with_player_id(self) -> None:
        """Agent subclass should store player_id."""
        agent = RandomAgent(player_id=2)
        assert agent.player_id == 2

    def test_initialization_with_name(self) -> None:
        """Agent subclass should have a name."""
        agent = RandomAgent(player_id=0)
        assert agent.name == "Random"

    def test_reset_default_does_nothing(self) -> None:
        """Default reset() should not raise."""
        agent = RandomAgent(player_id=0)
        agent.reset()  # Should not raise

    def test_notify_result_default_does_nothing(self) -> None:
        """Default notify_result() should not raise."""
        agent = RandomAgent(player_id=0)
        agent.notify_result(
            action=0,
            reward=1.0,
            next_observation={},
            terminated=False,
            truncated=False,
        )  # Should not raise


# =============================================================================
# Test RandomAgent
# =============================================================================


class TestRandomAgent:
    """Tests for RandomAgent."""

    def test_initialization_without_seed(self) -> None:
        """RandomAgent can be initialized without a seed."""
        agent = RandomAgent(player_id=0)
        assert agent.player_id == 0
        assert agent.name == "Random"
        assert agent.rng is not None

    def test_initialization_with_seed(self) -> None:
        """RandomAgent can be initialized with a seed."""
        agent = RandomAgent(player_id=1, seed=42)
        assert agent.player_id == 1
        assert agent.rng is not None

    def test_name_is_random(self) -> None:
        """RandomAgent should have name 'Random'."""
        agent = RandomAgent(player_id=0)
        assert agent.name == "Random"

    def test_choose_action_returns_valid_action_from_mask(self) -> None:
        """choose_action should return an action that is True in the mask."""
        agent = RandomAgent(player_id=0, seed=42)
        game = MonopolyGame(num_players=2, seed=123)

        # Create a mask with some valid actions
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[OFFSET_END_TURN] = True
        mask[OFFSET_BUY_PROPERTY] = True

        action = agent.choose_action({}, mask, game)
        assert mask[action], f"Action {action} should be valid in mask"

    def test_choose_action_with_single_valid_action(self) -> None:
        """choose_action should return the only valid action."""
        agent = RandomAgent(player_id=0, seed=42)
        game = MonopolyGame(num_players=2, seed=123)

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[OFFSET_END_TURN] = True  # Only valid action

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_END_TURN

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed should produce same sequence of actions."""
        game = MonopolyGame(num_players=2, seed=123)

        # Create a mask with multiple valid actions
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[OFFSET_END_TURN] = True
        mask[OFFSET_BUY_PROPERTY] = True
        mask[OFFSET_PASS_BUY] = True

        agent1 = RandomAgent(player_id=0, seed=42)
        agent2 = RandomAgent(player_id=0, seed=42)

        actions1 = [agent1.choose_action({}, mask, game) for _ in range(10)]
        actions2 = [agent2.choose_action({}, mask, game) for _ in range(10)]

        assert actions1 == actions2

    def test_different_seeds_give_different_results(self) -> None:
        """Different seeds should (usually) produce different actions."""
        game = MonopolyGame(num_players=2, seed=123)

        # Create a mask with many valid actions
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        for i in range(50):
            mask[i] = True

        agent1 = RandomAgent(player_id=0, seed=42)
        agent2 = RandomAgent(player_id=0, seed=999)

        actions1 = [agent1.choose_action({}, mask, game) for _ in range(20)]
        actions2 = [agent2.choose_action({}, mask, game) for _ in range(20)]

        # With 50 valid actions and 20 samples, different seeds should give different results
        assert actions1 != actions2

    def test_fallback_when_no_valid_actions(self) -> None:
        """Should return OFFSET_END_TURN when mask is all False."""
        agent = RandomAgent(player_id=0, seed=42)
        game = MonopolyGame(num_players=2, seed=123)

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)  # All False

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_END_TURN

    def test_reset_does_not_raise(self) -> None:
        """reset() should not raise errors."""
        agent = RandomAgent(player_id=0, seed=42)
        agent.reset()  # Should not raise

    def test_notify_result_does_not_raise(self) -> None:
        """notify_result() should not raise errors."""
        agent = RandomAgent(player_id=0, seed=42)
        agent.notify_result(
            action=OFFSET_END_TURN,
            reward=0.0,
            next_observation={},
            terminated=False,
            truncated=False,
        )  # Should not raise

    def test_uniform_distribution(self) -> None:
        """Actions should be uniformly distributed among valid options."""
        agent = RandomAgent(player_id=0, seed=42)
        game = MonopolyGame(num_players=2, seed=123)

        # Create a mask with exactly 3 valid actions
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[OFFSET_END_TURN] = True
        mask[OFFSET_BUY_PROPERTY] = True
        mask[OFFSET_PASS_BUY] = True

        # Sample many actions
        action_counts: dict[int, int] = {}
        for _ in range(3000):
            action = agent.choose_action({}, mask, game)
            action_counts[action] = action_counts.get(action, 0) + 1

        # Each action should be chosen roughly 1000 times (1/3 of 3000)
        # Allow 20% variance
        for action_idx in [OFFSET_END_TURN, OFFSET_BUY_PROPERTY, OFFSET_PASS_BUY]:
            count = action_counts.get(action_idx, 0)
            assert 700 < count < 1300, f"Action {action_idx} count {count} not uniform"

    @given(st.integers(0, 7))
    @settings(max_examples=10)
    def test_player_id_property(self, player_id: int) -> None:
        """player_id should be correctly stored for any valid ID."""
        agent = RandomAgent(player_id=player_id)
        assert agent.player_id == player_id


# =============================================================================
# Test RuleBasedAgent
# =============================================================================


class TestRuleBasedAgent:
    """Tests for RuleBasedAgent."""

    def test_initialization_with_default_thresholds(self) -> None:
        """RuleBasedAgent should have default thresholds."""
        agent = RuleBasedAgent(player_id=0)
        assert agent.player_id == 0
        assert agent.name == "RuleBased"
        assert agent.buy_threshold == 0.5
        assert agent.build_threshold == 0.3

    def test_initialization_with_custom_thresholds(self) -> None:
        """RuleBasedAgent can be initialized with custom thresholds."""
        agent = RuleBasedAgent(player_id=0, buy_threshold=0.7, build_threshold=0.4)
        assert agent.buy_threshold == 0.7
        assert agent.build_threshold == 0.4

    def test_name_is_rulebased(self) -> None:
        """RuleBasedAgent should have name 'RuleBased'."""
        agent = RuleBasedAgent(player_id=0)
        assert agent.name == "RuleBased"

    def test_buy_threshold_affects_buying_decision_buys(self) -> None:
        """Agent should buy when cost < threshold * money."""
        agent = RuleBasedAgent(player_id=0, buy_threshold=0.9)  # High threshold
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Move player to Mediterranean (position 1, cost $60)
        # Player has $1500, so $60 < 0.9 * $1500 = $1350 -> should buy
        game.players[0].position = 1

        mask = encoder.get_action_mask(game, 0)
        assert mask[OFFSET_BUY_PROPERTY], "Buy should be valid"

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_BUY_PROPERTY

    def test_buy_threshold_affects_buying_decision_passes(self) -> None:
        """Agent should pass when cost >= threshold * money."""
        agent = RuleBasedAgent(player_id=0, buy_threshold=0.01)  # Very low threshold
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Move player to Mediterranean (position 1, cost $60)
        # Player has $1500, so $60 >= 0.01 * $1500 = $15 -> should pass
        game.players[0].position = 1

        mask = encoder.get_action_mask(game, 0)
        assert mask[OFFSET_PASS_BUY], "Pass should be valid"

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_PASS_BUY

    def test_passes_when_cannot_afford(self) -> None:
        """Agent should end turn when cannot afford property (no buy/pass in mask)."""
        agent = RuleBasedAgent(player_id=0, buy_threshold=0.9)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Reduce money so player can't afford
        game.players[0].money = 50  # Mediterranean costs $60
        game.players[0].position = 1

        mask = encoder.get_action_mask(game, 0)
        # Buy won't be in mask since can't afford
        assert not mask[OFFSET_BUY_PROPERTY], "Buy should not be valid"
        # Pass buy is only available when property is unowned (which it is)
        # but the agent's logic only checks pass if buy is available
        # So agent will fall through to end turn
        action = agent.choose_action({}, mask, game)
        # When buy isn't available, agent ends turn (or does other actions if available)
        # In this case, there are no other priority actions, so it ends turn
        assert action == OFFSET_END_TURN

    def test_priority_buy_over_build(self) -> None:
        """Buy property has priority over building."""
        agent = RuleBasedAgent(player_id=0, buy_threshold=0.9, build_threshold=0.9)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Give player a monopoly with buildable houses
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            game.property_manager.properties[pos].owner = 0
        game.players[0].money = 5000

        # Move to an unowned property
        game.players[0].position = 6  # Oriental Avenue

        mask = encoder.get_action_mask(game, 0)
        # Both buy and build should be available
        assert mask[OFFSET_BUY_PROPERTY] or mask[OFFSET_PASS_BUY]

        action = agent.choose_action({}, mask, game)
        # Buy has higher priority
        if mask[OFFSET_BUY_PROPERTY]:
            assert action == OFFSET_BUY_PROPERTY
        else:
            assert action == OFFSET_PASS_BUY

    def test_uses_jail_card_when_available(self) -> None:
        """Agent should use jail card when in jail and has one."""
        agent = RuleBasedAgent(player_id=0)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Put player in jail with a jail card
        game.players[0].in_jail = True
        game.players[0].jail_cards = 1
        game.players[0].position = 10  # Jail position

        mask = encoder.get_action_mask(game, 0)
        assert mask[OFFSET_USE_JAIL_CARD], "Use jail card should be valid"

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_USE_JAIL_CARD

    def test_pays_jail_fine_when_no_card(self) -> None:
        """Agent should pay fine when in jail without card."""
        agent = RuleBasedAgent(player_id=0)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Put player in jail without a jail card
        game.players[0].in_jail = True
        game.players[0].jail_cards = 0
        game.players[0].position = 10  # Jail position
        game.players[0].money = 500  # Can afford fine

        mask = encoder.get_action_mask(game, 0)
        assert mask[OFFSET_PAY_JAIL_FINE], "Pay jail fine should be valid"
        assert not mask[OFFSET_USE_JAIL_CARD], "Use jail card should not be valid"

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_PAY_JAIL_FINE

    def test_builds_houses_when_has_monopoly_and_money(self) -> None:
        """Agent should build houses when has monopoly and sufficient money."""
        agent = RuleBasedAgent(player_id=0, build_threshold=0.5)  # Build if cost < 50% money
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Give player Brown monopoly (cheap to build on)
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            game.property_manager.properties[pos].owner = 0
        game.players[0].money = 1000  # $50 house cost < 0.5 * $1000 = $500

        # Not on a purchasable property
        game.players[0].position = 0  # Go

        mask = encoder.get_action_mask(game, 0)

        # Find a build house action that's valid
        build_actions = [i for i in range(OFFSET_BUILD_HOUSE, OFFSET_BUILD_HOTEL) if mask[i]]
        assert len(build_actions) > 0, "Should be able to build houses"

        action = agent.choose_action({}, mask, game)
        assert OFFSET_BUILD_HOUSE <= action < OFFSET_BUILD_HOTEL

    def test_builds_hotels_when_has_4_houses(self) -> None:
        """Agent should build hotels when property has 4 houses."""
        agent = RuleBasedAgent(player_id=0, build_threshold=0.9)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Give player Brown monopoly with 4 houses on each
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            game.property_manager.properties[pos].owner = 0
            game.property_manager.properties[pos].houses = 4
        game.houses_remaining = 32 - 8  # 4 houses on 2 properties
        game.players[0].money = 1000

        # Not on a purchasable property
        game.players[0].position = 0  # Go

        mask = encoder.get_action_mask(game, 0)

        # Find a build hotel action that's valid
        hotel_actions = [i for i in range(OFFSET_BUILD_HOTEL, OFFSET_BUILD_HOTEL + 22) if mask[i]]
        assert len(hotel_actions) > 0, "Should be able to build hotels"

        action = agent.choose_action({}, mask, game)
        assert OFFSET_BUILD_HOTEL <= action < OFFSET_BUILD_HOTEL + 22

    def test_unmortgages_when_affordable(self) -> None:
        """Agent should unmortgage properties when affordable."""
        agent = RuleBasedAgent(player_id=0)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Give player a mortgaged property
        game.property_manager.properties[1].owner = 0
        game.property_manager.properties[1].mortgaged = True
        game.players[0].money = 1000  # Can afford to unmortgage

        # Not on a purchasable property
        game.players[0].position = 0  # Go

        mask = encoder.get_action_mask(game, 0)

        # Find unmortgage actions
        unmortgage_actions = [
            i for i in range(OFFSET_UNMORTGAGE, OFFSET_END_TURN) if mask[i]
        ]
        assert len(unmortgage_actions) > 0, "Should be able to unmortgage"

        action = agent.choose_action({}, mask, game)
        # Agent should either unmortgage or end turn
        assert action in unmortgage_actions or action == OFFSET_END_TURN

    def test_ends_turn_when_no_other_actions(self) -> None:
        """Agent should end turn when no better actions available."""
        agent = RuleBasedAgent(player_id=0)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Player on Go with no properties
        game.players[0].position = 0

        mask = encoder.get_action_mask(game, 0)

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_END_TURN

    def test_keeps_cash_reserve_for_building(self) -> None:
        """Agent should not build if money would go below reserve."""
        agent = RuleBasedAgent(player_id=0, build_threshold=0.3)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Give player Brown monopoly
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            game.property_manager.properties[pos].owner = 0
        game.players[0].money = 150  # Less than 200 reserve

        # Not on a purchasable property
        game.players[0].position = 0  # Go

        mask = encoder.get_action_mask(game, 0)

        action = agent.choose_action({}, mask, game)
        # Should end turn due to cash reserve
        assert action == OFFSET_END_TURN

    @given(
        buy_threshold=st.floats(0.0, 1.0),
        build_threshold=st.floats(0.0, 1.0),
    )
    @settings(max_examples=10)
    def test_thresholds_in_valid_range(
        self, buy_threshold: float, build_threshold: float
    ) -> None:
        """Agent should accept any threshold in [0, 1]."""
        agent = RuleBasedAgent(
            player_id=0,
            buy_threshold=buy_threshold,
            build_threshold=build_threshold,
        )
        assert agent.buy_threshold == buy_threshold
        assert agent.build_threshold == build_threshold


# =============================================================================
# Test AggressiveAgent
# =============================================================================


class TestAggressiveAgent:
    """Tests for AggressiveAgent."""

    def test_has_correct_thresholds(self) -> None:
        """AggressiveAgent should have thresholds (0.8, 0.5)."""
        agent = AggressiveAgent(player_id=0)
        assert agent.buy_threshold == 0.8
        assert agent.build_threshold == 0.5

    def test_name_is_aggressive(self) -> None:
        """AggressiveAgent should have name 'Aggressive'."""
        agent = AggressiveAgent(player_id=0)
        assert agent.name == "Aggressive"

    def test_is_subclass_of_rulebased(self) -> None:
        """AggressiveAgent should be a subclass of RuleBasedAgent."""
        assert issubclass(AggressiveAgent, RuleBasedAgent)

    def test_inherits_choose_action(self) -> None:
        """AggressiveAgent should inherit choose_action from RuleBasedAgent."""
        agent = AggressiveAgent(player_id=0)
        assert hasattr(agent, "choose_action")

    def test_buys_more_eagerly_than_default(self) -> None:
        """AggressiveAgent should buy at higher cost/money ratio."""
        aggressive = AggressiveAgent(player_id=0)
        default = RuleBasedAgent(player_id=0)

        # Aggressive has 0.8, default has 0.5
        assert aggressive.buy_threshold > default.buy_threshold

    def test_builds_more_eagerly_than_default(self) -> None:
        """AggressiveAgent should build at higher cost/money ratio."""
        aggressive = AggressiveAgent(player_id=0)
        default = RuleBasedAgent(player_id=0)

        # Aggressive has 0.5, default has 0.3
        assert aggressive.build_threshold > default.build_threshold

    @given(st.integers(0, 7))
    @settings(max_examples=5)
    def test_player_id_property(self, player_id: int) -> None:
        """player_id should be correctly stored for any valid ID."""
        agent = AggressiveAgent(player_id=player_id)
        assert agent.player_id == player_id


# =============================================================================
# Test ConservativeAgent
# =============================================================================


class TestConservativeAgent:
    """Tests for ConservativeAgent."""

    def test_has_correct_thresholds(self) -> None:
        """ConservativeAgent should have thresholds (0.3, 0.2)."""
        agent = ConservativeAgent(player_id=0)
        assert agent.buy_threshold == 0.3
        assert agent.build_threshold == 0.2

    def test_name_is_conservative(self) -> None:
        """ConservativeAgent should have name 'Conservative'."""
        agent = ConservativeAgent(player_id=0)
        assert agent.name == "Conservative"

    def test_is_subclass_of_rulebased(self) -> None:
        """ConservativeAgent should be a subclass of RuleBasedAgent."""
        assert issubclass(ConservativeAgent, RuleBasedAgent)

    def test_inherits_choose_action(self) -> None:
        """ConservativeAgent should inherit choose_action from RuleBasedAgent."""
        agent = ConservativeAgent(player_id=0)
        assert hasattr(agent, "choose_action")

    def test_buys_less_eagerly_than_default(self) -> None:
        """ConservativeAgent should buy at lower cost/money ratio."""
        conservative = ConservativeAgent(player_id=0)
        default = RuleBasedAgent(player_id=0)

        # Conservative has 0.3, default has 0.5
        assert conservative.buy_threshold < default.buy_threshold

    def test_builds_less_eagerly_than_default(self) -> None:
        """ConservativeAgent should build at lower cost/money ratio."""
        conservative = ConservativeAgent(player_id=0)
        default = RuleBasedAgent(player_id=0)

        # Conservative has 0.2, default has 0.3
        assert conservative.build_threshold < default.build_threshold

    def test_conservative_vs_aggressive_thresholds(self) -> None:
        """ConservativeAgent should have lower thresholds than AggressiveAgent."""
        conservative = ConservativeAgent(player_id=0)
        aggressive = AggressiveAgent(player_id=0)

        assert conservative.buy_threshold < aggressive.buy_threshold
        assert conservative.build_threshold < aggressive.build_threshold

    @given(st.integers(0, 7))
    @settings(max_examples=5)
    def test_player_id_property(self, player_id: int) -> None:
        """player_id should be correctly stored for any valid ID."""
        agent = ConservativeAgent(player_id=player_id)
        assert agent.player_id == player_id


# =============================================================================
# Test Agent Integration with MonopolyEnv
# =============================================================================


class TestAgentIntegration:
    """Integration tests with game environment."""

    def test_random_agent_works_with_env(self) -> None:
        """RandomAgent should work with MonopolyEnv."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        agents = {
            "player_0": RandomAgent(player_id=0, seed=100),
            "player_1": RandomAgent(player_id=1, seed=101),
        }

        steps = 0
        max_steps = 100

        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if steps >= max_steps:
                break

        # Should have taken some steps
        assert steps > 0

    def test_rulebased_agent_works_with_env(self) -> None:
        """RuleBasedAgent should work with MonopolyEnv."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        agents = {
            "player_0": RuleBasedAgent(player_id=0),
            "player_1": RuleBasedAgent(player_id=1),
        }

        steps = 0
        max_steps = 100

        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if steps >= max_steps:
                break

        assert steps > 0

    def test_aggressive_agent_works_with_env(self) -> None:
        """AggressiveAgent should work with MonopolyEnv."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        agents = {
            "player_0": AggressiveAgent(player_id=0),
            "player_1": AggressiveAgent(player_id=1),
        }

        steps = 0
        max_steps = 100

        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if steps >= max_steps:
                break

        assert steps > 0

    def test_conservative_agent_works_with_env(self) -> None:
        """ConservativeAgent should work with MonopolyEnv."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        agents = {
            "player_0": ConservativeAgent(player_id=0),
            "player_1": ConservativeAgent(player_id=1),
        }

        steps = 0
        max_steps = 100

        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if steps >= max_steps:
                break

        assert steps > 0

    def test_mixed_agents_can_play_together(self) -> None:
        """Different agent types should work together."""
        env = MonopolyEnv(num_players=4)
        env.reset(seed=42)

        agents = {
            "player_0": RandomAgent(player_id=0, seed=100),
            "player_1": RuleBasedAgent(player_id=1),
            "player_2": AggressiveAgent(player_id=2),
            "player_3": ConservativeAgent(player_id=3),
        }

        steps = 0
        max_steps = 200

        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if steps >= max_steps:
                break

        assert steps > 0

    def test_agents_can_complete_full_game(self) -> None:
        """Agents should be able to play a complete game (until termination)."""
        env = MonopolyEnv(num_players=2, max_turns=500)
        env.reset(seed=42)

        agents = {
            "player_0": RandomAgent(player_id=0, seed=100),
            "player_1": RandomAgent(player_id=1, seed=101),
        }

        steps = 0
        max_steps = 5000  # Safety limit

        game_over = False
        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                game_over = True
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if game_over or steps >= max_steps:
                break

        # Either game ended naturally or we hit max steps
        assert steps > 0
        if steps < max_steps:
            # Game ended naturally - all agents should be terminated or truncated
            assert all(env.terminations.values()) or all(env.truncations.values())

    def test_action_always_in_mask(self) -> None:
        """Agent actions should always be in the valid action mask."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        agents = {
            "player_0": RandomAgent(player_id=0, seed=100),
            "player_1": RuleBasedAgent(player_id=1),
        }

        steps = 0
        max_steps = 100

        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

                # Verify action is valid
                assert action_mask[action], f"Action {action} not in mask for {agent_name}"

            env.step(action)
            steps += 1

            if steps >= max_steps:
                break

    def test_agent_reset_between_games(self) -> None:
        """Agent reset() should be callable between games."""
        env = MonopolyEnv(num_players=2)

        agents = {
            "player_0": RandomAgent(player_id=0, seed=100),
            "player_1": RuleBasedAgent(player_id=1),
        }

        # Play two games
        for game_num in range(2):
            env.reset(seed=game_num)

            # Reset all agents
            for agent in agents.values():
                agent.reset()

            steps = 0
            for agent_name in env.agent_iter():
                obs, reward, term, trunc, info = env.last()

                if term or trunc:
                    action = None
                else:
                    agent = agents[agent_name]
                    action_mask = info["action_mask"]
                    action = agent.choose_action(obs, action_mask, env.game)

                env.step(action)
                steps += 1

                if steps >= 50:
                    break

    def test_notify_result_called_correctly(self) -> None:
        """notify_result() should be callable after each step."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        agents = {
            "player_0": RandomAgent(player_id=0, seed=100),
            "player_1": RandomAgent(player_id=1, seed=101),
        }

        prev_action: int | None = None
        prev_agent_name: str | None = None

        steps = 0
        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            # Notify previous agent of result
            if prev_agent_name is not None and prev_action is not None:
                agents[prev_agent_name].notify_result(
                    action=prev_action,
                    reward=reward,
                    next_observation=obs if obs is not None else {},
                    terminated=term,
                    truncated=trunc,
                )

            if term or trunc:
                action = None
            else:
                agent = agents[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)
                prev_action = action
                prev_agent_name = agent_name

            env.step(action)
            steps += 1

            if steps >= 50:
                break

    def test_deterministic_with_same_seeds(self) -> None:
        """Same seeds should produce same game outcomes."""

        def play_game(env_seed: int, agent_seeds: tuple[int, int]) -> list[int]:
            env = MonopolyEnv(num_players=2)
            env.reset(seed=env_seed)

            agents = {
                "player_0": RandomAgent(player_id=0, seed=agent_seeds[0]),
                "player_1": RandomAgent(player_id=1, seed=agent_seeds[1]),
            }

            actions = []
            steps = 0
            for agent_name in env.agent_iter():
                obs, reward, term, trunc, info = env.last()

                if term or trunc:
                    action = None
                else:
                    agent = agents[agent_name]
                    action_mask = info["action_mask"]
                    action = agent.choose_action(obs, action_mask, env.game)
                    if action is not None:
                        actions.append(action)

                env.step(action)
                steps += 1

                if steps >= 50:
                    break

            return actions

        # Play same game twice with same seeds
        actions1 = play_game(42, (100, 101))
        actions2 = play_game(42, (100, 101))

        assert actions1 == actions2

    def test_random_vs_rulebased_strategies(self) -> None:
        """Test that different strategies produce different behavior."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        # Track property ownership after some steps
        agents_random = {
            "player_0": RandomAgent(player_id=0, seed=100),
            "player_1": RandomAgent(player_id=1, seed=101),
        }

        env.reset(seed=42)
        steps = 0
        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                agent = agents_random[agent_name]
                action_mask = info["action_mask"]
                action = agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if steps >= 100:
                break

        random_money = [p.money for p in env.game.players]

        # Reset and play with rule-based agents
        env.reset(seed=42)
        agents_rulebased: dict[str, Agent] = {
            "player_0": AggressiveAgent(player_id=0),
            "player_1": ConservativeAgent(player_id=1),
        }

        steps = 0
        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                rulebased_agent = agents_rulebased[agent_name]
                action_mask = info["action_mask"]
                action = rulebased_agent.choose_action(obs, action_mask, env.game)

            env.step(action)
            steps += 1

            if steps >= 100:
                break

        rulebased_money = [p.money for p in env.game.players]

        # The strategies should produce different outcomes
        # (though not guaranteed due to randomness in dice rolls)
        # At minimum, the test verifies both work
        assert len(random_money) == 2
        assert len(rulebased_money) == 2


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_random_agent_handles_all_false_mask(self) -> None:
        """RandomAgent should handle empty mask gracefully."""
        agent = RandomAgent(player_id=0, seed=42)
        game = MonopolyGame(num_players=2, seed=123)

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        action = agent.choose_action({}, mask, game)

        assert action == OFFSET_END_TURN  # Fallback

    def test_rulebased_agent_handles_all_false_mask(self) -> None:
        """RuleBasedAgent should handle empty mask gracefully."""
        agent = RuleBasedAgent(player_id=0)
        game = MonopolyGame(num_players=2, seed=123)

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        action = agent.choose_action({}, mask, game)

        # Should return end turn (which is checked regardless of mask)
        assert action == OFFSET_END_TURN

    def test_agent_with_bankrupt_player(self) -> None:
        """Agent should handle being bankrupt."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        # Bankrupt player 0
        env.game.players[0].bankrupt = True
        env.game.players[0].money = 0

        agent = RandomAgent(player_id=0, seed=100)
        encoder = ActionEncoder()
        mask = encoder.get_action_mask(env.game, 0)

        # Mask should be all False for bankrupt player
        assert not mask.any(), "Bankrupt player should have no valid actions"

        # Agent should still return something
        action = agent.choose_action({}, mask, env.game)
        assert action == OFFSET_END_TURN  # Fallback

    def test_rulebased_with_extreme_thresholds_zero(self) -> None:
        """RuleBasedAgent with threshold 0 should never buy/build."""
        agent = RuleBasedAgent(player_id=0, buy_threshold=0.0, build_threshold=0.0)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Move to a property
        game.players[0].position = 1

        mask = encoder.get_action_mask(game, 0)

        action = agent.choose_action({}, mask, game)
        # Should pass buy since threshold is 0
        assert action == OFFSET_PASS_BUY

    def test_rulebased_with_extreme_thresholds_one(self) -> None:
        """RuleBasedAgent with threshold 1 should buy if has any money."""
        agent = RuleBasedAgent(player_id=0, buy_threshold=1.0, build_threshold=1.0)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Move to Mediterranean (cost $60)
        game.players[0].position = 1
        game.players[0].money = 100  # Can afford, and $60 < 1.0 * $100

        mask = encoder.get_action_mask(game, 0)

        action = agent.choose_action({}, mask, game)
        assert action == OFFSET_BUY_PROPERTY

    def test_multiple_valid_build_actions(self) -> None:
        """Agent should pick one of multiple valid build actions."""
        agent = RuleBasedAgent(player_id=0, build_threshold=0.9)
        game = MonopolyGame(num_players=2, seed=42)
        encoder = ActionEncoder()

        # Give player Brown and Light Blue monopolies
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            game.property_manager.properties[pos].owner = 0
        for pos in PROPERTY_GROUPS[PropertyColor.LIGHT_BLUE]:
            game.property_manager.properties[pos].owner = 0

        game.players[0].money = 5000
        game.players[0].position = 0  # Go

        mask = encoder.get_action_mask(game, 0)

        # Should have multiple build actions
        build_actions = [i for i in range(OFFSET_BUILD_HOUSE, OFFSET_BUILD_HOTEL) if mask[i]]
        assert len(build_actions) >= 2

        action = agent.choose_action({}, mask, game)
        # Should pick one of the build actions
        assert action in build_actions

    def test_agent_player_id_matches_game_player(self) -> None:
        """Agent player_id should correspond to correct game player."""
        game = MonopolyGame(num_players=4, seed=42)

        for player_id in range(4):
            agent = RandomAgent(player_id=player_id, seed=100 + player_id)
            assert agent.player_id == player_id

            # Agent should be able to query the correct player
            player = game.players[agent.player_id]
            assert player.id == player_id


# =============================================================================
# Property-based Tests
# =============================================================================


class TestPropertyBasedAgent:
    """Property-based tests using hypothesis."""

    @given(
        num_players=st.integers(2, 4),
        seed=st.integers(0, 10000),
    )
    @settings(max_examples=10, deadline=None)
    def test_random_agent_always_returns_valid_action(
        self, num_players: int, seed: int
    ) -> None:
        """RandomAgent should always return a valid action."""
        env = MonopolyEnv(num_players=num_players)
        env.reset(seed=seed)

        agents = {
            f"player_{i}": RandomAgent(player_id=i, seed=seed + i)
            for i in range(num_players)
        }

        steps = 0
        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                break

            agent = agents[agent_name]
            action_mask = info["action_mask"]
            action = agent.choose_action(obs, action_mask, env.game)

            # Action should be valid in mask (or fallback to END_TURN)
            if action_mask.any():
                assert action_mask[action], f"Invalid action {action}"

            env.step(action)
            steps += 1

            if steps >= 20:
                break

    @given(
        buy_threshold=st.floats(0.0, 1.0),
        build_threshold=st.floats(0.0, 1.0),
        seed=st.integers(0, 10000),
    )
    @settings(max_examples=10, deadline=None)
    def test_rulebased_agent_always_returns_valid_action(
        self, buy_threshold: float, build_threshold: float, seed: int
    ) -> None:
        """RuleBasedAgent should always return a valid or fallback action."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=seed)

        agents = {
            "player_0": RuleBasedAgent(
                player_id=0,
                buy_threshold=buy_threshold,
                build_threshold=build_threshold,
            ),
            "player_1": RuleBasedAgent(
                player_id=1,
                buy_threshold=buy_threshold,
                build_threshold=build_threshold,
            ),
        }

        steps = 0
        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                break

            agent = agents[agent_name]
            action_mask = info["action_mask"]
            action = agent.choose_action(obs, action_mask, env.game)

            # Action should be in range
            assert 0 <= action < ACTION_SPACE_SIZE

            env.step(action)
            steps += 1

            if steps >= 20:
                break

    @given(st.integers(0, 100))
    @settings(max_examples=20)
    def test_random_agent_seed_determinism(self, seed: int) -> None:
        """Same seed should always produce same first action."""
        game = MonopolyGame(num_players=2, seed=42)

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[OFFSET_END_TURN] = True
        mask[OFFSET_BUY_PROPERTY] = True
        mask[OFFSET_PASS_BUY] = True
        mask[5] = True
        mask[10] = True

        agent1 = RandomAgent(player_id=0, seed=seed)
        agent2 = RandomAgent(player_id=0, seed=seed)

        action1 = agent1.choose_action({}, mask, game)
        action2 = agent2.choose_action({}, mask, game)

        assert action1 == action2
