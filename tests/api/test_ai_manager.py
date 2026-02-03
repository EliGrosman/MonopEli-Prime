"""
Tests for AI Manager service.
"""

import pytest

from api.services.ai_manager import AIManager, AI_TYPES
from api.services.game_manager import GameManager
from agents.random_agent import RandomAgent
from agents.rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent


@pytest.fixture
def ai_manager() -> AIManager:
    """Create an AI manager for testing."""
    return AIManager(think_delay_ms=0)  # No delay for tests


@pytest.fixture
def game_manager() -> GameManager:
    """Create a game manager for testing."""
    return GameManager()


# ============================================================================
# AI Types Tests
# ============================================================================


class TestAITypes:
    """Tests for AI type configuration."""

    def test_ai_types_available(self):
        """Test that all expected AI types are available."""
        assert "random" in AI_TYPES
        assert "rule_based" in AI_TYPES
        assert "aggressive" in AI_TYPES
        assert "conservative" in AI_TYPES

    def test_ai_types_classes(self):
        """Test that AI types map to correct classes."""
        assert AI_TYPES["random"] == RandomAgent
        assert AI_TYPES["rule_based"] == RuleBasedAgent
        assert AI_TYPES["aggressive"] == AggressiveAgent
        assert AI_TYPES["conservative"] == ConservativeAgent


# ============================================================================
# Agent Creation Tests
# ============================================================================


class TestAgentCreation:
    """Tests for creating AI agents."""

    def test_create_random_agent(self, ai_manager: AIManager):
        """Test creating a random agent."""
        agent = ai_manager.create_agent("game-1", player_id=0, ai_type="random")

        assert isinstance(agent, RandomAgent)
        assert agent.player_id == 0

    def test_create_rule_based_agent(self, ai_manager: AIManager):
        """Test creating a rule-based agent."""
        agent = ai_manager.create_agent("game-1", player_id=1, ai_type="rule_based")

        assert isinstance(agent, RuleBasedAgent)
        assert agent.player_id == 1

    def test_create_aggressive_agent(self, ai_manager: AIManager):
        """Test creating an aggressive agent."""
        agent = ai_manager.create_agent("game-1", player_id=2, ai_type="aggressive")

        assert isinstance(agent, AggressiveAgent)

    def test_create_conservative_agent(self, ai_manager: AIManager):
        """Test creating a conservative agent."""
        agent = ai_manager.create_agent("game-1", player_id=3, ai_type="conservative")

        assert isinstance(agent, ConservativeAgent)

    def test_create_agent_default_type(self, ai_manager: AIManager):
        """Test creating agent with default type."""
        agent = ai_manager.create_agent("game-1", player_id=0)

        assert isinstance(agent, RuleBasedAgent)

    def test_create_agent_invalid_type(self, ai_manager: AIManager):
        """Test creating agent with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown AI type"):
            ai_manager.create_agent("game-1", player_id=0, ai_type="invalid")


# ============================================================================
# Agent Lookup Tests
# ============================================================================


class TestAgentLookup:
    """Tests for looking up agents."""

    def test_get_agent_exists(self, ai_manager: AIManager):
        """Test getting an existing agent."""
        ai_manager.create_agent("game-1", player_id=0, ai_type="random")

        agent = ai_manager.get_agent("game-1", player_id=0)

        assert agent is not None
        assert isinstance(agent, RandomAgent)

    def test_get_agent_not_found(self, ai_manager: AIManager):
        """Test getting a non-existent agent."""
        agent = ai_manager.get_agent("nonexistent", player_id=0)

        assert agent is None

    def test_is_ai_player_true(self, ai_manager: AIManager):
        """Test checking if player is AI when they are."""
        ai_manager.create_agent("game-1", player_id=2, ai_type="random")

        assert ai_manager.is_ai_player("game-1", player_id=2) is True

    def test_is_ai_player_false(self, ai_manager: AIManager):
        """Test checking if player is AI when they're not."""
        assert ai_manager.is_ai_player("game-1", player_id=0) is False


# ============================================================================
# Agent Removal Tests
# ============================================================================


class TestAgentRemoval:
    """Tests for removing agents."""

    def test_remove_agent_success(self, ai_manager: AIManager):
        """Test removing an existing agent."""
        ai_manager.create_agent("game-1", player_id=0)

        removed = ai_manager.remove_agent("game-1", player_id=0)

        assert removed is True
        assert ai_manager.get_agent("game-1", player_id=0) is None

    def test_remove_agent_not_found(self, ai_manager: AIManager):
        """Test removing a non-existent agent."""
        removed = ai_manager.remove_agent("nonexistent", player_id=0)

        assert removed is False

    def test_remove_game_agents(self, ai_manager: AIManager):
        """Test removing all agents for a game."""
        ai_manager.create_agent("game-1", player_id=0)
        ai_manager.create_agent("game-1", player_id=1)
        ai_manager.create_agent("game-2", player_id=0)

        count = ai_manager.remove_game_agents("game-1")

        assert count == 2
        assert ai_manager.get_agent("game-1", player_id=0) is None
        assert ai_manager.get_agent("game-1", player_id=1) is None
        # game-2 agent should still exist
        assert ai_manager.get_agent("game-2", player_id=0) is not None


# ============================================================================
# Agent Count Tests
# ============================================================================


class TestAgentCount:
    """Tests for agent counting."""

    def test_agent_count_empty(self, ai_manager: AIManager):
        """Test agent count when empty."""
        assert ai_manager.agent_count() == 0

    def test_agent_count_after_creation(self, ai_manager: AIManager):
        """Test agent count after creating agents."""
        ai_manager.create_agent("game-1", player_id=0)
        assert ai_manager.agent_count() == 1

        ai_manager.create_agent("game-1", player_id=1)
        assert ai_manager.agent_count() == 2

        ai_manager.create_agent("game-2", player_id=0)
        assert ai_manager.agent_count() == 3

    def test_agent_count_after_removal(self, ai_manager: AIManager):
        """Test agent count after removal."""
        ai_manager.create_agent("game-1", player_id=0)
        ai_manager.create_agent("game-1", player_id=1)

        ai_manager.remove_agent("game-1", player_id=0)

        assert ai_manager.agent_count() == 1


# ============================================================================
# AI Type Lookup Tests
# ============================================================================


class TestAITypeLookup:
    """Tests for getting AI type from agent."""

    def test_get_ai_type_random(self, ai_manager: AIManager):
        """Test getting AI type for random agent."""
        ai_manager.create_agent("game-1", player_id=0, ai_type="random")

        ai_type = ai_manager.get_ai_type_for_agent("game-1", player_id=0)

        assert ai_type == "random"

    def test_get_ai_type_rule_based(self, ai_manager: AIManager):
        """Test getting AI type for rule-based agent."""
        ai_manager.create_agent("game-1", player_id=0, ai_type="rule_based")

        ai_type = ai_manager.get_ai_type_for_agent("game-1", player_id=0)

        assert ai_type == "rule_based"

    def test_get_ai_type_not_found(self, ai_manager: AIManager):
        """Test getting AI type for non-existent agent."""
        ai_type = ai_manager.get_ai_type_for_agent("nonexistent", player_id=0)

        assert ai_type is None
