"""Agents package for Monopoly RL environment.

This package provides agent implementations for playing Monopoly:
- Agent: Abstract base class defining the agent interface
- RandomAgent: Baseline agent that selects uniformly random valid actions
- RuleBasedAgent: Heuristic agent using configurable strategy rules
- AggressiveAgent: Rule-based agent that buys and builds more eagerly
- ConservativeAgent: Rule-based agent that keeps larger cash reserves
- MCTSAgent: Monte Carlo Tree Search agent with optional value network
- HybridAgent: Combined MCTS + LLM trading agent
"""

from .base import Agent
from .hybrid_agent import HybridAgent, HybridAgentConfig
from .mcts_agent import MCTSAgent, SearchStats
from .random_agent import RandomAgent
from .rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent

__all__ = [
    "Agent",
    "HybridAgent",
    "HybridAgentConfig",
    "MCTSAgent",
    "RandomAgent",
    "RuleBasedAgent",
    "AggressiveAgent",
    "ConservativeAgent",
    "SearchStats",
]
