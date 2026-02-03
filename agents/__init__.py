"""Agents package for Monopoly RL environment.

This package provides agent implementations for playing Monopoly:
- Agent: Abstract base class defining the agent interface
- RandomAgent: Baseline agent that selects uniformly random valid actions
- RuleBasedAgent: Heuristic agent using configurable strategy rules
- AggressiveAgent: Rule-based agent that buys and builds more eagerly
- ConservativeAgent: Rule-based agent that keeps larger cash reserves
"""

from .base import Agent
from .random_agent import RandomAgent
from .rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent

__all__ = [
    "Agent",
    "RandomAgent",
    "RuleBasedAgent",
    "AggressiveAgent",
    "ConservativeAgent",
]
