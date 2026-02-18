"""Agents package for Monopoly RL environment.

This package provides agent implementations for playing Monopoly:
- Agent: Abstract base class defining the agent interface
- RandomAgent: Baseline agent that selects uniformly random valid actions
- RuleBasedAgent: Heuristic agent using configurable strategy rules
- AggressiveAgent: Rule-based agent that buys and builds more eagerly
- ConservativeAgent: Rule-based agent that keeps larger cash reserves
- MCTSAgent: Monte Carlo Tree Search agent with optional value network (requires torch)
- HybridAgent: Combined MCTS + LLM trading agent (requires torch)
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

# MCTSAgent requires torch — only export when available
try:
    from .mcts_agent import MCTSAgent, SearchStats

    __all__ += ["MCTSAgent", "SearchStats"]
except ImportError:
    pass

# HybridAgent requires torch — only export when available
try:
    from .hybrid_agent import HybridAgent, HybridAgentConfig

    __all__ += ["HybridAgent", "HybridAgentConfig"]
except ImportError:
    pass
