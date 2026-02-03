"""Monopoly Gym - PettingZoo RL environment for Monopoly.

This package provides a PettingZoo AEC (Agent Environment Cycle) compatible
reinforcement learning environment for the Monopoly game engine.

Usage:
    from monopoly_gym import MonopolyEnv

    env = MonopolyEnv(num_players=4)
    env.reset(seed=42)

    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            action = None
        else:
            action = select_action(obs, info["action_mask"])
        env.step(action)
"""

from .action_space import (
    ACTION_SPACE_SIZE,
    BUYABLE_POSITIONS,
    DEVELOPABLE_POSITIONS,
    OFFSET_BUILD_HOTEL,
    OFFSET_BUILD_HOUSE,
    OFFSET_BUY_PROPERTY,
    OFFSET_END_TURN,
    OFFSET_MORTGAGE,
    OFFSET_PASS_BUY,
    OFFSET_PAY_JAIL_FINE,
    OFFSET_SELL_HOTEL,
    OFFSET_SELL_HOUSE,
    OFFSET_UNMORTGAGE,
    OFFSET_USE_JAIL_CARD,
    ActionEncoder,
)
from .env import MonopolyEnv
from .observation import (
    BOARD_SIZE,
    MAX_MONEY,
    NUM_PROPERTIES,
    PROPERTY_POS_TO_IDX,
    PROPERTY_POSITIONS,
    ObservationEncoder,
    flatten_observation,
    get_flat_observation_size,
)

__version__ = "0.1.0"
__all__ = [
    # Environment
    "MonopolyEnv",
    # Action space
    "ActionEncoder",
    "ACTION_SPACE_SIZE",
    "BUYABLE_POSITIONS",
    "DEVELOPABLE_POSITIONS",
    # Offsets
    "OFFSET_BUY_PROPERTY",
    "OFFSET_PASS_BUY",
    "OFFSET_BUILD_HOUSE",
    "OFFSET_BUILD_HOTEL",
    "OFFSET_SELL_HOUSE",
    "OFFSET_SELL_HOTEL",
    "OFFSET_MORTGAGE",
    "OFFSET_UNMORTGAGE",
    "OFFSET_END_TURN",
    "OFFSET_USE_JAIL_CARD",
    "OFFSET_PAY_JAIL_FINE",
    # Observation space
    "ObservationEncoder",
    "flatten_observation",
    "get_flat_observation_size",
    "MAX_MONEY",
    "BOARD_SIZE",
    "NUM_PROPERTIES",
    "PROPERTY_POSITIONS",
    "PROPERTY_POS_TO_IDX",
]
