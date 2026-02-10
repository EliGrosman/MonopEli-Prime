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

Phase 2.5a Trade Support:
    env = MonopolyEnv(num_players=4, enable_trades=True)
    # Action space expands from 149 to 907 actions
    # - 149-904: Simple 1-for-1 property trades
    # - 905: Accept trade
    # - 906: Reject trade
"""

from .action_space import (
    ACTION_SPACE_SIZE,
    BUYABLE_POSITIONS,
    DEVELOPABLE_POSITIONS,
    GAMEPLAY_ACTION_SPACE_SIZE,
    OFFSET_ACCEPT_TRADE,
    OFFSET_BUILD_HOTEL,
    OFFSET_BUILD_HOUSE,
    OFFSET_BUY_PROPERTY,
    OFFSET_END_TURN,
    OFFSET_MORTGAGE,
    OFFSET_PASS_BUY,
    OFFSET_PAY_JAIL_FINE,
    OFFSET_REJECT_TRADE,
    OFFSET_SELL_HOTEL,
    OFFSET_SELL_HOUSE,
    OFFSET_SIMPLE_TRADE,
    OFFSET_UNMORTGAGE,
    OFFSET_USE_JAIL_CARD,
    SIMPLE_TRADE_DIM,
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
from .single_agent_env import SingleAgentMonopolyEnv
from .trades import (
    TradeRewardConfig,
    calculate_trade_rewards,
    decode_simple_trade,
    encode_simple_trade,
    get_simple_trade_mask,
    get_strategic_property_value,
)

__version__ = "0.1.0"
__all__ = [
    # Environments
    "MonopolyEnv",
    "SingleAgentMonopolyEnv",
    # Action space
    "ActionEncoder",
    "ACTION_SPACE_SIZE",
    "GAMEPLAY_ACTION_SPACE_SIZE",
    "BUYABLE_POSITIONS",
    "DEVELOPABLE_POSITIONS",
    # Gameplay offsets
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
    # Trade offsets (Phase 2.5a)
    "OFFSET_SIMPLE_TRADE",
    "OFFSET_ACCEPT_TRADE",
    "OFFSET_REJECT_TRADE",
    "SIMPLE_TRADE_DIM",
    # Trade utilities (Phase 2.5a)
    "encode_simple_trade",
    "decode_simple_trade",
    "get_simple_trade_mask",
    "calculate_trade_rewards",
    "get_strategic_property_value",
    "TradeRewardConfig",
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
