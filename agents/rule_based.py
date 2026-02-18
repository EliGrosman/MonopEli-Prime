"""Rule-based heuristic agents for Monopoly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .base import Agent

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame


class RuleBasedAgent(Agent):
    """Heuristic agent using simple strategy rules.

    Configurable thresholds control aggressiveness.
    When trading is enabled (mask length > 149), the agent can propose
    and respond to 1-for-1 property trades.
    """

    def __init__(
        self,
        player_id: int,
        buy_threshold: float = 0.5,
        build_threshold: float = 0.3,
        trade_eagerness: float = 0.5,
        trade_accept_threshold: float = 0.0,
    ) -> None:
        """Initialize rule-based agent.

        Args:
            player_id: The player ID this agent controls
            buy_threshold: Buy if cost < this fraction of money
            build_threshold: Build if cost < this fraction of money
            trade_eagerness: Higher = proposes trades more often (0-1)
            trade_accept_threshold: Min net benefit delta to accept a trade ($)
        """
        super().__init__(player_id, name="RuleBased")
        self.buy_threshold = buy_threshold
        self.build_threshold = build_threshold
        self.trade_eagerness = trade_eagerness
        self.trade_accept_threshold = trade_accept_threshold
        self._rejected_trades: set[tuple[int, int]] = set()
        self._has_pending_outgoing_trade = False

    def reset(self) -> None:
        """Reset per-game state."""
        self._rejected_trades.clear()
        self._has_pending_outgoing_trade = False

    def _should_propose_trade(
        self,
        game: "MonopolyGame",
        action_mask: NDArray[np.bool_],
    ) -> int | None:
        """Check if we should propose a trade, and return the action index if so."""
        from mcts.trade_utils import suggest_valuable_trades
        from monopoly_gym.trades import (
            OFFSET_SIMPLE_TRADE,
            encode_simple_trade,
        )

        # Skip if we already have a pending outgoing trade
        if self._has_pending_outgoing_trade:
            return None

        candidates = suggest_valuable_trades(game, self.player_id, 3)

        # Filter to 1-for-1 candidates with sufficient value
        min_value = (1.0 - self.trade_eagerness) * 200
        for candidate in candidates:
            if (
                len(candidate.give_properties) != 1
                or len(candidate.want_properties) != 1
            ):
                continue
            if candidate.estimated_value < min_value:
                continue

            give = candidate.give_properties[0]
            want = candidate.want_properties[0]

            # Skip if previously rejected
            if (give, want) in self._rejected_trades:
                continue

            try:
                action_idx = encode_simple_trade(give, want)
            except ValueError:
                continue

            if action_idx < len(action_mask) and action_mask[action_idx]:
                return action_idx

        return None

    def _should_accept_trade(
        self,
        game: "MonopolyGame",
        action_mask: NDArray[np.bool_],
    ) -> bool | None:
        """Decide whether to accept or reject a pending incoming trade.

        Returns True to accept, False to reject, None if no trade pending.
        """
        from mcts.trade_utils import trade_impact
        from monopoly_gym.trades import (
            OFFSET_ACCEPT_TRADE,
            OFFSET_REJECT_TRADE,
            find_trade_for_player,
        )

        # Check if accept/reject are available in the mask
        if OFFSET_ACCEPT_TRADE >= len(action_mask):
            return None
        if not action_mask[OFFSET_ACCEPT_TRADE] and not action_mask[OFFSET_REJECT_TRADE]:
            return None

        trade_id = find_trade_for_player(game, self.player_id)
        if trade_id is None:
            return None

        trade_data = game.state.pending_trades[trade_id]
        impact = trade_impact(game, trade_data)

        from_id = trade_data["from_player"]
        my_id = self.player_id

        # Reject if it breaks our monopoly
        for pid, _color in impact["monopolies_broken"]:
            if pid == my_id:
                return False

        # Check monopoly creation
        i_get_monopoly = any(
            pid == my_id for pid, _ in impact["monopolies_created"]
        )
        they_get_monopoly = any(
            pid == from_id for pid, _ in impact["monopolies_created"]
        )

        # Accept if we get a monopoly and they don't
        if i_get_monopoly and not they_get_monopoly:
            return True

        # Reject if they get a monopoly and we don't
        if they_get_monopoly and not i_get_monopoly:
            return False

        # Score-based: accept if net change favours us
        my_net = impact["to_player_net_change"]
        their_net = impact["from_player_net_change"]
        return bool(my_net - their_net >= self.trade_accept_threshold)

    def choose_action(
        self,
        observation: dict[str, Any],
        action_mask: NDArray[np.bool_],
        game: "MonopolyGame",
    ) -> int:
        """Select action based on heuristic rules."""
        from monopoly_engine.rules import get_building_cost, get_unmortgage_cost
        from monopoly_gym.action_space import (
            BUYABLE_POSITIONS,
            DEVELOPABLE_POSITIONS,
            GAMEPLAY_ACTION_SPACE_SIZE,
            OFFSET_BUILD_HOTEL,
            OFFSET_BUILD_HOUSE,
            OFFSET_BUY_PROPERTY,
            OFFSET_END_TURN,
            OFFSET_PASS_BUY,
            OFFSET_PAY_JAIL_FINE,
            OFFSET_UNMORTGAGE,
            OFFSET_USE_JAIL_CARD,
        )

        player = game.players[self.player_id]
        trades_enabled = len(action_mask) > GAMEPLAY_ACTION_SPACE_SIZE

        # Priority 0: Respond to incoming trade (accept/reject)
        if trades_enabled:
            from monopoly_gym.trades import OFFSET_ACCEPT_TRADE, OFFSET_REJECT_TRADE

            accept_result = self._should_accept_trade(game, action_mask)
            if accept_result is True:
                self._has_pending_outgoing_trade = False
                return OFFSET_ACCEPT_TRADE
            if accept_result is False:
                # Track rejection so we don't re-propose same trade
                trade_id = None
                from monopoly_gym.trades import find_trade_for_player
                trade_id = find_trade_for_player(game, self.player_id)
                if trade_id is not None:
                    td = game.state.pending_trades[trade_id]
                    for gp in td["give_properties"]:
                        for wp in td["want_properties"]:
                            self._rejected_trades.add((wp, gp))
                self._has_pending_outgoing_trade = False
                return OFFSET_REJECT_TRADE

        # Priority 1: Buy property if affordable
        if action_mask[OFFSET_BUY_PROPERTY]:
            prop = game.property_manager.get(player.position)
            if prop is not None:
                from monopoly_engine.rules import get_property_cost

                prop_cost = get_property_cost(player.position)
                if prop_cost < player.money * self.buy_threshold:
                    return OFFSET_BUY_PROPERTY
            # If can't afford threshold, pass
            return OFFSET_PASS_BUY

        # Priority 2: Get out of jail
        if player.in_jail:
            if action_mask[OFFSET_USE_JAIL_CARD]:
                return OFFSET_USE_JAIL_CARD
            if action_mask[OFFSET_PAY_JAIL_FINE]:
                return OFFSET_PAY_JAIL_FINE

        # Priority 3: Propose trade if beneficial candidate exists
        if trades_enabled:
            trade_action = self._should_propose_trade(game, action_mask)
            if trade_action is not None:
                self._has_pending_outgoing_trade = True
                return trade_action

        # Priority 4: Build houses/hotels on monopolies
        if player.money > 200:  # Keep cash reserve
            build_house_dim = len(DEVELOPABLE_POSITIONS)
            for i in range(build_house_dim):
                if action_mask[OFFSET_BUILD_HOUSE + i]:
                    prop_pos = DEVELOPABLE_POSITIONS[i]
                    cost = get_building_cost(prop_pos)
                    if cost < player.money * self.build_threshold:
                        return OFFSET_BUILD_HOUSE + i

            # Build hotels
            for i in range(build_house_dim):
                if action_mask[OFFSET_BUILD_HOTEL + i]:
                    prop_pos = DEVELOPABLE_POSITIONS[i]
                    cost = get_building_cost(prop_pos)
                    if cost < player.money * self.build_threshold:
                        return OFFSET_BUILD_HOTEL + i

        # Priority 5: Unmortgage properties if affordable
        mortgage_dim = len(BUYABLE_POSITIONS)
        for i in range(mortgage_dim):
            if action_mask[OFFSET_UNMORTGAGE + i]:
                prop_pos = BUYABLE_POSITIONS[i]
                cost = get_unmortgage_cost(prop_pos)
                if cost < player.money * 0.3:
                    return OFFSET_UNMORTGAGE + i

        # Default: End turn
        return OFFSET_END_TURN


class AggressiveAgent(RuleBasedAgent):
    """More aggressive - builds and buys more eagerly."""

    def __init__(self, player_id: int) -> None:
        """Initialize aggressive agent.

        Args:
            player_id: The player ID this agent controls
        """
        super().__init__(
            player_id,
            buy_threshold=0.8,
            build_threshold=0.5,
            trade_eagerness=0.8,
            trade_accept_threshold=-50.0,
        )
        self.name = "Aggressive"


class ConservativeAgent(RuleBasedAgent):
    """More conservative - keeps larger cash reserve."""

    def __init__(self, player_id: int) -> None:
        """Initialize conservative agent.

        Args:
            player_id: The player ID this agent controls
        """
        super().__init__(
            player_id,
            buy_threshold=0.3,
            build_threshold=0.2,
            trade_eagerness=0.3,
            trade_accept_threshold=50.0,
        )
        self.name = "Conservative"
