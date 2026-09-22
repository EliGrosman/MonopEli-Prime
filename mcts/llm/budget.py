"""Response caching and LLM budget management.

Provides two components:

- ``TokenBudget``: Tracks LLM call count and token usage per game,
  enforcing configurable limits. When the budget is exhausted, callers
  should fall back to heuristic evaluation.

- ``ResponseCache``: Caches LLM responses keyed by a hash of the game
  state and prompt. Avoids redundant LLM calls when the game situation
  has not materially changed.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from monopoly_engine.game import MonopolyGame
from monopoly_engine.rules import calculate_net_worth


@dataclass
class TokenBudget:
    """Tracks and limits LLM usage per game.

    When either ``calls_used >= max_calls_per_game`` or
    ``tokens_used >= max_tokens_per_game``, :attr:`budget_remaining`
    returns ``False`` and callers should skip the LLM.
    """

    max_calls_per_game: int = 20
    max_tokens_per_game: int = 50_000
    calls_used: int = field(default=0, repr=True)
    tokens_used: int = field(default=0, repr=True)

    @property
    def budget_remaining(self) -> bool:
        """Whether there is budget left for another LLM call."""
        return (
            self.calls_used < self.max_calls_per_game
            and self.tokens_used < self.max_tokens_per_game
        )

    def record_usage(self, tokens: int = 0) -> None:
        """Record one LLM call and its token usage."""
        self.calls_used += 1
        self.tokens_used += tokens

    def reset(self) -> None:
        """Reset counters (e.g., at the start of a new game)."""
        self.calls_used = 0
        self.tokens_used = 0


class ResponseCache:
    """Caches LLM responses keyed by (state_hash, prompt_hash).

    Uses an LRU-style eviction: when ``max_size`` is exceeded, the
    oldest entry is removed. This keeps memory bounded even across
    long games.
    """

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(
        self,
        state_hash: str,
        prompt_hash: str,
    ) -> dict[str, Any] | None:
        """Look up a cached response. Returns ``None`` on miss."""
        key = f"{state_hash}:{prompt_hash}"
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(
        self,
        state_hash: str,
        prompt_hash: str,
        response: dict[str, Any],
    ) -> None:
        """Store a response in the cache, evicting oldest if full."""
        key = f"{state_hash}:{prompt_hash}"
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = response
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of entries currently cached."""
        return len(self._cache)

    @staticmethod
    def hash_game_state(game: MonopolyGame, player_id: int) -> str:
        """Create a hash of game state relevant to trading decisions.

        Includes: player money, property ownership, building counts,
        mortgaged status, and monopoly status. Excludes volatile state
        like dice rolls, event logs, and card decks.
        """
        state_parts: list[Any] = [
            player_id,
            game.state.turn_number,
        ]

        # Player financial state
        for p in game.players:
            state_parts.append(
                (
                    p.id,
                    p.money,
                    p.bankrupt,
                    calculate_net_worth(p, game.property_manager),
                )
            )

        # Property ownership and building state
        prop_data: list[tuple[int, int | None, int, bool]] = []
        for pos in sorted(game.property_manager.properties):
            prop = game.property_manager.properties[pos]
            prop_data.append((pos, prop.owner, prop.houses, prop.mortgaged))
        state_parts.append(prop_data)

        raw = json.dumps(state_parts, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Create a hash of a prompt string."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
