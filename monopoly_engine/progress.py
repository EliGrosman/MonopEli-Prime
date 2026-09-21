"""Pure progress checks shared by all game drivers."""

from collections import Counter

from .game import MonopolyGame


def semantic_state(game: MonopolyGame) -> str:
    s = game.state
    return repr(
        (
            s.current_player,
            s.decision_player,
            s.phase,
            s.roll_owed,
            s.doubles_count,
            s.last_roll,
            s.obligations,
            s.continuation,
            [
                (p.money, p.position, p.in_jail, p.jail_turns, p.jail_cards, p.bankrupt)
                for p in game.players
            ],
            [(p.owner, p.houses, p.mortgaged) for p in game.property_manager.properties.values()],
            s.chance_deck.to_dict(),
            s.chest_deck.to_dict(),
        )
    )


class ProgressGuard:
    def __init__(self) -> None:
        self.turn = -1
        self.count = 0
        self.seen: Counter[str] = Counter()

    def check(self, game: MonopolyGame) -> None:
        if self.turn != game.turn_number:
            self.turn, self.count = game.turn_number, 0
            self.seen.clear()
        self.count += 1
        key = semantic_state(game)
        self.seen[key] += 1
        if self.count > 1000 or self.seen[key] >= 20:
            raise RuntimeError("stalled")
