"""Serialize Monopoly game state to natural language for LLM consumption.

The output is designed to be concise (<1000 tokens) and unambiguous.
Position numbers are always shown in brackets so the LLM can reference
them in trade proposals without hallucinating property names.
"""

from __future__ import annotations

from monopoly_engine.board import Board, PropertySpace, RailroadSpace, UtilitySpace
from monopoly_engine.game import MonopolyGame
from monopoly_engine.property import PropertyManager
from monopoly_engine.rules import calculate_net_worth
from monopoly_engine.types import PROPERTY_GROUPS, PropertyColor, TradeOfferData


def serialize_game_state(game: MonopolyGame, perspective_player: int) -> str:
    """Convert game state to natural language for LLM consumption.

    The output is from the perspective of ``perspective_player``, showing
    their assets prominently and opponents as summaries.

    Target: <1000 tokens of context.
    """
    player = game.players[perspective_player]
    pm = game.property_manager
    lines: list[str] = []

    # Header
    lines.append(f"You are Player {perspective_player} ({player.name}). You have ${player.money}.")
    lines.append("")

    # Our properties
    our_props = pm.get_owned_by(perspective_player)
    if our_props:
        lines.append("YOUR PROPERTIES:")
        lines.extend(
            _format_player_properties(
                game,
                perspective_player,
                our_props,
                show_near_monopoly=True,
            )
        )
    else:
        lines.append("YOUR PROPERTIES: (none)")
    lines.append("")

    # Opponents
    lines.append("OPPONENTS:")
    for pid, opp in enumerate(game.players):
        if pid == perspective_player or opp.bankrupt:
            continue
        lines.append(_format_opponent_summary(game, pid))

        # Show near-monopoly hints for opponents
        hints = _near_monopoly_hints(game, pid)
        for hint in hints:
            lines.append(f"  {hint}")
    lines.append("")

    # Unowned properties
    unowned = _get_unowned_properties(pm)
    if unowned:
        unowned_strs = [f"{Board.get_space(pos).name} [{pos}]" for pos in unowned]
        lines.append(f"UNOWNED: {', '.join(unowned_strs)}")
    else:
        lines.append("UNOWNED: (none)")
    lines.append("")

    # Game info
    space_name = Board.get_space(player.position).name
    jail_note = " (IN JAIL)" if player.in_jail else ""
    lines.append(
        f"GAME: Turn {game.state.turn_number}, "
        f"you are on {space_name} (position {player.position}){jail_note}."
    )
    lines.append("")
    lines.append(
        "Use position numbers (in brackets) when specifying properties in trade proposals."
    )

    return "\n".join(lines)


def serialize_player_summary(game: MonopolyGame, player_id: int) -> str:
    """Summarize a single player's position."""
    player = game.players[player_id]
    pm = game.property_manager
    props = pm.get_owned_by(player_id)
    nw = calculate_net_worth(player, pm)
    monopolies = pm.get_monopolies(player_id)

    prop_strs = [f"{Board.get_space(p).name} [{p}]" for p in sorted(props)]
    mono_strs = [
        c.name for c in monopolies if c not in (PropertyColor.RAILROAD, PropertyColor.UTILITY)
    ]

    parts = [
        f"Player {player_id} ({player.name}): ${player.money} (net worth ${nw})",
    ]
    if prop_strs:
        parts.append(f"  Properties: {', '.join(prop_strs)}")
    if mono_strs:
        parts.append(f"  Monopolies: {', '.join(mono_strs)}")
    if player.in_jail:
        parts.append("  STATUS: In jail")
    if player.bankrupt:
        parts.append("  STATUS: Bankrupt")

    return "\n".join(parts)


def serialize_property_landscape(game: MonopolyGame) -> str:
    """Describe the property ownership landscape.

    Groups by color, shows which properties are owned/unowned,
    and highlights near-complete monopolies.
    """
    pm = game.property_manager
    lines: list[str] = []

    for color in PropertyColor:
        positions = PROPERTY_GROUPS.get(color, ())
        if not positions:
            continue

        group_entries: list[str] = []
        owners: dict[int, int] = {}  # player_id -> count

        for pos in positions:
            space = Board.get_space(pos)
            prop = pm.properties.get(pos)
            if prop is not None and prop.owner is not None:
                owner_name = game.players[prop.owner].name
                houses_str = ""
                if isinstance(space, PropertySpace) and prop.houses > 0:
                    houses_str = " (HOTEL)" if prop.houses == 5 else f" ({prop.houses}h)"
                mort_str = " [M]" if prop.mortgaged else ""
                group_entries.append(
                    f"{space.name} [{pos}] -> P{prop.owner} ({owner_name}){houses_str}{mort_str}"
                )
                owners[prop.owner] = owners.get(prop.owner, 0) + 1
            else:
                group_entries.append(f"{space.name} [{pos}] -> unowned")

        # Header with monopoly status
        group_size = len(positions)
        status = ""
        for pid, count in owners.items():
            if count == group_size:
                status = f" ** P{pid} MONOPOLY **"
                break
            elif count == group_size - 1:
                status = f" (P{pid} has {count}/{group_size})"

        lines.append(f"{color.name}{status}:")
        for entry in group_entries:
            lines.append(f"  {entry}")

    return "\n".join(lines)


def serialize_trade_proposal(trade: TradeOfferData, game: MonopolyGame) -> str:
    """Convert a trade proposal to readable text."""
    from_p = game.players[trade["from_player"]]
    to_p = game.players[trade["to_player"]]

    give_parts: list[str] = []
    for pos in trade["give_properties"]:
        give_parts.append(f"{Board.get_space(pos).name} [{pos}]")
    if trade["give_money"] > 0:
        give_parts.append(f"${trade['give_money']}")

    want_parts: list[str] = []
    for pos in trade["want_properties"]:
        want_parts.append(f"{Board.get_space(pos).name} [{pos}]")
    if trade["want_money"] > 0:
        want_parts.append(f"${trade['want_money']}")

    give_str = ", ".join(give_parts) if give_parts else "(nothing)"
    want_str = ", ".join(want_parts) if want_parts else "(nothing)"

    return (
        f"Player {trade['from_player']} ({from_p.name}) offers "
        f"Player {trade['to_player']} ({to_p.name}):\n"
        f"  GIVES: {give_str}\n"
        f"  WANTS: {want_str}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_player_properties(
    game: MonopolyGame,
    player_id: int,
    positions: list[int],
    *,
    show_near_monopoly: bool = False,
) -> list[str]:
    """Format a player's property list with details."""
    pm = game.property_manager
    lines: list[str] = []
    seen_colors: set[PropertyColor] = set()

    for pos in sorted(positions):
        space = Board.get_space(pos)
        prop = pm.properties[pos]

        if isinstance(space, PropertySpace):
            color = space.color
            houses_str = (
                "hotel"
                if prop.houses == 5
                else f"{prop.houses} houses"
                if prop.houses > 0
                else "0 houses"
            )
            mort_str = "mortgaged" if prop.mortgaged else "unmortgaged"
            lines.append(f"- {space.name} [{pos}] ({color.name}) - {mort_str}, {houses_str}")

            # Near-monopoly hint (once per color)
            if show_near_monopoly and color not in seen_colors:
                seen_colors.add(color)
                hint = _color_group_hint(game, player_id, color)
                if hint:
                    lines.append(f"  {hint}")

        elif isinstance(space, RailroadSpace):
            mort_str = "mortgaged" if prop.mortgaged else "unmortgaged"
            lines.append(f"- {space.name} [{pos}] - {mort_str}")

        elif isinstance(space, UtilitySpace):
            mort_str = "mortgaged" if prop.mortgaged else "unmortgaged"
            lines.append(f"- {space.name} [{pos}] - {mort_str}")

    return lines


def _color_group_hint(
    game: MonopolyGame,
    player_id: int,
    color: PropertyColor,
) -> str:
    """Generate a hint about a player's progress in a color group."""
    pm = game.property_manager
    positions = PROPERTY_GROUPS.get(color, ())
    if not positions:
        return ""

    owned = [
        p
        for p in positions
        if pm.properties.get(p) is not None and pm.properties[p].owner == player_id
    ]
    missing = [p for p in positions if p not in owned]

    if len(owned) == len(positions):
        return f"[{color.name}: MONOPOLY]"
    if not missing:
        return ""

    total = len(positions)
    missing_strs: list[str] = []
    for pos in missing:
        space = Board.get_space(pos)
        prop = pm.properties.get(pos)
        if prop is not None and prop.owner is not None:
            missing_strs.append(f"{space.name} [{pos}] owned by Player {prop.owner}")
        else:
            missing_strs.append(f"{space.name} [{pos}] (unowned)")

    return f"[{color.name}: you own {len(owned)}/{total}, {', '.join(missing_strs)}]"


def _format_opponent_summary(game: MonopolyGame, player_id: int) -> str:
    """Format a compact summary line for an opponent."""
    player = game.players[player_id]
    pm = game.property_manager
    props = pm.get_owned_by(player_id)
    monopolies = [
        c
        for c in pm.get_monopolies(player_id)
        if c not in (PropertyColor.RAILROAD, PropertyColor.UTILITY)
    ]

    prop_strs = [f"{Board.get_space(p).name} [{p}]" for p in sorted(props)]
    props_text = ", ".join(prop_strs) if prop_strs else "(no properties)"

    mono_text = f"{len(monopolies)} monopolies" if monopolies else "0 monopolies"
    jail_text = " (IN JAIL)" if player.in_jail else ""

    return (
        f"- Player {player_id} ({player.name}): "
        f"${player.money} | {props_text} | {mono_text}{jail_text}"
    )


def _near_monopoly_hints(game: MonopolyGame, player_id: int) -> list[str]:
    """Generate near-monopoly hints for a player."""
    pm = game.property_manager
    hints: list[str] = []

    for color in PropertyColor:
        if color in (PropertyColor.RAILROAD, PropertyColor.UTILITY):
            continue
        positions = PROPERTY_GROUPS.get(color, ())
        if not positions:
            continue

        owned = [
            p
            for p in positions
            if pm.properties.get(p) is not None and pm.properties[p].owner == player_id
        ]
        total = len(positions)

        if 0 < len(owned) < total:
            missing = [p for p in positions if p not in owned]
            missing_strs: list[str] = []
            for pos in missing:
                space = Board.get_space(pos)
                prop = pm.properties.get(pos)
                if prop is not None and prop.owner is not None:
                    missing_strs.append(f"{space.name} [{pos}] (P{prop.owner})")
                else:
                    missing_strs.append(f"{space.name} [{pos}] (unowned)")

            if len(owned) == total - 1:
                hints.append(
                    f"[{color.name}: owns {len(owned)}/{total}, missing {', '.join(missing_strs)}]"
                )

    return hints


def _get_unowned_properties(pm: PropertyManager) -> list[int]:
    """Get all unowned buyable property positions."""
    unowned: list[int] = []
    for pos, prop in sorted(pm.properties.items()):
        if prop.owner is None:
            unowned.append(pos)
    return unowned
