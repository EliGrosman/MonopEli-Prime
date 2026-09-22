"""Local browser test server. Never import this fixture app in production."""

from api.config import Settings
from api.main import create_app
from monopoly_engine.cards import CardType
from monopoly_engine.foundation import charge, settle

app = create_app(Settings(rate_limit_per_minute=10000, rate_limit_burst=10000))


@app.post("/__scenario/{scenario}")
async def scenario_game(scenario: str):
    manager = app.state.game_manager
    rules_id = (
        "foundation-trade-v1"
        if scenario in {"trade", "trade-human-bot", "trade-bot-human", "trade-debt"}
        else "foundation-v1"
    )
    game_id = await manager.create_game(num_players=2, seed=4, rules_id=rules_id)
    active = await manager.get_game(game_id)
    game = active.game
    session_id = "browser-" + game_id
    await manager.claim_player_slot(game_id, 0, session_id)
    if scenario.startswith("jail"):
        game.send_to_jail(0)
        game.state.phase = "jail_decision"
        if scenario == "jail-card":
            deck = game.state.chance_deck
            card = next(c for c in deck.cards if c.card_type == CardType.GET_OUT_OF_JAIL)
            deck.cards.remove(card)
            game.state.jail_card_sources[0] = ["chance"]
            game.players[0].jail_cards = 1
        game.roll_dice = lambda: (1, 2)
    elif scenario in {"debt", "trade-debt"}:
        game.current_player = 1
        game.state.decision_player = 1
        game.state.phase = "asset_management"
        game.state.roll_owed = False
        game.players[0].money = 0
        game.property_manager.properties[39].owner = 0
        charge(game, 0, 100, 1)
        settle(game)
    elif scenario == "purchase":
        game.players[0].position = 6
        game.state.phase = "purchase_decision"
        game.state.roll_owed = True
        game.last_roll = (3, 3)
        game.doubles_count = 1
    elif scenario in {"trade", "trade-human-bot", "trade-bot-human"}:
        game.state.phase = "asset_management"
        game.state.roll_owed = False
        game.property_manager.properties[1].owner = 0
        game.property_manager.properties[6].owner = 0
        game.property_manager.properties[3].owner = 1
        game.property_manager.properties[8].owner = 1
        game.property_manager.properties[9].owner = 1
        if scenario == "trade-human-bot":
            app.state.ai_manager.create_agent(game_id, 1, "trading_rule_based")
            return {"game_id": game_id, "session_id": session_id}
        recipient_session_id = "browser-recipient-" + game_id
        await manager.claim_player_slot(game_id, 1, recipient_session_id)
        if scenario == "trade-bot-human":
            app.state.ai_manager.create_agent(game_id, 0, "trading_rule_based")
            await app.state.ai_manager.process_ai_turns_for_game(manager, game_id)
        return {
            "game_id": game_id,
            "session_id": session_id,
            "recipient_session_id": recipient_session_id,
        }
    else:
        raise ValueError("Unknown browser scenario")
    return {"game_id": game_id, "session_id": session_id}
