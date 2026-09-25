"""Native asynchronous evaluation for the guided Jev integration."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agents.jev import FakeProvider
from agents.jev.types import QuestionBatch
from api.config import Settings
from api.services.ai_manager import AIManager
from api.services.game_manager import GameManager

from .runner import invariant

JEV_EVALUATOR_VERSION = "guided-jev-evaluation-v1"


def deterministic_fake_answer(request: QuestionBatch) -> dict[str, Any]:
    """Return stable typed answers without approximating live model output."""

    answers: dict[str, Any] = {}
    for question_id, question in request.questions.items():
        criteria = question["criteria"]
        keys = list(criteria)
        preferred: tuple[str, ...]
        if question_id == "cash_reserve":
            preferred = ("reserve_200",)
        elif question_id == "action":
            preferred = (
                "accept_trade",
                "buy_property",
                "build_house",
                "build_hotel",
                "unmortgage_property",
                "roll_dice",
                "consider_trade",
                "pass_buy",
                "end_turn",
            )
        elif question_id == "trade_path":
            preferred = tuple(key for key in keys if key.startswith("suggestion_")) + (
                tuple(key for key in keys if key.startswith("construct_offer_to_"))
            ) + (
                "skip_trade",
            )
        elif question_id == "post_trade_action":
            preferred = ("end_turn", "roll_dice", "pass_buy")
        elif question_id in {"second_give_property", "second_want_property"}:
            preferred = ("none",)
        elif question_id == "cash_direction":
            preferred = ("none",)
        elif question_id == "complete_offer":
            preferred = ("propose",)
        else:
            preferred = tuple(keys)
        selected = next((key for key in preferred if key in criteria), keys[0])
        answers[question_id] = {
            "type": "choice",
            "choice": selected,
            "probabilities": {
                key: 1.0 if key == selected else 0.0 for key in criteria
            },
            "confidence": 1.0,
        }
    return {
        "model": "fake-jev-deterministic-v1",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": len(answers)},
    }


async def play_fake_game(
    *,
    seed: int,
    focal_seat: int,
    opponent: str,
    horizon: int = 1_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one game through AIManager and GameManager with an offline provider."""

    provider = FakeProvider(default_step=deterministic_fake_answer)
    ai = AIManager(think_delay_ms=0, settings=Settings(), provider=provider)
    games = GameManager(background_ai_scheduling=False)
    games.set_ai_manager(ai)
    game_id = await games.create_game(
        2,
        player_names=["Seat 1", "Seat 2"],
        seed=seed,
        rules_id="foundation-trade-v1",
        owner_session_id=f"offline-{seed}",
    )
    active = await games.get_game(game_id)
    assert active is not None
    for player_id in range(2):
        ai_type = "jev" if player_id == focal_seat else f"trading_{opponent}"
        ai.create_agent(
            game_id,
            player_id,
            ai_type,
            seed=seed + player_id + 1,
            session_budget_id=f"offline-{seed}",
        )
        active.player_slots[player_id].is_ai = True
        active.player_slots[player_id].ai_type = ai_type
    initial = active.game.to_dict()
    started = time.monotonic()
    status = "cutoff"
    error: str | None = None
    try:
        while not active.game.game_over:
            if (
                active.game.state.phase != "trade_response"
                and active.game.turn_number >= horizon
                and (
                    active.game.decision_player == focal_seat
                    or active.game.players[focal_seat].bankrupt
                )
            ):
                break
            invariant(active.game)
            actor = active.game.decision_player
            taken = await ai.process_ai_turn(games, game_id, actor, max_actions=1)
            if taken != 1:
                raise RuntimeError("AI decision made no progress")
            invariant(active.game)
        if active.game.game_over:
            status = "completed"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    trace = list(active.agent_action_log)
    final = active.game.to_dict()
    event_counts: Counter[str] = Counter()
    for item in trace:
        event_counts.update(event["type"] for event in item["structured_events"])
    jev_rows = [
        item for item in trace if item.get("source") in {"jev", "forced", "fallback"}
    ]
    strategic = [item for item in jev_rows if item.get("source") != "forced"]
    latencies = sorted(float(item.get("latency_seconds", 0.0)) for item in jev_rows)
    usage = ai.usage_for_game(game_id) or {}
    record = {
        "seed": seed,
        "focal_seat": focal_seat,
        "opponent": opponent,
        "status": status,
        "error": error,
        "winner": active.game.winner,
        "win": status == "completed" and active.game.winner == focal_seat,
        "loss": status == "completed" and active.game.winner != focal_seat,
        "eliminated": active.game.players[focal_seat].bankrupt,
        "turns": active.game.turn_number,
        "decisions": len(trace),
        "jev_decisions": len(jev_rows),
        "forced_decisions": sum(item.get("source") == "forced" for item in jev_rows),
        "fallback_decisions": sum(item.get("source") == "fallback" for item in jev_rows),
        "strategic_decisions": len(strategic),
        "fallback_rate": (
            sum(item.get("source") == "fallback" for item in strategic) / len(strategic)
            if strategic
            else 0.0
        ),
        "proposals": event_counts["trade_proposed"],
        "acceptances": event_counts["trade_accepted"],
        "rejections": event_counts["trade_rejected"],
        "request_latency_p95": (
            latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            if latencies
            else 0.0
        ),
        "usage": usage,
        "runtime_seconds": time.monotonic() - started,
        "rules_id": "foundation-trade-v1",
        "evaluator_version": JEV_EVALUATOR_VERSION,
        "provider": "fake",
        "model": "fake-jev-deterministic-v1",
    }
    replay = {"initial": initial, "trace": trace, "final": final}
    await ai.shutdown()
    return record, replay


async def run_fake_smoke(output: Path, seed_root: int, horizon: int = 1_000) -> None:
    """Run the seat-rotated smoke matrix and write replayable artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    replay_dir = output / "replays"
    replay_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    index = 0
    for opponent in ("rule_based", "aggressive"):
        for focal_seat in range(2):
            seed = seed_root + index
            record, replay = await play_fake_game(
                seed=seed,
                focal_seat=focal_seat,
                opponent=opponent,
                horizon=horizon,
            )
            replay_content = json.dumps(replay, sort_keys=True)
            replay_path = replay_dir / f"game-{index:03d}.json"
            replay_path.write_text(replay_content)
            record["replay_path"] = str(replay_path.relative_to(output))
            record["replay_sha256"] = hashlib.sha256(replay_content.encode()).hexdigest()
            records.append(record)
            index += 1
    (output / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    summary = {
        "games": len(records),
        "counts": dict(Counter(record["status"] for record in records)),
        "wins": sum(record["win"] for record in records),
        "losses": sum(record["loss"] for record in records),
        "fallback_decisions": sum(record["fallback_decisions"] for record in records),
        "strategic_decisions": sum(record["strategic_decisions"] for record in records),
        "proposals": sum(record["proposals"] for record in records),
        "acceptances": sum(record["acceptances"] for record in records),
        "rejections": sum(record["rejections"] for record in records),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    manifest = {
        "evaluator_version": JEV_EVALUATOR_VERSION,
        "provider": "fake",
        "model": "fake-jev-deterministic-v1",
        "prompt_protocol": "guided-jev-v1",
        "seed_root": seed_root,
        "horizon": horizon,
        "games": len(records),
        "live_inference": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
