"""Supplement frozen AEC certification with seeded learner and API/bot parity.

Learners stop at their own elimination. Their remaining opponents then finish through
the underlying AEC environment so the complete action hash can still be compared to
the original tournament. No learner is stepped after its terminal transition.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from agents.trading_agent import TradingAgent
from api.services.ai_manager import AIManager
from api.services.game_manager import GameManager
from monopoly_engine import MonopolyGame
from monopoly_gym import SingleAgentMonopolyEnv
from monopoly_gym.action_space import ActionEncoder
from monopoly_gym.observation import ObservationEncoder, flatten_observation
from scripts.certify_trading_integration import _agent
from training.pettingzoo_selfplay import SelfPlayEnv

CONSUMERS = ("single_agent", "self_play", "api_bot")
COMPARISON_FIELDS = (
    "status",
    "winner",
    "turns",
    "decisions",
    "elimination_order",
    "cash",
    "proposals",
    "acceptances",
    "rejections",
    "trace_sha256",
)


class TraceCheck:
    """Observe actual consumer actions and check each against a separate engine."""

    def __init__(self, record: dict[str, Any]):
        self.game = MonopolyGame(
            len(record["policies"]), seed=record["seed"], rules_id=record["rules_id"]
        )
        self.encoder = ActionEncoder(rules_id=record["rules_id"])
        self.observer = ObservationEncoder(
            len(record["policies"]), rules_id=record["rules_id"], action_encoder=self.encoder
        )
        self.trace: list[dict[str, int]] = []

    def transition(self, actor: int, index: int, actual: MonopolyGame) -> None:
        command = self.encoder.decode(index, actor, self.game)
        self.game.apply_action(actor, command)
        assert self.game.to_dict() == actual.to_dict(), "Consumer/direct state divergence"
        self.trace.append({"actor": actor, "action": int(index)})

    def boundary(self, env: SingleAgentMonopolyEnv, observation: np.ndarray) -> None:
        seat = env.learner_seat
        expected = flatten_observation(self.observer.encode(self.game, seat))
        np.testing.assert_array_equal(observation, expected)
        np.testing.assert_array_equal(
            env.action_masks(), self.encoder.get_action_mask(self.game, seat)
        )
        assert env.observation_space.contains(observation)

    def result(self, truncated: bool = False) -> dict[str, Any]:
        game = self.game
        events = game.state.structured_event_log
        return {
            "status": "completed" if game.game_over else "cutoff" if truncated else "unfinished",
            "winner": game.winner,
            "turns": game.turn_number,
            "decisions": len(self.trace),
            "elimination_order": game.state.elimination_order,
            "cash": [player.money for player in game.players],
            "proposals": sum(event["type"] == "trade_proposed" for event in events),
            "acceptances": sum(event["type"] == "trade_accepted" for event in events),
            "rejections": sum(event["type"] == "trade_rejected" for event in events),
            "trace_sha256": hashlib.sha256(
                json.dumps(self.trace, sort_keys=True).encode()
            ).hexdigest(),
        }


def run_learner(record: dict[str, Any], consumer: str) -> dict[str, Any]:
    wrapper = SingleAgentMonopolyEnv if consumer == "single_agent" else SelfPlayEnv
    focal, policies, seed = record["focal_seat"], record["policies"], record["seed"]
    opponents = {name for seat, name in enumerate(policies) if seat != focal}
    assert len(opponents) == 1
    env = wrapper(
        len(policies),
        opponent_type="trading_" + opponents.pop(),
        learner_seat=focal,
        rules_id=record["rules_id"],
        max_turns=1000,
    )
    check = TraceCheck(record)
    original_step = env._env.step

    def checked_step(index: int | None) -> None:
        actor = env._env.agent_name_mapping[env._env.agent_selection]
        original_step(index)
        if index is not None:
            check.transition(actor, index, env._env.game)

    env._env.step = checked_step
    policy_seeds = [seed * 11 + seat for seat in range(len(policies))]
    observation, info = env.reset(options={"engine_seed": seed, "policy_seeds": policy_seeds})
    check.boundary(env, observation)
    agent = _agent(policies[focal], focal, policy_seeds[focal], "mutual")
    assert isinstance(agent, TradingAgent)
    boundaries = responses = 0
    reward_total = 0.0
    while True:
        if info.get("pending_terminal"):
            index = 0  # The wrapper delivers a terminal event that happened during reset.
        else:
            responses += int(check.game.state.phase == "trade_response")
            command = agent.choose_native_action(env._env.game, env._env.action_encoder)
            index = env._env.action_encoder.encode_current(command, env._env.game, focal)
        observation, reward, terminated, truncated, info = env.step(index)
        boundaries += 1
        check.boundary(env, observation)
        expected_terminated = check.game.game_over or check.game.players[focal].bankrupt
        assert terminated == expected_terminated
        expected_reward = (
            (1.0 if check.game.winner == focal else -1.0) if expected_terminated else 0.0
        )
        assert reward == expected_reward, "Learner reward belongs to the wrong player/boundary"
        assert not (terminated and truncated)
        reward_total += reward
        if terminated or truncated:
            break
    learner_revision = check.game.state.revision
    before = env._env.game.to_dict()
    try:
        env.step(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Learner accepted a second terminal step")
    assert env._env.game.to_dict() == before
    # Preserve the live policies' RNG and queued post-trade actions for the continuation.
    agents = {env._env.agent_name_mapping[name]: policy for name, policy in env._opponents.items()}
    agents[focal] = agent
    while not check.game.game_over and not any(env._env.truncations.values()):
        name = env._env.agent_selection
        if env._env.terminations[name]:
            env._env.step(None)
            continue
        actor = env._env.agent_name_mapping[name]
        command = agents[actor].choose_native_action(env._env.game, env._env.action_encoder)
        env._env.step(env._env.action_encoder.encode_current(command, env._env.game, actor))
    result = check.result(any(env._env.truncations.values()))
    result.update(
        learner_boundaries=boundaries,
        learner_responses=responses,
        learner_terminal_revision=learner_revision,
        learner_reward=reward_total,
    )
    env.close()
    return result


async def run_api(record: dict[str, Any]) -> dict[str, Any]:
    manager, bots = GameManager(), AIManager(think_delay_ms=0)
    game_id = await manager.create_game(
        len(record["policies"]), seed=record["seed"], rules_id=record["rules_id"]
    )
    active = await manager.get_game(game_id)
    assert active is not None
    # Use the real bot action loop one decision at a time. No background scheduler or
    # transport delays are attached to this in-process parity driver.
    for seat, name in enumerate(record["policies"]):
        bots.create_agent(game_id, seat, "trading_" + name, seed=record["seed"] * 11 + seat)
    check = TraceCheck(record)
    execute = manager.execute_action

    async def checked_execute(game_id, action, **kwargs):
        actor = active.game.decision_player
        index = check.encoder.encode_current(action, check.game, actor)
        success, reason = await execute(game_id, action, **kwargs)
        assert success, reason
        check.transition(actor, index, active.game)
        return success, reason

    manager.execute_action = checked_execute
    truncated = False
    while not active.game.game_over:
        count = await bots.process_ai_turn(
            manager, game_id, active.game.decision_player, max_actions=1
        )
        assert count == 1, "API bot stalled"
        game, focal = active.game, record["focal_seat"]
        if (
            not game.game_over
            and game.turn_number >= 1000
            and game.state.phase != "trade_response"
            and (game.decision_player == focal or game.players[focal].bankrupt)
        ):
            truncated = True
            break
    return check.result(truncated)


def run_job(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
    record, consumer = job
    result = (
        asyncio.run(run_api(record)) if consumer == "api_bot" else run_learner(record, consumer)
    )
    differences = [field for field in COMPARISON_FIELDS if result[field] != record[field]]
    assert not differences, (consumer, record["label"], record["seed"], differences)
    return {
        "consumer": consumer,
        "label": record["label"],
        "seed": record["seed"],
        "focal_seat": record["focal_seat"],
        **result,
    }


def select_records(run: Path, blocks: int) -> list[dict[str, Any]]:
    manifest = json.loads((run / "manifest.json").read_text())
    records = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
    selected = [row for row in records if row["seed"] < manifest["seed_root"] + blocks]
    expected = {
        (label, manifest["seed_root"] + block, seat)
        for label in manifest["matchups"]
        for block in range(blocks)
        for seat in range(int(label[0]))
    }
    actual = {(row["label"], row["seed"], row["focal_seat"]) for row in selected}
    assert actual == expected and len(selected) == len(expected), "Incomplete seed/seat schedule"
    assert all(row["rules_id"] == "foundation-trade-v1" for row in selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocks < 1 or args.workers < 1:
        parser.error("blocks and workers must be positive")
    records = select_records(args.run, args.blocks)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        {
            path
            for root in (
                "monopoly_engine",
                "monopoly_gym",
                "agents",
                "api",
                "training",
                "evaluation",
                "scripts",
                "frontend/src",
                "frontend/e2e",
                "tests",
            )
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix in (".py", ".ts", ".tsx")
        }
    ) + [Path("uv.lock"), Path("frontend/package-lock.json")]
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path))
    manifest = {
        "purpose": "milestone1d-shared-consumer-parity",
        "blocks_per_matchup": args.blocks,
        "consumers": CONSUMERS,
        "games_per_consumer": len(records),
        "source_run": str(args.run),
        "reference_records_sha256": hashlib.sha256(
            (args.run / "records.jsonl").read_bytes()
        ).hexdigest(),
        "reference_manifest_sha256": hashlib.sha256(
            (args.run / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_archive_sha256": hashlib.sha256(
            (args.output / "source.tar.gz").read_bytes()
        ).hexdigest(),
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy_seed_schedule": "engine_seed * 11 + player_id (explicit reset policy_seeds)",
        "jobs": [{key: row[key] for key in ("label", "seed", "focal_seat")} for row in records],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [(record, consumer) for record in records for consumer in CONSUMERS]
    total = 0
    with (
        ProcessPoolExecutor(args.workers) as pool,
        (args.output / "records.jsonl").open("w") as output,
    ):
        for count, result in enumerate(pool.map(run_job, jobs, chunksize=1), start=1):
            output.write(json.dumps(result) + "\n")
            output.flush()
            total += result["decisions"]
            if count % 36 == 0:
                print(f"Matched {count}/{len(jobs)} consumer games", flush=True)
    summary = {"games": len(jobs), "checked_transitions": total, "mismatches": 0}
    summary["records_sha256"] = hashlib.sha256(
        (args.output / "records.jsonl").read_bytes()
    ).hexdigest()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
