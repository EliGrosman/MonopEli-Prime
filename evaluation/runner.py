"""Reproducible full-game execution independent of training rewards."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from typing import Any

import numpy as np

from agents import AggressiveAgent, ConservativeAgent, RandomAgent, RuleBasedAgent
from monopoly_engine import MonopolyGame
from monopoly_engine.progress import ProgressGuard
from monopoly_gym.action_space import ActionEncoder

POLICIES = {
    "random": RandomAgent,
    "rule_based": RuleBasedAgent,
    "aggressive": AggressiveAgent,
    "conservative": ConservativeAgent,
}
EVALUATOR_VERSION = "foundation-evaluation-v1"


def invariant(game: MonopolyGame) -> None:
    props = list(game.property_manager.properties.values())
    assert game.houses_remaining + sum(p.houses for p in props if p.houses < 5) == 32
    assert game.hotels_remaining + sum(p.houses == 5 for p in props) == 12
    assert all(0 <= p.houses <= 5 for p in props)
    assert all(p.owner is None or not game.players[p.owner].bankrupt for p in props)
    assert all(p.money >= 0 for p in game.players)
    assert all(not p.mortgaged or p.houses == 0 for p in props)
    if game.game_over:
        assert [p.id for p in game.players if not p.bankrupt] == [game.winner]
    else:
        assert not game.players[game.decision_player].bankrupt
        assert game.winner is None


def play_game(
    policies: list[str],
    seed: int,
    focal_seat: int = 0,
    max_turns: int = 1000,
    capture: bool = False,
    policy_instances: list[Any] | None = None,
) -> dict[str, Any]:
    n = len(policies)
    stream = np.random.SeedSequence(seed).spawn(n + 1)
    derived = [int(s.generate_state(1)[0]) for s in stream]
    game = MonopolyGame(n, seed=derived[0])
    if policy_instances is None:
        agents = [
            POLICIES[name](i, seed=derived[i + 1]) if name == "random" else POLICIES[name](i)
            for i, name in enumerate(policies)
        ]
    else:
        agents = policy_instances
        if any(type(a).__name__ == "HybridAgent" for a in agents):
            raise ValueError("Hybrid side effects are disabled in foundation-v1")
    for agent in agents:
        agent.reset()
    encoder = ActionEncoder()
    guard = ProgressGuard()
    trace = []
    digest = hashlib.sha256()
    decisions, inference = 0, 0.0
    initial = game.to_dict()
    start = time.monotonic()
    status, error = "cutoff", None
    try:
        while not game.game_over:
            if game.turn_number >= max_turns and (
                game.decision_player == focal_seat or game.players[focal_seat].bankrupt
            ):
                break
            guard.check(game)
            invariant(game)
            pid = game.decision_player
            mask = encoder.get_action_mask(game, pid)
            if not mask.any():
                raise RuntimeError("Live decision has no legal actions")
            t = time.monotonic()
            action = int(agents[pid].choose_action(None, mask, game))
            inference += time.monotonic() - t
            if not 0 <= action < len(mask) or not mask[action]:
                raise ValueError(f"Illegal policy action: {action}")
            result = game.apply_action(pid, encoder.decode(action, pid, game))
            item = {
                "actor": pid,
                "action": action,
                "events": result.events,
                "revision": result.revision,
            }
            digest.update(json.dumps(item, sort_keys=True).encode())
            trace.append(item)
            decisions += 1
            invariant(game)
        if game.game_over:
            status = "completed"
    except Exception as exc:
        status = "stalled" if str(exc) == "stalled" else "error"
        error = f"{type(exc).__name__}: {exc}"
    won = status == "completed" and game.winner == focal_seat
    result = {
        "seed": seed,
        "derived_seeds": derived,
        "focal_seat": focal_seat,
        "policies": policies,
        "policy_configurations": [
            {
                "class": type(a).__module__ + "." + type(a).__name__,
                "player_id": i,
                "buy_threshold": getattr(a, "buy_threshold", None),
                "build_threshold": getattr(a, "build_threshold", None),
                "seed": derived[i + 1],
                "mode": "direct",
            }
            for i, a in enumerate(agents)
        ],
        "diagnostic_cash": [p.money for p in game.players],
        "cutoff_reason": "turn_limit" if status == "cutoff" else None,
        "illegal_action_failures": int(error is not None and "Illegal policy action" in error),
        "status": status,
        "winner": game.winner,
        "game_over": game.game_over,
        "win": won,
        "loss": status == "completed" and not won,
        "eliminated": game.players[focal_seat].bankrupt,
        "elimination_order": game.state.elimination_order,
        "turns": game.turn_number,
        "decisions": decisions,
        "horizon": max_turns,
        "overshoot": max(0, game.turn_number - max_turns),
        "error": error,
        "trace_sha256": digest.hexdigest(),
        "runtime_seconds": time.monotonic() - start,
        "inference_seconds": inference,
        "rules_id": "foundation-v1",
        "action_version": "action-v2",
        "observation_version": "observation-v2",
        "reward_version": "terminal-v1",
        "evaluator_version": EVALUATOR_VERSION,
    }
    if capture or status in ("error", "stalled"):
        result["replay"] = {"initial": initial, "trace": trace}
    return result


def summarize(records: list[dict[str, Any]], bootstrap_seed: int = 12345) -> dict[str, Any]:
    total = len(records)
    counts = Counter(r["status"] for r in records)
    blocks: dict[int, list[int]] = {}
    for r in records:
        blocks.setdefault(r["seed"], []).append(int(r["win"]))
    means = np.asarray([np.mean(values) for values in blocks.values()])
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(means, (2000, len(means)), replace=True).mean(axis=1)
    wins = sum(r["win"] for r in records)
    return {
        "games": total,
        "counts": dict(counts),
        "wins": wins,
        "losses": sum(r["loss"] for r in records),
        "eliminations": sum(r["eliminated"] for r in records),
        "win_rate": wins / total,
        "win_rate_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "completed_win_rate": wins / counts["completed"] if counts["completed"] else None,
        "completion_rate": counts["completed"] / total,
        "transitions": sum(r["decisions"] for r in records),
        "ready": counts["completed"] / total >= 0.95
        and not counts["error"]
        and not counts["stalled"],
    }


class PolicyAgent:
    """Direct MaskablePPO inference with explicit encoding compatibility checks."""

    def __init__(self, model: Any, player_id: int, num_players: int, deterministic: bool = True):
        from monopoly_gym.action_space import GAMEPLAY_ACTION_SPACE_SIZE
        from monopoly_gym.observation import ObservationEncoder, get_flat_observation_size

        if model.action_space.n != GAMEPLAY_ACTION_SPACE_SIZE or model.observation_space.shape != (
            get_flat_observation_size(num_players),
        ):
            raise ValueError("Checkpoint requires observation-v2/action-v2; no implicit remapping")
        self.model, self.player_id = model, player_id
        self.encoder = ObservationEncoder(num_players)
        self.deterministic = deterministic

    def reset(self):
        pass

    def choose_action(self, observation, action_mask, game):
        from monopoly_gym.observation import flatten_observation

        obs = flatten_observation(self.encoder.encode(game, self.player_id))
        action, _ = self.model.predict(
            obs, action_masks=action_mask, deterministic=self.deterministic
        )
        return int(action)


def evaluate_policy(
    model: Any,
    opponent: str,
    games: int,
    players: int = 2,
    horizon: int = 1000,
    seed: int = 17000000,
    deterministic: bool = True,
):
    records = []
    if games % players:
        raise ValueError("Seat-balanced evaluation requires games divisible by players")
    for i in range(games):
        focal = i % players
        game_seed = seed + i // players
        seeds = np.random.SeedSequence(game_seed).spawn(players + 1)
        policies = [opponent] * players
        agents = []
        for seat in range(players):
            if seat == focal:
                agents.append(PolicyAgent(model, seat, players, deterministic))
            else:
                cls = POLICIES[opponent]
                agents.append(
                    cls(seat, seed=int(seeds[seat + 1].generate_state(1)[0]))
                    if opponent == "random"
                    else cls(seat)
                )
        policies[focal] = "direct_policy"
        records.append(play_game(policies, game_seed, focal, horizon, policy_instances=agents))
    return records
