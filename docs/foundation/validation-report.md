# Foundation milestone implementation and validation

Status: **implemented candidate; readiness blocked**. Keep `foundation-v1`, the 1,000-turn
horizon and the 95% completion requirement unchanged, as requested on 2026-09-21.
No substantive model training was resumed. Do not promote this candidate as certified Monopoly
strength or proceed to training on the basis of these results.

The engine now owns phases, decision ownership, explicit rolls, purchase refusal, obligations,
liquidation, bankruptcy and replay state. API humans/bots, AEC, the shared learner adapter and
the headless runner use `apply_action`. Sparse terminal rewards, fresh observations/masks,
independent episode streams and explicit incompatible-mode/checkpoint rejection replace the
old training shortcuts. Search remains experimental; learned search and hybrid trading reject.

## Tournament evidence

Two runs each executed 12,000 games, including all seat rotations for 500 two-player or 250
four-player seed blocks per matchup. Each run covered **15,069,861 transitions**. Outcomes,
seeds, elimination sequences, decisions and action/event trace hashes agree across every pair.
No errors, illegal-action fallbacks or stalled games were recorded. Thirty-six saved full replays
were restored with current code: **39,979 transitions** reproduced their events and revisions.
Every scheduled game remains in the primary denominator. A cutoff has no winner.

| Matchup | Wins | Losses | Cutoffs | Completion | All-game win rate (95% seed-block CI) |
|---|---:|---:|---:|---:|---:|
| 2p-rule_based-vs-random | 1000 | 0 | 0 | 100.0% | 100.0% (100.0%–100.0%) |
| 2p-rule_based-vs-rule_based | 443 | 443 | 114 | 88.6% | 44.3% (43.0%–45.6%) |
| 2p-aggressive-vs-random | 1000 | 0 | 0 | 100.0% | 100.0% (100.0%–100.0%) |
| 2p-aggressive-vs-rule_based | 484 | 407 | 109 | 89.1% | 48.4% (46.3%–50.5%) |
| 2p-conservative-vs-random | 1000 | 0 | 0 | 100.0% | 100.0% (100.0%–100.0%) |
| 2p-conservative-vs-rule_based | 409 | 504 | 87 | 91.3% | 40.9% (38.7%–43.2%) |
| 4p-rule_based-vs-random | 1000 | 0 | 0 | 100.0% | 100.0% (100.0%–100.0%) |
| 4p-rule_based-vs-rule_based | 88 | 264 | 648 | 35.2% | 8.8% (7.3%–10.3%) |
| 4p-aggressive-vs-random | 1000 | 0 | 0 | 100.0% | 100.0% (100.0%–100.0%) |
| 4p-aggressive-vs-rule_based | 126 | 224 | 650 | 35.0% | 12.6% (11.0%–14.2%) |
| 4p-conservative-vs-random | 999 | 0 | 1 | 99.9% | 99.9% (99.7%–100.0%) |
| 4p-conservative-vs-rule_based | 46 | 339 | 615 | 38.5% | 4.6% (3.5%–5.8%) |

All six matchups against rule-based opponents fail the 95% completion gate. Four-player
rule-based self-match completion is 35.2%, with 648 cutoffs. This is an observed limitation of
the approved no-auction/no-trade task, not a reason to adjudicate wealth as a win. No superiority
claim is made from these point estimates. The CI resamples complete seed blocks, keeping seats
together; future policy-difference reports must pair those same blocks.

## Reproduction and provenance

```sh
uv sync --dev --extra training
.venv/bin/python scripts/benchmark_foundation.py --games 200 --workers 8 --output artifacts/development-new
.venv/bin/python scripts/benchmark_foundation.py --games 1000 --workers 8 --namespace certification --output artifacts/certification-new
.venv/bin/python scripts/verify_foundation_run.py artifacts/foundation-certification-final --compare artifacts/foundation-certification
```

Run directories must not already exist. Development roots start at 11000000; certification at
17000000. These roots are reserved separately from future training. The frozen manifest contains
all seed/seat assignments, dependency-lock hash, dirty-tree status, evaluator/encoding/rules/reward
versions and complete source-file hashes. Each result identifies policy class, thresholds, seed,
status, elimination order, actual horizon/overshoot, timings and trace hash. Timings are excluded
from deterministic comparison. Historical checkpoints were not silently adapted.

Local raw results: `artifacts/foundation-certification-final/`. Generated JSON/JSONL records are
ignored by Git at the user's request; documentation manifests and browser fixtures remain tracked.
The measured candidate was a dirty checkout based on main `b9d8f95`; the archived source, rather
than that base commit alone, identifies exactly what ran. The first run lacked an untracked-source
archive; the second remedies that gap. Later test, API seed/configuration and replay-type fixes
were validated separately; the saved tournament replays also pass under the current code.

- Frozen run-manifest SHA-256: `9856ae19abe63952414e5e136e8e7ccf094f72d367b84c752b113ee123e89efe`.
- Source archive SHA-256: `42fbcf49960a0b4f3b315c226377a22f335cc64a29f333504e809d78520869fd` (`source.tar.gz` in that directory).
- Dependency-lock SHA-256: `7993429362e8a6351af8695ffe948e5af7b3a3f4a4ca5dd2a681c06a2746774c`.

## Validation and limits

- Full locked-environment backend run: 1,870 passed, five external-LLM skips, 57 strict expected
  failures preserving deferred learned-search/trading positive-path tests. Those features are not
  counted as implemented. The suite includes a 200-transition API/AEC/direct-engine parity test.
  Foundation scenarios include exact debt, card-rent, bankruptcy, jail, purchase,
  replay, seed, reward, dead-agent and timeout behavior.
- Backend Ruff (`monopoly_engine api`) and strict engine mypy pass. Frontend: 345 unit tests,
  ESLint, TypeScript/build and source formatting pass.
- Gymnasium/PettingZoo contract checks and a 16-step CPU MaskablePPO collection/save/load smoke
  test pass; DummyVecEnv preserves terminal observation and timeout bootstrap semantics.
- Tournament invariants check building supplies, ownership, mortgage/building compatibility,
  nonnegative cash, valid actor and unique engine winner. Exact financial scenario tests cover
  bank/inter-player payments. These checks are not a separately implemented bank ledger.
- All eight browser scenarios pass: three scripted transport tests and five live Chromium →
  FastAPI → engine tests. They cover reconnection, all three jail choices, out-of-turn debt
  rescue and doubles purchase refusal.
- This machine tested Python 3.13.13, Gymnasium 1.2.3, PettingZoo 1.25.0 and SB3/sb3-contrib 2.7.1.
  Remote CI's other Python versions were not run locally. PettingZoo emits two advisory warnings
  about the intentionally structured observation space. The test driver additionally installed
  pytest-timeout 2.4.0 for hang detection; this is not a runtime or lockfile dependency.

## Branch and artifact outcome

All five audited branch heads have local `archive/milestone1/*` tags. The verified recovery
bundle and Git/LFS artifact distinction are documented in [branch inventory](../branch-inventory.md)
and [artifact manifest](artifact-manifest.json). Missing LFS dataset payloads remain unavailable
locally; a bundle preserves their pointers, not their data.

No remote branches were deleted: the release gate is blocked. Retain synthetic-data, phase-two
training and mixed MCTS-updates branches for their recorded future milestones. The two fully
merged refs remain deletion candidates only after the documented validation/recovery gates.
No old branch snapshot was merged into the repaired code.

## Explicitly deferred

Auctions; trading/negotiation UI; immediate inherited-mortgage transfer interest; guided Jev;
new datasets, BC/critic recovery, scaled PPO/PPG and leagues; reward shaping; learned/chance-aware
MCTS; caching optimizations. Historical README/training accuracy and wealth-adjudicated labels
are not current strength evidence. The next milestone remains blocked until the retained
readiness criteria are addressed explicitly; the rules and gate have not been relaxed.

## Browser commands

```sh
cd frontend
npm ci
npx playwright install chromium
npx playwright test e2e/foundation.e2e.ts --project=chromium
npx playwright test --config playwright.foundation.config.ts
```

The live configuration starts a local Vite server and a **test-only** FastAPI app under
`tests/foundation/browser_app.py`. Its scenario endpoint is never registered in the production
application. The frontend/API commands and replies are real; only initial game fixtures and
jail dice are controlled.
