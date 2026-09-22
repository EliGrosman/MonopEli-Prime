# Milestone 1b: why foundation-v1 games fail to finish

**Readiness remains blocked.** The main measured obstacle is persistent split ownership,
followed by low-pressure developed positions; some games genuinely need longer. A separate
mortgage/build validation defect is confirmed and remains unrepaired in this diagnostic milestone.
No rules, baseline agents, production horizon, completion gate, branches, or training changed.

Investigated candidate: `02a5e73912a25e7cfc0bf8c1fe44329c380dd4a3`,
`codex/foundation-v1`. Starting working tree contained only the existing untracked `AGENTS.md`
and `PROJECT_REVIEW.md`. The review, foundation contract, validation report and branch inventory
were read as evidence, then checked against current execution paths.

## Measured diagnosis

The main development sample has **1,200 games / 1,754,140 decisions**, using 50 seed blocks
`12000000..12000049`, all focal seats, and unchanged foundation-v1 at 1,000 completed player
turns with the existing focal-boundary overshoot. This is not a new certification tournament.
There are 900 games against rule-based opponents and 300 rule-based-versus-random controls.
The same seed blocks are shared across policies; seats and policy comparisons are correlated,
not 1,200 independent draws. Counts below describe this sample, not precise population estimates.

| Focal / opponents | Two-player completed | Four-player completed |
|---|---:|---:|
| Rule-based / rule-based | 88/100 | 68/200 |
| Aggressive / rule-based | 86/100 | 68/200 |
| Conservative / rule-based | 91/100 | 68/200 |
| Rule-based / random | 100/100 | 200/200 |

Zero execution errors, stalls, invalid-action fallbacks or cash-ledger reconciliation failures
were recorded. These checks do **not** imply correct building legality: the separate defect below
passes the engine's current validator and original benchmark invariants.

### Overlapping patterns, including completed controls

| Observation | 2p cutoffs (35) | 4p cutoffs (396) | Completed heuristic games (469) |
|---|---:|---:|---:|
| No complete color group at stop | 22 (62.9%) | 328 (82.8%) | 0 |
| Development at stop | 13 | 68 | 465 |
| Complete group but no buildings at stop | 0 | 0 | 4 |
| All 28 properties owned | 35 | 396 | 268 |
| No ownership/building/mortgage/elimination change for ≥500 turns | 35 | 393 | 0 |
| At least one legal build declined at end-turn | 12 | 12 | 436 |
| A sold property subsequently rebuilt | 6 | 9 | 120 |
| Observed mortgaged-sibling build defect | 1 | 1 | 155 |

All **431 cutoffs** had every property owned, no mortgages, and either no complete group or
**every complete group fully developed to hotels**. No cutoff had a declined legal building
opportunity at or after turn 500. All 35 two-player cutoffs retained two players; 394/396
four-player cutoffs retained four, and two retained two. Thus absent purchases, forgotten
late-game building, insufficient houses, and unredeemed mortgages do not explain these endpoints.

Of the 68 developed four-player cutoffs, 66 had only brown, one only light blue, and one light
blue plus magenta. The 13 developed two-player cutoffs had brown only (7), brown plus light
blue (5), or brown plus dark blue (1). Hotels alone do not ensure sufficiently concentrated rent
pressure. Median total cash at cutoff was $29,430 in two-player and $34,500 in four-player games.
Rent circulates cash between players; GO and cards can replenish vulnerable players while much
of the board remains split and unimproved. Aggregate cash growth alone is not proof that each
player is safe, so the records retain individual cash and classified per-player flows.

Declining an affordable build is often the intended reserve/threshold heuristic, and rebuilding
after forced liquidation can be sensible recovery. Both occur more frequently in completed
controls than in cutoffs. These descriptive flags are not evidence of irrational loops. Likewise,
final monopolies in completed games can result from bankruptcy transfers; final ownership alone
is not evidence that those monopolies caused the win. The random controls are economically weak
and should not be mistaken for a strong-agent completion baseline.

## Longer horizons: persistence and genuine slow games

A purposive, seat-balanced extension includes **48 games at a diagnostic-only 10,000-turn
limit**. Selection is reproducible in `select_long()` in the analysis script:

1. First up to three rule-based seed blocks per player count for each cutoff category
   (no complete group / developed). Only two two-player developed blocks existed. This gives
   34 games in 11 blocks.
2. All cutoff blocks with structural change within the last 500 turns or complete colors other
   than brown/light blue: 14 games in four additional blocks. Include every focal seat, even
   when some rotations had already finished at 1,000 turns.

The rule-based blocks are 2p seeds `12000000, 12000005, 12000011, 12000013, 12000017`;
4p seeds `12000000, 12000001, 12000002, 12000008, 12000012, 12000015`.
Additional blocks: 2p aggressive `12000017`; 4p aggressive `12000025`; 4p conservative
`12000012, 12000021`. These are selected examples, **not** a population completion estimate.

Of 43 original cutoffs, **2 finished**, **41 remained cutoffs**, and three additional eliminations
occurred across three games. Forty retained exactly the same final property ownership, buildings
and mortgages; all players gained cash in 39 (including zeros for previously eliminated players
makes this a deliberately strict count). The five already-completed controls reproduced unchanged.
All 34 selected rule-based games remained unfinished, with no eliminations through 10,000 turns.
This finite experiment does not prove any game can never finish.

| Representative | At 1,000-turn boundary | At extended boundary |
|---|---|---|
| 4p rule-based, seed 12000000, focal 0 | Four survivors; no colors; cash `[8566,7307,8183,7704]` | Four survivors; same properties; `[91478,87788,86737,75677]` at turn 10000 |
| 4p rule-based, seed 12000008, focal 0 | Four survivors; player 2 has brown hotels; `[13968,4683,14646,1844]` | Same owners/buildings; `[134336,18432,175211,12827]` at turn 10000 |
| 2p aggressive, seed 12000017, focal 1 | Both alive; brown/dark-blue hotels | Player 1 wins at turn 1332 |
| 4p conservative, seed 12000021, focal 0 | Players 1/2 alive; light-blue/magenta hotels | Player 1 wins at turn 4154 |
| 4p conservative, seed 12000012, focal 1 | Four survivors; brown hotels | Player 1 eliminated at turn 1450; three survivors, no winner at turn 10000 |

The first example's 1,000-turn ledger records $39,000 GO income and $4,925 card income;
$5,690 purchases, $8,800 taxes, $2,300 jail fines and $1,375 card payments leave $25,760 net bank
inflow. Add the initial $6,000 to obtain $31,760 cash exactly. Separately, $13,833 rent and $1,080
inter-player card payments circulated among players. No rent/income was inferred from net worth.
In the brown-hotel example, total cash grew from $6,000 to $35,141 despite $31,812 in rent transfers.
The full per-player ledger and 50-turn snapshots allow inspection of the different players' exposure.

## Verified paths and correctness finding

- `evaluation.runner.play_game` → `ActionEncoder` mask/decode → `MonopolyGame.apply_action` →
  `foundation.apply_action` is the measured path, also used by the API/environment foundations.
  No private auto-roll, trading side effect or wealth adjudication was introduced.
- Purchase refusal is `PassBuy`; unowned properties remain available for later landings. In this
  sample every cutoff eventually bought the entire board. Auctions could change its initial
  distribution, but an auction of an unowned property cannot exchange already-owned split sets.
- `PropertyManager.has_monopoly` checks all group owners. `can_build_house` enforces ownership,
  even building, target-property mortgage, cash, and building supplies. House actions correctly
  convert four houses into a hotel; rent indexes use the resulting development level.
- All three heuristic classes roll first, buy at their existing thresholds, develop while cash
  exceeds $200 and the construction threshold, then redeem mortgages before ending a turn.
  They liquidate only to resolve debt. No policy thresholds were changed. Three deterministic
  rich-orange-monopoly scenarios each build 15 units into three hotels, return all houses,
  and eliminate a $500 opponent on a forced orange landing. The identical undeveloped scenario
  charges $28 and leaves that opponent alive. A $200 reserve scenario separately verifies a
  legal-but-declined build. A new policy ablation was unnecessary to establish this distinction;
  profitable trading behavior itself remains untested.
- `property_landing` computes rent; `charge` queues obligations; `settle` pays or pauses for
  liquidation; `bankrupt` transfers assets and credits actual remaining cash/building proceeds.
  The observer retains the charge's cause across those pauses. Tests cover rent funded by a
  mortgage, creditor bankruptcy, bank bankruptcy, and exact post-transition cash reconciliation.
- GO credits originate in `move_player` / `move_player_to`, card income in `execute_card`, and
  taxes/fines/card debts in `charge`. No missing mandatory payment was found in the sampled runs.
- `ProgressGuard` detects repeated semantic states or >1,000 decisions within a turn. Moving
  players with growing cash are different states and finish turns normally. Economic persistence
  should be diagnosed separately, not converted into an execution stall or a fabricated winner.

**Confirmed defect:** own Mediterranean and Baltic, mortgage Baltic, then request a house on
Mediterranean. Current `BuildHouse.validate`, action mask and execution accept it. The group-wide
mortgage prohibition is absent from `rules.can_build_house` and the basic property helper.
It is required by the ordinary building/mortgage rules retained by foundation-v1; see the
[Hasbro rulebook](https://www.hasbro.com/common/instruct/Monopoly.pdf) and
[Hasbro's explicit group-wide restriction](https://www.hasbro.com/common/documents/dad288661c4311ddbd0b0800200c9a66/6138541519B9F369102F22309F59478F.pdf).
This is a correctness repair, not a new economic variant.

`test_mortgaged_sibling_must_block_engine_and_mask_build` preserves the desired rejection as a
**strict expected failure**, including mask and nonmutation assertions to activate when repaired.
In the development sample it occurred in 2/431 heuristic cutoffs (four build actions),
155/469 completed heuristic games, and 161/300 random-control games. It therefore matters to
validity, even though it cannot explain the 350 no-monopoly cutoffs, which never developed.
Its causal effect on other outcomes is **not measured** without a paired repaired-engine run.
The previous invariant only prohibited buildings on the mortgaged property itself; it missed
mortgaged siblings. No engine repair was silently folded into these diagnostics.

## Tooling, reproduction and evidence

- `evaluation/economics.py`: diagnostic-only synchronous observer around the unchanged runner.
  Semantic hooks classify actual cash transfers, preserve debt provenance, and reconcile all
  player balances after every decision. Bankruptcy salvage is recorded only when actually
  credited to a player creditor; bank bankruptcy does not invent a sale/payment cash cycle.
  Hooks restore on exceptions; do not use them concurrently in threads. The CLI uses processes.
- Per-game records: final color ownership/fragmentation, unowned properties, physical supplies,
  development/mortgages, eliminations with turns/revisions, per-player action counts, categorized
  flows, 50-turn snapshots, declined end-turn builds, sell/rebuild and mortgage/redemption cycles.
  Structural age measures ownership/building/mortgage/elimination changes; cash-transfer age is
  separate. This explicitly distinguishes rising cash from static productive assets.
- Sample replays include initial/final complete engine snapshots, every action/event/revision,
  transaction ledger and checksums. All other records retain seed/seat/configuration sufficient
  for regeneration. Policy IDs, thresholds, derived RNG seeds and encoding versions come from
  the existing runner. Manifests freeze jobs, candidate SHA, dirty status, lock/source hashes;
  `source.tar.gz` preserves the exact Python sources and dependency files before execution.

From the repository root, using fresh output directories:

```sh
.venv/bin/python -m scripts.diagnose_foundation --output artifacts/milestone1b/pilot-new --blocks 2 --workers 4
.venv/bin/python -m scripts.diagnose_foundation --output artifacts/milestone1b/baseline-new --blocks 50 --workers 4
.venv/bin/python -m scripts.analyze_foundation_economics artifacts/milestone1b/baseline-new --select-long artifacts/milestone1b/jobs-new.json --verify-replays
.venv/bin/python -m scripts.diagnose_foundation --output artifacts/milestone1b/extensions-new --jobs-from artifacts/milestone1b/jobs-new.json --workers 4
.venv/bin/python -m scripts.analyze_foundation_economics artifacts/milestone1b/baseline-new --extended artifacts/milestone1b/extensions-new
.venv/bin/python -m scripts.analyze_foundation_economics artifacts/milestone1b/extensions-new --verify-replays
.venv/bin/pytest tests/foundation --no-cov -q
.venv/bin/ruff check evaluation/economics.py scripts/diagnose_foundation.py scripts/analyze_foundation_economics.py tests/foundation/test_economics.py
```

The initial 48-game pilot took 2.91 seconds with four workers, justifying the bounded expansion.
Final evidence directories are `artifacts/milestone1b/baseline-final/` and
`artifacts/milestone1b/extensions-final/`. Their `records.jsonl` name each representative replay;
look up by `(policies, seed, focal_seat)`, not by a filename guessed from seed order.
Exploratory directories retain earlier observer iterations; use the final directories for cash
flow totals. Engine outcomes and action/event hashes were repeatedly compared across iterations.
Generated artifacts remain ignored by Git; source tools, tests and this report are retained.
Keep/copy the final directories if transferring this investigation to another machine.

Validation measurements and exact evidence hashes are in the accompanying
[1b evidence manifest](economic-evidence.md). This investigation did not rerun the entire
certification tournament or frontend/browser checks: no production/frontend code changed.

## Recommended Milestone 1c

**First repair group-wide mortgage/build validation; then test a narrowly scoped, separately
versioned trading variant with deterministic agents.** Increasing the global horizon, weakening
the completion gate, changing GO/taxes, or tuning construction thresholds is not justified by
these results. Trading is the smallest proposed new decision mechanism that can rearrange the
already-owned split sets. It is a hypothesis for improving completion, not a promise of 95%.
Auctions remain valuable for fuller Monopoly later, but auctions alone cannot repair the observed
frozen ownership endpoints. Do not implement auctions and negotiation simultaneously merely to
hide which intervention changes outcomes.

1. **Correctness prerequisite, separate commit.** Update `monopoly_engine/rules.py` and
   `property.py` so no property in a mortgaged color group can be developed; preserve ordinary
   unimproved monopoly rent. Remove the strict xfail after it passes. Extend engine/mask/API
   regression scenarios and `evaluation.runner.invariant` to cover the entire color group.
   Gate: validation, masks and execution agree; invalid actions preserve RNG/state; zero group
   mortgage/build violations on 100,000 seeded transitions. Rerun paired development games to
   quantify changed outcomes before attributing them to trading. Keep old evidence pinned to
   `02a5e73`; a rules correction does not retroactively validate its results.
2. **Approve and implement a separate trading rules ID.** In `foundation.py`, `state.py` and
   `actions.py`, introduce authoritative offer/response decisions with explicit proposer,
   responder, revision, complete asset/cash bundle and suspended turn continuation. Start with
   one outstanding offer and a bounded number of proposals per turn, legal only on the
   proposer's own asset-management phase without debt. Reject/end-turn must always be available
   appropriately. Acceptance atomically revalidates both balances and ownership; rejection resumes
   the precise interrupted phase. Document group-building sale restrictions and mortgage-transfer
   treatment explicitly. Record proposals, responses and transfers as replayable structured events;
   never mutate inside `choose_action`. No auctions, counteroffer trees or LLMs in this first test.
3. **Deterministic agents and shared consumers.** Add bounded stable candidate enumeration and
   a separately named offer/acceptance heuristic under `agents/`, keeping current heuristics as
   no-trade controls. Begin with simple mutually set-completing swaps and explicit cash limits,
   using public state only. Feed response ownership through `monopoly_gym`'s shared adapter and
   API manager, expose legal proposal/response choices in API/frontend, and version any changed
   action/observation encoding. Do not reuse the historical hybrid side-effect path or merge old
   trading branches wholesale. Gate: accept/reject/invalid/stale/debt/terminal/reconnection/replay
   scenarios and API/AEC/direct traces agree; budgets cannot create offer loops.
4. **Measure intervention before claiming readiness.** Pair corrected no-trade and trading agents
   on fresh development seed blocks, all seats, at the unchanged 1,000-turn horizon. Report
   accepted/rejected offers, ownership fragmentation, time to first development, rent concentration,
   actual winners, all cutoff/error/stall rates, and seed-block uncertainty. Require zero accounting,
   invariant, legality and replay failures plus a measured reduction in split-set cutoffs, rather
   than merely more accepted trades. Only after development validation freeze a new rules/policy
   manifest and fresh certification seeds. The existing ≥95% per-matchup completion gate remains;
   if it still fails, report it and investigate further instead of relabeling wealth or extending
   the production horizon. No substantive model training until a useful task is certified.

Uncertainties: only 50 development seeds per matchup; purposive long-run selection; policy action
changes can alter RNG consumption despite seed pairing; the mortgage defect confounds some
completed outcomes; deterministic trading incentives/acceptance coverage have not been tested.
Other rules defects may remain. These results strongly support the economic diagnosis for this
candidate, not an assurance about full Monopoly, future trading, or model playing strength.
