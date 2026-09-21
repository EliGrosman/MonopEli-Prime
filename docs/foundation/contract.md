# foundation-v1

This is a deliberately limited US Monopoly variant, not full standard Monopoly.
Auctions and trading are disabled; rejected purchases and bank-bankruptcy assets remain unowned.
Asset management occurs on one's turn or during one's debt decision. Inherited mortgages remain
mortgaged; immediate inheritance interest is deferred until normal redemption. These omissions
reduce price discovery and set completion, increase landing luck, and can substantially lengthen
four-player games. Results do not measure trading strength.

## Engine

`MonopolyGame.apply_action(actor, action)` validates then executes a single decision and all
mandatory effects, returning events, decision player, phase, revision, eliminations and winner.
A debt can give decision control to someone other than the turn owner. Cash shortfalls pause for
building sales/mortgages; full color-group liquidation works despite hotel downgrade shortages.
Insolvency uses attainable liquidation value. Payments to multiple players use seat order.
Bankruptcy returns buildings once and preserves creditor mortgages. Last survivor is the winner.

Phases: pre_roll, jail_decision, purchase_decision, asset_management, debt_resolution, terminal.
Rolls are explicit. Purchase refusal is independent of turn ending. Doubles require another roll;
third doubles send to jail. Jail release doubles do not grant a second roll. Third failed jail
attempt queues the fine and then the rolled movement. Pending debts precede optional actions.
Nearest-railroad/utility cards preserve special rent, with fresh utility dice.

Snapshots include phase, obligation continuation, all RNG states, deck contents and jail-card
provenance. API and learning observations exclude secret deck/RNG information. Legacy snapshots
without those fields cannot promise replay equivalence.

## Learning

AEC steps reset the acting player's cumulative reward, deliver all players' transition rewards,
and expose dead agents for one final last()/step(None). Sparse reward is +1 once to the winner,
-1 once on elimination, zero otherwise. Dense/rank/trade modes reject explicitly.
SingleAgentMonopolyEnv and SelfPlayEnv share one learner adapter with configurable seat.
Each step executes one learner decision and opponents until the next learner decision or death.
Opponent-turn debt decisions interrupt that interval. Repeated calls after done reject.
Pre-decision elimination at reset is exposed as pending_terminal; first step delivers it once
without executing its action.

The 1,000 completed-turn horizon is external, with doubles part of one turn. Stop at the next
focal decision boundary (or next survivor decision after focal elimination); record overshoot.
Termination takes precedence; ordinary cutoff is truncated only and has a real final observation
for bootstrapping. Discount per returned learner decision. Errors/stalls invalidate trajectories.

Action-v2 preserves indices 0-148 except index 1 is true PassBuy; index 149 is RollDice;
150-157 sell entire groups in board order (1,6,11,16,21,26,31,37). Legacy trade policies reject.
Observation-v2 includes 13 decision features: six phase indicators, turn owner/7, actor/7,
roll owed, doubles/3, debt/10000 (capped), (creditor+1)/8 or zero for bank,
and purchase (position+1)/40 or zero. Caching is disabled. Checkpoint size mismatches reject.

Explicit reset seeds restart streams; unseeded resets advance. Engine, decks, policy opponents
and workers require separately derived seeds. Terminal rewards are never evaluation results.

## Evaluation and release

`python scripts/benchmark_foundation.py --games 200 --output artifacts/dev-run`
then `--games 1000 --namespace certification --output artifacts/cert-run`.
The manifest is frozen before execution. Twelve matchups cover two/four players, three heuristics,
and random/rule-based opponents; rule-based self-matches are symmetric controls. Rotate seats.
Bootstrap entire seed blocks (2,000 draws, seed 12345); primary win rate includes all scheduled games.
Completed-only rates are secondary. Errors, stalls, cutoffs and eliminations are separate fields.
Stall: >1000 decisions per turn or 20 repetitions of semantic state. Never fabricate winners.

Release requires exact scenario/replay/API parity, 100,000 clean legal transitions, no benchmark
errors or stalls, >=95% completion in EACH matchup, browser debt/jail flows, locked full CI,
and verified branch/artifact recovery. A failing completion gate must remain visible. Changes
in horizon/rules require development diagnosis and a new protocol/version before certification.
No substantive model training, search certification or branch retirement precedes these gates.

## Diagnostic follow-up (Milestone 1b)

[The economic diagnosis](economic-diagnosis.md) preserves this ruleset and its production horizon.
The ordinary mortgage rules require clearing every mortgage in a color group before building;
`02a5e73` misses this check for sibling properties. A strict regression reproduces the defect;
repair is a Milestone 1c prerequisite, not a newly approved variant. Readiness stays blocked.
