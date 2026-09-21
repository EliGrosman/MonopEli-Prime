# Milestone 1c: building repair and headless trading experiment

Status: **the headless experiment supports integrating authoritative trading into the shared
environment/API path; foundation readiness remains blocked pending that integration and a new
certification run.** `foundation-v1`, its 1,000-turn horizon, and the 95% per-matchup completion
gate are unchanged. No model training, auctions, browser trading, learned search, or branch cleanup
was performed.

## Building-legality repair

Commit `ad6c632` rejects construction anywhere in a color group containing a mortgaged property.
The basic property helper, authoritative validator, action mask, execution path, and shared
invariant now agree. Redemption re-enables construction, and doubled unimproved monopoly rent is
unchanged.

The repaired engine reran the exact 1b development manifest: 1,200 games, the same seeds, seats,
policies, and 1,000-turn boundary. It executed 1,766,692 transitions with zero defects, errors,
stalls, actor failures, or mask/execution mismatches. All matchup outcome counts were unchanged:

| Focal / opponents | Two-player completed | Four-player completed |
|---|---:|---:|
| Rule-based / rule-based | 88/100 | 68/200 |
| Aggressive / rule-based | 86/100 | 68/200 |
| Conservative / rule-based | 91/100 | 68/200 |
| Rule-based / random | 100/100 | 200/200 |

The repair changed 157 action/event traces because previously legal builds were delayed until
redemption; it changed no completion, winner, elimination, or cutoff classification in this paired
sample. These data isolate the later trading effect from the correctness repair. Historical 1b
traces remain evidence for their archived source and are not expected to execute identically on
the corrected engine.

## Experiment contract

Commit `f7c7a2c` adds the opt-in `foundation-trade-v1` ruleset and leaves default construction on
`foundation-v1`. The authoritative action path owns proposals, response decisions, revalidation,
atomic settlement, proposal budgets, exact turn resumption, snapshots, structured events, and the
cash ledger. Native action/replay version `native-action-v1` carries parameterized offers without
changing Gym `action-v2`. API, browser, Gym, LLM negotiation, and the historical 907-action encoder
remain disabled or explicitly deferred.

The frozen deterministic wrapper uses public state and the documented price/group-bonus valuation.
It considers one- or two-property bundles, requires at least $25 gain for both parties, caps cash
adjustments at $500, preserves a $200 payer reserve, and will not break a complete group. The engine
does not impose those policy restrictions. See the [rules and policy contract](trading-contract.md).

## Development result

The development namespace used all 50 seed blocks `13000000..13000049`, every focal seat, three
focal styles, rule-based-style opponents, and all three arms: 2,700 games / 4,401,915 transitions.
There were zero errors, stalls, accounting failures, or split-set cutoffs in the mutual arm.
The rejection arm made 705,435 proposals and rejected all of them; after removing proposal/response
bookkeeping, every outcome, turn count, final cash vector, elimination sequence, and gameplay trace
matched its paired baseline.

| Players / focal style | Baseline completed | Rejection completed | Mutual completed |
|---|---:|---:|---:|
| 2p rule-based | 82/100 | 82/100 | 100/100 |
| 2p aggressive | 78/100 | 78/100 | 100/100 |
| 2p conservative | 88/100 | 88/100 | 100/100 |
| 4p rule-based | 68/200 | 68/200 | 200/200 |
| 4p aggressive | 66/200 | 66/200 | 200/200 |
| 4p conservative | 77/200 | 77/200 | 200/200 |

The pooled four-player completion difference was **+64.8 percentage points**, with a paired
seed-block bootstrap 95% interval of **+54.2 to +75.3 points**. The mutual arm proposed and accepted
4,050 offers. Identical proposal/acceptance counts are expected here because proposer and responder
apply the same frozen benefit test; candidate rejections are retained by reason.

## Fresh validation result

Fresh blocks `14000000..14000099` compared the corrected baseline with mutual trading: 3,600 games /
3,389,252 transitions. Every one of the 1,800 trading games finished. The baseline reproduced the
known cutoff pattern.

| Players / focal style | Baseline completion | Mutual completion | Baseline split cutoffs | Mutual split cutoffs |
|---|---:|---:|---:|---:|
| 2p rule-based | 86.0% | 100.0% | 16 | 0 |
| 2p aggressive | 85.5% | 100.0% | 16 | 0 |
| 2p conservative | 89.5% | 100.0% | 10 | 0 |
| 4p rule-based | 34.0% | 100.0% | 184 | 0 |
| 4p aggressive | 34.8% | 100.0% | 184 | 0 |
| 4p conservative | 36.5% | 100.0% | 184 | 0 |

The pooled completion differences were **+13.0 points** for two players (paired 95% CI
**+8.3 to +18.3**) and **+64.9 points** for four players (**+56.2 to +73.1**). Median first complete
group moved from turns 40–44 to 20–21 in two-player games and from turns 54–63 to turn 25 in
four-player games. Median first build moved from turns 44–46 to 26–27 and from turns 62–63 to
28–30, respectively.

The mutual arm proposed and accepted 7,693 offers and transferred $2,257,260 in trade cash.
It recorded $11,918,467 in rent transfers, compared with $22,439,631 over the longer baseline
trajectories; aggregate rent totals are exposure/runtime diagnostics, not strength scores. Median
decision count fell from 950 to 484. Every scheduled game remains in the denominator, and only an
engine winner counts as a win.

No error or stall occurred. All 36 sampled validation replays reproduced 43,336 transitions,
including actions, structured events, final snapshots, RNG state, and ledgers. A separate rerun of
the first two validation blocks reproduced all 72 non-timing records exactly. Development replay
verification separately covered 54 games / 108,182 transitions. The corrected no-trade and trade
runs each exceed the required 100,000 checked transitions by more than an order of magnitude.

## Validation and limits

- Deterministic trade, stale/invalid action, mortgage, developed-group, proposal-budget, turn
  resumption, snapshot, candidate-cache, reserve/benefit, controlled-outcome, and disabled-consumer
  regressions pass. Invalid actions preserve state and RNG.
- Backend: 1,861 passed, 5 skipped, 90 expected deferred, and 4 negative negotiation checks XPASS.
  The latter still pass even though their obsolete positive integration path is marked deferred.
  The full run needs local FastAPI test-client binding outside the filesystem sandbox.
- Strict engine mypy and Ruff on every changed source/test file pass. Repository-wide Ruff still
  reports 178 pre-existing findings in unrelated legacy files, so the global lint gate remains
  unresolved rather than being represented as green.
- Frontend: 345 tests pass; TypeScript check, production build, and Prettier check pass. No frontend
  implementation changed.
- The development/validation experiments use homogeneous deterministic benefit tests. They do not
  establish negotiation quality, robustness against strategic refusal, or playing strength. The
  policy excludes preparatory trades, larger bundles, and three-party exchanges.
- This is not the full certification tournament. The 95% gate remains visible and cannot certify
  shared consumers that do not yet support the ruleset.

## Evidence and reproduction

```sh
.venv/bin/python -m scripts.experiment_trading \
  --output artifacts/milestone1c/development-new --blocks 50 --seed 13000000 --workers 4
.venv/bin/python -m scripts.experiment_trading \
  --output artifacts/milestone1c/validation-new --blocks 100 --seed 14000000 \
  --workers 4 --arms baseline mutual
.venv/bin/python -m scripts.analyze_foundation_economics \
  artifacts/milestone1c/validation-new --verify-replays
```

Generated JSON, JSONL, replays, and source archives are ignored by Git. The frozen directories are
`artifacts/milestone1c/repair-baseline/`, `development/`, `validation/`, and `validation-repeat/`.

| Evidence | Manifest SHA-256 | Records SHA-256 | Source archive SHA-256 |
|---|---|---|---|
| Repair rerun | `c8569616b0c1ea8c143d09338c939a3fb28333a43ad87103345bd7666b4d5b32` | `8037b3407cd247069216979f4fcaeba6f87a54bf3a81dd12c8d96f8577b5a570` | `8ae9c7910af2697af0618f6b9711bea6cfa103bfd05e6b23c9d5a37744012caf` |
| Development | `157854509270328c1022d832a7bd4d4735625da130d15a2c0366691970a2cc2a` | `da39568b7757a85d3a4f4549fbf78b6fa2086507fe18a01e959559c3b2205571` | `d3721cf1cfc6eaee210caa3ebf125413434c654f6197ce3148b7ba16533a8143` |
| Validation | `e6704b8574697629bc66c6c567e6c613d16ae1d96015378e59124e4afe3142bf` | `ddcf6d4a28a98fcffba896780118e913e24fd4fee2736ef17c24c228f5af6a92` | `b8173114fe1e9f3c6aa09bb5a7c55b96765ea5759fe4ba288a4d0fc87f90de13` |

## Recommendation

Proceed to a bounded shared-consumer integration milestone: expose this authoritative trade
decision/response contract through the learner adapter and API, then add the browser controls and
cross-path parity tests before a fresh certification tournament. Keep `foundation-v1` as the
no-trade control and `foundation-trade-v1` opt-in. Do not tune this policy on the validation seeds,
resume training, relax the horizon, or declare foundation readiness from this headless result.
