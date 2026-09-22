# Milestone 1d: authoritative trading integration

Status: **implementation and certification complete.** `foundation-v1` remains the
default and the separately reported no-trade control. The readiness gate remains 1,000 completed
turns and at least 95% completion in every `foundation-trade-v1` matchup.

## Integrated contract

`decision-contract-v1` exposes detached public state, exact decision ownership and revision,
phase-specific legal decisions, proposal capacity, and complete pending terms. Engine validation
and settlement remain authoritative. The deterministic $25 benefit, $200 reserve, $500 adjustment,
and group-completion rules belong only to `foundation-benefit-candidates-v1` and the named trading
reference policies.

No-trade consumers retain `action-v2`/`observation-v2`. Trading consumers use 192-action
`action-v3` and complete `observation-v3` offer/candidate rows. Native humans and agents may submit
any legal offer outside that 32-slot learner shortlist. Checkpoint sidecars reject rules, shape,
seat-count, provider, or capacity mismatches before inference.

API actions authenticate the actor from the live session, require revision and request IDs for
trading games, reject duplicate/stale requests without mutation, and return refreshed state.
Browser drafts are reviewed before submission, pending state is acknowledgement-based, old state
updates are ignored, and reconnect restores a pending response without automatic acceptance.

## Correctness evidence

- Engine and cross-consumer tests cover response-only enforcement, stale/nonmutating actions,
  exact cash/property/mortgage settlement, proposal budgets, candidate stability, flattened offer
  fidelity, API/AEC/direct parity, snapshot detachment, and checkpoint compatibility.
- Backend after closeout: 1,925 passed, five skipped, 79 strict deferred, zero XPASS.
- Repository Ruff and strict engine mypy pass.
- Frontend: 356 unit tests plus lint, type, format, and production build pass.
- Nine live Chromium scenarios pass, including human/human trade composition, review, reconnect,
  acceptance and resumption, human/bot acceptance, and bot/human refusal. Existing jail, debt,
  and purchase cases remain; trading-enabled debt resolution is also exercised after reload.
- A frozen 48-game smoke run passed direct-engine/AEC parity, saved-replay restoration, and paired
  always-reject/no-trade outcome and action-stream checks. Smoke results are not readiness evidence.

## Certification protocol

The frozen runner uses seed roots 18,000,000 for development and 19,000,000 for certification,
all twelve 2/4-player focal/opponent matchups, every focal seat, named mutual-benefit trading
wrappers, a 1,000-turn soft boundary, and engine winners only. It archives Python/API/frontend
sources and both dependency locks. Every trading transition executes through AEC v3 and is replayed
through a direct engine clone before continuing. The paired no-trade arm is recorded separately;
development also contains one seat-balanced always-reject and mixed-response block per matchup.

The verifier requires zero errors, at least 100,000 checked trading transitions, 95% completion in
each trading matchup, exact captured replay restoration, and exact paired no-trade equivalence for
the always-reject diagnostics. A failed or post-tuning run does not certify readiness.

The shared-consumer gate additionally requires the first 20 complete seed blocks of every matchup
through `SingleAgentMonopolyEnv`, `SelfPlayEnv`, and the real `GameManager`/`AIManager` bot action
path. Engine seeds and each policy's seeds are explicit, preserving the original tournament's
`engine_seed * 11 + player_id` policy schedule. Normal learner resets keep their existing independent
seed streams. The consumer verifier rejects missing seats, missing consumers, duplicate records,
insufficient blocks, outcome/action-hash mismatches, and incorrect learner terminal rewards.
Both wrappers must actually exercise incoming responses, wins, losses, and four-player learner
eliminations before game completion; a run lacking that lifecycle coverage cannot certify them.

## Certification result

The frozen certification run completed all 12,000 scheduled `foundation-trade-v1` games. Every
one of the twelve matchups completed 1,000/1,000 games, with zero errors and zero cutoffs. The
runner checked 8,673,947 authoritative AEC transitions against a direct-engine clone. It recorded
40,834 proposals and 40,834 acceptances under the named mutual-benefit responder. All 72 repeated
games matched their original non-timing records, and every captured replay restored and replayed
to the saved final snapshot. The complete-seed-block bootstrap intervals are 100%–100% for every
matchup.

The separately reported 12,000-game `foundation-v1` arm continues to reproduce the economic
limitation. Completion ranges from 87.8%–100% in two-player matchups and 35.1%–99.9% in
four-player matchups; all three four-player rule-based-opponent matchups remain near 35%–37%.
Those control results do not enter the trading readiness gate.

- Certification manifest SHA-256: `87efb96e965b297302d2d15c11a0a70892106859391c1e34129003bbe9bd796f`.
- Frozen source archive SHA-256: `d1fca2411ee925520b8eb8b9009f9e2c6ec8df735b8265e031788a792036e6f4`.
- Certification records SHA-256: `0feebc8eb6db969d15ee7869567d2c07510e65e9b3d4eb585bccb6fa47cbeb9b`.
- Development manifest SHA-256: `a08aa453c9862939ec6fe5ee0fb79ab1659875ba8f2a1a9b88afbdad71017737`.

```sh
uv run python -m scripts.certify_trading_integration \
  --namespace development --games 200 \
  --output artifacts/milestone1d/development
uv run python -m scripts.certify_trading_integration \
  --namespace certification --games 1000 \
  --output artifacts/milestone1d/certification
uv run python -m scripts.certify_trading_consumers \
  artifacts/milestone1d/certification --blocks 20 --workers 4 \
  --output artifacts/milestone1d/consumer-parity
uv run python -m scripts.verify_trading_integration \
  artifacts/milestone1d/certification \
  --consumer-run artifacts/milestone1d/consumer-parity
```

Use new output directories when reproducing these commands. `--aec-only` explicitly verifies
historical AEC evidence without claiming that learner/API coverage is complete.

## Review closeout

Review found that debt buttons called the socket directly while metadata creation was duplicated
between hooks. A `rulesId`/`rules_id` mismatch caused the server to reject those trading-game
actions. The socket now owns metadata and pending state for every action entry point. Matching
success acknowledgements retain pending state until the resulting revision arrives; unrelated
acknowledgements and state updates cannot release it. Failures retain request correlation,
disconnected decisions are discarded, and reconnect waits for fresh state. Eleven focused socket
regressions and the live trading-debt scenario cover this repair.

The original 12,000-game evidence above covered AEC versus the direct engine. Its source archive
remains historical evidence rather than being relabeled as the final source. The supplement freezes
the closeout sources separately and compares consumer runs to those original records without
tuning policies or changing the rules, horizon, or completion threshold. Learner checks include
observations/masks at every learner boundary and terminal reward delivery exactly once. A learner
stops on its own elimination; its surviving opponents continue through its underlying AEC instance
solely to compare the complete game hash. The learner is never stepped after termination. API
parity uses the real bot action loop one decision at a time, with background scheduling and delays
disabled; the separate live browser scenarios cover WebSocket transport and human interaction.

The completed supplement matched **2,160/2,160 consumer games**, covering **1,713,657 checked
transitions** with zero mismatches. Each consumer reproduced 720 original games and 571,219
transitions across the first 20 complete seed blocks of all twelve matchups and every focal seat.
Each learner wrapper exercised 109,715 learner boundaries, 1,214 incoming trade responses,
473 wins, 247 losses, and 122 four-player eliminations before the game ended. Its terminal rewards,
full action hashes, winners, cash, trade counts, turns and elimination sequences matched the
reference records. The final combined verifier passes.

Supplement evidence: `artifacts/milestone1d/consumer-parity/`.

- Manifest SHA-256: `957bcf97b2d7466c4bfe25ca981e3219ffb51f48dae81bcf488bae8201a7fa59`.
- Records SHA-256: `ddf80797e171eac2ee84ce3dfea1559bbae8046744ddb071ae7ada17f1daf789`.
- Source archive SHA-256: `587874eeb9fb8b1e48b2b1e7235a61a4cbc607663f9123e584e99cb7a9062870`.

After the supplement source freeze, only the offline verifier and its tests were strengthened to
require positive response/win/loss/early-elimination coverage; documentation was then updated with
the measured results. Game, policy, adapter, API and browser runtime sources remained unchanged.

## Claim boundary

Passing evidence supports correct integrated trading and completion under the named deterministic
reference policies. It does not establish negotiation strength, learned-policy quality, arbitrary
refusal robustness, or readiness of `foundation-v1`. Jev calls, new training, auctions,
counteroffers, inherited-mortgage interest, and learned MCTS remain deferred.
