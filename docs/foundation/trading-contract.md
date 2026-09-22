# foundation-trade-v1 headless contract

`foundation-trade-v1` is an opt-in extension of corrected `foundation-v1`. The default engine,
Gym, API, lobby, and browser rules ID remains `foundation-v1`. Trading games use the same
authoritative engine through the AEC environment, learner adapter, API, bots, and browser.
Auctions, counteroffers, LLM negotiation, and model training remain deferred.

## Authoritative decisions

A proposal is legal only for the turn owner in asset management after all mandatory decisions.
It contains one live recipient, zero to two distinct properties per side, and optionally one
nonnegative integer cash payment in one direction. At least one property must move. The payer
must hold the cash; trading cannot resolve debt. A property is untradeable while any property in
its color group has buildings. Mortgages transfer unchanged; this variant charges no immediate
transfer interest and retains the normal redemption cost.

Each turn permits two proposals and at most one to any opponent. An offer consumes its budget
whether accepted or rejected. Only one offer can be pending. Proposal changes the decision player
to its recipient and enters `trade_response` without changing the turn owner or RNG. Acceptance
or rejection restores the proposer to the exact asset-management continuation. A pending response
is resolved before the tournament runner applies its soft focal-player horizon boundary, making
the proposal/response interaction atomic for cutoff accounting.

Offer IDs increase monotonically. Snapshots preserve the offer, creation revision, next ID,
used targets and interrupted phase. Acceptance revalidates ownership, development and cash before
an atomic transfer. Rejection remains legal if acceptance becomes stale. Invalid decisions leave
state, RNG and revision unchanged. String events remain available for existing traces; structured
events record complete proposal terms and accepted/rejected transfers.

## Deterministic experiment policy

The separately named trading wrappers retain their base rule-based purchase, construction, jail
and liquidation behavior. Immediately before ending a turn, they enumerate stable bundles that
can complete an entirely unmortgaged color group without breaking either participant's existing
complete group. Candidate enumeration uses public state and never mutates the live game.

The frozen valuation is cash at face value; an unmortgaged property at printed purchase price; a
mortgaged property at printed price minus normal redemption cost; and each complete, unmortgaged
color group at an additional twice its total printed price. Both players must gain at least $25.
Cash adjustment is capped at $500 and cannot leave its payer below $200. Offers rank by proposer
gain, recipient gain, then stable opponent/property/cash order. The recipient independently checks
the same constraints. The score is a deterministic candidate heuristic, not a reward or estimate
of winning probability.

The policy cannot express bundles larger than two properties per side, preparatory exchanges,
three-party arrangements, counteroffers or deals that one participant values negatively. A
failure to reach the completion gate therefore evaluates this policy and variant together; it
does not prove that trading in general is ineffective.

## Compatibility and versions

`decision-contract-v1` is the immutable public decision view. It identifies the turn owner and
decision player separately, carries the exact revision, phase-specific legal actions, public
balances and board state, proposal capability, and complete pending terms. It omits deck order,
RNG state, seeds, valuations, and agent internals. Consumer commands carry an expected revision;
stale commands fail before validation or mutation.

Parameterized native decisions and replays remain `native-action-v1`. No-trade learners retain
`action-v2`/`observation-v2`. Trading learners use `action-v3`/`observation-v3`: ordinary indices
0–157, accept/reject at 158–159, and 32 revision-bound proposal slots at 160–191. Candidate rows
contain the complete bundles, mortgage masks, identities, and cash terms. The named
`foundation-benefit-candidates-v1` provider is a learner shortlist, not an engine legality rule.
The obsolete 907-action encoding remains rejected.

Checkpoint sidecars record the rules ID, action and observation versions, player count, candidate
provider, and capacity. A missing or mismatched sidecar is rejected before trading inference;
legacy v2 loading requires an explicit compatibility option. The engine rejects unknown rules IDs
and snapshots always record the selected rules ID.

Generated experiments must report actual engine winners, cutoffs, errors and stalls; never infer
a winner from wealth. The production horizon remains 1,000 completed player turns and the
foundation release gate remains at least 95% completion in every certified matchup.
