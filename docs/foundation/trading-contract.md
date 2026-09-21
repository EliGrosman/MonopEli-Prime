# foundation-trade-v1 headless contract

`foundation-trade-v1` is an opt-in experimental extension of corrected `foundation-v1`.
It exists to test whether mutually beneficial property trading reduces persistent split ownership.
The default engine rules ID remains `foundation-v1`; Gym/AEC, API and browser games continue to
reject trading. Auctions, counteroffers, LLM negotiation and training encodings remain deferred.

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

Parameterized trades use `native-action-v1` in headless traces. Existing ordinary decisions
retain `action-v2`; `observation-v2` and `terminal-v1` are unchanged. The obsolete 907-action
experimental encoding is not revived. Checkpoints and learning environments remain no-trade.
The engine rejects unknown rules IDs and snapshots always record the selected rules ID.

Generated experiments must report actual engine winners, cutoffs, errors and stalls; never infer
a winner from wealth. The production horizon remains 1,000 completed player turns and the
foundation release gate remains at least 95% completion in every certified matchup.
