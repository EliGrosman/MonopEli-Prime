# Guided Jev agent contract

`guided-jev-v1` is a native asynchronous policy for `foundation-trade-v1`. It does not change
`decision-contract-v1`, the engine rules, or the Gym action and observation contracts.

## State boundary

The policy accepts a detached `DecisionView` and memory owned by the same game, seat, and agent
generation. Provider state contains only the decision contract, public board and player facts,
static rules facts, seat labels, and that agent's bounded strategy memory. Engine snapshots, random
state, deck order, session data, player supplied names, and other agents' memory are excluded.

## Provider boundary

`DecisionProvider.evaluate` accepts a `QuestionBatch` and returns strictly validated typed answers.
The production adapter calls `POST https://api.typesafe.ai/v1/systemone` with Bearer authentication.
It supports documented Choice, Score, and Noul answers; the guided policy currently uses Choice.
Tests and offline evaluation inject `FakeProvider`, so normal validation never calls the network.

The server reads `TYPESAFE_API_KEY` as an excluded `SecretStr`. It never sends the credential to the
browser or writes it to action records. The provider adapter exposes sanitized error codes instead
of response bodies.

## Decision and commit rules

One question selects an exact legal ordinary command for the captured revision. Strategy objective
and reserve questions share one request because they are independent. Dependent trade questions are
serialized because TypeSafe evaluates batched questions independently against the same state. Every
step receives a detached snapshot of the selected recipient, property bundles, cash direction,
active cash range, and known cash amounts. Construction ends with exact integer cash selection and a
complete-offer confirmation. A selected command is validated and applied by the engine under the
game lock only if game, agent generation, decision owner, revision, and request identity remain
current. Strategy updates commit in the same critical section. Stale answers and staged memory are
discarded.

Sole executable decisions are recorded as `forced`. Provider selected decisions are `jev`.
Timeouts, invalid answers, exhausted budgets, and local progress guards use explicit `fallback`
records. Incoming trades fall back to rejection; other phases use the existing public rule based
policy.

## Memory and inspection

Memory holds one objective pair, target group, soft cash reserve, refresh markers, and the eight most
recent relevant public trade outcomes. It lives for the active game and resets when the agent is
replaced or the game is deleted. The browser receives only objectives, reserve, status, a short
action summary, and a sanitized fallback reason. No chain of thought is requested or displayed.

## Budgets and evaluation

Every admitted attempt reserves the documented 64k request ceiling before entering the shared
concurrency gate, including attempts that wait in the queue. The decision deadline covers that queue
wait, inference, and retry backoff. A request that expires or is cancelled before dispatch releases
its reservation. Trustworthy usage from an HTTP response replaces the reservation even when its
answer is rejected. Once dispatch begins, missing or malformed usage, timeouts, and cancellation
retain the reservation conservatively. Settlement updates the per-game, session, and process scopes
together exactly once. Default per-game caps are 2,000 attempts, five million input tokens, and
$0.25 exposure. The process caps are 20,000 attempts, 50 million input tokens, and $3 exposure.

## Probability validation and diagnostics

Choice and Score probability maps must contain exactly the requested keys. Every value must be a
finite number from zero through one. Totals are computed with `math.fsum` and accepted only when the
absolute deviation from one is at most `0.0001`, regardless of option count. Accepted values are not
normalized. TypeSafe's [Choice](https://docs.typesafe.ai/primitives/choice) and
[Score](https://docs.typesafe.ai/primitives/score) documentation says totals equal one, while its
[OpenAPI schema](https://api.typesafe.ai/openapi.json) describes them as summing approximately to
one; none specifies a numerical tolerance. The bounded tolerance above is therefore a local
validation policy rather than a provider guarantee.

A rejected total logs only the question type, option count, computed sum, absolute deviation, and
tolerance. Provider bodies, question contents, game state, option names, and credentials are not
logged or returned to the browser. No available local log captured the probability sum from the
historical `malformed_probability_sum` fallback, so rounding drift is not established as its cause.
The new diagnostics are intended to distinguish small drift from materially invalid future results.

`scripts.evaluate_jev --provider fake` exercises the same AI and authoritative execution services,
stores exact executed commands separately from provider inputs, and makes no external calls.
`scripts.verify_jev_run` restores each private initial snapshot and replays those commands. Replay
proves engine execution, not reproduction of external model answers.
