# Project review and agent plans

Reviewed September 21, 2026, at `main` commit `b9d8f95`. This is an architecture review and proposal, not an implementation change. Remote-tracking branches were inspected locally without switching branches or fetching new history.

## What exists

| Layer | Main entry points | Responsibility and current limits |
| --- | --- | --- |
| Rules and state | `monopoly_engine/game.py`, `state.py`, `actions.py`, `rules.py` | Python game state, board, cards, rent, ownership, construction, mortgages, trades, bankruptcy and serialization. Shared by gameplay and training. Rules need an audit before calling this complete standard Monopoly. |
| Learning environments | `monopoly_gym/env.py`, `single_agent_env.py`, `observation.py`, `action_space.py` | PettingZoo AEC environment, Gymnasium wrapper, numeric observations, legal-action masks. Ordinary gameplay has 149 action indices; the optional trade environment has 907, with simple one-property-for-one-property trades. |
| Agents | `agents/base.py`, `rule_based.py`, `mcts_agent.py`, `hybrid_agent.py` | `choose_action(observation, action_mask, game) -> int`; random, rule-based, aggressive, conservative, search and hybrid agents. The hybrid performs trades as side effects before returning a normal gameplay action. |
| PPO training | `training/train.py`, `pettingzoo_selfplay.py`, `curriculum.py` | MaskablePPO, configurable opponents, curriculum, self-play support, evaluation and checkpointing. There are multiple wrapper/reward implementations, with different defects and behavior. |
| Search and learned evaluation | `mcts/search.py`, `network.py`, `data.py`, `training.py` | Tree search, a small PyTorch policy/value network, replay buffer, data generation and an intended AlphaZero-style loop. Critical connections from the network back into search are unfinished. |
| Hosted-model trading | `mcts/llm/`, `negotiation.py`, `trade_utils.py`, `trade_verifier.py` | State summaries, proposal generation, accept/reject/counteroffer logic, budgets, caching, candidate trades and simulation-based verification. Providers currently include Claude, OpenAI and Ollama. |
| Human gameplay | `api/`, `frontend/src/` | FastAPI, sessions, lobbies, WebSocket updates, React/TypeScript board, player panels, dice and property/building controls. |

The server uses engine actions directly. The RL path adds observation encoding and action-index translation around the same engine. Jev belongs in an agent/controller layer; it does not require rebuilding the board or rules.

### Human interface

The browser UI has home, lobby and game routes, including `/lobby/:lobbyId` and `/game/:gameId`. It supports human slots and the four baseline AI types. The server registry in `api/services/ai_manager.py:25` does **not** currently register trained PPO, MCTS, hybrid or Jev opponents.

Trading is not complete through the human interface. Frontend trade hooks exist but have no consuming trade panel, and `api/models/action.py:37` labels trade support as future work; its action converter has no propose/accept/reject cases. Engine and experimental agent trading should not be mistaken for complete browser trading.

Documented launch paths, not exercised during this review:

```bash
# From the repository root, if .env does not already exist:
cp .env.example .env
make up
# Browser: http://localhost:3000
```

Or run `uv sync --dev`, `uv run uvicorn api.main:app --reload`, and, in a second terminal in `frontend/`, `npm ci` followed by `npm run dev`. Vite's default local address is port 5173.

### Rules that affect learning

There are auction enum values but no implemented auction action/flow in the inspected engine/API. The rent handler immediately bankrupts a player who cannot pay from cash, without offering a debt-resolution phase to mortgage assets or sell buildings. `RollDice.validate` and `EndTurn.validate` do not enforce a complete turn-phase state machine. The RL environment automatically rolls at turn boundaries while the browser exposes rolling explicitly.

These differences matter: a policy can optimize cash reserves or ending turns for this simulator and behave differently under standard Monopoly rules. Standard rules include auctions after declining a purchase and raising money before bankruptcy. Compare against the [Hasbro rules](https://www.hasbro.com/common/instruct/Monopoly.pdf) while defining one explicit ruleset for both training and human games.

## Recovered history

No logs, results or model directories are present in the working checkout. `.gitignore` excludes most training artifacts. However, locally available remote-tracking branches preserve more work:

| Branch | Recovered work |
| --- | --- |
| `origin/feature/synthetic_data_generation` | Expert-data generation, behavioral cloning, a saved SB3 policy archive, a TensorBoard event file, dataset statistics and a proposed fine-tuning command. |
| `origin/feature/training_phase2` | Rank rewards, observation/reward normalization, separate value-function learning-rate work and a PPG auxiliary training phase. These are not all present on main. |
| `origin/feature/mcts_updates` | Later trading heuristics, multiple-choice trade proposals, evaluation fixes and result-comparison scripts. It still contains the zero-returning value-network placeholder. |

The synthetic-data report records 10,000 rule-based four-player games, 8,090,230 decisions and no trades. Seat wins were 2,858 / 2,563 / 2,381 / 2,198, illustrating why seat-balanced evaluation matters.

The recovered cloning log has ten epochs:

| Metric | First epoch | Last epoch |
| --- | ---: | ---: |
| Training loss | 0.033770 | 0.006053 |
| Validation loss | 0.020547 | 0.009950 |
| Training action accuracy | 98.70% | 99.76% |
| Validation action accuracy | 99.09% | 99.62% |

This is evidence that a network learned to reproduce recorded actions. It is not evidence of winning games. The generator shuffles individual experiences into training and validation, allowing decisions from the same game into both sets. Accuracy also needs to be separated by action type and by whether there was a meaningful choice. No per-action frequency report was recovered.

The checkpoint metadata describes separate `[256, 256]` policy and value MLPs. Its zero SB3 timestep counter is consistent with supervised training outside the SB3 rollout loop; it does not mean the checkpoint was never trained. The cloning implementation trains the policy while leaving the value network untrained, so the transition into PPO deserves an explicit critic warm-up experiment.

The large HDF5 datasets are Git LFS pointers in the local history; this review did not retrieve their contents. The archive and small TensorBoard file themselves were readable from Git. No corresponding PPO/PPG learning curves or trustworthy end-to-end tournament results were recovered. The README's 94% versus random and 22% versus rule-based claims remain historical claims without a recovered evaluation protocol.

Read the surviving records without checking out another branch:

```bash
git show origin/feature/synthetic_data_generation:data/expert/test/stats.json
git show origin/feature/synthetic_data_generation:models/bc/a
git log --oneline main..origin/feature/training_phase2
git log --oneline main..origin/feature/mcts_updates
```

## Confirmed problems before further training

These findings describe the inspected code, not a proven explanation of every historical run.

1. **The MCTS network is disconnected.** `mcts/search.py:427` returns zero for every player's value even when a network is supplied. Both expansion calls (`:496`, `:516`) omit learned policy priors. The `network` rollout policy also falls back to random actions. A spy network received zero prediction calls through both simulation and a small search. Thus improving its weights cannot improve this search through the intended policy/value connections. An existing test explicitly expects zero values from the placeholder.

2. **The older single-agent wrapper can return the wrong player's reward.** `monopoly_gym/single_agent_env.py:159` calls `last()` after stepping, when control can already belong to an opponent, and reads reward before rolling through opponents. An isolated transition with player rewards 0.1 and 0.3 returned 0.3 for the learner. This is distinct from `SelfPlayEnv`, which calculates its own reward.

3. **AEC cumulative rewards are not cleared for normal actions.** In `monopoly_gym/env.py`, consecutive newly earned rewards of 0.1 and 0.05 produced `last()` rewards of 0.1 and 0.15. The standard AEC contract resets the acting agent's cumulative reward before accumulating new transition rewards. See [PettingZoo environment creation](https://pettingzoo.farama.org/main/content/environment_creation/). Fix this together with wrapper boundaries and terminal reward delivery.

4. **Truncations can be recorded as wins.** `training/pettingzoo_selfplay.py:245` uses a helper that combines termination and truncation to set `terminated`; `:376` assigns relative-net-worth reward when there is no winner; `:832` counts positive reward as a win. A controlled turn-limit episode returned both flags true and positive reward while `game_over` was false and `winner` was null. The evaluator also uses shorter caps than training. Read the actual winner, separate cutoffs, and implement bootstrapping according to the chosen horizon semantics.

5. **Observation caches can be stale.** In `monopoly_gym/observation.py:624`, repeated observations for the same player skip opponent updates even when opponent money changed. A controlled change from $1,500 to $1,300 still encoded $1,500. At `:643`, one shared game snapshot is compared against per-player cached observations: after another player updates that snapshot, a player can retain the previous turn number. Correctness checks must cover actual state changes, not just which player was most recently observed.

6. **The older wrapper restarts the configured seed on every unseeded reset.** `monopoly_gym/single_agent_env.py:143` falls back to its stored seed each time; vectorized training supplies fixed seeds per worker. It therefore repeats each worker's initial shuffle and RNG sequence rather than producing a fresh episode stream. This finding does not apply identically to `SelfPlayEnv`, whose reset implementation differs.

Additional search issues from inspection: each action child stores one sampled post-action state, including chance outcomes; revisiting it does not resample that outcome. Seeded root games do not make all search randomness reproducible. Data generation records only player 0 against rule-based opponents, despite the self-play naming. A trained policy head therefore needs both actual integration and a genuine opponent-population loop. Once values are connected, explicitly define player perspective and output-seat mapping; current training examples are all player-0 observations.

Validation: 178 existing environment, incremental-observation and reward tests passed in a temporary environment with Gymnasium 1.3.0 and PettingZoo 1.27.0. These were not a full project/CI run with the locked environment. Seven small diagnostic probes reproduced the behaviors above. No browser session, full training run, hosted-model call or fresh win-rate benchmark was performed.

## Track A: a guided Jev agent

Assumption: Jev means TypeSafe AI's model. The current official interface accepts a state and typed Choice, Score and Noul questions. It returns decisions/probabilities rather than generated explanations; questions in one request are evaluated independently. This matches a guided decision controller well. See [TypeSafe's introduction](https://docs.typesafe.ai/introduction).

Use an explicit controller around the model:

1. Construct a player-visible `DecisionContext`: phase, cash and liabilities, positions, property ownership/buildings, bank supplies, offers, recent public events, and persistent strategy selections. Compute prices, legal moves, set completion, rent exposure and before/after trade effects in Python. Do not expose hidden deck order or RNG state from full engine serialization.
2. Generate legal candidate actions and trade bundles in code, reusing `suggest_valuable_trades` and `trade_impact`. Include `no_trade` and `end_turn` only where legal. Candidate generation determines which strategies are available, so measure its coverage and add diversity rather than relying exclusively on existing heuristics.
3. Ask a small set of focused questions. Use the answer to select a legal candidate or choose which deeper calculation to run.
4. Persist selected short- and long-term objectives in controller memory; refresh them on meaningful state changes. They are conditional preferences, not permanent constraints that prevent adaptation.
5. Execute one action through the authoritative engine/service path, then rebuild context. Route trade proposals/responses through explicit game events instead of mutating a live game inside `choose_action`.

Example question contracts:

| Situation | Question | Output |
| --- | --- | --- |
| Strategy checkpoint | Which objective has the best prospects now: preserve liquidity, complete a specific available set, develop an owned set, or block a rival? | Choice over concrete achievable objectives |
| Optional trade | Is there a worthwhile trade opportunity among these candidates? | Noul or Choice including no trade |
| Offer selection | Which full partner/property/cash bundle best improves our position after accounting for the partner's improvement? | Choice over candidate IDs |
| Incoming offer | Accept, reject, or select one of these legal counteroffers? | Choice |
| Development | Which affordable building plan best balances rent gains and remaining cash? | Choice |
| End of turn | Which legal action is worth taking before ending the turn? | Choice including end turn |

Avoid independently choosing a partner, a property and an amount and hoping the answers form a coherent offer. Give each option a complete bundle. When action selection depends on a newly chosen goal, use a subsequent request containing that goal, or evaluate complete conditional options; independent parallel questions do not consume each other's answers. Choice currently accepts up to 255 options. See [Choice documentation](https://docs.typesafe.ai/primitives/choice).

Implementation seams: add a typed decision client beside the text-completion client, a controller/agent, common decision logging, and registration in API/backend/frontend AI selections. Reuse public-state summaries and trade math. Run inference outside the server event loop and validate the game revision and action again before committing a response. Add timeouts, a bounded decision budget and a legal fallback.

First experiment: fixed scenario suite plus headless games, comparing (a) existing rule-based decisions, (b) one flat Jev action choice, and (c) the guided controller. Add search verification as a separate ablation once the search defects are fixed. Measure actual wins, decision latency, token cost, invalid/stale choices, trade acceptance and post-trade outcomes. Jev's confidence is not a measured probability of winning Monopoly.

Pin the tested model version. TypeSafe currently lists `jev-1.13.0` and states that customer fine-tuning/LoRA is not available; request design guides its answers without updating per-account weights. Calling it repeatedly will not train a persistent Monopoly policy for this project. See [model reference](https://docs.typesafe.ai/models).

## Track B: train our own model without Jev

RL is appropriate for decisions with consequences over many turns. Neural networks are the function approximators; RL, imitation learning and supervised value prediction are training methods that can be combined. A compact policy trained from game data satisfies the original goal without language-model pretraining or a hosted dependency.

Recommended sequence:

1. **Repair and define the task.** One ruleset, one transition contract, correct observations/rewards/masks, stable terminal semantics, and reproducible evaluation. Start with two-player no-trade experiments for debugging, then validate the actual four-player/trading target. Track these as different benchmarks.
2. **Recover a credible imitation baseline.** Reuse the old cloning approach, but regenerate or validate data against corrected rules/encoding. Split whole games/seeds, balance rare consequential decisions, exclude forced moves from headline decision accuracy, and evaluate closed-loop games. Ask an expert to relabel states the learner visits (dataset aggregation) so small mistakes do not leave it in unfamiliar states indefinitely. Pretrain value estimates on game outcomes before PPO fine-tuning; retain the original policy as a comparison and use a conservative update schedule.
3. **Establish two small competing learning paths.** A repaired MaskablePPO baseline and a simulator-assisted candidate/value learner. For the latter, sample reachable situations, apply each candidate, simulate many independent continuations against an opponent mixture, and learn action rankings/values. Keep sampling uncertainty and the assumed continuation policies explicit; rollout estimates are not true optimal values. Improve the teacher with search and iterate.
4. **Use structured state and actions.** Begin with a small MLP over reliable features. If necessary, compare a shared property encoder and player encoder with pooling/attention, preserving board position, color groups and turn order. Score `(state, candidate action)` rather than creating an enormous fixed output index for every possible trade. Values should estimate per-player outcomes under an explicitly defined seat mapping. Try recurrence only if public history/opponent behavior adds measurable value.
5. **Introduce a league.** Sample frozen past checkpoints and several heuristic styles, rotate the learning seat, and evaluate on held-out seeds. Add trade proposals, trade responses and then richer bundles through a curriculum. Avoid a policy that wins only against its current training opponent.
6. **Distill if runtime search is too expensive.** Train a compact policy on stronger search decisions. It can play directly using its own weights, with optional search only for difficult decisions. [Policy distillation](https://arxiv.org/abs/1511.06295) is an established way to transfer a stronger teacher's behavior into a smaller policy.

For dense rewards, treat net worth and set completion as shaping signals rather than the objective. Prefer bounded potential-based differences, `gamma * Phi(next_state) - Phi(state)`, with consistent terminal handling, and compare against terminal-only rewards. Check that repeated optional actions cannot harvest rewards. This form has a theoretical basis in [potential-based shaping for stochastic games](https://arxiv.org/abs/1401.3907); it is not a guarantee of practical convergence. Test the discount horizon: at gamma 0.99, an outcome 500 learner decisions away has a direct discounted weight of about 0.0066.

If retaining MCTS, connect both heads, test that changing weights changes search, represent or resample chance outcomes correctly, seed all randomness, and train with representative opponents. Do not copy two-player deterministic-game assumptions into four-player Monopoly. We already possess a simulator, so learning the game dynamics with MuZero/Dreamer adds work before there is a demonstrated need. [MuZero](https://arxiv.org/abs/1911.08265) learns a dynamics model; the useful part to borrow first is planning plus learned policy/value estimates.

Other viable experiments: gradient-boosted models to predict candidate quality or trade acceptance from simulation data; evolutionary/Bayesian optimization of heuristic thresholds as an inexpensive performance baseline. Neither supplies full sequential strategic learning by itself. Offline RL is possible, but online simulation is available and avoids making a fixed narrow dataset the only source of experience.

### What would count as progress

Use isolated scenario competence, closed-loop win rate and confidence intervals, not PPO loss or average imitation accuracy. Initial smoke comparisons can use roughly 200 games per matchup; serious comparisons should use 1,000+ held-out games, all seats, and several independent training seeds. Game counts are starting budgets, not guarantees of statistical significance. Compare candidates under both fixed simulation and wall-clock budgets when search is involved.

Log actual winner, elimination order, cutoff rate, reason for termination, strategy decisions, trades, cash/rent outcomes, checkpoint, rules version, encoding version, opponent identities and seeds. Evaluate direct policy inference separately from policy-plus-search. Use paired seed/seat assignments, recognizing that different actions can change subsequent RNG consumption. Report uncertainty over both games and training runs; see [reliable deep-RL evaluation](https://arxiv.org/abs/2108.13264).

In a symmetric, seat-balanced four-player matchup, equal strength implies roughly 25% wins. For a two-player matchup it is 50%. Compare against a rule-based agent occupying the same evaluation role, rather than imposing an arbitrary 50% four-player threshold.

## Proposed first milestones

1. Correct environment, rules and evaluation defects; preserve diagnostic cases as regression tests. Deliver a reproducible baseline tournament and recover relevant branch work selectively.
2. Define shared player-visible contexts, legal candidates and decision logs. These serve both tracks.
3. Build and benchmark a headless guided Jev controller, then register it and complete trading in the human UI.
4. Reestablish imitation baseline on corrected data; compare repaired PPO with simulator-trained candidate/value learning. Scale only after scenario results and held-out gameplay show real improvements.

The strongest current explanation is an incomplete and inconsistent training pipeline, with direct evidence that some learned weights were not connected to decisions. Historical runs still need their exact branch, configuration and checkpoint matched before assigning a single cause. There is sufficient reusable infrastructure to investigate both agent tracks without a rewrite.
