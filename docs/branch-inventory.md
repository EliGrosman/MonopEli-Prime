# Branch decision log — foundation milestone

Audited against main b9d8f95 on 2026-09-21. Full heads, merge bases, unique commit lists,
file lists and `git cherry` results are in [branch-manifest](foundation/branch-manifest.json).
Only main existed locally; live remote heads matched the five tracking refs.

| Remote branch | Decision | Purpose, evidence and remaining work |
| --- | --- | --- |
| codex/foundation-v1 | Trading integration and review closeout; submit through normal PR review | Contains authoritative trading, browser request fixes, and the learner/API certification supplement. Preserve the documented evidence and retain the branch after merge as the foundation recovery line. Merge requires green remote CI and normal review. No merge or deletion is performed by this closeout. |
| feature/mcts_engine | Redundant ref; deletion pending release validation | Fully merged by a18305a (PR #7), zero unique commits. Search/network/data/tests live on main. Learned heads are disconnected, chance sampling biased. Later search milestone owns these defects. |
| feature/llm_trading | Redundant ref; deletion pending release validation | Fully merged by de9030d (PR #8), zero unique commits. Clients, negotiation, verifier and hybrid on main. Later guided-agent/trading milestone must replace side effects with events and complete browser trading. |
| feature/mcts_updates | Mixed; retain | 14 unique commits directly after main. Heuristic/candidate trading and budget improvements useful later. Keep 994c401, 79d9694, f340fe9, 2bc5b5a, ce7d792 and associated tests. Its evaluator rolls on every action and still mishandles bankruptcy; supersede those patches with foundation runner. Comparison/export concepts reimplemented, not cherry-picked. No unique model/data artifact in delta. |
| feature/synthetic_data_generation | Retain for imitation milestone | Two unique commits; 44 main commits absent. Generator, BC scripts, HDF5 integration, policy checkpoint and records survive. Connected policy-only training, untrained critic; row-level split leakage, fabricated wealth winners, dead-agent workarounds. Port standalone modules later, regenerate labels/data, split entire games, test critic warm-up. Do not merge old env/lock files. |
| feature/training_phase2 | Retain for training experiments | Six unique commits; 37 main commits absent. Connected rank reward, normalization, value-LR and PPG options. No unique training artifacts found. Validate masked distributions, callback ordering, LR scheduling and optimizer/normalization resume before selective reuse. Rank shaping excluded from certified objective. |

No patch-equivalent unique commits were found on main for the three retained branches.
Old optimization, experiment/trading, misc API fixes, frontend, FastAPI and RL training
branches survive in merged history, not live refs. Do not recreate them automatically.

## Recovery and cleanup

Archive tags: `archive/milestone1/{mcts-engine,llm-trading,mcts-updates,synthetic-data,training-phase2}`.
A complete bundle was created and verified at `artifacts/recovery/milestone1.bundle`.
Its hash and the checkpoint/log/LFS inventory are in [artifact-manifest](foundation/artifact-manifest.json).
The bundle is a local recovery artifact, excluded from source commits. Copy it to durable
backup before discarding this checkout. Restore with `git fetch <bundle> <archive-tag>`
and create a branch from that tag. Tags also retain the objects in this checkout.

LFS pointers do NOT include the missing 657,602,363-byte train or 83,762,111-byte validation
payload. The 1,591,161-byte BC ZIP, 3,874-byte event file and text records are actual Git blobs.
Their outcomes are historical evidence, not validated strength measurements.

Only deletion candidates: remote `refs/heads/feature/mcts_engine`,
`refs/heads/feature/llm_trading`, and their matching `refs/remotes/origin/...` tracking refs.
No local feature heads exist. Re-read live heads before deleting; changed heads must be retained.
No unique work would be lost at the audited heads. Cleanup remains gated on recovery verification
and validation; retain all refs if the release gates fail.

Normal merge new validated foundation work. Selectively port old synthetic/training work and mixed
MCTS updates afterward; whole merges risk restoring obsolete wrappers and dependency state.
Record each later integration's source commit, destination commit, tests and superseded pieces here.

Final cleanup outcome: all five remote feature refs remain intact. The user chose to keep the
approved rules and blocked completion gate; no deletion was attempted. The recovery bundle was
verified again after implementation. See [validation report](foundation/validation-report.md).
