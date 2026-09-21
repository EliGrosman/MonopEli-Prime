# Milestone 1b evidence manifest

Candidate: `02a5e73912a25e7cfc0bf8c1fe44329c380dd4a3` on `codex/foundation-v1`.

Rules, agents and production runner are unchanged. Diagnostic source files were untracked during
execution; each frozen manifest records that dirty state and the complete source hashes. Its
`source.tar.gz` preserves the actual source independently of the base commit. All archived source
hashes were checked against the delivered Python files. See [the diagnosis](economic-diagnosis.md).

## Final runs

| Run | Games | Cash-reconciled decisions | Elapsed seconds (4 workers) | Saved replays |
|---|---:|---:|---:|---:|
| `baseline-final` | 1200 | 1,754,140 | 61.17 | 24 |
| `extensions-final` | 48 | 935,874 | 43.18 | 48 |

Root: `artifacts/milestone1b/`. These two runs overlapped in wall time; timings are not
performance benchmarks. Initial bounded pilot: 48 games, 83,724 transitions, 2.91 seconds.

## File checksums

| Run / file | SHA-256 |
|---|---|
| `baseline-final/manifest.json` | `a48c7670aca139c59eb2be3248337bbcc5dc053d7b182df413e827453a869eb6` |
| `baseline-final/source.tar.gz` | `78e2eca000447bbd6d614928ec0573783bf53749cf16a31435b492b269998478` |
| `baseline-final/records.jsonl` | `c5500dae2e8b7da1d667d6b50fdad207ef798a3e801c8481f0c59ea1d345df8a` |
| `baseline-final/summary.json` | `54b0375c707c910f9fdec06699b8e5191103ca0ecea6cec0aff431aa2981d05b` |
| `extensions-final/manifest.json` | `c8adbc77d504d41f7288e8b3d4c7b840431622a64dd0f65858beaecb8fae6c65` |
| `extensions-final/source.tar.gz` | `ec26454c2a32f99b2344abc3307154d812c3f2d6d23f36082d2d090f425b83b9` |
| `extensions-final/records.jsonl` | `a66d234edb20585f76ea60ffafa7d644d0a33db4cef0bdd808cb93f552aa5a60` |
| `extensions-final/summary.json` | `8f2566fe10826a4775cdbf7575718e3805a7e173780f2401b34cf183becc2caf` |

The seed/seat/policy jobs are embedded in each manifest. Dependency lock SHA-256:
`7993429362e8a6351af8695ffe948e5af7b3a3f4a4ca5dd2a681c06a2746774c`.

## Validation

- All 1,200 baseline outcomes and action/event trace hashes match the earlier development run.
- All 48 extended outcomes and trace hashes match the earlier selected-extension runs.
- 72 saved replays reproduced **981,242 transitions**, full final snapshots including RNG state,
  and final per-player cash reconstructed independently from the classified transaction ledger.
- All **2,690,014** final-run decisions passed engine invariants and per-player cash reconciliation.
- `pytest tests/foundation --no-cov -q`: **41 passed, 1 strict expected failure**, two existing
  PettingZoo structured-observation advisories. The expected failure is the newly reproduced
  mortgage/build defect; it is a readiness blocker, not a passing correctness check.
- Scoped Ruff: passed. `mypy monopoly_engine`: passed. `git diff --check`: passed.
- Engine, policy, environment, API/frontend implementations, dependency files, production horizon
  and branch refs were not edited. No full certification or substantive training was run.

Analysis outputs: `baseline-analysis.json` (prevalence, paired extensions, repeat/replay checks)
and `extensions-analysis.json` (extension prevalence/replay verification) under the root above.
Raw results, source archives and replays remain ignored by Git, as requested. Copy those final
directories with this report for archival; documentation alone does not contain their payloads.
