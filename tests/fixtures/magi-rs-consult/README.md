# magi-rs consult fixtures

Every file here comes from a real `magi-rs consult --output-format json --structured-verdicts` run
against the pinned binary. The variants are edits of a capture, never shapes written from memory.

| File | Origin | Edit |
|---|---|---|
| `three-agents.json` | Captured run, magi-rs 0.21.0, `--timeout 16820`, 2026-09-29 (run `5180-2c75dd2da48dd8fa`) | None; re-indented only |
| `ceiling-above-sanity-true.json` | Captured run, magi-rs 0.21.0, `--timeout 168200`, 2026-09-29 (run `33460-9c25190c7ecda9fb`) | None; re-indented only. `applied_caps.ceiling_above_sanity` is `true` |
| `two-agents-degraded.json` | `three-agents.json` | Caspar's verdict removed from `consult.agents`; `consult.degraded` set to `true`; `consult.failed_agents` = `{"caspar": "exhausted its attempts (derived variant)"}` |
| `no-agents.json` | `three-agents.json` | `consult.agents` and `consult.consensus` removed |
| `consult-null.json` | `three-agents.json` | `consult` set to `null`; `error` = `{"kind": "consult_failed", "message": "fewer than two mages completed (derived variant)"}`; `stop_reason` = `"error"` |
| `clean-magi-report.json` | A `magi-report.json` from a healthy Claude-path run | None. It carries no magi-rs block, so a pin move does not touch it |

When the exact-shape test in `tests/test_magi_rs_result.py` goes red after a pin move, re-verify the
contract with a real run first and only then recapture these files.
