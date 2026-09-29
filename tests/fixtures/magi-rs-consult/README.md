# magi-rs consult fixtures

Every file here comes from a real `magi-rs consult --output-format json --structured-verdicts` run
against the pinned binary. The variants are edits of a capture, never shapes written from memory.
The `applied_caps.ceiling_above_sanity: true` case is derived in the tests themselves by flipping
that one boolean on `three-agents.json`; its real shape was confirmed on a `--timeout 168200` run.

| File | Origin | Edit |
|---|---|---|
| `three-agents.json` | Captured code-review run over two of this plugin's own modules, magi-rs 0.21.0, `--timeout 16820`, 2026-09-29 (run `8160-5788b0503f996696`) | None; re-indented only |
| `two-agents-degraded.json` | `three-agents.json` | Caspar's verdict removed from `consult.agents`; `consult.degraded` set to `true`; `consult.failed_agents` = `{"caspar": "exhausted its attempts (derived variant)"}` |
| `no-agents.json` | `three-agents.json` | `consult.agents` and `consult.consensus` removed |
| `consult-null.json` | `three-agents.json` | `consult` set to `null`; `error` = `{"kind": "consult_failed", "message": "fewer than two mages completed (derived variant)"}`; `stop_reason` = `"error"` |
| `clean-magi-report.json` | A `magi-report.json` from a healthy Claude-path run | None. It carries no magi-rs block, so a pin move does not touch it |

When the exact-shape test in `tests/test_magi_rs_result.py` goes red after a pin move, re-verify the
contract with a real run first and only then recapture these files.
