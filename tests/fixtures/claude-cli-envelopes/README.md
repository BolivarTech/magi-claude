# Claude CLI telemetry-envelope fixtures

These fixtures pin the **telemetry** fields of a real
`claude -p --output-format json` envelope — the parts `served_model.py` and
`cost.py` read, not the agent payload `parse_agent_output.py` parses.

They exist for the same reason as `../claude-cli-outputs/`: nothing in the unit
suite spawns `claude -p` (it needs the CLI and a paid call), so without a
captured sample the field names `modelUsage` and `canonicalModel` are an
assumption about the upstream CLI rather than a pinned contract. If Anthropic
renames or restructures them, attribution goes quietly empty and the only
symptom is a report that stops saying which model judged the input.

## Why this is a separate directory from `claude-cli-outputs/`

`test_claude_cli_fixture_contract.py` auto-discovers every `.json` in that
sibling directory and asserts it round-trips through `parse_agent_output` into a
valid seven-key agent output. These fixtures deliberately carry a **redacted**
`result`, so they would fail that invariant. Two fixture sets, two contracts.

## Adding a fixture

Capture a real envelope from a run, then redact: keep the telemetry keys
verbatim, replace `session_id` / `uuid` with `REDACTED`, and replace `result`
with a note. Never commit a real verdict payload or a session identifier here.

| File | Captured from | Pins |
|---|---|---|
| `modelusage-firstparty.json` | a v5.4.0 gate run on `claude-opus-5` (2026-07-30) | `modelUsage` keyed by model ID, entries carrying `canonicalModel` and `provider` |
