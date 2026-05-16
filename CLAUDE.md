# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAGI is a Claude Code **plugin** implementing a multi-perspective analysis system inspired by the MAGI supercomputers from Neon Genesis Evangelion. Three specialized AI agents — Melchior (Scientist), Balthasar (Pragmatist), Caspar (Critic) — independently analyze the same input through different lenses, then their verdicts are synthesized via majority vote.

`docs/MAGI-System-Documentation.md` is the full technical reference.

## Development Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Full verification (lock sync + tests + lint + format + types)
make verify

# Individual checks
make lockcheck | make test | make lint | make format | make typecheck

# Run analysis (parallel mode, requires claude CLI)
python skills/magi/scripts/run_magi.py <code-review|design|analysis> <file_or_text> \
  [--model opus|sonnet|haiku] [--timeout 900] [--output-dir <dir>] [--keep-runs 5] [--no-status]

# Run synthesis standalone
python skills/magi/scripts/synthesize.py agent1.json agent2.json [agent3.json] --output report.json

# Test plugin locally
claude --plugin-dir .
```

## Plugin Structure

```
.claude-plugin/
  plugin.json                 — Plugin manifest
  marketplace.json            — Local marketplace config
skills/magi/
  SKILL.md                    — Orchestrator (mode detection, workflow, fallback)
  agents/                     — melchior.md, balthasar.md, caspar.md (system prompts)
  scripts/
    run_magi.py               — Async orchestrator + tracked_launch
    temp_dirs.py              — magi-run-* LRU cleanup, realpath traversal guard
    subprocess_utils.py       — windows_kill_tree, stderr drain, timeouts
    status_display.py         — Live tree renderer (ANSI + plain, UTF-8 + ASCII)
    stderr_shim.py            — _buffered_stderr_while context
    models.py                 — MODEL_IDS, resolve_model, MODE_DEFAULT_MODELS
    synthesize.py             — Facade re-exporting validate/consensus/reporting
    validate.py               — ValidationError + schema check
    consensus.py              — VERDICT_WEIGHT + determine_consensus
    reporting.py              — format_banner + format_report (ASCII)
    parse_agent_output.py     — Claude CLI JSON extractor (3 output formats)
tests/
  fixtures/claude-cli-outputs/  — Pinned claude -p samples (contract tests)
  test_synthesize.py            — 142 tests
  test_parse_agent_output.py    — 27 tests
  test_run_magi.py              — 105 tests
  test_status_display.py        — 46 tests
  test_entry_point_invocation.py — 6 tests (python -m hardening)
pyproject.toml                — Python >= 3.9, dual license, tool config
conftest.py                   — tdd-guard pytest plugin + sys.path setup
Makefile                      — verify, test, lint, format, typecheck targets
```

### Cross-file contract: Agent JSON schema

All three agent prompts and every script depend on this schema — change it everywhere or not at all:

```json
{
  "agent": "melchior | balthasar | caspar",
  "verdict": "approve | reject | conditional",
  "confidence": 0.0-1.0,
  "summary": "string",
  "reasoning": "string",
  "findings": [{"severity": "critical|warning|info", "title": "string", "detail": "string"}],
  "recommendation": "string"
}
```

### Consensus logic (consensus.py)

Weight-based scoring with `VERDICT_WEIGHT = {approve: 1, conditional: 0.5, reject: -1}`:

```
score = sum(VERDICT_WEIGHT[verdict] for each agent) / num_agents
```

| Score | Condition | Consensus |
|-------|-----------|-----------|
| 1.0 | — | STRONG GO |
| -1.0 | — | STRONG NO-GO |
| > 0 | has conditionals | GO WITH CAVEATS (N-M) |
| > 0 | no conditionals | GO (N-M) |
| 0 | — | HOLD -- TIE |
| < 0 | — | HOLD (N-M) |

Labels carry a dynamic `(N-M)` split for every non-unanimous and non-tie outcome. `HOLD -- TIE` maps to `consensus_verdict: "reject"` (ties default to "do not proceed").

**Single-source-of-truth invariant (2.1.1)**: `consensus_verdict` is derived from `score` alone. The agent partition (`majority_agents` vs `dissent_agents`) is taken from whichever side matches the verdict — approve and conditional resolve to the approve side, reject to the reject side. The `(N-M)` split derives from the partition. This keeps the rendered label, `majority_agents`, and `_compute_confidence` referencing the same side on every vector.

**Confidence formula**:

```
base_confidence = sum(majority_confidence) / num_agents   # denominator is num_agents
weight_factor   = (abs(score) + 1) / 2                    # symmetric for approve/reject
confidence      = clamp(base_confidence * weight_factor, 0.0, 1.0)
```

- **Dissent dilution**: denominator is `num_agents`, not `len(majority_agents)`. A minority that disagrees dilutes the numerator, so unanimous wins yield higher confidence than bare-majority wins. Moderate confidence on a narrow win means "the split itself reduces certainty".
- **Symmetric weighting**: `abs(score)` makes unanimous reject produce high confidence (matching approve), not zero. At score=0 weight_factor=0.5 halves confidence.

Key behaviors:
- `conditional` maps to `approve` for majority identification, conditions preserved in report.
- Unanimous `conditional` → `GO WITH CAVEATS (3-0)` at ~0.68 confidence, not `STRONG GO`.
- `consensus.conditions` is sourced from each conditional agent's `summary`; `consensus.recommendations` from each agent's `recommendation`. Both fields must render distinct text.
- Findings deduplicated by case-insensitive title with `sources` list; highest severity wins.
- Minimum 2 agents (raises `ValueError`). Duplicate agent names raise `ValueError`.

Implementation split: `_consensus_short_verdict`, `_format_consensus_label`, `_deduplicate_findings`, `_compute_confidence`.

### Import convention

The `synthesize.py` facade re-exports public symbols from `validate.py`, `consensus.py`, `reporting.py`. Always import from `synthesize`:

```python
from synthesize import load_agent_output, determine_consensus, format_report
```

Do not import from sub-modules — the facade is the stable API.

### Orchestrator (run_magi.py)

- Launches 3 `claude -p` subprocesses concurrently via `asyncio.create_subprocess_exec` with per-agent `--timeout` (default 900s).
- `--model` (default `opus`) selects LLM for all agents. Valid: `opus`, `sonnet`, `haiku`. `VALID_MODELS` is derived from `MODEL_IDS.keys()` — single source.
- User prompt sent via **stdin** (avoids ~32K Windows CLI arg limit). A copy is saved to `{agent}.prompt.txt` for debugging.
- System prompts passed via `--system-prompt-file` using the original `.md` path (no temp copy).
- Validates subprocess exit code before parsing — non-zero raises `RuntimeError` with stderr context.
- Each agent's raw output → `parse_agent_output.py` → `load_agent_output()` validation.
- < 3 agents succeed: warning to stderr, `"degraded": true` in report, proceed with ≥ 2.
- < 2 agents succeed: raises `RuntimeError`.
- Cross-platform temp via `tempfile.mkdtemp(prefix="magi-run-")`, cleaned up on failure.
- `--keep-runs N` (default 5): LRU cleanup sorted by `st_mtime`, resolved via `realpath` with temp-root validation to prevent symlink traversal. Disabled with `--keep-runs 0`.
- Live status tree (`StatusDisplay`) wired around `asyncio.gather` via `tracked_launch` mapping `launch_agent` exit paths to display states. Disabled with `--no-status`. Catches both `asyncio.TimeoutError` and `TimeoutError` (3.9/3.10 compat).

### Model selection

Default `opus` applied uniformly — differentiation comes from system prompts, not model. `models.py` is the only place mapping short names → Anthropic IDs (`claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`).

**Recommended per mode** (opus ≈ $0.25/agent, ~$0.75/run):

| Mode | Recommended | Rationale |
|------|-------------|-----------|
| `code-review` | `opus` | Dense reasoning; subtle interaction tracing. |
| `design` | `opus` | Multi-level abstraction; smaller models lose cohort confidence. |
| `analysis` | `opus` | See 2.2.5 — sonnet caused ≥33% Caspar drops under sbtdd. |
| Smoke / fixtures | `haiku` | ~10× cheaper; validates schema/parsing without burning opus. |

When `--model` is omitted, `MODE_DEFAULT_MODELS` (`models.py`) resolves it. Explicit `--model X` always wins.

### Status display (status_display.py)

Stdlib-only tree renderer:

- **ANSI mode** (TTY): in-place redraw every 200ms via `\033[NA` cursor movement and `\033[2K` erase. Background async drives spinner. On Windows, `ENABLE_VIRTUAL_TERMINAL_PROCESSING` is enabled via `ctypes` with narrow exception handling.
- **Plain mode** (pipe/captured stream): one line per `update()` call, no escape codes.
- **Glyph fallback**: probes `stream.encoding` against `"●○✓✗⏱├─└─⠋"`; falls back to ASCII (`* . v x ~ |- \-`) on cp1252. Streams without bound encoding (`io.StringIO`) are treated as unicode-capable. Timeout glyph is `~` (avoids letter `T` collision).
- **Invariant**: plain-mode and ANSI refresh writes are mutually exclusive — `_use_ansi` selects exactly one path.
- `stop()` is idempotent.

### Parser (parse_agent_output.py)

Handles three Claude CLI output formats:

1. `{"result": "..."}` — standard `--output-format json`
2. `{"content": [{"type": "text", "text": "..."}]}` — content-block format
3. Plain string — raw text

Strips markdown code fences and validates extracted JSON. Raises `ValueError` for unrecognised output types (no silent fallback).

### Execution pipeline

```
User input → SKILL.md (gate + mode) → run_magi.py launches 3× claude -p
  → each writes JSON to temp dir → parse_agent_output.py extracts JSON
  → validate.load_agent_output() → consensus.determine_consensus()
  → reporting.format_report() → stdout banner + JSON to output dir
```

Fallback (no `claude -p`): SKILL.md simulates three perspectives sequentially (Caspar first to reduce anchoring).

## Key Design Decisions

- **Disagreement is a feature**. Unanimous agreement on non-trivial input may indicate insufficiently differentiated prompts.
- **Caspar is adversarial by design**. Most likely to vote `reject` — intentional red-teaming.
- **Weight-based scoring** via `VERDICT_WEIGHT`. Unanimous `conditional` maps to moderate confidence, not high.
- **Agent prompts enforce English output** regardless of input language.
- **Prompt injection guard** in all agent prompts; schema validation (`load_agent_output`) is the technical enforcement layer.
- **Failure alerting**. Degraded mode (< 3 agents) is explicitly flagged in report and stderr.

## Distribution & Installation

Decentralized marketplace — no central registry. A marketplace is any public GitHub repo with `.claude-plugin/marketplace.json` cataloging plugins.

### For users

```bash
/plugin marketplace add BolivarTech/magi-claude
/plugin install magi@bolivartech-plugins
/magi
# Update after publishes:
/plugin marketplace update
```

### For development

```bash
# Option A — plugin flag:
claude --plugin-dir /path/to/magi

# Option B — symlink for auto-discovery:
mkdir -p .claude/skills
ln -s ../../skills/magi .claude/skills/magi
claude
# /reload-plugins picks up changes.
```

`.claude/skills/` auto-discovery is project-scoped. For user-wide availability use `/plugin install` or symlink into `~/.claude/skills/`. `plugin.json` requires `"skills": "./skills/"`.

### Publishing updates

1. Bump `"version"` in `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json` (both occurrences), `pyproject.toml`, and `uv.lock` (`[[package]] name = "magi"`).
2. Run `make verify`. Lock sync + all tests + zero lint + clean format + no type errors.
3. Commit `fix|feat|chore: <summary> and release <version>`. Push to `main`.
4. Annotated tag: `git tag -a v<version> -m "Release <version>: <summary>"` then `git push origin v<version>`. Every release from 2.1.4 onward carries an annotated tag in `v<MAJOR>.<MINOR>.<PATCH>` form. Lightweight tags are not acceptable.
5. Users pick up via `/plugin marketplace update`.

### Pre-merge MAGI self-review (standard procedure)

For any release shipping behaviour-affecting code (`feat:`, medium-severity+ `fix:`, anything touching `tracked_launch` / `launch_agent` / the schema), run MAGI on the diff before merging.

1. **Build the diff bundle** as `magi-review-<version>-context.md`: one-paragraph what+why, full `git diff main feature/<branch>` in a ```diff``` fence, 4-6 non-leading review questions ("Is X correct?", "Could Y break Z?").
2. **Invoke**: `python skills/magi/scripts/run_magi.py code-review magi-review-<version>-context.md --timeout 900 --no-status`. Wall time 1-3 min, opus cost ~$0.75.
3. **Read raw per-agent `.json`** in the run's temp dir. Per-finding `detail` carries reasoning the rendered table truncates.
4. **Process findings via `superpowers:receiving-code-review`**: every finding gets a verdict — `valid`, `wrong`, `style`, `out-of-scope`, or `YAGNI`. Push-backs require concrete proof (file:line, test, counter-example). Performative agreement forbidden.
5. **Address valid findings** in a follow-up `fix:` commit on the same branch. Push-back rationale goes in the commit body. ≥ 3 valid findings of the same shape → stop and re-plan.
6. **Delete the context bundle** before merging. `magi-report.json` on disk is the audit artifact.
7. **Merge `--no-ff`**, tag, push.

**Skip when**: pure docs, version bumps, gitignore, reformatting, dependency updates with no behaviour change. The $0.75 cost is real.

**Escalate beyond MAGI** when: `STRONG NO-GO`, or two consecutive `GO WITH CAVEATS` with overlapping conditions. Pull in a second human reviewer.

### Marketplace structure

| File | Purpose |
|------|---------|
| `plugin.json` | Plugin identity: name, version, author, repository, license, skills path |
| `marketplace.json` | Catalog: owner, plugin list with sources, categories, tags |

This repo hosts only `magi` with `source: "./"`.

## Test Coverage

326 tests across 5 files (325 passing, 1 skipped on Windows). Run with `python -m pytest tests/ -v` or `make test`.

| File | Tests | Covers |
|------|-------|--------|
| `test_synthesize.py` | 142 | Validation, type/length checks, weight-based consensus, confidence formula, findings dedup, dynamic labels, HOLD--TIE, duplicate-agent rejection, banner alignment, verdict-suffix preservation, report ordering, SKILL.md parity, zero-width Unicode (U+2060-U+206F) |
| `test_parse_agent_output.py` | 27 | Fence stripping, text extraction (3 formats), fail-fast on unknown shapes, pipeline, pinned `claude -p` fixture contract (auto-discovered) |
| `test_run_magi.py` | 105 | Arg parsing, --no-status, model passthrough, orchestration, degraded mode, cleanup LRU/symlink, tracked_launch states, kill-tree order, stderr replay OSError safety, retry on ValidationError + JSONDecodeError, retried_agents telemetry, MODE_DEFAULT_MODELS ↔ VALID_MODES lockstep, cp1252 hardening, UTF-8 console reconfigure |
| `test_status_display.py` | 46 | Init/update/render, ASCII fallback, async lifecycle, stop idempotency, write-path invariant tripwire, OSError + non-OSError refresh-loop resilience, retrying-state glyphs (UTF-8 ↻, ASCII `r`), cp1252 fallback |
| `test_entry_point_invocation.py` | 6 | `python -m skills.magi.scripts.<name>` and direct-script invocation of `run_magi`, `synthesize`, `parse_agent_output` |

## Resolved Issues (consolidated)

All issues from MAGI self-analysis (2026-04-01 migration) and four rounds of self-review (R1-R4) are resolved. Highlights:

- **Schema/Validation**: weight-based scoring replaces unanimous-conditional → STRONG GO bug; non-dict top-level JSON guarded with `ValidationError`; agent-name uniqueness validated; `_extract_text` fails fast on unknown shapes.
- **Subprocess robustness**: prompt via stdin (32K Windows arg limit); `--timeout` enforced; exit-code validated; Windows kill-tree runs *before* `proc.kill()`; stderr drain hardened against OSError.
- **Filesystem**: cross-platform tempdirs; LRU cleanup via `st_mtime` with `realpath` + temp-root validation; `OSError` warnings on cleanup failure.
- **Display**: status tree never fails the run (`refresh_loop` catches `Exception`); `stop()` idempotent; cp1252 ASCII fallback; verdict suffix preserved under overlong labels.
- **Telemetry**: zero-width Unicode (`U+2060-U+206F` word joiner, invisible math operators, tag controls) cleaned in finding titles; banner uses integer percent; `HOLD -- TIE` replaces misleading HOLD with conditionals.
- **Test contract**: pinned `claude -p` output fixtures under `tests/fixtures/claude-cli-outputs/` auto-discovered, with a non-empty guard test so renames cannot degrade the contract to a vacuous pass.

Detail of each fix lives in git history; only the LOCKED debt entry below requires re-reading.

### Known residual limitations

- **TOCTOU**: narrow race between `realpath()` and `rmtree()` in `cleanup_old_runs`. Acceptable for dev tooling, not security-critical environments.
- **Windows subprocess orphans**: only if `taskkill` is missing from PATH or `_TASKKILL_TIMEOUT` exceeded — emits a warning naming the pid.
- **Temp-dir scan**: may be slow on machines with thousands of temp entries (shared CI).
- **Windows non-VT TTY**: falls through to plain mode (append-only); modern Windows Terminal / ConEmu / WSL unaffected. Use `--no-status` if append output is undesirable.
- **`_StderrBufferShim` bypass paths**: `os.write(2, ...)`, subprocess fd 2 inheritance, and pre-cached `sys.stderr` references (modules that capture `err = sys.stderr` at import time bypass the shim).
- **Buffered diagnostics on hard process death**: `_buffered_stderr_while` flushes in `finally`, surviving exceptions / CancelledError / KeyboardInterrupt / SystemExit. Lost only on `SIGKILL`, segfault, or `os._exit()`.

## Open technical debt

Real items with concrete fix paths, deferred to avoid bloating unrelated releases.

### Backwards-compatibility re-exports in `run_magi.py`

Lines 56-67 re-export `cleanup_old_runs`, `create_output_dir`, `MAGI_DIR_PREFIX` for legacy test imports. Comment explicitly says future code should import from `temp_dirs` directly.

**Fix** (~1-2h): rewrite the few `from run_magi import cleanup_old_runs` call sites (mostly in `tests/test_run_magi.py`) to `from temp_dirs import ...`; delete the three names from `__all__`. `make verify` flags misses via mypy.

**Acceptance**: `git grep "from run_magi import cleanup_old_runs"` returns zero. Orchestration symbols (`MODEL_IDS`, `VALID_MODELS`, `resolve_model`) stay re-exported because they legitimately straddle modules.

**Why deferred**: pure cleanup, zero behaviour change. Bundle into the next refactor release (e.g., 2.3.0).

### `run_magi.py` size and orchestrator layering

628 LOC — largest module. Carries arg parsing, `launch_agent`, `_DisplayLogGate`, `_safe_display_update`, `_build_retry_prompt`, `run_orchestrator`, `tracked_launch` closure (7 captured variables), and `main`. R4-4 already pulled out filesystem/subprocess pieces; what remains is genuinely entangled.

**Fix** (~3-5h, minor release): extract `launch_agent` + timeout/stderr into `agent_runner.py`; extract per-agent state machine (running → retrying → terminal) into a `LaunchTracker` class in `tracking.py`. `run_orchestrator` shrinks to ~80 LOC.

**Acceptance**: no module > ~300 LOC; `tracked_launch` no longer a closure; tracker class has unit tests independent of the orchestrator.

**Why deferred**: warrants a minor release (2.3.0) and a brainstorm — touches the hottest async path. The 2026-05-15 telemetry routine should land first; it may suggest specific tracker responsibilities (per-agent budget, retry-count) the refactor should accommodate from day one.

### `ValueError` boundary conflates parser-shape and config errors

`tracked_launch` does not retry on `ValueError`, which lumps two failure modes:

- `resolve_model("gpt4")` — config error, retry would re-fail identically. Correct.
- `parse_agent_output._extract_text({"weird_shape": ...})` — structural CLI change, retry might recover.

**Fix** (~1-2h): introduce `class ParseShapeError(ValueError)` in `parse_agent_output.py`; replace the two `raise ValueError` sites in `_extract_text` with `raise ParseShapeError` (file-too-large stays plain `ValueError`); extend `tracked_launch` catch to `(ValidationError, json.JSONDecodeError, ParseShapeError)`. Add regression test for shape-fail → retry; rename existing `test_value_error_from_parse_does_not_retry` to target `resolve_model`-style errors.

**Acceptance**: a future Anthropic CLI shape change produces a single recoverable failure per agent, not 2-of-3 catastrophic loss. The `ValueError`-from-config boundary remains pinned.

**Why deferred**: zero observed cost today. If a production incident surfaces, this becomes the next patch with the same TDD shape as 2.2.4.

### `synthesize` import gap for code outside `skills/magi/scripts/` — `[LOCKED]`

`synthesize.py` and the other orchestration modules live in `skills/magi/scripts/`, which is **not** a standard importable package. `conftest.py:34-36` injects it into `sys.path` for tests; the entry-point scripts (`run_magi.py`, `synthesize.py`, `parse_agent_output.py`) self-inject their own directory via a 4-line bootstrap at module top (covers `python -m skills.magi.scripts.<name>`); direct invocation is covered by Python's script-dir injection.

Any **non-entry-point** module created outside `skills/magi/scripts/` fails with `ImportError: No module named 'synthesize'` unless the operator prepends `PYTHONPATH=skills/magi/scripts` or applies the bootstrap.

**Discovered**: 2026-05-14 during the rejected premortem A/B experiment (`experiments/premortem/`). Branch was deleted; the underlying convention-coupling for new outside callers remains.

**Fix (LOCKED — implement when needed)**: add this bootstrap at the top of any new outside-callers module, **before** the first `from synthesize import ...`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[N] / "skills" / "magi" / "scripts"))
```

Where `N` is parent-directories up to the repo root. `experiments/<topic>/foo.py` → N=2; `tools/foo.py` → N=1. Standard PEP 8 import-block ordering is unchanged.

**Rejected alternatives**: full packaging (invasive — touches plugin distribution + CI); conftest-style side-effect helper (magic import, only worth it with 3+ consumers); restructuring `synthesize.py` location (invasive — every existing import + agent-prompt reference needs updating).

**Acceptance**: next outside-caller module on `main` carries the bootstrap. Its README documents `python -m <module>` directly. No new conftest hacks.

**Dual-import caveat**: a module loaded via the bootstrap is registered in `sys.modules` under its bare name (e.g., `synthesize`) — same key as in-tree imports. If a caller ALSO does a dotted import (`from skills.magi.scripts.synthesize import ...`), Python treats the two names as distinct cache keys and loads TWO module objects with independent state — globals, class identity checks, singletons diverge silently. Outside callers MUST keep imports under the bare name. Same constraint applies to the 2.2.8 entry-point hardening.

**Why locked**: architectural decision made (option B from 2026-05-14). Future implementers apply the snippet, don't re-litigate.

## Post-release hardening (changelog)

Brief release notes preserving load-bearing context. Implementation details live in git history and test docstrings.

### 2.2.0 — Single-shot agent retry on `ValidationError`

`tracked_launch` catches `ValidationError`, emits `retrying` display state, rebuilds prompt via `_build_retry_prompt` (original + `---RETRY-FEEDBACK---` delimiter + error + 7-key schema restatement), re-invokes `launch_agent` with the full `--timeout`. Retry count fixed at 1. Worst-case wall time per retried agent: `2 × --timeout`. `StatusDisplay.VALID_STATES` gains `"retrying"` (UTF-8 glyph `↻`, ASCII `r`). `_UNICODE_PROBE` widened to include `↻`. Non-goals (each pinned by a regression test): retry on subprocess timeout, retry on non-schema exceptions, retry count > 1.

### 2.2.1 — `retried_agents` telemetry

`run_orchestrator` records every agent that took the retry path; serialises sorted alphabetically to `retried_agents` in the report. Conditional presence (like `degraded`, `failed_agents`). Downstream can compose `retried_agents - failed_agents` (retry-recovered) and `retried_agents & failed_agents` (retry-also-failed).

### 2.2.2 — cp1252 retry glyph fallback

`_UNICODE_PROBE` updated so cp1252 streams fall through to ASCII glyphs **before** the first retry, not on it.

### 2.2.3 → 2.2.5 — Analysis-mode default model

- 2.2.3: switched `MODE_DEFAULT_MODELS["analysis"]` to `sonnet` for cost relief.
- 2.2.5: reverted to `opus`. Production data showed Caspar (longest output, 3.6-6.7K tokens) failed in ≥33% of sbtdd verifications under sonnet's ~8K max-output ceiling. 2.2.4 retry could not recover because the failure was structural, not stochastic. Losing Caspar specifically biases consensus toward false-positive approval — the adversarial lens disappears, so `GO_WITH_CAVEATS (2-0)` from Mel+Bal is structurally different from a clean 3-agent result. Cost reverts to ~$0.75/run for bare-`analysis`. `MODE_DEFAULT_MODELS` plumbing preserved for future per-mode work.

Open follow-ups (deferred until the 2026-05-15 telemetry routine surfaces enough data to choose):

1. **Per-agent model**: Caspar on opus, Mel/Bal on sonnet. Saves ~$0.30/run; needs per-agent plumbing.
2. **`--max-tokens` flag** in `launch_agent`: override model default; needs Anthropic CLI option parser.

### 2.2.4 — Retry extended to `json.JSONDecodeError`

After an iter-2 sbtdd run lost two of three agents to `JSONDecodeError` raised **before** `load_agent_output` could wrap. `tracked_launch` now catches `(ValidationError, json.JSONDecodeError)`. `_build_retry_prompt` reworded ("rejected by the parsing pipeline") to be accurate for both layers; 7-key schema reminder retained.

Still **out of scope** (each pinned by a regression test): `ValueError` from `resolve_model` or `_extract_text`, `TimeoutError`, `RuntimeError`, `OSError`, `CancelledError`.

### 2.2.6 — Windows cp1252 hardening

Two crash sites fixed:

1. **Encode-side**: replaced four `⚠ WARNING` prints with ASCII `[!] WARNING` (U+26A0 is not in cp1252's codepage; `print` raised `UnicodeEncodeError` whenever an agent failed under captured stderr).
2. **Decode-side**: extracted `_load_input_content(input_arg) -> tuple[str, str]` using `open(..., encoding="utf-8", errors="replace")` so cp1252-only bytes decode as U+FFFD and the run continues.

Em-dash U+2014 left intact — cp1252 encodes it at 0x97.

### 2.2.7 — UTF-8 console reconfigure

`_enable_utf8_console_io()` is the **first** statement in `main()`. On `win32`, switches `sys.stdout`/`sys.stderr` to `encoding="utf-8"` with `errors="backslashreplace"` via `stream.reconfigure()`. `backslashreplace` is the non-negotiable error policy — produces ASCII so output is encodable in any codepage. No-op on POSIX. Streams without `reconfigure` (custom sinks, pytest capture) silently skipped.

Supersedes 2.2.6's whack-a-mole: fixes the channel itself so any LLM-emitted Unicode in finding titles survives without per-codepoint patches. The 2.2.6 ASCII markers stay — they read cleaner than escape-sequence noise.

### 2.2.8 — Entry-point hardening for `python -m`

`run_magi.py`, `synthesize.py`, `parse_agent_output.py` self-inject their own directory at module top via a 4-line `sys.path` bootstrap. Covers `python -m skills.magi.scripts.<name>` invocations. Pinned by 6 parametrized subprocess tests in `tests/test_entry_point_invocation.py`. Direct-script invocation is covered automatically by Python's script-dir injection.

### Agent prompt schema-key reinforcement

Production-log scan (10 runs × 3 agents, 30 outputs) found one schema violation (Caspar, 2026-04-19, missing `recommendation`). Incidence 1/30 ≈ 3.3%, exclusively on Caspar. The `IMPORTANT:` closer in all three agent prompts now enumerates the seven required top-level keys and states explicitly that omission causes rejection. Applied to all three — failure mode is LLM-schema-drift, not agent-specific.

### Premortem A/B experiment for Caspar — rejected (2026-05-14)

Tested adding a "premortem frame" to `caspar.md` (project six months into a failed future, write findings as past-tense post-mortem). N=16 paired inputs from sbtdd's `.magi_review_input/`.

| Metric | Baseline | Treatment | Cliff's δ | Wilcoxon p |
|---|---:|---:|---:|---:|
| length (words) | 504.4 | 616.9 (+22%) | 0.289 | **0.0019** |
| past-tense density | 0.0365 | 0.0362 | 0.055 | 0.717 |
| unique findings | 6.0 | 6.0 | −0.090 | 0.878 |
| judge specificity (1-5) | 2.584 | 2.691 | 0.148 | 0.258 |

Technical rule: `inconclusive` (judge δ=0.148 crossed the 0.147 small-effect cutoff by 0.001). **Operator rejected** because length +22% with flat density/unique findings matches the "more reasons, not better reasons" Mitchell-trap shape, and the judge effect is at noise floor (Wilcoxon p=0.258). +22% Opus tokens not justified.

**Methodology amendment preserved**: title-based judge pairing failed (zero matched pairs after 2-input smoke). Replaced with per-finding individual rating + per-run mean aggregation, preserving input-level pairing for Wilcoxon. **Generalisation**: any future LLM-prompt A/B should not assume title-based pairing survives stylistic interventions.

**Cost calibration**: ~$26 total (Opus 32 × ~$0.50 + Haiku 196 × ~$0.04). Haiku per-call via `claude -p` ≈ $0.04 (Claude Code context priming inflates input tokens), not the bare-API $0.001.

**Branch fate**: `premortem` kept local-only on the operator's machine until 2026-05-29 for raw-data inspection (16 baseline + 16 treatment JSONs + judge scores + results.json + report.md under `experiments/premortem/`). Never pushed to origin. After 2026-05-29 operator runs `git branch -D premortem` locally. No automated scheduling — operator's calendar reminder. This entry is the durable record.

**Why this entry exists**: protect against re-litigation. The proposal was tested at scale with pre-registered metrics. Future sessions raising the idea should be pointed here; only new evidence (different rubric, structural Caspar change, different corpus) justifies re-running.

## Breaking changes (2.0.0)

- `GO WITH CAVEATS` now renders with `(N-M)` split suffix (e.g., `GO WITH CAVEATS (3-0)`). Downstream parsers grepping for an exact flat string must tolerate the trailing split.
- `consensus.conditions[*].condition` is sourced from each conditional agent's `summary`, not `recommendation`. Consumers relying on the duplication must switch to `recommendations[agent]`.
- `validate.clean_title` is public (previously `_clean_title`). Re-exported through `synthesize.clean_title`.
- `StatusDisplay._write_plain_event` raises `RuntimeError` (previously `AssertionError`) when invoked under ANSI mode. Invariant survives `python -O`.

## Breaking changes (1.1.0)

- `## Consensus Summary` section removed from `format_report` output. Banner → `## Key Findings` directly. `consensus.majority_summary` field remains in JSON for downstream consumers — parse that, don't grep markdown.
- `## Dissenting Opinion` shows `summary` only, not full `reasoning`. Full reasoning preserved in JSON and raw agent output files.

## Dependencies

| Component | Required | Notes |
|-----------|----------|-------|
| Claude Code CLI (`claude -p`) | For parallel mode | Fallback available |
| Python 3.9+ | Yes | `dict[str, Any]`, `asyncio` |
| pytest + pytest-asyncio | Dev only | Async tests |
| ruff | Dev only | Lint + format |
| mypy | Dev only | Strict mode |

## License

Dual licensed under `MIT OR Apache-2.0` (Rust ecosystem convention). See `LICENSE` (MIT) and `LICENSE-APACHE`.
