# MAGI — Multi-Perspective Analysis Plugin for Claude Code

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-994%20passing-brightgreen.svg)](#running-tests)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A Claude Code plugin that implements a **multi-perspective analysis system** inspired by the [MAGI supercomputers](https://evangelion.fandom.com/wiki/Magi) from *Neon Genesis Evangelion*.

Three specialized AI agents independently analyze the same problem from complementary — and deliberately adversarial — perspectives, then synthesize their verdicts via weight-based majority vote.

---

## Why Three Adversarial Perspectives?

### The MAGI in Evangelion

In *Neon Genesis Evangelion* (1995, Hideaki Anno / Gainax), the MAGI are three supercomputers that govern Tokyo-3's critical decisions. Each embodies a different facet of their creator, Dr. Naoko Akagi: **Melchior** (the scientist), **Balthasar** (the mother), and **Caspar** (the woman). Decisions require consensus — no single perspective dominates.

This design reflects a profound insight: **complex decisions benefit from structured disagreement**. A single decision-maker, no matter how capable, carries blind spots. Three independent evaluators with different priorities surface risks, trade-offs, and opportunities that any one of them would miss.

### The Theory in Practice

The adversarial multi-perspective model addresses well-documented cognitive biases in software engineering:

| Bias | How MAGI Mitigates It |
|------|----------------------|
| **Confirmation bias** | Three agents with different evaluation criteria are unlikely to share the same blind spots |
| **Anchoring** | Agents analyze independently — no agent sees the others' output before forming its own verdict |
| **Groupthink** | Caspar (Critic) is designed to be adversarial; its role is to find fault, not agree |
| **Optimism bias** | The weight-based scoring penalizes reject (-1) more heavily than approve (+1), making negative signals harder to override |
| **Status quo bias** | Each agent evaluates from first principles against its own criteria, not against "how things are done" |

The key insight is that **disagreement between agents is a feature, not a failure**. When Melchior (Scientist) approves but Caspar (Critic) rejects, the dissent surfaces a genuine tension between technical correctness and risk tolerance. Unanimous agreement on non-trivial input may indicate insufficiently differentiated prompts, not actual consensus.

In practice, the system works best for decisions with:
- **Genuine uncertainty** — multiple valid approaches exist
- **Significant consequences** — the cost of a wrong decision is high
- **Hidden trade-offs** — benefits and risks are not immediately obvious

For trivial questions with one clear answer, the complexity gate skips the full system and responds directly.

---

## Documentation

For the full technical reference, see [`docs/MAGI-System-Documentation.md`](docs/MAGI-System-Documentation.md).

**New in v4.0.0 — Ollama backend (opt-in):** run the MAGI *gate* on local/LAN/cloud
open-weight models with genuine cross-lineage diversity (a distinct model per mage),
without changing the default Claude path. Quick start: `/magi --ollama` (or
`run_magi.py ... --ollama`); `--ollama-init` scaffolds the config. Full guide —
rationale, configuration, default models, and recommended hardware tiers — in
[`docs/ollama-backend.md`](docs/ollama-backend.md).

**New in v5.0.0 — fallback rotation (BREAKING, Ollama only):** when a mage's model
exhausts its attempts, MAGI now **rotates it to a declared `[[fallback]]` model of a
different lineage** instead of losing the mage to degraded mode — preserving the three
independent perspectives the ensemble depends on. This **changes the `magi-ollama.toml`
schema**: `[models]` entries go from a bare string to a table with an explicit `lineage`
(`melchior = { model = "qwen3.5:397b-cloud", lineage = "alibaba" }`). A v4 config now
fails closed with an actionable error — run `python skills/magi/scripts/validate_magi_toml.py` to see
exactly what to change (the lineage is never inferred). **Kill-switch:**
`MAGI_OLLAMA_MAX_ROTATIONS=0` disables rotation entirely (a shadow-rollout mode that keeps
the new preflight/probe active while rotation is off). **Fail-closed by default (v5.2.0):**
`strict_context_guard` now defaults to **`true`** — on an endpoint that does **not** expose
`/api/show` (so the window cannot be measured), MAGI **aborts** rather than run on an
unverified estimate. Set `strict_context_guard = false` to opt out and proceed with an
estimated guard. Rotation, its telemetry, the probe cost, and migration are documented in
[`docs/ollama-backend.md`](docs/ollama-backend.md).

**New in v5.0.2 — the scaffold shows every knob:** `--ollama-init` now writes all nine
rotation / context-window settings as active keys at their defaults (top-level, before
`[models]`), so the tuning surface — and the `max_rotations = 0` kill-switch — is visible
and editable without reading the docs. In plain terms: **`max_attempts_per_model`** = how
many times a mage retries the same model before rotating; **`max_rotations`** = how many
fallbacks it may try (`0` turns rotation off); **`output_headroom_tokens`** = context space
reserved for the model's reply so it is never cut off; **`input_margin_pct`** = safety
cushion when estimating whether the input fits a model's window; **`strict_context_guard`**
= refuse a model whose window cannot be measured (default `true`; set `false` to estimate);
**`retry_backoff_seconds`** = **base** of the exponential retry backoff (v5.3.0; `0` = retry
immediately), capped by **`retry_backoff_max_seconds`**; a server `Retry-After` is honored up to
**`retry_after_max_seconds`**; **`timeout`** = per-agent request timeout (`--timeout` overrides);
**`preflight_timeout_seconds`** / **`probe_timeout_seconds`** = timeouts for the metadata calls and
the context probe. An untouched scaffold behaves exactly as the built-in defaults. Full table in
[`docs/ollama-backend.md`](docs/ollama-backend.md).

> **`--ollama` runs the gate Claude-free, end-to-end.** The consensus verdict and the
> output banner are produced by **deterministic local Python** (`consensus.determine_consensus`
> + `reporting.format_report`), not by any LLM — so with `--ollama` the *entire* cycle
> (three mages, synthesis, and report) runs without any Claude/Anthropic API call. It is
> runnable standalone as a CI gate: `python skills/magi/scripts/run_magi.py <mode> <file_or_text> --ollama`.

**New in v5.1.0 — the verdict sentinel (BREAKING, agent prompt contract):** MAGI no
longer *searches* an agent's raw output for something that looks like a verdict — it
*extracts* the JSON between two literal marker lines, `<MAGI_VERDICT>` /
`</MAGI_VERDICT>`, each alone on its own line. A response missing either marker, or
carrying more than one block, is rejected outright and retried with corrective feedback
instead of being scanned for a lookalike. This closes a long-standing residual where a
truncated or ambiguous response could cause the old parser to silently accept the
**worked example already sitting in the agent's own system prompt** as its verdict — a
fabricated `approve`, in the adversarial seat, indistinguishable from a real one. A new
installation-time guard also refuses to start if any of the three shipped agent prompts
has malformed markers or a complete example sitting *between* them (see
[`docs/faq-prompt-guard.md`](docs/faq-prompt-guard.md) for every `[FATAL]` message and how
to fix it). Full rationale, evidence, and the alternatives considered and rejected:
[`docs/adr/0001-no-runtime-heuristic-fallback.md`](docs/adr/0001-no-runtime-heuristic-fallback.md).

> **There is no runtime fallback to the old heuristic — read this before you need it.**
> If a model in your trio cannot reliably emit the marker lines, MAGI retries it with a
> cause-specific correction and, absent a fix, drops that mage to a degraded run rather
> than silently guessing (watch `extraction_failures` in `magi-report.json` — it is the
> counter that tells you if this is happening). There is intentionally no config flag or
> environment variable to re-enable the pre-5.1.0 heuristic: a flag like that would be a
> switch that restores silent fabrication of an `approve`, not a safety valve (the ADR
> above has the full argument, made after the proposal was raised and rejected seven
> times during design review).
>
> **The downgrade path is the only production safety net for this change**, so it is
> documented here instead of buried in a changelog — a net nobody knows about is not a
> net. Every release is tagged (`vX.Y.Z`, annotated), so reverting to the last pre-sentinel
> release is a normal local checkout, not an emergency patch:
> ```bash
> git clone https://github.com/BolivarTech/magi-claude.git
> cd magi-claude
> git checkout v5.0.3
> claude --plugin-dir "$(pwd)"
> ```
> If you installed via the marketplace (`/plugin install magi@bolivartech-plugins`), the
> checkout above (dev-mode `--plugin-dir`) is the reliable way to pin an older version,
> since the marketplace itself tracks whatever ref its source points to. Uninstall the
> marketplace copy first (`/plugin uninstall magi@bolivartech-plugins`) to avoid running
> two copies of the plugin at once.

#### What the sentinel does *not* protect you from

Stated plainly, because a guarantee you misunderstand is worse than one you don't have:

- **A mage that copies the prompt's worked example and edits a word.** The anti-echo canary
  matches that example's exact fingerprint, so a *verbatim* copy is rejected — a *modified*
  copy is not. This is deliberate: widening the fingerprint would shrink a false-positive
  rate already measured at 0 in 170 real verdicts while enlarging the evasion, and every
  past attempt in this codebase to "tighten" a check by widening an exclusion shipped a
  fail-open. The real protection is upstream: **between the markers the prompt shows a
  placeholder, never a valid verdict**, so there is nothing there worth copying. A mage that
  emits an altered copy of the example is not a parser failure — it is a **degraded seat**,
  indistinguishable from a mage that simply reasons badly, and no parser can catch that.
  Rotate the model; don't argue with it.
- **A verdict that is well-formed and worthless.** The sentinel guarantees the verdict came
  from *that mage, between its own markers*. It cannot tell you the reasoning behind it was
  any good.
- **Silence.** Watch `extraction_failures` in `magi-report.json`. If a seat's marker-omission
  rate climbs past ~10% of attempts, iterate that prompt or rotate that model — the answer is
  never to restore a heuristic that guesses. If a run dies before it can write a report, the
  same counts are printed to stderr, so the cause survives the run.

---

## Agents

| Agent | Codename | Lens | Personality |
|-------|----------|------|-------------|
| **Melchior** | Scientist | Technical rigor and correctness | Precise, evidence-based, favors proven solutions |
| **Balthasar** | Pragmatist | Practicality and maintainability | Grounded, trade-off oriented, advocates for the team |
| **Caspar** | Critic | Risk, edge cases, and failure modes | Adversarial by design, finds what others miss |

---

## Installation

### From GitHub (for users)

```bash
# 1. Add this repo as a marketplace source
/plugin marketplace add BolivarTech/magi-claude

# 2. Install the plugin
/plugin install magi@bolivartech-plugins

# 3. Use it
/magi
```

To update after new versions are published:

```bash
/plugin marketplace update
```

### Local Development

```bash
# Option 1: Plugin flag
claude --plugin-dir /path/to/magi-claude

# Option 2: Symlink for auto-discovery (no flags needed)
mkdir -p .claude/skills
ln -s ../../skills/magi .claude/skills/magi
claude
```

Changes are picked up with `/reload-plugins` without restarting.

---

## Usage

Invoke with `/magi` or natural trigger phrases:

```
MAGI review this code
Give me three perspectives on this design
MAGI analysis of this problem
```

### Modes

| Mode | When to Use | Example |
|------|-------------|---------|
| `code-review` | Reviewing code or diffs | "MAGI review this PR" |
| `design` | Evaluating architecture decisions | "MAGI analyze this migration plan" |
| `analysis` | General problem analysis, trade-offs | "MAGI should we use Redis or Postgres for this?" |

### CLI (Direct Execution)

```bash
python skills/magi/scripts/run_magi.py <mode> <file_or_text> [--model opus] [--timeout 300] [--output-dir <dir>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `opus` | LLM model for all agents (`opus`, `sonnet`, `haiku`) — Claude backend |
| `--ollama` | off | Use the OpenAI-compatible **Ollama** backend (distinct model per mage) instead of `claude -p`. Mutually exclusive with `--model`. See [`docs/ollama-backend.md`](docs/ollama-backend.md) |
| `--ollama-init` | — | Scaffold `./.claude/magi-ollama.toml` from defaults and exit |
| `--timeout` | `300` | Per-agent timeout in seconds |
| `--output-dir` | auto | Directory for agent outputs (default: temp dir) |
| `-o`, `--out` | — | Redirect the human-readable verdict report to a file (suppressing it on stdout); on a write failure it warns and falls back to stdout so the verdict is never lost |

---

## How It Works

```
User input
  |
  v
SKILL.md (complexity gate + mode detection)
  |
  v
run_magi.py launches 3x claude -p (parallel, async)
  |               |               |
  v               v               v
Melchior        Balthasar       Caspar
(Scientist)     (Pragmatist)    (Critic)
  |               |               |
  v               v               v
parse_agent_output.py (extract the verdict: Claude envelope, or bare/fenced content)
  |               |               |
  v               v               v
validate.load_agent_output() (schema validation)
  |
  v
consensus.determine_consensus() (weight-based scoring)
  |
  v
reporting.format_report() (banner + markdown report)
```

### Step by Step

1. **Complexity gate** — Simple questions are answered directly without invoking three agents.
2. **Parallel dispatch** — Three sub-agents run concurrently via `asyncio` + `claude -p`, each with a distinct system prompt written to a temp file.
3. **Independent analysis** — Each agent evaluates the same input through its unique lens and produces a structured JSON verdict.
4. **Validation** — Each agent's output is parsed and validated against the [agent JSON schema](#agent-json-schema).
5. **Weight-based vote** — The consensus engine computes a weighted score, deduplicates findings, and generates a consensus report.

### Consensus Rules

Verdicts are weighted: `approve = 1`, `conditional = 0.5`, `reject = -1`.

```
score = sum(weight[verdict] for each agent) / num_agents
```

| Score | Consensus |
|-------|-----------|
| 1.0 (unanimous approve) | **STRONG GO** |
| -1.0 (unanimous reject) | **STRONG NO-GO** |
| > 0 with conditionals | **GO WITH CAVEATS** |
| > 0 without conditionals | **GO (N-M)** |
| <= 0 | **HOLD (N-M)** |

Labels are dynamic: `(N-M)` reflects the actual majority/minority split (e.g., `GO (2-1)` or `HOLD (1-1)` in degraded mode).

### Confidence Formula

```
weight_factor = (abs(score) + 1) / 2    # symmetric for approve and reject
base_confidence = sum(majority_confidence) / num_agents
confidence = base_confidence * weight_factor
```

Using `abs(score)` ensures that both unanimous approve and unanimous reject produce high confidence. At `score = 0` (exact tie), confidence is halved — appropriate for an undecided split.

### Output Example

```
+==================================================+
|          MAGI SYSTEM -- VERDICT                  |
+==================================================+
|  Melchior (Scientist):   APPROVE (90%)           |
|  Balthasar (Pragmatist): CONDITIONAL (85%)       |
|  Caspar (Critic):        REJECT (78%)            |
+==================================================+
|  CONSENSUS: GO WITH CAVEATS                      |
+==================================================+

## Key Findings
[!!!] **[CRITICAL]** SQL injection in query builder _(from melchior, caspar)_
[!!]  **[WARNING]**  Missing retry logic for API calls _(from balthasar)_
[i]   **[INFO]**     Consider adding request timeout _(from caspar)_

## Dissenting Opinion
**Caspar (Critic)**: Risk of data loss outweighs shipping speed...

## Conditions for Approval
- **Balthasar**: Add integration tests before merge

## Recommended Actions
- **Melchior** (Scientist): Fix SQL injection, add parameterized queries
- **Balthasar** (Pragmatist): Ship after adding integration tests
- **Caspar** (Critic): Rework query layer before proceeding
```

### Degraded Mode

When an agent fails (timeout, parse error, validation error):
- Warning printed to stderr identifying the failed agent and reason
- Synthesis proceeds if >= 2 agents succeeded
- Report flagged with `"degraded": true` and `"failed_agents": [...]`

### Fallback Mode

When `claude -p` is unavailable, the skill simulates all three perspectives sequentially within a single response, with **Caspar first** to reduce anchoring bias.

---

## Agent JSON Schema

All agents must produce output matching this schema:

```json
{
  "agent": "melchior | balthasar | caspar",
  "verdict": "approve | reject | conditional",
  "confidence": 0.0-1.0,
  "summary": "One-line verdict summary",
  "reasoning": "Detailed analysis (2-5 paragraphs)",
  "findings": [
    {
      "severity": "critical | warning | info",
      "title": "Short title (non-empty)",
      "detail": "Explanation"
    }
  ],
  "recommendation": "What this agent recommends"
}
```

---

## Project Structure

```
.claude-plugin/
  plugin.json                 -- Plugin manifest (name, version, author, repository)
  marketplace.json            -- Local marketplace config for development
skills/magi/
  SKILL.md                    -- Orchestrator (mode detection, model selection, workflow, fallback)
  agents/
    melchior.md               -- Scientist system prompt
    balthasar.md              -- Pragmatist system prompt
    caspar.md                 -- Critic system prompt (adversarial by design)
  scripts/
    __init__.py               -- Python package marker
    run_magi.py               -- Async orchestrator with --model flag
    synthesize.py             -- Facade: re-exports from validate, consensus, reporting
    validate.py               -- ValidationError + load_agent_output schema validation
    consensus.py              -- VERDICT_WEIGHT + determine_consensus (weight-based scoring)
    reporting.py              -- AGENT_TITLES + format_banner + format_report (ASCII)
    parse_agent_output.py     -- transport unwrap + verdict extraction between the markers
tests/
  test_synthesize.py          -- validation, consensus, confidence, dedup, labels
  test_verdict_markers.py     -- the sentinel: extraction, both marker predicates, hypothesis
  test_parse_agent_output.py  -- transport envelopes, the delimited block, fail-closed paths
  test_prompt_guard.py        -- the installation-time guard and its dry run
  test_run_magi.py            -- args, orchestration, rotation, retry feedback, telemetry
docs/
  MAGI-System-Documentation.md  -- Full technical reference (Spanish)
pyproject.toml                -- Python >= 3.12, dual license, dev deps, tool config
conftest.py                   -- tdd-guard pytest plugin + sys.path setup
Makefile                      -- verify, test, lint, format, typecheck targets
```

### Module Architecture

The synthesis engine is split into focused, single-responsibility modules:

| Module | Responsibility | Key Exports |
|--------|---------------|-------------|
| `validate.py` | Schema validation | `ValidationError`, `load_agent_output` |
| `consensus.py` | Weight-based scoring | `VERDICT_WEIGHT`, `determine_consensus` |
| `reporting.py` | ASCII banner + markdown report | `format_banner`, `format_report` |
| `synthesize.py` | Facade (re-exports all above) | All public symbols |

**Import convention:** Always import from `synthesize` (the facade), not directly from sub-modules:

```python
from synthesize import load_agent_output, determine_consensus, format_report
```

---

## Running Tests

```bash
# All tests (994)
python -m pytest tests/ -v

# Full verification (tests + lint + format + types)
make verify

# Individual checks
make test        # pytest
make lint        # ruff check
make format      # ruff format --check
make typecheck   # mypy
```

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Claude Code CLI (`claude -p`) | For parallel mode | Fallback available without it |
| Python 3.12+ | Yes | Uses `asyncio`, `dict[str, Any]` syntax |
| [`uv`](https://docs.astral.sh/uv/) | **To develop** (not to use) | Every `make` target runs through `uv run`, so the toolchain comes from `uv.lock` instead of from whichever venv is active. Using the plugin needs none of this. |

### Dev Dependencies

```bash
uv sync          # installs the toolchain pinned in uv.lock
make verify      # lockcheck + tests + ruff check + ruff format --check + mypy
```

**`uv` is required for every `make` target** (each one runs through `uv run`). That is
deliberate: with a bare `pip install`, the toolchain you get depends on which venv is
active, and two unpinned toolchains gave *opposite verdicts on the same code* — `mypy`
1.x reports a `no-any-return` that 2.x does not. A gate that answers differently
depending on which shell you are in is not a gate. `uv sync` also brings in `hypothesis`,
which the property-based tests need.

---

## License

Dual licensed under [MIT](LICENSE) OR [Apache-2.0](LICENSE-APACHE), at your option.

---

## Credits

The MAGI concept originates from [*Neon Genesis Evangelion*](https://en.wikipedia.org/wiki/Neon_Genesis_Evangelion) (1995) by Hideaki Anno / Gainax. The three supercomputers — Melchior, Balthasar, and Caspar — govern critical decisions through structured consensus, each embodying a different facet of their creator Dr. Naoko Akagi.

This plugin is a creative adaptation of that multi-perspective decision-making philosophy for software engineering, where the three "facets" become three analytical lenses: technical rigor, pragmatism, and adversarial risk assessment.
