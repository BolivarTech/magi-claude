# MAGI Ollama Backend (`--ollama`)

> **Opt-in.** Without `--ollama`, MAGI behaves exactly as before (Claude CLI via
> `claude -p`). This backend changes **only** the engine of MAGI's *analysis gate*;
> code generation stays in Claude Code. Available since **v4.0.0**.

---

## 1. Why an Ollama backend for MAGI

MAGI is not code generation: it is **ensemble judgment** — three perspectives
(Melchior/Balthasar/Caspar) analyzing the same input and voting by majority. That
shifts the cost-vs-token calculus compared to coding:

- **Cost:** one MAGI cycle ingests spec+plan+diff ×3 agents + reasoning +
  synthesis (~150–250k tokens), and it iterates as a recurring gate. It is the
  pattern where per-token cost hurts. Local Ollama ≈ 0 marginal; Ollama Cloud ~$20 flat.
- **Real diversity:** three instances of the *same* frontier model share the same
  blind spots. With Ollama you run **three open-weight models of distinct lineages**
  as the three mages → genuine architecture/training diversity, exactly the spirit
  of MAGI. Majority voting over heterogeneous models is more robust, and human
  review acts as a backstop.

| Task | Backend |
|-------|---------|
| Generate code | **Claude Code** (frontier) — unchanged |
| MAGI analysis gate | **Claude** (default) or **Ollama** (`--ollama`, opt-in) |

---

## 2. How to use it

### Activation
```bash
# Skill:
/magi --ollama
# Direct orchestrator:
python skills/magi/scripts/run_magi.py <code-review|design|analysis> <file_or_text> --ollama [--timeout 900]
```
- `--ollama` is **mutually exclusive** with `--model` (per-mage models come from
  the config, not the CLI). Passing both is an error.
- Each mage uses its **own model** (see §3).

### Scaffolding the config
```bash
python skills/magi/scripts/run_magi.py --ollama-init
```
Generates `./.claude/magi-ollama.toml` (does not overwrite if it already exists) as
an editable template, with the local `base_url` active, the default trio, and a
2-mode header.

### Configuration (layered, per-key merge)
Precedence (first present wins): **env > repo `./.claude/magi-ollama.toml` >
global `~/.claude/magi-ollama.toml` > built-in defaults**.

| Key | Chain |
|---|---|
| `base_url` | `MAGI_OLLAMA_HOST` → repo → global → `OLLAMA_HOST` → `http://localhost:11434/v1` |
| `api_key` | `MAGI_OLLAMA_API_KEY` → repo → global → `OLLAMA_API_KEY` → *(none)* |
| `models.<mage>` | `MAGI_OLLAMA_MODEL_<MAGE>` → repo `[models]` → global `[models]` → default trio |

Example `magi-ollama.toml` (minimal — just `base_url`/`api_key`/`models`; see §3b below for
the full schema with `lineage` and `[[fallback]]`, required since **v5.0.0**):
```toml
base_url = "http://localhost:11434/v1"   # OpenAI-compatible base (any path is honored verbatim; a bare host:port gets /v1)
# api_key = "sk-..."                       # cloud/auth only; local does not need it
[models]
melchior  = { model = "qwen3.5:397b-cloud",   lineage = "alibaba"  }
balthasar = { model = "kimi-k2.6:cloud",      lineage = "moonshot" }
caspar    = { model = "glm-5.2:cloud",        lineage = "zhipu"    }
```

> **Security:** the `api_key` is never logged nor written to artifacts, and it is
> redacted in error messages. Do not commit a TOML with a real key. An
> `MAGI_OLLAMA_API_KEY=""` (empty) means **explicitly no auth** (it does not inherit
> the one from the file) — useful in CI.

### Two cloud modes (same REST client)
- **A) Local daemon + `ollama signin`** *(recommended)*: local `base_url`, **no**
  `api_key`. The `:cloud` models run in Ollama's cloud **without downloading weights**
  (just a manifest); the daemon attaches your credentials. This is the default.
- **B) Direct cloud API**: cloud-endpoint `base_url` + `api_key`. For machines
  without a local daemon.

### Verdict extraction (v5.1.0 — replaces the old `structured`/`response_format` setting)
Reliability no longer comes from asking the server to constrain its own output. Through
**v5.0.x**, MAGI could request `response_format` (JSON schema) via the `structured` setting,
with a one-retry-without-it downgrade on a 400. **Both were removed in v5.1.0**: a model
constrained to raw JSON cannot also wrap its answer in the `<MAGI_VERDICT>` /
`</MAGI_VERDICT>` marker lines the verdict sentinel now requires (see
[`docs/faq-prompt-guard.md`](faq-prompt-guard.md) and the
[ADR](adr/0001-no-runtime-heuristic-fallback.md)) — the two mechanisms are mutually
exclusive, and the sentinel is the one that closes the fabrication residual, so it wins.
There is no `structured` key and no `MAGI_OLLAMA_STRUCTURED` environment variable to set
anymore; a `magi-ollama.toml` that still sets `structured` gets a `WARNING: unknown key
'structured' ... (ignored)` on stderr and otherwise runs normally — it is not an error, but
it no longer does anything. Reliability instead comes from the marker contract
itself, cause-specific retry feedback (`retry_feedback.py`), and — since fallback rotation
landed in v5.0.0 — rotating a mage to a declared substitute model rather than losing it.

**What to watch, and what to do about it.** Removing `response_format` means nothing but the
prompt now pushes a model toward well-formed output, so the honest expectation is that some
models drift more than they used to — and the drift is *visible* rather than silently
"corrected" by a parser that guesses. The counter is `extraction_failures` in
`magi-report.json`, broken down per mage and per cause (`missing_markers`,
`unterminated_block`, `ambiguous_markers`, `invalid_json`, `echoed_example`,
`agent_identity`, `schema`). If a run dies before it can write a report, the same counts go
to stderr, so the cause outlives the run.

| What you see | What it means | What to do |
|---|---|---|
| Occasional `missing_markers`, run still valid | the retry absorbed it — this is the system working | nothing |
| One seat above ~10% of attempts, persistently | that model is drifting from the marker contract | iterate that agent's prompt, or move the model down its `[[fallback]]` list and promote a better one |
| `unterminated_block` climbing | the model is being truncated | its window is too small for your payload — check `context_guard` in the report, raise `output_headroom_tokens`, or split the input |
| `echoed_example` | the model copied the prompt's worked example | the seat is degraded, not the parser — rotate the model |
| Every seat failing at once | this is not the models | check the endpoint before touching any prompt |

What is *never* the answer is restoring a parser that searches for a verdict instead of
reading the one the mage marked. That is not a safety valve, it is a switch that re-enables
the silent fabrication of an `approve` — see the
[ADR](adr/0001-no-runtime-heuristic-fallback.md).

### Preflight (fail-fast)
Before launching agents, MAGI verifies that the host responds and that the trio is
available. If the entire `:cloud` trio is missing and the daemon lists no `:cloud`
models, the message suggests `ollama signin`. A missing `/models` (404/501) →
warn-and-proceed.

---

## 3. Default models (what the plugin ships)

Default trio = **Maximum** tier (cloud), **three distinct lineages** on purpose; the
most capable model goes to **Caspar** (the critic is the highest-leverage seat —
"losing Caspar biases toward false-positive approval"):

| Mage | Role | Default model | Lineage |
|------|-----|----------------|--------|
| **Melchior** | Scientist | `qwen3.5:397b-cloud` | Alibaba |
| **Balthasar** | Pragmatist | `kimi-k2.6:cloud` | Moonshot |
| **Caspar** | Critic | `glm-5.2:cloud` | Zhipu |

> ⚠️ **Model recommendations current as of 2026-07-03.** The Ollama catalog
> changes over time; **re-verify the current tags** at
> [ollama.com](https://ollama.com/search?c=cloud) before pinning them. The defaults
> are an easily updatable constant (`DEFAULT_MODELS` in `ollama_config.py`).

### 3 recommended tiers (as of 2026-06-07)

| Tier | Melchior | Balthasar | Caspar | Hardware |
|------|----------|-----------|--------|----------|
| **Light** | `qwen3:8b` | `gpt-oss:20b` | `deepseek-r1:8b` | 1 GPU ~16-24 GB (may serialize) |
| **Balanced** | `qwen3:32b` | `gpt-oss:20b` | `deepseek-r1:32b` | ~48-64 GB |
| **Maximum (default)** | `qwen3.5:397b-cloud` | `kimi-k2.6:cloud` | `glm-5.2:cloud` | Ollama Cloud (`ollama signin`) or 80 GB+ |

Lineage per role: **Melchior = Qwen** (Alibaba). In the Light/Balanced tiers
**Balthasar = OpenAI** (`gpt-oss`, adjustable effort) and **Caspar = DeepSeek-R1**
(a strong local reasoner); the **default Maximum tier** instead runs
**Balthasar = `kimi-k2.6:cloud` (Moonshot)** and **Caspar = `glm-5.2:cloud`
(Zhipu)** — the latter after `deepseek-v4-pro:cloud` proved unreliable at
chat-time (timeouts/5xx). The three defaults keep distinct lineages
(Qwen / Moonshot / Zhipu). The Maximum tier uses `:cloud` tags (no weight download,
requires `ollama signin`).

> VRAM/concurrency note: MAGI launches the 3 agents in parallel; if the models do
> not coexist in VRAM, Ollama serializes them. Size the trio to your budget, or use
> the Maximum tier in the cloud.

---

## 3b. Fallback rotation (v5.0.0, BREAKING)

When a mage's active model exhausts its attempts — whether from **transport** failures
(HTTP 5xx, host unreachable, timeout) or **schema** drift (invalid/truncated JSON) — MAGI
rotates that mage to a declared substitute instead of dropping it to degraded mode. The
substitute is chosen from a per-run **`[[fallback]]`** list, and rotation is governed by
one rule: **one lineage, one mage** — a candidate is skipped if its lineage is already in
play, already failed this mage, condemned run-wide by a transport failure, or if its
context window cannot hold the payload. This preserves the three *independent* perspectives
that make the ensemble meaningful. A run where a mage rotated to a **declared** fallback is
**VALID** (not degraded); only a mage that finds no eligible candidate, or falls to an
arbitrary model, degrades the run.

### Schema (BREAKING)

```toml
[models]
# v4 was a bare string; v5 is a table with an explicit lineage (never inferred).
melchior  = { model = "qwen3.5:397b-cloud",    lineage = "alibaba"  }
balthasar = { model = "kimi-k2.6:cloud",        lineage = "moonshot" }
caspar    = { model = "deepseek-v4-pro:cloud",  lineage = "deepseek" }

# Ordered most -> least capable, one model per lineage, none sharing a trio lineage.
# With max_rotations = 2 only the first three are ever reached.
[[fallback]]
model   = "glm-5.2:cloud"
lineage = "zhipu"
[[fallback]]
model   = "gpt-oss:120b-cloud"
lineage = "openai"
[[fallback]]
model   = "minimax-m3:cloud"
lineage = "minimax"

# Rotation and context-window settings (apply to ALL mages; top-level, before [models]).
max_attempts_per_model    = 2     # tries per model before rotating to a fallback (>= 1)
max_rotations             = 2     # fallback models a mage may rotate through (0 disables rotation)
max_probe_attempts        = 3     # fallback candidates to size-check before a mage gives up (>= 1)
output_headroom_tokens    = 8192  # context tokens reserved for the model's answer plus its thinking
input_margin_pct          = 40    # extra margin when checking the input fits a model's window, percent
strict_context_guard      = true  # default true: abort if a window cannot be measured; false to estimate
strict_lineage            = false # if true, abort when a model's architecture contradicts its declared lineage
retry_backoff_seconds     = 2.0   # seconds to wait between transport retries (0 = no wait)
preflight_timeout_seconds = 30    # timeout for preflight metadata calls, seconds
probe_timeout_seconds     = 120   # timeout for the context-probe call, seconds
```

`--ollama-init` scaffolds this shape — **all nine settings are emitted as active keys at
their defaults**, so every knob (and the kill-switch) is visible and editable without
reading the docs. Editing any of them is optional; an untouched scaffold behaves exactly
as the built-in defaults. A **v4 config fails closed** with an actionable error;
**`python skills/magi/scripts/validate_magi_toml.py [path]`** reports exactly what to
change — it never guesses a lineage (two mages sharing a lineage would give a consensus
that only *looks* like three perspectives). It ships with the plugin, so it is there the
moment the fail-closed error tells you to run it; pointing it at a path that does not
exist is an error (exit 2), never an `OK` on defaults it silently fell back to.

### Settings — what each one does

These are top-level keys (they apply to all three mages) and must appear **before**
`[models]`. Every one has a safe default; change them only if you have a reason.

| Setting | Default | Plain-language meaning |
|---|---|---|
| `max_attempts_per_model` | 2 | How many times a mage retries the **same** model before it gives up and rotates to a fallback. |
| `max_rotations` | 2 | How many fallback models a mage may move through. **`0` turns rotation off** (the kill-switch). |
| `max_probe_attempts` | 3 | When rotating, how many candidate models to size-check (does the payload fit?) before the mage gives up. |
| `output_headroom_tokens` | 8192 | Context space reserved for the model's **answer plus its thinking**, so the reply is never cut off. Raise it for very verbose reasoning models. |
| `input_margin_pct` | 40 | Safety cushion (percent) when estimating whether the input fits a model's context window, for the models MAGI can only estimate rather than measure exactly. |
| `strict_context_guard` | **true** | Default `true` (v5.2.0): **abort** if a model's context window **cannot be measured**, instead of proceeding on an estimate. Set `false` to opt out and run with an estimated guard. |
| `strict_lineage` | false | If `true`, **abort** when a model's real architecture family (from `/api/show`) contradicts its declared `lineage`; default `false` makes a contradiction a non-fatal warning. See "Lineage identity" below. |
| `retry_backoff_seconds` | 2.0 | Seconds to wait between **connection/transport** retries (e.g. after a 503). `0` = no wait. Schema-error retries never wait. |
| `preflight_timeout_seconds` | 30 | Timeout for the small preflight metadata calls (`/models`, `/api/show`). |
| `probe_timeout_seconds` | 120 | Timeout for the context probe, which processes the **whole prompt** once — larger than the metadata timeout on purpose. |

### Kill-switch and shadow rollout

`max_rotations = 0` (or, without touching the TOML, **`MAGI_OLLAMA_MAX_ROTATIONS=0`**)
turns rotation off while leaving the new preflight, the context probe, and the schema
validation **active** — a shadow-rollout mode to validate the v5 config parsing and the
measurement paths in production before enabling live rotation.

### Context guard, the probe, and its cost

Before launching a single agent, the preflight **measures** the payload with each trio
model's own tokenizer (an exact `max_tokens=1` probe against `/chat/completions`) and reads
each model's context window from `/api/show`. A model whose payload would not fit its
window **does not run** (truncation produces a verdict indistinguishable from a legitimate
one). The probe processes the full prompt once per trio model (concurrently) — on cloud
models this is a real, if small, cost paid once per run. The report's **`context_guard`**
field is `"enforced"` only when the payload was measured **and** every window is known; it
reads `"estimated"` (and the banner says so) whenever measurement was not possible — for a
rotated mage too.

### Hard limitation — no `/api/show`, no window

On an endpoint that does **not** expose `/api/show` (a generic OpenAI-compatible server),
the window cannot be read. Since **v5.2.0** `strict_context_guard` defaults to **`true`**, so
this now **fails closed**: the preflight aborts (and rotation skips such a model) rather than
run on an unverified estimate. Set **`strict_context_guard = false`** to opt back into the
old behaviour — a noisy warning names the affected models and the run proceeds on the
estimator, with **no truncation protection at all** for those models.

### Lineage identity (`strict_lineage`, and the digest check)

Three **distinct lineages** is the whole point of the ensemble. MS1 already enforces that the
declared `lineage` strings differ per mage. **v5.2.0** adds two deeper identity checks:

- **Family check (`strict_lineage`, opt-in, default `false`):** when `true`, MAGI reads each
  model's real architecture family from `/api/show` and compares it to the declared `lineage`;
  a contradiction (e.g. `lineage = "deepseek"` but the architecture maps to another vendor)
  **aborts** the run. With the default `false`, a contradiction is a non-fatal **warning** in
  `lineage_warnings`, not an abort. The architecture-to-vendor map is deliberately
  **non-exhaustive** — only unambiguous vendor-specific families (e.g. `qwen3.5 -> alibaba`,
  `deepseek4 -> deepseek`); ambiguous bases like `llama`/`mistral` are excluded. An unmapped or
  unknown architecture **never blocks** — it fails **open** (the TOML declaration always wins;
  the map is a typo detector, never an authority). Adding entries as new models appear is a
  maintenance point.

- **Digest uniqueness (always on):** two mages must never resolve to the **same model digest**
  — running the same model twice is forbidden on purpose, because it would collapse the
  ensemble into a single opinion wearing three hats. This is checked in the preflight and again
  at every rotation commit. **Cloud caveat (honest):** the `:cloud` trio's `/api/show` does
  **not** report a digest, so for cloud models the digest check has nothing to compare and
  **degrades gracefully** to the lineage-string uniqueness (MS1) plus the family check above;
  the digest check is only fully active for **local** models that report a digest. A *non-cloud*
  model that unexpectedly omits its digest **fails closed** (uniqueness cannot be proven).

```toml
strict_lineage = true   # opt in: a family contradiction aborts, not just warns
```

### Telemetry (never a silent fallback)

Every rotation is announced on **stderr** at the moment it happens, marked in the report
**banner** (`[fallback: <model>]`), and recorded in `magi-report.json` per agent
(`model_configured`, `model_used`, `rotations`, structured `fallback_reason`) and per run
(`fallback_agents`, `context_guard`, `lineage_warnings`, `token_estimate_delta`).

---

## 4. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `Cannot reach Ollama at <host>` | daemon down / wrong host | start Ollama; check `base_url`/`OLLAMA_HOST` |
| `No :cloud models available ... Run ollama signin` | `:cloud` trio without a cloud session | `ollama signin` (mode A) or set `api_key`+cloud base_url (mode B) |
| `Missing models on <host>: [...]` | local models not pulled | `ollama pull <model>`, or edit the TOML to the tier you have |
| `Auth failed (401/403)` | invalid/absent api_key | fix `MAGI_OLLAMA_API_KEY` / the TOML |
| `--model does not apply with --ollama` | you passed both | drop `--model`; models go in the TOML/env |
| A mage is dropped although its `*.raw.json` looks like a valid verdict | the model wrapped its JSON in a markdown fence or a `<think>` block | **fixed in 4.0.6** — the parser now reads the raw file as text before assuming an envelope. If you still see it on 4.0.6+, keep the `raw.json`: it is a parser bug, not a config problem |
| `[FATAL]` at startup naming a file under `agents/` | one of the three shipped agent prompts has malformed `<MAGI_VERDICT>` markers, or a complete verdict sitting between them | see [`docs/faq-prompt-guard.md`](faq-prompt-guard.md) — every message explained, with the fix |
| `extraction_failures` non-empty in `magi-report.json` (v5.1.0+) | a model omitted the marker lines, emitted more than one block, or copied the prompt's worked example | check the per-cause counts; a model that fails this repeatedly needs a prompt iteration or a model swap, not a config change (see the ADR at `docs/adr/0001-no-runtime-heuristic-fallback.md`) |

---

## 5. References

- Implementation: `skills/magi/scripts/ollama_backend.py`, `ollama_config.py`,
  `ollama_preflight.py`, `ollama_init.py`, `backend.py`, `claude_backend.py`,
  `fallback_policy.py`, `model_context.py`.
- Verdict extraction (v5.1.0): `skills/magi/scripts/verdict_markers.py`,
  `prompt_guard.py`, `retry_feedback.py` — see
  [`docs/faq-prompt-guard.md`](faq-prompt-guard.md) and
  [`docs/adr/0001-no-runtime-heuristic-fallback.md`](adr/0001-no-runtime-heuristic-fallback.md).
- Ollama cloud catalog: <https://ollama.com/search?c=cloud>
