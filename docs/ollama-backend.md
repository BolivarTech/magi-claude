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
| `structured` | `MAGI_OLLAMA_STRUCTURED` → repo → global → `"schema"` |

Example `magi-ollama.toml`:
```toml
base_url = "http://localhost:11434/v1"   # OpenAI-compatible base (any path is honored verbatim; a bare host:port gets /v1)
# api_key = "sk-..."                       # cloud/auth only; local does not need it
[models]
melchior  = "qwen3.5:397b-cloud"
balthasar = "gpt-oss:120b-cloud"
caspar    = "deepseek-v4-pro:cloud"
# structured = "schema"                    # "schema" | "object" | "off"
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

### Structured output + reliability
The request uses `response_format` (JSON schema, `strict:false` for portability). If a
server responds 400 rejecting `response_format`, MAGI does **one retry without it**
(downgrade) and relies on the existing parser+retry. `structured="off"` disables it.

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
| **Balthasar** | Pragmatist | `gpt-oss:120b-cloud` | OpenAI |
| **Caspar** | Critic | `deepseek-v4-pro:cloud` | DeepSeek |

> ⚠️ **Model recommendations current as of 2026-06-07.** The Ollama catalog
> changes over time; **re-verify the current tags** at
> [ollama.com](https://ollama.com/search?c=cloud) before pinning them. The defaults
> are an easily updatable constant (`DEFAULT_MODELS` in `ollama_config.py`).

### 3 recommended tiers (as of 2026-06-07)

| Tier | Melchior | Balthasar | Caspar | Hardware |
|------|----------|-----------|--------|----------|
| **Light** | `qwen3:8b` | `gpt-oss:20b` | `deepseek-r1:8b` | 1 GPU ~16-24 GB (may serialize) |
| **Balanced** | `qwen3:32b` | `gpt-oss:20b` | `deepseek-r1:32b` | ~48-64 GB |
| **Maximum (default)** | `qwen3.5:397b-cloud` | `gpt-oss:120b-cloud` | `deepseek-v4-pro:cloud` | Ollama Cloud (`ollama signin`) or 80 GB+ |

Consistent lineage per role across tiers: **Caspar = DeepSeek** (the strongest
reasoner), **Balthasar = OpenAI** (gpt-oss, adjustable effort), **Melchior = Qwen**
(Alibaba). The Maximum tier uses `:cloud` tags (no weight download, requires
`ollama signin`).

> VRAM/concurrency note: MAGI launches the 3 agents in parallel; if the models do
> not coexist in VRAM, Ollama serializes them. Size the trio to your budget, or use
> the Maximum tier in the cloud.

---

## 4. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `Cannot reach Ollama at <host>` | daemon down / wrong host | start Ollama; check `base_url`/`OLLAMA_HOST` |
| `No :cloud models available ... Run ollama signin` | `:cloud` trio without a cloud session | `ollama signin` (mode A) or set `api_key`+cloud base_url (mode B) |
| `Missing models on <host>: [...]` | local models not pulled | `ollama pull <model>`, or edit the TOML to the tier you have |
| `Auth failed (401/403)` | invalid/absent api_key | fix `MAGI_OLLAMA_API_KEY` / the TOML |
| 400 + degraded on every run | server does not support `response_format` | use `structured="object"` or `"off"` |
| `--model does not apply with --ollama` | you passed both | drop `--model`; models go in the TOML/env |

---

## 5. References

- Implementation: `skills/magi/scripts/ollama_backend.py`, `ollama_config.py`,
  `ollama_preflight.py`, `ollama_init.py`, `backend.py`, `claude_backend.py`,
  `agent_schema.py`.
- Ollama cloud catalog: <https://ollama.com/search?c=cloud>
