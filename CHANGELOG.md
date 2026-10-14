# Changelog

All notable changes to MAGI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Install and update through the marketplace: `/plugin marketplace add BolivarTech/magi-claude` →
`/plugin install magi@bolivartech-plugins`, then `/plugin marketplace update` to pick up a new
release.

## [7.0.1] - 2026-10-14

### Fixed

- A `degraded` flag from `magi-rs` that is not a JSON boolean now reads as degraded. It used to
  be read by truthiness, so an off-contract `[]` or `null` reported a healthy run. An absent
  flag still reads as healthy, and `magi-rs` itself always sends a real boolean. The DEGRADED
  warning on stderr names that cause ("magi-rs sent a degraded flag that is not a JSON
  boolean") instead of blaming a seat for delivering no verdict. `magi_rs.degraded` in
  `magi-report.json` still carries the value exactly as `magi-rs` sent it.

### Documentation

- `docs/ollama-backend.md` section 5 now describes `capped_agents` the way the code computes
  it: any seat whose last attempt ended with something other than a clean `stop`, which is
  usually the completion cap. An unknown finish reason is listed rather than passed over; an
  attempt with no finish reason at all, such as a timeout, is not.

## [7.0.0] - 2026-10-14

### Breaking

- `.magi/` has to be regenerated. The pinned `magi-rs` 0.24.1 refuses a `.magi/magi.toml` that
  has no `[decision]` section (required since `magi-rs` 0.23.0), and every `.magi/` created
  with plugin 6.x lacks it. An `--ollama` run against such a project stops with `magi-rs`'s
  message (`... has no [decision] section: it was written by magi-rs 0.22.0 or earlier`).
  `magi-rs init` never edits an existing file, so from the project root: move `.magi` aside
  (`mv .magi .magi.bak-0.21`), run `--ollama-init`, and copy your trio, lineages, fallback pool
  and `max_rotations` from the old `magi.toml` into the new one. Do not restore the old
  `.magi-rs-memory.db`: `magi-rs` 0.22.0 no longer supports databases written by 0.21.0 or
  earlier, and the plugin never reads it. Steps in `docs/ollama-backend.md`, section 2.
- The migration is one-way. Plugin 6.2.0 (on `magi-rs` 0.21.0) rejects the regenerated file (``unknown field
  `decision` ``), so going back to 6.2.0 means restoring the backup.

### Changed

- The pinned `magi-rs` moves from v0.21.0 to **v0.24.1**, on the same four platforms
  (`windows-x86_64`, `linux-x86_64`, `linux-aarch64` (rpi5), `macos-universal2`). Archive
  digests are verified three ways and the archive layout is unchanged.
- Nothing else changes on the plugin side: same flags, same defaults, same report. Measured on
  2026-10-13 with 0.24.1, `--timeout 16820` still derives a per-mage ceiling of 2335 s, an
  operation budget of 1401 s per attempt and a 700 s per-request clock, and the `consult` JSON
  has exactly the shape the 6.2.0 fixtures pin.
- The `[decision]` section that `--ollama-init` now writes configures `magi-rs`'s decision
  models (`clef-flash`, then `nimble`). The plugin never calls them, because it always passes
  the mode and runs with `--no-memory`, so they do not need to be pulled.
- What `magi-rs` 0.22.0 to 0.24.1 bring, inherited with the binary: SQLite 3.53.2; the `decide`
  command; per-section API keys (`MAGI_API_KEY`, `EMBEDDING_API_KEY`, `DECISION_API_KEY`) and
  the main key withheld from other origins; a grouped `magi init` scaffold. The plugin passes
  no key (its child environment does not carry `MAGI_API_KEY`), and an Ollama trio on the root
  `base_url` needs none. See the
  [`magi-rs` releases](https://github.com/BolivarTech/magi/releases) for details.

### Fixed

- Under `--ollama`, each verdict was written to `{agent}.json` using the `agent` field of the
  `magi-rs` output unchecked, so a value such as `../x` wrote a file outside the run directory
  before validation rejected it. Only `melchior`, `balthasar` and `caspar` name the file now;
  any other value is written as `agent-<index>.json` and fails validation as before.

## [6.2.0] - 2026-09-29

### Changed

- The pinned `magi-rs` moves from v0.19.1 to **v0.21.0** (`magi-core` `=4.1.0` to `=4.2.0`),
  on the same four platforms: `windows-x86_64`, `linux-x86_64`, `linux-aarch64` (rpi5) and
  `macos-universal2`. `linux-x86_64-compat` and the `.7z` archives stay out of the pin.
- Under `--ollama`, `--timeout` now defaults to **16820** (was 1800). It is the value
  `magi-rs` 0.21.0 publishes for a headless gate: it derives a per-mage ceiling of 2335 s, an
  operation budget of 1401 s per attempt and a 700 s per-request clock (confirmed locally on
  2026-09-29). The Claude path keeps 900. A default `--ollama` run measured on 2026-09-29 took
  about 2 minutes on a 75 KB bundle; a run where seats rotate through every model can take
  hours (worst case about 4.7 h).
- The `--timeout` warning threshold under `--ollama` moves from 1200 to 16820. Any explicit
  positive value is still honoured; below 16820 the warning says a lower wall timeout also
  cuts every seat's reasoning budget.
- `magi-rs` 0.21.0 raises its completion cap from 16 384 to **65 536** tokens. It is adjustable
  with `max_tokens` under `[magi]` in your `.magi/magi.toml`; the plugin does not validate that
  file's model settings. A model whose maximum output is below the cap answers HTTP 400 and its
  lineage is dropped for the run. Measured on 2026-09-29: `kimi-k2.6:cloud`, `glm-5.2:cloud`,
  `gpt-oss:120b-cloud`, `mistral-large-3:675b-cloud` and `gemma4:cloud` all accepted it.
- Each record of `consult.completions` gains two keys, `reasoning` and `control`, and they
  travel verbatim into the `magi_rs` block of `magi-report.json`. `schema_version` is unchanged.
- To get the 6.1.0 behaviour back: `--timeout 1800`, and in `[magi]` of `.magi/magi.toml`,
  `max_tokens = 16384` and `agent_timeout_secs = 90`.
- `magi-rs` 0.20.0 added four `[magi]` keys (`reasoning`, `reasoning_spelling`, `max_tokens`,
  `reasoning_trace`); see `docs/REASONING-BUDGET.md` in the `magi-rs` repository.
- `skills/magi/SKILL.md` now tells the agent to launch the `--ollama` orchestrator in the
  background and wait for its completion notification, and how to find and stop a leftover
  `magi-rs` process.
- `tools/measure_marker_adherence.py` uses the plugin's `--ollama` default (16820) for
  `--ollama-bundle` runs when its own `--timeout` is omitted, instead of 900. Claude bundles
  keep 900.

### Added

- A `WARNING` line, printed after the run, when `magi-rs` flags the `--timeout` as a probable
  typo (a derived per-mage ceiling above 2400 s, for example 168200 instead of 16820). The
  plugin cannot warn before the launch because the derivation is internal to `magi-rs`, and it
  also depends on `max_rotations`: with `max_rotations = 0` even 16820 is flagged, and a
  flagged value that is not above 16820 gets a warning pointing at `max_rotations` instead.
- The completion-cap warning now names the reasoning state of the cut attempt: `Measured N
  chars`, `NotMeasured`, `Unsupported (<backend>, N chars)`, `not reported` or `unknown
  (<variant or type>)`. It never prints model text.
- The `magi_rs` block of every `--ollama` report records `timeout_seconds` (the `--timeout`
  passed to `magi-rs`) and `ceiling_above_sanity` (`applied_caps.ceiling_above_sanity`: `true`,
  `false`, or `null` when `magi-rs` published no boolean).

### Fixed

- When `magi-rs` exits non-zero, the error message shows the tail of its stderr (the last 2000
  characters, prefixed with `[... earlier stderr elided ...]` when cut) instead of the head.
  `magi-rs` writes its `WARN` lines at startup, so the cause comes last.
- A segmented `--ollama` run refreshes its run lock before every segment, so a concurrent
  `--keep-runs` cleanup can no longer remove a long run that is still alive.

### Security

Inherited with `magi-rs` 0.20.0 (details in its
[release notes](https://github.com/BolivarTech/magi/releases/tag/v0.20.0)):

- A credential in `base_url` is redacted in the connection-failure hint.
- Memory-distillation failures go through the audited logging path.
- `Authorization: Basic` is masked.
- OAuth login no longer echoes the code, verifier or token.
- `-p` accepts values that start with `-`.
- The passphrase-strength floor applies to every envelope bootstrap, including the first open
  of a `.magi/` scaffolded by `init` without a passphrase under headless `consult`. The plugin
  runs without a passphrase and was verified to keep working.

## [6.1.0] - 2026-09-20

### Removed

- The 5.x Ollama HTTP client and everything that existed to sustain it: `ollama_backend.py`,
  `ollama_preflight.py`, `fallback_policy.py`, `model_context.py`, `lineage_identity.py`,
  `backoff.py`, `redaction.py`, `ollama_config.py` and `validate_magi_toml.py`, together with
  the per-mage backend abstraction (`backend.py`, `claude_backend.py`) that `ClaudeTrio`
  replaced in 6.0.0. Nothing in it has been reachable from `--ollama` since 6.0.0: rotation,
  context guard, backoff and redaction all belong to `magi-rs` now. The migration
  command `validate_magi_toml.py` goes with the schema it migrated: there is no plugin TOML
  left to migrate.
- `--ollama-init` no longer carries the 5.x template code. It is one `magi-rs init` call,
  exactly as documented in 6.0.0.

### Changed

- `--max-attempts` passed together with `--ollama` now says that it has no effect on that
  path (`magi-rs` owns every retry and rotation from `.magi/magi.toml`). It used to claim a
  TOML override that no longer existed.

### Documentation

- `docs/ollama-backend.md` gains the two migration notes that only a document can carry: a
  `magi.toml` written by a `magi-rs` older than 0.18.0 is refused over its `[headless]` log
  keys, and the `linux-x86_64-compat` build is not in the pin.

## [6.0.0] - 2026-09-19

### Changed (BREAKING)

- `--ollama` is executed by **`magi-rs`** (pinned v0.19.1). The plugin builds the payload as
  before and hands it to one `magi-rs consult` call, which runs the three mages and returns
  three structured verdicts that enter the existing consensus and rendering unchanged. The
  flag keeps its name and its meaning; who runs the trio changed.
- `.claude/magi-ollama.toml` is no longer read, validated or mentioned. The trio, its
  fallbacks, its context guard and the endpoint live in `.magi/magi.toml`, created by
  `--ollama-init`, which now does exactly one thing: call `magi-rs init -w <cwd>`. Delete the
  old file when you like; the environment chain (`MAGI_OLLAMA_*`, repo, global) went with it.
- `--timeout` on the open-weight path defaults to 1800 and also sets each mage's per-attempt
  budget inside `magi-rs`. Values below 1200 are honoured with a stderr warning.

### Added

- The bootstrap: on first use the plugin downloads the `magi-rs` release asset for the host
  platform into its own `bin/`, verifies its sha256 against `skills/magi/magi-rs.pin.json`,
  extracts it (every archive member validated; size-bounded), leaves exactly one executable,
  and re-verifies digest and `--version` before every run. Nothing unverified is executed.
- The child process runs with an allowlisted environment: platform, proxy, TLS/CA variables
  and `OLLAMA_HOST` only. No passphrase, no credential and no dynamic-linker variable reaches
  `magi-rs`, so the same command takes the same path whoever launches it. Process failures
  name the allowlist as a suspect and list the variables it withheld.
- `magi-report.json` gains a `magi_rs` block with the `consult` telemetry verbatim plus
  `capped_agents`, `model_used`, `run_id` and `gate_ran_on_unverified_context`; the raw
  envelope is saved as `magi-rs-raw.json`. The cost block reads `unpriced` (`total_usd: null`)
  because `magi-rs` exposes no usage.
- Four stderr warnings, silent on a clean run: a degraded run (the gate is invalid), a seat
  whose accepted attempt hit the completion cap, seats that ran with an unmeasured context
  window (with the remedy), and upstream extraction failures.
- Bundle segmentation: input above 65 % of `magi-rs`'s 262 144-byte ceiling is split at
  markdown headings (up to `###`) and `diff --git` boundaries, the map is written before the
  first segment runs, segments run one at a time into `segment-N/`, every verdict is tabled
  verbatim and the gate is approved only when all pass. A section too large on its own is
  refused, never truncated.
- Fail-closed errors at the CLI boundary are one `[FATAL]` or `ERROR:` line, never a
  traceback.

### Not yet removed

- The 5.x Ollama HTTP client (`ollama_backend.py`, `ollama_preflight.py`, the rotation and
  context-guard modules, `ollama_config.py`, `validate_magi_toml.py`) is still in the tree but
  is no longer reached by `--ollama`. It is deleted in the next release, after this one has
  been installed from the marketplace and the bootstrap has survived the update. (Done in
  6.1.0.)

## [5.4.0] - 2026-07-30

### Changed

- The Claude path now defaults to **Opus 5**. Model names resolve to floating aliases without a
  date suffix: `opus` → `claude-opus-5`, `sonnet` → `claude-sonnet-5`, `haiku` → `claude-haiku-4-5`.
  All three modes still default to `opus`; per-mode defaults are unchanged.
- The three agent prompts no longer ask the mage to verify or re-check its own output. Opus 5 does
  that on its own, and instructing it to caused redundant work rather than added rigor.

### Added

- `magi-report.json` carries a `model` block — `{backend, short_name, source, resolved, served,
  unattributed}` — so a saved verdict says which model produced it. The shape is identical for both
  backends (Ollama fields are explicitly `null`, never absent), so consumers never have to branch.
- `served` is read back out of each agent's raw response, not assumed from the requested name.
  Because every model id is now a floating alias, the requested name alone cannot tell two servings
  of `claude-opus-5` apart — which is the drift the block exists to expose.
- `unattributed` lists agents whose model identity could not be confirmed, and a Claude-path run
  that attributes none of them warns on stderr. Silence about a total attribution failure would be
  its own kind of lie.

## [5.3.0] - 2026-07-16

### Added

- Exponential retry backoff for transient HTTP failures (`408`/`429`/`500`/`502`/`503`/`504`) on the
  Ollama path, capped by the new `retry_backoff_max_seconds` (default 60).
- A server-sent `Retry-After` header now wins over the computed backoff, in both delta-seconds and
  HTTP-date form, capped by the new `retry_after_max_seconds` (default 300) so a hostile or buggy
  value cannot hang a run.
- Scalar `timeout` key in the TOML (default 900). Precedence is `--timeout` > TOML > default.

### Changed

- A timeout, a connection reset, or any retryable transport failure that is not a known transient
  HTTP status uses the **flat** backoff, not the exponential one: waiting does not clear a network
  partition, and growing the wait would only inflate the worst case.
- README and `docs/ollama-backend.md` publish the worst-case duration formula and a tuning guide,
  including an explicit upper bound and the thundering-herd caveat of running without jitter.

### Notes

With the default of two attempts per model, the exponential formula produces the same wait as the
old flat one. At defaults, the behavioral change is that `Retry-After` is now honored; the growth
curve and its ceiling only come into play if you raise `max_attempts_per_model`.

## [5.2.0] - 2026-07-16

### Changed

- **`strict_context_guard` now defaults to `true`.** A model whose context window cannot be measured
  aborts the run instead of proceeding on an estimate. Silent truncation is indistinguishable from a
  legitimate verdict, which is exactly why it must not be allowed to happen quietly. Set
  `strict_context_guard = false` to opt back into the old behavior.

### Added

- `strict_lineage` (default `false`): when `true`, a model whose real architecture family contradicts
  its declared `lineage` aborts the run; otherwise it is a non-fatal warning.
- Digest-uniqueness enforcement in preflight and at every rotation commit, so two mages cannot end
  up running the same weights under different names.

### Notes

Cloud models do not expose a digest, so for them uniqueness degrades to the lineage string plus the
family check. A **non-cloud** model missing a digest fails closed.

## [5.1.2] - 2026-07-15

### Changed

- The agent prompts emphasize the verdict-marker instructions more strongly.

## [5.1.1] - 2026-07-13

### Changed

- Repository cleanup; no behavior change.

## [5.1.0] - 2026-07-13

### Changed

- **The parser no longer searches for a verdict — it extracts one.** A mage's verdict must now sit
  between two literal marker lines, `<MAGI_VERDICT>` and `</MAGI_VERDICT>`, and nothing outside them
  is ever inspected.

  The old behavior decoded every JSON-shaped object in a response and kept whichever one *looked*
  like a verdict. That had two failure modes: two candidates dropped the mage (noisy but safe), or
  the only decodable object was the worked example baked into the agent's own prompt — which, for
  the adversarial seat, carries `"verdict": "approve"` — and that became the verdict, silently.
  There is nothing left to disambiguate, so neither can happen.

### Added

- An installation-time prompt guard that re-checks the three agent prompts before a run starts, with
  `--check-prompts` to run it on demand. It catches a known Windows failure mode where a symlinked
  plugin silently degrades into a stale copy. Every `[FATAL]` message it can emit is documented in
  `docs/faq-prompt-guard.md`.
- Marker-adherence telemetry, announced on stderr, plus `--max-attempts`.

### Removed

- The heuristic verdict scanner, with no flag and no fallback. `docs/adr/0001-no-runtime-heuristic-fallback.md`
  records why a config switch back to it was proposed and rejected.

## [5.0.3] - 2026-07-12

### Fixed

- The v5 config validator now ships with the plugin instead of living in an untracked directory —
  four published surfaces told users to run a file they did not have. Also fixes a fail-open in it
  and makes the gate independent of which shell you are in.

## [5.0.2] - 2026-07-12

### Fixed

- `--ollama-init` emits the complete config template: all nine scalars, with plain-language comments.

## [5.0.1] - 2026-07-12

### Added

- `-o`/`--out` redirects the verdict report to a file.

## [5.0.0] - 2026-07-12

### Added

- **Fallback rotation.** When a mage exhausts its attempts it now rotates to a declared fallback
  model of a different lineage and still returns a verdict, instead of dropping and degrading the
  run. One lineage per mage is the invariant, enforced structurally rather than by convention.
- A context guard that measures the payload with each model's own tokenizer and refuses to run a
  model the payload would not fit.
- Fast-fail: two distinct lineages failing at connection level means the endpoint is down, not the
  models, so the run aborts instead of grinding through every fallback.

### Changed

- A **transport** failure condemns a lineage run-wide; a **schema** failure stays local to the mage
  that saw it. The model answered in the second case — it just failed that mage's contract, and it
  may still serve another.

### BREAKING

- **The config schema changed and there is no shim.** `[models]` must declare an explicit `lineage`
  per mage; it is never inferred from the tag. A v4 config fails closed with an error naming the
  migration command. Run `python skills/magi/scripts/validate_magi_toml.py` for assisted migration.

## [4.0.6] - 2026-07-11

### Fixed

- **The Ollama path could not parse its own agents.** The raw output was JSON-decoded before code
  fences were stripped. Claude's backend returns an envelope so it always decoded; Ollama returns
  bare content, so a model that fenced its JSON produced a decode error, the agent was discarded,
  and the run degraded — meaning the gate approved nothing. The output is now read as text first.

## [4.0.5] - 2026-07-06

### Changed

- Agents dispatch Caspar-first, to reduce anchoring.

## [4.0.4] - 2026-07-05

### Changed

- Balthasar defaults to `kimi-k2.6:cloud`.

## [4.0.3] - 2026-07-04

### Changed

- Caspar defaults to `glm-5.2:cloud`.

## [4.0.2] - 2026-06-09

### Changed

- Melchior defaults to `qwen3.5:397b-cloud`.

## [4.0.1] - 2026-06-07

### Fixed

- `-cloud` tag detection and the Ollama cost warning.

## [4.0.0] - 2026-06-07

### Added

- **Opt-in Ollama backend** via `--ollama`, using the OpenAI-compatible API, with zero new runtime
  dependencies. `--ollama-init` scaffolds `.claude/magi-ollama.toml`. Per-mage models, layered
  configuration, structured outputs, and a fail-fast preflight that validates the host and every
  configured model before spending a run.

## [3.0.1] - 2026-05-24

### Fixed

- Finding de-duplication now requires a positive line number; a non-positive one falls back to the
  title-based key, mirroring what validation already did.

## [3.0.0] - 2026-05-24

### Added

- Optional structured `file`, `line` and `category` fields on findings, with stable finding ids and
  de-duplication that merges by id when a finding is located and by title otherwise.
- A diff-grounded hallucination guard for code-review runs: a finding citing a file absent from the
  diff is dropped, and one outside the changed range is annotated rather than removed. The guard is
  vote-neutral — it never changes the consensus score.
- Per-run cost and input-size telemetry, plus `--warn-input-tokens`.

### Changed

- Prompt calibration: an explicit likelihood vocabulary and worked examples, which cut spurious
  warnings substantially while preserving every critical finding in the validation study.

### Notes

The seven-key agent contract is unchanged; all new finding fields are optional and fail soft.

## [2.6.0] - 2026-05-23

### Added

- Per-project run isolation and concurrency hardening. Runs live under a per-project namespace, each
  writing a liveness lock, and cleanup never prunes a directory whose owning process is still alive.

## [2.5.0] - 2026-05-22

### Added

- Deterministic review-context enrichment for code-review runs: the content of touched files and the
  definitions of referenced symbols are injected so agents are grounded without needing runtime
  tools. Flags: `--base` (default `main`), `--no-enrich`, `--enrich-max-chars` (default 512 KB).

### Notes

Enrichment only runs on a clean working tree, so every read shares one coherent source. **Use
`--no-enrich` when reviewing a pull request from an untrusted contributor** — enrichment widens the
prompt-injection surface.

## [2.4.2] - 2026-05-21

### Fixed

- The parser recovers a JSON verdict embedded in prose. A `design` run had failed outright because
  all three agents emitted a natural-language preamble before their JSON. Recovery is schema-aware
  and bounded, and fails closed when more than one candidate qualifies.

## [2.4.1] - 2026-05-16

### Fixed

- `InvalidInputError` is a sibling of `ValidationError`, not a subclass, so the orchestrator's retry
  cannot silently swallow a fail-closed event. The distinction is enforced by the type system rather
  than by convention.

## [2.4.0] - 2026-05-16

### Added

- A prompt-hardening pipeline: newline normalization, invisible-character stripping, header
  neutralization, and nonce-wrapped delimiters that fail closed on a nonce collision.

## [2.3.0] - 2026-05-16

### Changed

- Repository renamed to `magi-claude`.

## [2.2.8] - 2026-05-14

### Fixed

- Entry points are hardened against `python -m` invocation.

## [2.2.7] - 2026-04-28

### Fixed

- stdout and stderr are reconfigured to UTF-8 on Windows, in an ASCII-safe way for any code page.

## [2.2.6] - 2026-04-27

### Fixed

- Windows cp1252 crash hardening: ASCII warning markers and tolerant input decoding.

## [2.2.5] - 2026-04-26

### Changed

- **Reverted the `analysis` default back to `opus`.** Sonnet's output ceiling dropped Caspar in at
  least a third of runs. The failure was structural, not random, so the retry could not recover it —
  and losing the adversarial seat biases the system toward false-positive approval.

## [2.2.4] - 2026-04-25

### Changed

- The single-shot retry also covers JSON decode errors.

## [2.2.3] - 2026-04-25

### Added

- Per-mode default model resolution.

## [2.2.2] - 2026-04-24

### Fixed

- Header sync, retrying-state coverage, and documentation of open debt.

## [2.2.1] - 2026-04-24

### Added

- `retried_agents` telemetry in the orchestrator report.

## [2.2.0] - 2026-04-24

### Added

- Single-shot agent retry on a validation error: the prompt is rebuilt with a restatement of the
  seven-key schema and the agent is given exactly one more chance.

## [2.1.4] - 2026-04-24

### Changed

- The agent prompt schema contract is reinforced: each prompt now enumerates the seven required keys
  and states that omitting any of them causes rejection.

---

## Earlier releases

Releases before 2.1.4 predate this project's annotated-tag policy and are not individually
catalogued here. Their breaking changes are recorded for completeness:

**2.0.0**
- `GO WITH CAVEATS` renders with an `(N-M)` suffix — tolerate it when grepping.
- `consensus.conditions[*].condition` comes from the agent's `summary`, not its `recommendation`;
  use `recommendations[agent]` for the old text.
- `validate.clean_title` is public and re-exported through `synthesize`.
- `StatusDisplay._write_plain_event` raises `RuntimeError` instead of `AssertionError` under ANSI
  mode.

**1.1.0**
- `## Consensus Summary` was removed from `format_report`. Read `consensus.majority_summary` from the
  JSON instead of grepping the rendered markdown.
- `## Dissenting Opinion` shows the dissenter's `summary` only; the full reasoning stays in the JSON.

[5.4.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.4.0
[5.3.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.3.0
[5.2.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.2.0
[5.1.2]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.1.2
[5.1.1]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.1.1
[5.1.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.1.0
[5.0.3]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.0.3
[5.0.2]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.0.2
[5.0.1]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.0.1
[5.0.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v5.0.0
[4.0.6]: https://github.com/BolivarTech/magi-claude/releases/tag/v4.0.6
[4.0.5]: https://github.com/BolivarTech/magi-claude/releases/tag/v4.0.5
[4.0.4]: https://github.com/BolivarTech/magi-claude/releases/tag/v4.0.4
[4.0.3]: https://github.com/BolivarTech/magi-claude/releases/tag/v4.0.3
[4.0.2]: https://github.com/BolivarTech/magi-claude/releases/tag/v4.0.2
[4.0.1]: https://github.com/BolivarTech/magi-claude/releases/tag/v4.0.1
[4.0.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v4.0.0
[3.0.1]: https://github.com/BolivarTech/magi-claude/releases/tag/v3.0.1
[3.0.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v3.0.0
[2.6.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.6.0
[2.5.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.5.0
[2.4.2]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.4.2
[2.4.1]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.4.1
[2.4.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.4.0
[2.3.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.3.0
[2.2.8]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.8
[2.2.7]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.7
[2.2.6]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.6
[2.2.5]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.5
[2.2.4]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.4
[2.2.3]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.3
[2.2.2]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.2
[2.2.1]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.1
[2.2.0]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.2.0
[2.1.4]: https://github.com/BolivarTech/magi-claude/releases/tag/v2.1.4
