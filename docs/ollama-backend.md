# MAGI open-weight path (`--ollama`), executed by `magi-rs`

- **Author:** Julian Bolivar
- **Version:** 7.0.1
- **Date:** 2026-10-14

Since 6.0.0, `--ollama` no longer talks to Ollama itself. The plugin builds the payload
exactly as before and hands it to **`magi-rs`**, a pinned, digest-verified Rust binary that
runs the three mages in one `consult` call. The three structured verdicts then enter the
pipeline that already existed: `load_agent_output`, `determine_consensus`, `format_report`.
For you the flag still means the same thing, an open-weight trio against an Ollama endpoint
with one model per mage and no Claude call anywhere. What changed is who runs the trio,
where the configuration lives, and what `--timeout` does.

## 1. Quick start

```bash
# Once per project: creates .magi/ (config, run log) through `magi-rs init`
python skills/magi/scripts/run_magi.py --ollama-init

# Any gate, any mode
python skills/magi/scripts/run_magi.py code-review <file_or_text> --ollama
python skills/magi/scripts/run_magi.py design      <file_or_text> --ollama --timeout 16820
```

The first `--ollama` (or `--ollama-init`) after an install downloads `magi-rs` (pinned at
v0.24.1) into the plugin's `bin/`, checks its sha256 against `skills/magi/magi-rs.pin.json`,
and only then runs it. Every later run verifies the digest and `--version` again before executing. When
the binary cannot be obtained or does not verify, the run stops with one `[FATAL]` line and
nothing is executed. There is no fallback to another architecture or to an unverified copy.
The one escape hatch is to place the pinned executable in `bin/` by hand, where it is
verified like any download.

The pin covers four builds: `windows-x86_64`, `linux-x86_64`, `linux-aarch64` (the rpi5
build) and `macos-universal2`. One build is not in it: `linux-x86_64-compat`, the variant
`magi-rs` publishes for older glibc (the `.7z` archives are out too). On such a host the
bootstrap downloads the regular `linux-x86_64` asset. The escape hatch does not help either:
the digest the plugin expects is the regular build's.
If the regular build does not start on your machine, that is the limit of 7.x as shipped;
say so in an issue and it can be pinned.

Run both commands **from the project root**. `--ollama-init` and the gate pass the current
directory to `magi-rs` as `-w`, so they always agree with each other. A gate run from a
directory without `.magi/magi.toml` fails and names the command to run; it never
initialises a project on its own.

## 2. Configuration lives in `.magi/magi.toml`

The plugin has no configuration file of its own any more. `.claude/magi-ollama.toml`, repo
or global, is not read, not validated and not mentioned, and you can delete it. Everything
it used to declare has exactly one owner now:

| Setting | Where it lives |
|---|---|
| provider, `base_url` | `.magi/magi.toml` (top level) |
| the trio: `melchior_model`, `balthasar_model`, `caspar_model` and their `*_lineage` | `[magi]` |
| fallbacks, `max_rotations`, `enforce_diversity`, `strict_context_guard` | `[magi]` / `[[magi.fallback]]` |
| per-attempt timeouts, memory, everything else | `magi.toml`, documented inline by `magi-rs init` |

You edit that file directly. Before a run the plugin only checks that it **exists**; its
contents are `magi-rs`'s business, and a file that does not parse is reported by `magi-rs`
in its own words. The environment chain of 5.x (`MAGI_OLLAMA_*`, repo, global, built-in)
went away with the file, and since 6.1.0 so did the plugin's own Ollama client: rotation,
context guard, backoff and redaction are all `magi-rs`'s, and there is no second validator
of `magi.toml` on this side.

**Upgrading from 6.x: `.magi/` has to be regenerated (7.0.0).** Since `magi-rs` 0.23.0 a
`magi.toml` needs a `[decision]` section, and one without it no longer loads. `magi-rs`
stops before doing anything else, and the plugin relays its message:
`... has no [decision] section: it was written by magi-rs 0.22.0 or earlier`. `magi-rs init`
never edits an existing file, so `--ollama-init` cannot fix it in place. From the project root:

1. Move the old state directory aside: `mv .magi .magi.bak-0.21`.
2. Run `python skills/magi/scripts/run_magi.py --ollama-init`. It writes a fresh `.magi/`
   whose `magi.toml` has the `[decision]` section.
3. Copy your own settings (the trio, its lineages, the `[[magi.fallback]]` pool,
   `max_rotations`, anything else you changed) from `.magi.bak-0.21/magi.toml` into the new
   file. Comment lines can be ignored; the new scaffold groups the pool right under `[magi]`.

Do not bring back the old `.magi-rs-memory.db`. `magi-rs` 0.22.0 moved its bundled SQLite to
3.53.2 and no longer supports databases written by 0.21.0 or earlier, and the plugin never
reads it (see below). The migration is one-way: plugin 6.2.0 (on `magi-rs` 0.21.0) rejects the new file
(``unknown field `decision` ``), so going back to 6.2.0 means restoring the backup.

The `[decision]` section configures `magi-rs`'s small decision models (`clef-flash`, with
`nimble` as fallback, in the scaffold). They pick a consult's mode when none is given and
judge memory contradictions. The plugin always passes the mode and runs with `--no-memory`,
so it never calls them and you do not have to pull them. The section still has to be
present, because `magi-rs` refuses to load a file without it.

An older migration works the same way, with `magi-rs`'s message as the guide. In a
`magi.toml` written by a `magi-rs` older than 0.18.0, the
`[headless]` keys `log_retention`, `log_max_bytes` and `log_level` are fatal (`unknown
field`) from 0.18.0 on, and nothing migrates them for you. Their replacements live under
`[logging]`: `log_dir`, `file_filter`, `compress`, `compress_after_days`, `retain_days`,
`max_total_bytes`. Retention is counted in days there, not runs. The plugin relays
`magi-rs`'s error as it is, so this paragraph is the migration guide. Since 0.18.1 the
diagnostics that used to reach stderr (the per-mage ceiling, the probe notices) go to
`.magi/logs/magi-<date>.log`, one line per event tagged with the `run_id` the report
persists; stderr carries only `ERROR` and `WARN`.

`magi-rs` reads credentials from an encrypted vault only when given a passphrase, and this
plugin never passes one. The provider is Ollama, which needs no key, so the vault would have
nothing to hold. That is also why `--no-memory` is passed on every run: it is the flag that
lets `magi-rs consult` proceed without a passphrase. It is not a memory switch; `consult`
never touches the conversational memory in the first place. Should you ever point
`magi.toml` at a provider that needs a key, that decision comes back.

## 3. What the child process sees

The child runs with an **allowlisted environment**: `PATH`, `HOME`/`USERPROFILE`, the temp
variables, `SYSTEMROOT`, `COMSPEC`, the locale variables, the proxy variables
(`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` and their lowercase forms), the TLS/CA variables
(`SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`,
`NODE_EXTRA_CA_CERTS`) and `OLLAMA_HOST`. Nothing else travels: no `MAGI_PASSPHRASE`, no
`OPENAI_API_KEY` or `MAGI_API_KEY`, no dynamic-linker variables. The same command therefore takes the same path
whoever launches it. The list is enumerated rather than prefix-matched, and it grows only on
the evidence of a run that failed. When the child dies or hangs, the error names the
allowlist as a suspect and lists the parent variables that were withheld, by name only.

## 4. `--timeout` has two effects

`--timeout` is still the wall-clock ceiling of the run. It is also the number `magi-rs`
derives each mage's per-attempt budget from. The default on this path is **16820**, not the
Claude path's 900. That is the value `magi-rs` 0.21.0 publishes for a headless gate. It
derives a per-mage ceiling of 2335 s, an operation budget of 1401 s per attempt and a
700 s per-request clock, confirmed locally on 2026-09-29 and again with 0.24.1 on
2026-10-13. Any explicit positive value is
honoured. Below 16820 a warning goes to stderr, because a lower wall timeout also cuts the
reasoning budget of every seat, and the symptom (mages that stop completing) does not look
like a timeout setting.

A default run measured on 2026-09-29 took about 2 minutes on a 75 KB bundle; `magi-rs`'s
own replay took 531 s. A run in which seats rotate through every model and exhaust their
attempts can take hours (worst case about 4.7 h). When the orchestrator is launched from a
tool with its own time limit, run it in the background.

**Completion cap.** `magi-rs` 0.21.0 raised its default completion cap from 16 384 to
**65 536** tokens. You can change it with `max_tokens` under `[magi]` in your own
`.magi/magi.toml`; the plugin does not read or validate that file's model settings. A model
whose maximum output is below the cap answers HTTP 400, and its lineage is dropped for that
run. Measured on 2026-09-29, `kimi-k2.6:cloud`, `glm-5.2:cloud`, `gpt-oss:120b-cloud`,
`mistral-large-3:675b-cloud` and `gemma4:cloud` all accepted 65 536. A reasoning model that
spends the cap on thinking still shows up as a seat that fails; the remedy is a different
model in that seat, not a longer timeout.

**Reasoning keys.** `magi-rs` 0.20.0 added four `[magi]` keys: `reasoning`
(`default`/`disabled`/`enabled`), `reasoning_spelling` (`openai-compat` only), `max_tokens`
and `reasoning_trace` (opt-in, log only). They are documented in
[`docs/REASONING-BUDGET.md`](https://github.com/BolivarTech/magi/blob/v0.24.1/docs/REASONING-BUDGET.md)
in the `magi-rs` repository.

**A `--timeout` that is too high.** A high value is obeyed as given. `magi-rs` flags a
`--timeout` whose derived per-mage ceiling exceeds 2400 s as a probable typo (168200 for
16820, say). The plugin records the flag in the report and prints one `WARNING` line
**after** the run; it cannot warn before the launch, because the derivation is internal to
`magi-rs`. The derivation also depends on `max_rotations` in `.magi/magi.toml`: with
`max_rotations = 0` even 16820 derives 7005 s and is flagged, while the scaffold default
`max_rotations = 2` derives 2335 s and is not. When the flagged value is not above 16820, the
warning points at `max_rotations` instead of recommending a timeout.

**Restoring the 6.1.0 behaviour.** Pass `--timeout 1800`, and set `max_tokens = 16384` and
`agent_timeout_secs = 90` under `[magi]` in `.magi/magi.toml`.

## 5. What the report carries

`magi-report.json` keeps its shape (`agents`, `consensus`, `cost`, `guard`, `input_size`,
`model`) plus one block this path adds:

- **`magi_rs`**: the `consult` telemetry verbatim (`degraded`, `mode`, `mode_source`,
  `extraction_failures`, `input_size`, `report_truncated`, `endpoint_divergence`,
  `timeout_below_formula`, `failed_agents`, `rotations`, `ran_unmeasured`, `completions`,
  `pool_eligibility`; `degraded` stays exactly as `magi-rs` sent it, and only the stderr warning
  reads a non-boolean value as degraded), plus `capped_agents` (seats whose last attempt ended for
  a reason other than a clean `stop`, usually the completion cap: the verdict counts, but the
  trailing prose may be lost; an unknown finish reason is listed rather than passed over, while
  an attempt with no finish reason at all, such as a timeout, is not), `timeout_seconds` (the
  `--timeout` passed to `magi-rs`), `ceiling_above_sanity` (`applied_caps.ceiling_above_sanity` from
  `magi-rs`: `true`, `false`, or `null` when it published no boolean), `model_used` (the tag
  each seat's accepted attempt actually ran on), `run_id` (the pointer into `.magi/logs/magi-<date>.log`,
  where `magi-rs` writes the per-mage ceiling and the probe notices) and
  `gate_ran_on_unverified_context` (`true` only when **every** seat ran with an unmeasured
  context window).
- **`cost`** is `{"total_usd": null, "per_agent": {}, "cost_source": "unpriced"}`:
  `magi-rs` exposes no usage per mage, and an unpriced run is not a free one.
- The `model` block has the same shape as on the Claude path, with every Claude-only field
  `null`. `served` stays `null` here because nothing confirms the served identity.

The raw envelope is saved next to it as **`magi-rs-raw.json`**, exactly as received, for
audit. Each record of `completions` carries two keys added in 0.20.0, `reasoning` and
`control`, and they reach the report verbatim; `schema_version` did not change. Its embedded
transcript is capped at 64 KiB by `magi-rs`; the prompt itself is what the plugin sent,
unchanged.

Five warnings can reach stderr; none appears on a clean run. The run is DEGRADED, because a
seat delivered nothing, a delivered verdict failed the seven-key contract here, or `magi-rs`
sent a `degraded` flag that is not a JSON boolean; in each case the gate is invalid whatever
the verdict says. A seat's last attempt ended without a clean `stop`, usually at the
completion cap; the message also names that attempt's reasoning state (`Measured N chars`,
`NotMeasured`, `Unsupported (<backend>, N chars)`, `not reported` or `unknown (<variant or
type>)`) and never prints model text. The context window was not measured for some seats; the message names the
remedy, `strict_context_guard = true` in `.magi/magi.toml`, and raises its tone when nobody
measured anything. There were extraction failures upstream, with their `cause` relayed as
spelled. Last, `magi-rs` flagged the `--timeout` as a probable typo (see section 4).

When `magi-rs` exits non-zero, the error shows the tail of its stderr: the last 2000
characters, prefixed with `[... earlier stderr elided ...]` when cut. `magi-rs` writes its
`WARN` lines at startup, so the cause comes last.

## 6. Bundles that do not fit are split

`magi-rs consult` rejects any stdin above **262 144 bytes** (about 65 500 tokens), instantly
and without truncating. A bundle that does not fit under 65 % of that ceiling is split at
section boundaries and never mid-section: markdown headings up to `###`, and `diff --git`
file boundaries, including inside a ```` ```diff ```` fence, which is closed and reopened
per piece. The split is written to `magi-report.json` under `segmentation` **before** the
first segment runs. Segments then run one at a time into `segment-N/`, every verdict is
tabled verbatim, and the gate is approved only when every segment passed. A segment whose
run fails is tabled as `FAILED (...)` while the rest still run; a section too large on its
own is refused with its name, size and bound. A section may declare the interfaces it owns
with a `<!-- defines: A, B -->` line right under its heading; when the split separates a
definition from a segment that mentions it, an extra integration segment runs on the lines
that mention those names. A bundle that fits is never split.

## 7. Troubleshooting

| Symptom | Meaning / what to do |
|---|---|
| `[FATAL] ... is not initialised for magi-rs ... run /magi --ollama-init` | No `.magi/magi.toml` under the current directory. Run the init from the project root, then the gate from the same place. |
| `ERROR: magi-rs exited 2; ... has no [decision] section` | The `.magi/magi.toml` predates `magi-rs` 0.23.0. Regenerate it as described in section 2 (upgrading from 6.x). |
| `[FATAL] digest mismatch` / `version mismatch` | The binary in `bin/` is not the pinned one. Delete it and re-run; the bootstrap downloads and verifies again. |
| `ERROR: magi-rs exited N; stderr: ...` | The child failed; its stderr is relayed verbatim. If it looks like a missing certificate, proxy or host, the allowlist (section 3) is the first suspect: the message lists what it withheld. |
| `ERROR: magi-rs did not finish within --timeout` | Raise `--timeout` (section 4), or re-seat a model that keeps overrunning. |
| `WARNING: the run is DEGRADED ... The gate is INVALID` | A seat died, a verdict failed the contract, or `magi-rs` sent a non-boolean `degraded` flag. Resolve the cause and re-run; a degraded run approves nothing. |
| `WARNING: NOBODY MEASURED the context window` | Every seat ran on an estimated window. Set `strict_context_guard = true` in `.magi/magi.toml` to fail closed instead. |
| `WARNING: the bundle was SPLIT into N segments` | Section 6. Each verdict judges one segment; read the table at the end. |

## 8. References

- `skills/magi/magi-rs.pin.json`: the pinned version and per-platform digests.
- `magi-rs`: [BolivarTech/magi](https://github.com/BolivarTech/magi) releases (assets and `SHA256SUMS.txt`).
- `docs/MAGI-System-Documentation.md`: the full technical reference of the plugin.
