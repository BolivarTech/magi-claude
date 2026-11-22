# MAGI open-weight path (`--ollama`), executed by `magi-rs`

- **Author:** Julian Bolivar
- **Version:** 8.0.0
- **Date:** 2026-11-21

Since 6.0.0, `--ollama` no longer talks to Ollama itself. The plugin builds the payload
exactly as before and hands it to **`magi-rs`**, a pinned, digest-verified Rust binary that
runs the three mages in one `consult` call. The three structured verdicts then enter the
pipeline that already existed: `load_agent_output`, `determine_consensus`, `format_report`.
For you the flag still means the same thing, an open-weight trio against an Ollama endpoint
with one model per mage and no Claude call anywhere. What changed is who runs the trio,
where the configuration lives, and what `--timeout` does.

Version 8.0.0 changes two more things. `code-review --ollama` now hands the whole review to
`magi-rs`'s own `code-review` command (section 1a), and every `--ollama` run opens the
project's `magi-rs` state with a key the plugin generates and keeps (section 2a). A `.magi/`
created by 7.x has to be regenerated (section 2). For `code-review --ollama` the plugin builds
no payload and makes no `consult` call; read the paragraphs above with that exception (section 1a).

## 1. Quick start

```bash
# Once per project: creates .magi/ (config, run log, key, vault) through `magi-rs init`
python skills/magi/scripts/run_magi.py --ollama-init

# Store an API key your magi.toml needs in the vault (the value comes from stdin or a hidden prompt)
python skills/magi/scripts/run_magi.py --ollama-vault-set <NAME>

# design and analysis take a file or inline text
python skills/magi/scripts/run_magi.py design <file_or_text> --ollama --timeout 16820

# code-review takes a path or a git revision, never a prompt
python skills/magi/scripts/run_magi.py code-review --ollama
python skills/magi/scripts/run_magi.py code-review --ollama --commit <rev>
```

The first `--ollama` (or `--ollama-init`) after an install downloads `magi-rs` (pinned at
v0.25.0) into the plugin's `bin/`, checks its sha256 against `skills/magi/magi-rs.pin.json`,
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

`--ollama-init` exits 1 with one `[FATAL]` line when it cannot finish. In 7.x it relayed the
refusal and exited 0, so a script that checked the exit code never saw the failure.

## 1a. `code-review --ollama` reviews paths and git revisions

Since 8.0.0 the plugin no longer builds a bundle for this mode. It hands the target to
`magi-rs code-review`, which does the batching, finds definitions and callers, pulls in
documentation and earlier findings from its memory, masks secrets and runs its own
false-positive gate. The enrichment, the diff-grounded guard and the bundle splitter stay
in the plugin for the Claude path and, for the splitter, for `design` and `analysis`.

The target is exactly one of these, passed through to `magi-rs` as given:

| Target | Reviews |
|---|---|
| `<path>` | a file, or a whole directory |
| `--commit <rev>` | one commit |
| `--from <rev> --to <rev>` | a range |
| `--base <rev> --head <rev>` | what `--head` added since it diverged from `--base` |
| `--uncommitted` | staged, unstaged and untracked changes |

With no target the plugin reviews `--base main --head HEAD`, the whole branch. Standing on
`main` that range is empty, and the plugin stops with an error asking for an explicit
target; it does not fall back to the last commit. Invalid combinations (`--commit` with
`--uncommitted`, `--from` without `--to`, free text that is not an existing path) are
rejected before `magi-rs` starts, so nothing is spent on them. A path that resolves inside
`.magi/` or inside a rotation backup is refused too. The banner always prints the target
that was reviewed.

**Breaking in 8.0.0: a `.md` is a file to review.** In 7.x, `code-review --ollama bundle.md`
read the file as the prompt. Now any file given as the input, a markdown bundle included,
is reviewed as a file, and its content is never read as instructions.

**The embedder is a prerequisite.** `magi-rs code-review` gets documentation and earlier
findings through the embedder configured in `[embedding]` of `.magi/magi.toml` (the
`base_url` is inherited from the top level when `[embedding]` does not set one). Before any
model is called, the plugin sends one small request to that endpoint. If the connection
fails or times out, the run stops with one `[FATAL]` line that names the endpoint and the
model. The probe measures reachability only: any HTTP answer counts, a 401 or 403 included,
so a key that is missing or wrong passes the probe and fails later, during the review, with
`magi-rs`'s own message. An embedder that goes down during the review yields a report
marked `degraded`: the cause is in the report, a warning goes to stderr, and the run does
not approve any gate. A report that carries `without_memory` is a contract violation (the
plugin never asks for a review without memory) and ends as one `ERROR:` line.

**What the report adds.** The `code_review` object of `magi-rs` is copied verbatim to
`magi_rs.code_review`. The banner shows the batches, what was discarded, what the gate
filtered, `judged_fraction`, what was skipped or masked, and `without_embedder` when it
applies. An effort block records batches, duration, findings sent to the gate and judged,
and tokens per batch and model when `magi-rs` reports them; a field the envelope does not
carry is `null`. `input_size` and `guard` are `null` on this path, because the plugin
builds no payload. `magi_rs.model_used` is `null` as well, with a `model_used_note`: the
envelope names the three models as one comma-joined list, not one per seat.

If the false-positive gate cannot use `clef:cloud`, `magi-rs` reports it in the
`code_review` block (`gate_stopped` and `gate_causes`), and the plugin prints those two
fields as `magi-rs` sent them. Section 2b says what the default `magi.toml` declares as
fallbacks.

**Time.** `--timeout` keeps its default of 16820 s and is passed to `magi-rs` as the total
for the review. If it is too small for the number of batches, `magi-rs` refuses it before
spending anything (measured with 0.25.0: a floor of 114 s per batch), and the plugin shows
that as one `ERROR:` line naming `--timeout`. It does not retry with another value. After
the report is printed, `magi-rs` records the review history, which can take up to 30 s. The
plugin gives the child 45 s beyond `--timeout` to finish that, and only then ends it. If
the history could not be recorded, `magi-rs` says so on stderr; the plugin detects that
line and prints its own warning, and the report already written stays valid.

**Size of a batch.** `magi-rs` lets one batch, code and context together, take up to half of the
smallest context window in the pool (the trio plus its fallbacks). A model whose window it could
not measure counts as 32 768 tokens, and the report lists those models in
`magi_rs.code_review.unmeasured_models`. With the cloud trio the windows go unmeasured, so a batch
is capped near 16 000 tokens. A changed unit that does not fit in one batch by itself stops the
review before any model is called: in our measurement with 0.25.0, a new Python file of 806 lines
needed 19 281 tokens against a limit of 16 268. The plugin shows the refusal as one `ERROR:` line.
Review a smaller target instead, such as a commit or a range whose changed units fit.

**Framing a gate.** Without a bundle, what the mages are told is settled comes from the
documentation `magi-rs` indexes: `README.md` and `CLAUDE.md` at the root, `docs/**/*.md`,
`adr/**/*.md` and doc comments. For a commit, a range or a branch it reads those files from
the revision under review, through git, never from your working tree, so a file that is
untracked or ignored does not reach the mages. Put the decisions you want respected in
tracked `docs/` or `adr/` files. Both folders travel with the project, so write them for
readers, never as process notes. Retrieval is not guaranteed: `magi-rs` brings back the
eight chunks nearest to each batch, and a decision can fall outside them.

## 2. Configuration lives in `.magi/magi.toml`

The plugin has no configuration file of its own any more. `.claude/magi-ollama.toml`, repo
or global, is not read, not validated and not mentioned, and you can delete it. Everything
it used to declare has exactly one owner now:

| Setting | Where it lives |
|---|---|
| provider, `base_url` | `.magi/magi.toml` (top level) |
| the trio: `melchior_model`, `balthasar_model`, `caspar_model` and their `*_lineage` | `[magi]` |
| fallbacks, `max_rotations`, `enforce_diversity`, `strict_context_guard` | `[magi]` / `[[magi.fallback]]` |
| per-attempt timeouts, `[embedding]`, `[code_review]`, everything else | `magi.toml`, documented inline by `magi-rs init` |

You edit that file directly. Before a run the plugin only checks that it **exists**; its
contents are `magi-rs`'s business, and a file that does not parse is reported by `magi-rs`
in its own words. The environment chain of 5.x (`MAGI_OLLAMA_*`, repo, global, built-in)
went away with the file, and since 6.1.0 so did the plugin's own Ollama client: rotation,
context guard, backoff and redaction are all `magi-rs`'s, and there is no second validator
of `magi.toml` on this side.

**Upgrading from 6.x or 7.x: `.magi/` has to be regenerated (8.0.0).** A `.magi/` created
by 7.x has no key, and the plugin does not adapt it. Every `--ollama` mode stops with a
`[FATAL]` line that gives the procedure. From the
project root:

1. Move the old state directory aside, to somewhere outside the project. A backup left in
   the working tree is untracked, so `--uncommitted` would pick it up.
2. Run `python skills/magi/scripts/run_magi.py --ollama-init`. It generates the key and
   runs `magi-rs init` with it, which writes a fresh `.magi/` (`magi.toml`, database and
   vault).
3. Reload each API key you use with `--ollama-vault-set <NAME>` (section 2a).
4. Re-apply by hand any model choices you still want from the old `magi.toml`. The new file
   is the binary's default and nothing is copied across, because a model the old file names
   may have been retired since. Changing a seat is your call, and it deserves a measured run
   (section 2b).

Do not bring back the old `.magi-rs-memory.db`; the plugin never reads it. `--ollama-init`
on a project that already has a `.magi/` touches nothing in it (config, key and database)
and says so. A key that exists is never overwritten, and a database is never deleted or
recreated.

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

## 2a. The key, the vault and the memory

`--ollama-init` generates one key for the project: 32 random bytes, base64-encoded, saved as
`.magi/.secret/magi.key`. It opens both the encrypted memory database and the vault of
`magi-rs`. The plugin gives it to the child process as `MAGI_PASSPHRASE` in the child's
environment and nowhere else: never in the arguments (those show in a process list), never
in a log, a report, an error message or a `repr`. A key that is missing or empty stops
every `--ollama` run with one `[FATAL]` line, before the input is read and before any model
is called. The plugin never falls back to running without memory.

**What the key protects, and what it does not.** The key sits in plain text next to the
database it opens. That keeps a copy of the database or the vault safe when it travels
without the `.magi/` directory, and no more. Anyone who can read your project's `.magi/`
can read the key. The plugin offers nothing stronger than that.

**How the key file is protected.** On creation, and again before every `--ollama` run and
every vault command, the plugin verifies that only you can reach it:

- On POSIX, `.secret/` is `0700` and `magi.key` is `0600`, and the owner is the current user.
  A stricter mode such as `0400` passes; any group or other bit fails. POSIX extended ACLs
  are not read.
- On Windows the key's DACL must be protected (no inheritance) and grant access only to the
  current user's SID, which the plugin sets with `icacls` and reads back by SID, not by
  account name. The file's owner is not checked.
- Links and reparse points are refused. The permissions of `.magi/` itself are not
  verified; the plugin assumes it belongs to you and that group and others cannot write to it.

If the check fails, the run stops with `[FATAL]` naming the file and the rotation command.
The plugin never repairs permissions silently: with a loosened key it cannot know whether
the file was copied.

**Rotating an exposed key.** `python skills/magi/scripts/run_magi.py --ollama-rotate-key`
replaces the key, the database and the vault, and nothing rotates by itself. In a terminal
it asks `y/N`; without one it needs `--yes`, and without either it does nothing. It does not
use `vault passwd`, because re-wrapping the same database key would not protect a copy that
already left. Instead it:

1. moves the old `.magi/` to a timestamped backup outside every working tree, under your
   per-user state directory (`%LOCALAPPDATA%\magi\rotated` on Windows, otherwise
   `$XDG_STATE_HOME/magi/rotated`, or `~/.local/state/magi/rotated` when that variable is
   unset), with the same owner-only protection;
2. creates a new `.magi/` through `--ollama-init`, with a new key and a new database;
3. lists the names (never the values) of the API keys the old vault held. Revoke each one at
   its provider and load the new one with `--ollama-vault-set`.

The memory and the review history are lost, and the new `magi.toml` is the binary's default.
Destroy the backup once you no longer need it. If a rotation stops midway, its one-line
message says where the old `.magi/` is and whether a new one must be removed before you move
it back; follow it.

**There is no recovery of a lost key.** If the key file is gone, so is what it protected: the
memory and the vault contents. Regenerate the project's `.magi/` as described in section 2.

**The vault.** `magi-rs` reads each credential from the environment first and from the vault
second. The plugin's child environment forwards no API-key variable (section 3), so the vault
is the one source. Store a key with
`python skills/magi/scripts/run_magi.py --ollama-vault-set <NAME>`, where `<NAME>` is one of
`OPENAI_API_KEY`, `MAGI_API_KEY`, `CODE_REVIEW_API_KEY`, `EMBEDDING_API_KEY` or
`DECISION_API_KEY`. The value comes from stdin or, in a terminal, from a hidden prompt, and
never from an argument. When `magi-rs` reports a missing key, the plugin's message names this
command; it cannot tell which `<NAME>` was missing.

**What memory does in each mode.** `code-review --ollama` uses it: documentation, and earlier
findings that git confirms, reach the mages as context. In `magi-rs` 0.25.0, `design` and
`analysis` open the database only to read the vault: they store and retrieve no memory, do
not call the embedder and never report `without_memory` or `without_embedder`. The plugin
passes the key in those modes only to open the vault, so read that as a documented limit of
those two modes, not as a promise of memory.

## 2b. What `--ollama-init` writes

The `magi.toml` is exactly what `magi-rs init` writes, and the plugin edits nothing in it.
Know what it contains before you rely on it. The `[decision]` section configures the small
decision models of `magi-rs` (`clef-flash`, with `nimble` as fallback); the plugin passes
the mode explicitly, so it does not use them to choose one, but `magi-rs` refuses to load a
file without the section. The `[code_review]` section names `clef:cloud` for the
false-positive gate, with the local models `clef-flash` and `nimble` as fallbacks, and the
default trio puts `deepseek-v4-pro` in Caspar's seat. If `clef:cloud` fails and the local
models are not installed on your machine, `magi-rs` reports it in the `code_review` block
(`gate_stopped`, `gate_causes`) and the plugin shows it as received; this guide does not
describe what `magi-rs` does between those two points. Re-seating a mage is your decision
and should follow a measured run, not a guess.

## 2c. Where your data goes

- Each batch of a review, and each consult, goes to the endpoint of the trio. With the cloud
  trio that is Ollama's cloud.
- With `code-review --ollama`, every finding the gate judges, plus 40 lines of code on each
  side of it, goes to the model in `[code_review]`. With `clef:cloud` that is `ollama.com`,
  and it spends quota there.
- The documentation and the code of each batch go to the endpoint in `[embedding]`.

## 3. What the child process sees

The child runs with an **allowlisted environment**: `PATH`, `HOME`/`USERPROFILE`, the temp
variables, `SYSTEMROOT`, `COMSPEC`, the locale variables, the proxy variables
(`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` and their lowercase forms), the TLS/CA variables
(`SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`,
`NODE_EXTRA_CA_CERTS`) and `OLLAMA_HOST`. The one addition is `MAGI_PASSPHRASE`, which the
plugin sets itself from the project's key (section 2a) and never inherits from your
environment. Nothing else travels: no `OPENAI_API_KEY` or `MAGI_API_KEY`, no dynamic-linker
variables. The same command therefore takes the same path whoever launches it. The list is enumerated rather than prefix-matched, and it grows only on
the evidence of a run that failed. When the child dies or hangs, the error names the
allowlist as a suspect and lists the parent variables that were withheld, by name only.

## 4. `--timeout` has two effects

For `code-review --ollama` there is no payload and no consult: `--timeout` is the total for
the review (section 1a).

`--timeout` is still the wall-clock ceiling of the run. It is also the number `magi-rs`
derives each mage's per-attempt budget from. The default on this path is **16820**, not the
Claude path's 900. That is the value `magi-rs` 0.21.0 publishes for a headless gate. It
derives a per-mage ceiling of 2335 s, an operation budget of 1401 s per attempt and a
700 s per-request clock, confirmed locally on 2026-09-29, again with 0.24.1 on
2026-10-13 and with 0.25.0 on 2026-11-18. Any explicit positive value is
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
[`docs/REASONING-BUDGET.md`](https://github.com/BolivarTech/magi/blob/v0.25.0/docs/REASONING-BUDGET.md)
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
- A few keys exist only for `code-review --ollama`. Inside `magi_rs` they are `code_review`
  (the `magi-rs` object, verbatim), `effort` (batches, duration, gate questions, tokens per batch
  with the batch `index`, so a failed batch leaves a visible gap) and `model_used_note`; there
  `model_used` and `gate_ran_on_unverified_context` are `null`, because the review envelope names
  no per-seat model and carries no unmeasured-window list. At the top level, `degraded_causes`
  appears only when a cause exists: `without_embedder`, `failed_batches` (a batch failed, or fewer
  batches completed than were planned) or `undelivered_units` (changed units that never reached
  the mages). Each cause degrades the run and prints one stderr warning; `input_size` and `guard`
  are `null`.
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

This applies to `design` and `analysis`. `code-review --ollama` has no bundle; `magi-rs`
batches the review itself (section 1a).

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

The release check on marker adherence (`tools/measure_marker_adherence.py`) runs
`analysis --ollama` in a throwaway workspace with a fresh `.magi/`, never in your project and
never with your project's key. Its results start a new series at 8.0.0: the 7.x numbers measured
a different mode and are not comparable.

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
| `[FATAL]` about a missing or empty memory key, or a `.magi/` without its key | A `.magi/` from 7.x or a lost key. Regenerate it as described in section 2. |
| `[FATAL]` about the key's permissions | The key file or its directory is reachable by others. Treat the database and the vault as compromised and run `--ollama-rotate-key` (section 2a). |
| `[FATAL]` naming the embedder endpoint | `code-review --ollama` could not connect to the `[embedding]` endpoint. Start it, or fix `base_url` in `.magi/magi.toml` (section 1a). |
| `ERROR:` naming `--timeout` on a `code-review --ollama` | `magi-rs` refused the total as too small for the batches before spending anything. Pass a larger `--timeout` (section 1a). |
| `ERROR:` with `does not fit in one batch` on a `code-review --ollama` | A changed unit is larger than one batch, which is capped near 16 000 tokens when the trio's windows are unmeasured. Review a smaller commit or range (section 1a). |
| `degraded: true` with `without_embedder` | The embedder failed during the review. The report is written but approves no gate; fix the embedder and run again. |

## 8. References

- `skills/magi/magi-rs.pin.json`: the pinned version and per-platform digests.
- `magi-rs`: [BolivarTech/magi](https://github.com/BolivarTech/magi) releases (assets and `SHA256SUMS.txt`).
- `docs/MAGI-System-Documentation.md`: the full technical reference of the plugin.
