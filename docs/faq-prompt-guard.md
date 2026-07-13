# FAQ — The Agent Prompt Guard (`[FATAL]` at startup)

Since **v5.1.0** (MS2, the verdict sentinel), MAGI validates its own three agent prompt
files (`agents/melchior.md`, `agents/balthasar.md`, `agents/caspar.md`) before it launches
a single agent. If a prompt is stale, corrupted, or edited in a way that can produce a
fabricated verdict, MAGI refuses to start and prints a `[FATAL]` message naming the exact
file and the exact defect.

This exists for one reason: the anchoring test (`tests/test_agent_prompt_contract.py`)
only sees the prompts **in this repository**. It cannot see the prompts **on a user's
machine** — and a known Windows failure mode (`mklink /D` silently degrading to a stale
*copy* instead of a real link, see `CLAUDE.techdebt.md`) can leave an installation running
old prompts against a new parser. `AgentPromptGuard` (`skills/magi/scripts/prompt_guard.py`)
is the check that runs where the repo's tests cannot reach: the installation itself.

If you never edit the files under `agents/`, you will never see this guard fire. This FAQ
is for the case where you do.

---

## Why this is strict: what the sentinel replaced

Before v5.1.0, the parser *searched* an agent's raw output for something that looked like a
verdict. That heuristic had a silent failure mode: if the only JSON object it could decode
was the **worked example already sitting in the agent's own system prompt** — which, for
Caspar, is a complete verdict whose `"verdict"` field is literally `"approve"` — the parser
would accept it as the agent's real answer. A fabricated approval in the adversarial seat,
indistinguishable from a real one. See
[`docs/adr/0001-no-runtime-heuristic-fallback.md`](adr/0001-no-runtime-heuristic-fallback.md)
for the full history of that residual and why it took several attempts to close correctly.

The sentinel fixes this by **extracting**, never **searching**: the agent's real verdict
must appear between two literal marker lines, `<MAGI_VERDICT>` and `</MAGI_VERDICT>`, each
alone on its own line. The parser reads only what is between them. Nothing outside the
markers is ever inspected — which means a worked example living *outside* the markers, in
the same prompt, is structurally never a candidate.

That guarantee only holds if the three prompt files actually keep the markers as a
**delimiter around an empty slot**, never as a wrapper around a real, complete example. The
guard exists to enforce that shape before any agent runs.

---

## The `[FATAL]` messages, one by one

Each of these aborts the run immediately, before any agent is launched or any API/model
call is made. None of them is retryable — retrying calls the same model with the same
broken prompt, so the fix has to happen in the prompt file itself, not in the run.

### `<path>: cannot read prompt file. Reinstall the plugin.`

The file could not be opened (permissions, deleted mid-run, corrupted filesystem entry).
**Fix:** reinstall the plugin (`/plugin marketplace update`, or re-link the dev checkout).

### `<path>: prompt file is not valid UTF-8.`

The file's bytes do not decode as UTF-8 (a UTF-8 BOM is handled transparently and does
**not** trigger this — see the note on encoding below). **Fix:** re-save the file as UTF-8,
or reinstall.

### `<path>: found N open and M close marker lines (expected exactly 1 of each). ...`

The file does not contain **exactly one** `<MAGI_VERDICT>` line and **exactly one**
`</MAGI_VERDICT>` line. The rest of the message depends on which of the two causes it is —
they have opposite fixes, and a single message would have misdirected one of the two
audiences:

- **`N=0, M=0` — a pre-5.1.0 prompt.** v5.0.x agent prompts predate the sentinel and never
  had marker lines at all. The message says **reinstall the plugin**, because that is the fix.
- **Too many — the markers were repeated.** Almost always someone customising the prompt who
  demonstrated the format a second time (a walkthrough that shows the markers again to explain
  them). The message does **not** tell you to reinstall: that would throw your work away and
  would not even fix it. Every extra marker line is a rival — a model reading a prompt with two
  open/close pairs has been shown two places a verdict can go, and may emit two blocks. **Fix:**
  keep exactly one pair, where the "Output format" section places it, and explain the format
  elsewhere in prose or with a placeholder — do not paste the marker lines again, not even
  inside a fenced code block.

Check your edit before you spend a run on it: `run_magi.py --check-prompts`.

### `<path>: <sentinel error message>` (wraps `MissingVerdictMarkers` / `UnterminatedVerdictBlock` / `AmbiguousVerdictMarkers`)

This fires when the marker **count** check above passes (exactly one exact `<MAGI_VERDICT>`
line and one exact `</MAGI_VERDICT>` line) but something is still wrong with their
**position** or their **exact form**. The two checks are deliberately different in
strictness (see "Two predicates, on purpose" below), and this is where that gap shows up:

- **`close marker precedes the open marker (open at line X, close at line Y)`** — the file
  has `</MAGI_VERDICT>` physically before `<MAGI_VERDICT>`. This is a corrupted or
  hand-edited file, not a normal authoring mistake.
- A count of exact markers can also disagree with what the extractor scans for if the file
  has a **near-duplicate marker line** — the same text plus an invisible character (a soft
  hyphen, a stray variation selector) that the exact, byte-for-byte installation check does
  not tolerate but the extractor's permissive scan (built for the *model's* untrusted
  output, not for files we ship) still matches. That combination is the signature of a
  **corrupted install** (a bad copy, a bad merge, an editor that silently inserted an
  invisible character), not of ordinary prose editing. **Fix in both cases:** reinstall the
  plugin from a clean checkout.

### `<path>: content between markers is a valid verdict. The model can COPY it and the copy would be accepted as its verdict; a PLACEHOLDER goes between the markers, not an example; the worked example goes OUTSIDE the markers.`

**This is the one to read carefully if you are editing a prompt yourself.**

The guard parses whatever is between the markers as JSON and checks whether it is a
complete verdict object — one that already has all seven required keys (`agent`,
`verdict`, `confidence`, `summary`, `reasoning`, `findings`, `recommendation`). If it does,
the guard refuses to start.

**Why this matters more than it looks:** it is easy to "improve" a prompt by turning the
placeholder into a fully worked example, on the reasoning that a concrete example helps the
model understand the format better:

```
<MAGI_VERDICT>
{"agent": "caspar", "verdict": "conditional", "confidence": 0.85, "summary": "...", ...}
</MAGI_VERDICT>
```

This looks harmless — it is *just* an example. It is not harmless. A model under any kind
of pressure (a confusing input, a truncated context, a bad day) can **copy that example
verbatim** instead of producing its own analysis. And because the copy is between the real
markers, in the real shape, with all seven keys present, the parser has no way to tell it
apart from a genuine verdict. This is exactly the residual the sentinel exists to close —
reintroduced by the prompt itself instead of by the parser.

**The rule, stated once:** between the markers goes a **placeholder** — the literal text
`{ ...your 7-key JSON object... }` (or equivalent prose that is obviously not real JSON).
The **worked example** — the fully fleshed-out, field-by-field illustration that helps the
model understand `verdict`/`findings`/`severity` values — goes **outside** the markers, in
its own fenced block, in the "Output format" section that already exists in each prompt
(see `agents/caspar.md`'s "The object has this shape:" block for the pattern to copy). The
placeholder cannot be echoed and mistaken for a verdict; the worked example, sitting
outside the markers, is never even a candidate — that is the whole point of extracting
instead of searching.

---

## Two known limitations — said plainly, not softened

### 1. The guard's canary does not catch a *modified* echo of the example

Retry feedback (`retry_feedback.py`) and the orchestrator's echo canary
(`verdict_markers.ECHO_CANARY`) catch a model that copies the worked example **verbatim** —
the canary compares the model's `summary`/`recommendation` fields against the exact
placeholder text those fields carry in the shipped example. If a model echoes the example
but changes a word or two, the canary does not fire, and the run proceeds with that
verdict.

This is **not a parser failure** — the object is a real, well-formed verdict with all seven
keys, decoded from between the real markers, exactly as the sentinel is supposed to accept.
What it actually indicates is a **degraded judge**: the model produced a verdict that is
mostly borrowed scaffolding rather than its own analysis — the same class of problem as a
Caspar seat that is "present but under-powered" (`CLAUDE.md`, Key Design Decisions), just
caused by echoing instead of by weak reasoning. Detecting this kind of failure — a seat
that is degraded **without erroring** — is an open question tracked as its own milestone
(`sbtdd/spec-behavior-base-MS6.md`, "seat degradation without failure"), not something this
guard is meant to solve. Until that lands, treat it as a signal to iterate the prompt or
rotate the model, and spot-check a sample of runs for near-echoes as part of any milestone
or release review — this is a manual inspection step, deliberately not an automated one:
distinguishing "similar because it echoed" from "similar because the analysis converged" is
a judgment call, not a string comparison.

### 2. The guard rejects ANY complete 7-key object between the markers — by design, not by accident

There is no special case that lets a "clearly labeled example" through. Any JSON object
between the markers that has all seven required keys is rejected, full stop, regardless of
how obviously placeholder-ish its values look to a human reader. This is deliberate: **a
complete 7-key object between the markers is what a worked example *is*.** There is no
reliable way to distinguish "an example the author intentionally left in" from "an example
the author forgot to remove" from the object's shape alone — and giving the guard a way to
let some complete objects through would reopen exactly the ambiguity the sentinel exists to
close. If you have a legitimate reason to show a complete verdict for illustration, it goes
outside the markers, not inside them with a comment explaining that it's "just an example."

---

## A note on encoding: why a BOM doesn't trigger the marker-count FATAL

Prompt files are read with `encoding="utf-8-sig"`, which strips a leading UTF-8 byte-order
mark before any line is examined. This exists because of an earlier design mistake in this
same guard: an earlier version used one shared strictness level for both "is this the
model's untrustworthy output" and "is this our own shipped file," and a BOM at the top of a
`.md` file — invisible in every editor, harmless to Markdown renderers — was enough to make
the first line fail an exact-match check and abort the run with a false `[FATAL]`. The fix
was not to loosen the check; it was to resolve the encoding question at the encoding layer,
before any line comparison happens at all. See the docstring of `VerdictSentinel` in
`skills/magi/scripts/verdict_markers.py` for the two-predicates design this reflects (one
permissive, for the model's untrusted output; one strict, for files we ship — and why a
single shared predicate failed twice, in both directions, before this split was found).

---

## Customising a prompt: check it before you run it

The guard is strict, and a rejected prompt aborts the run. You do not have to discover that by
starting one:

```bash
python skills/magi/scripts/run_magi.py --check-prompts
```

It validates the shipped `agents/` directory against the marker contract and exits — `0` if
every prompt is fine, `1` with the offending file and the reason if not. No tokens, no
network, and it is the *same* guard the run uses, not a second implementation that could drift
from it.

---

## Line breaks: which separators count

A marker must be alone on its line, and "a line" means what JSON escapes inside a string:
`\n`, `\r\n`, `\r`. It deliberately does **not** include `U+2028` (LINE SEPARATOR) or `U+2029`
(PARAGRAPH SEPARATOR), because those are **legal raw inside a JSON string** — `json.loads`
accepts them. Splitting on them would mean that a verdict quoting `</MAGI_VERDICT>` in one of
its own fields (which is exactly what MAGI produces when it reviews itself) could have its
block cut short, and a perfectly valid verdict would be thrown away.

Trailing or leading separators are harmless: `U+2028`/`U+2029` are whitespace, and a candidate
line is stripped before it is compared, so `<MAGI_VERDICT>` followed by a `U+2028` is still the
marker. The only case that fails is a model that uses `U+2028` as its *sole* line terminator —
and that fails **closed**, with a retry whose feedback shows the model the exact ASCII to emit.

---

## Characters that *look* like the markers but are not

The parser normalises **invisible** characters out of a candidate line before comparing it —
every Unicode `Cf` (format) and `Mn` (non-spacing mark) codepoint, which covers zero-width
spaces, the BOM, soft hyphens, variation selectors, and the rest. A model that emits
`<MAGI_VERDICT>` with a zero-width space wedged inside it still emits the marker, and killing
that mage over an invisible character would be a retry thrown away.

It does **not** normalise homoglyphs, and that is deliberate. A fullwidth `＜` (`U+FF1C`) is
not an invisible character — it is a **different character**, in a different Unicode category
(`Sm`). Accepting it would mean accepting, as the marker, something that is not the marker,
which is precisely the laxity this whole mechanism exists to remove. So a model that writes
`＜MAGI_VERDICT＞` has not written the marker: extraction fails closed, and the mage is retried.

**This is not a dead end for the model.** The retry's corrective feedback carries the marker
lines *literally* (`retry_feedback.py` interpolates the real `VERDICT_OPEN` / `VERDICT_CLOSE`
constants into the instruction), so the model is shown the exact ASCII it must emit. If a
particular model does this *persistently*, that is a model-selection problem, not a parser
problem: watch `extraction_failures` (cause `missing_markers`) for that seat, and iterate its
prompt or rotate it to a different model. Loosening the marker comparison to accept
lookalikes would trade a noisy, recoverable failure for a silent, unrecoverable one.

Case, on the other hand, **is** forgiven: `<magi_verdict>` is accepted. A model that lowercased
the tag still emitted the marker, and there is no way for a lowercase mention to be mistaken
for anything else — it can only ever fail closed, never open.

---

## References

- Implementation: `skills/magi/scripts/prompt_guard.py`, `skills/magi/scripts/verdict_markers.py`
- Tests: `tests/test_prompt_guard.py`, `tests/test_agent_prompt_contract.py`
- Design rationale: [`docs/adr/0001-no-runtime-heuristic-fallback.md`](adr/0001-no-runtime-heuristic-fallback.md)
- The prompts themselves: `skills/magi/agents/melchior.md`, `balthasar.md`, `caspar.md`
  (see the "Output format" section of any of them for the shape to follow if you add a
  fourth agent or edit these).
