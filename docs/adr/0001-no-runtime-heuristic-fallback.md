# ADR 0001 — No runtime fallback to the heuristic verdict recovery

- **Status:** Accepted (v5.1.0, MS2 — the verdict sentinel)
- **Date:** 2026-07-13
- **Deciders:** Julian Bolivar, with the MAGI gate (Melchior / Balthasar / Caspar)

## Why this record exists

The proposal below was raised — and rejected — **seven times** during the MS2 design gate.
Every time by the pragmatist seat, every time in good faith, and every time for the same
reasonable-sounding reason: *"what if the measurement surprises us?"*

A rejection that lives only in a review log **gets re-litigated the first day someone is in a
hurry**. And the day someone is in a hurry is exactly the day a switch that "just makes the
problem go away" becomes irresistible.

So it is written down here, with its evidence, so that the next person does not have to
re-derive it under pressure.

## The proposal

> Ship a feature flag (or config key) that re-enables the old heuristic verdict recovery, so
> that if the sentinel turns out to reject too many verdicts in production, operators can turn
> it back on without a rollback.

## The decision

**Rejected. There will be no runtime fallback to the heuristic.**

## Why — and this is the whole point

**A flag that re-enables the heuristic is not a safety valve. It is a switch that restores the
silent fabrication of an `approve` in the adversarial seat.**

The old parser did not *find* the verdict — it **searched** for it: it decoded every JSON object
in the agent's output and kept the one that *looked* like a verdict. That heuristic has two
failure modes, and one of them is silent:

| Failure | What happens |
|---|---|
| **False negative** | Two candidates → ambiguity guard → the mage is dropped. **Noisy.** The run degrades and nobody is fooled. |
| **False positive** | The only decodable object is the **example from the agent's own system prompt** — which literally carries `"verdict": "approve"` — so **that** becomes the verdict. **SILENT.** The consensus cannot distinguish it from a real one. |

The false positive is the one that matters, and it lands in **Caspar's chair** — the adversarial
seat, whose entire job is to be the hardest to convince. A fabricated `approve` there is not "a
bug in a parser". It is MAGI approving something **nobody reviewed**, while reporting that three
independent judges agreed.

**A flag does not make that failure less likely. It makes it a supported configuration.**

## The evidence

- Across **171 captured agent outputs** from real runs, the heuristic **fabricated nothing** —
  it rescued 7 genuine verdicts. That is the honest number, and it is why the residual stayed
  open for so long: it never bit us in production.
- But **it was demonstrated with constructed inputs**, and pinned by characterization tests that
  *asserted the bug*. Three of the four residual variants were **silent fabrications**. The
  residual sat `LOCKED` in `CLAUDE.techdebt.md` for months — not because it was theoretical, but
  because **nobody could close it without the rotation of MS1**: being strict meant trading
  *silent fabrication* for *frequent degraded runs*, and nobody would sign that trade.
- MS1 removed that cost. MS2 closes the residual. **Re-introducing the fallback re-introduces the
  residual entirely** — and would make the whole milestone theatre.

## What we have instead — three real nets, and they are not the same net

1. **R18 — adherence telemetry.** `magi-report.json` carries, per agent, a counter of extraction
   failures by cause. This is the one that **sees** the problem: if a model starts omitting the
   markers after a silent update under the same tag, the counter moves. Without it, the symptom
   would look like *"MAGI is slow and rotates a lot"* — a symptom nobody would know how to read.
2. **`--max-attempts`.** Stochastic drift is absorbed by the retry, which now carries
   **cause-specific corrective feedback** ("you did not emit the markers").
3. **Downgrade to v5.0.x.** The plugin is installed **versioned**, so reverting is a user action,
   not an emergency re-implementation. **This is the production safety net** — which is exactly
   why it is documented in the release notes: *a net the user does not know about is not a net.*

## What would change this decision

Real data, not a hypothetical. If R18 shows a **systematic** marker-omission rate in production
that the retry cannot absorb, the answer is to **fix the seat** — iterate the prompt, or rotate
the model — **not to re-enable a parser that guesses**.

If we ever want tolerance for a model that cannot follow the contract, the honest form of it is a
*different contract*, not a *silent search*.

## The line to remember

> **Between a failure that is loud and expensive (a degraded run, re-run the gate) and one that is
> silent and cheap (a fabricated `approve` nobody detects), take the loud one. Every time.**
