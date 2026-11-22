# magi-rs code-review fixture

`one-commit.json` is the unedited stdout of a real `magi-rs code-review` run (magi-rs 0.25.0), captured on
2026-11-20 13:50 ET:

```
magi-rs code-review --commit 6dfc985 --output-format json --structured-verdicts -w <workspace> --timeout 16820
```

`<workspace>` was a throwaway clone of this public repository with a fresh `.magi/` created by `--ollama-init`
(the binary's default `magi.toml`). The run exited 0 in 136 s with three agent verdicts (`STRONG GO`, 3-0), one
batch (`batches_planned = 1`), and neither `without_memory` nor `without_embedder` set.

No byte was edited: the envelope holds no absolute path, user name or key (checked by grep before committing).
The envelope embeds the reviewed code, which is why the reviewed commit is a published one.

This file pins the envelope shape the plugin parses (`CodeReviewResult`). When the magi-rs pin is bumped, re-capture
it with the same command; a test that starts failing then is the signal that the envelope changed.
