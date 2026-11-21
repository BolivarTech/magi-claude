# magi-rs vault fixture

`ls.txt` is the unedited output of `magi-rs vault ls -w <workdir>` (magi-rs 0.25.0), captured on
2026-11-20 against a throwaway workspace whose vault held two names. Each line is
`NAME · created · updated`, separated by U+00B7 (middle dot). Values are never listed.

`ls-empty.txt` is the same command against an empty vault (measurement M1): the single line
`(vault empty)`, exit 0.
