# Windows ACL fixtures

Real output of the two tools the Windows permission guard reads, captured on
Windows 11 on 2026-11-20 (measurement M9). The tests in
`tests/test_magi_key_permissions.py` replay these shapes instead of an imagined
one.

| File | Command |
|---|---|
| `whoami-nh.txt` | `%SystemRoot%\System32\whoami.exe /user /fo csv /nh` |
| `sddl-file-hardened.txt` | `icacls <file> /inheritance:r /grant:r *<SID>:F`, then `powershell.exe -NoProfile -NonInteractive -Command "(Get-Acl -LiteralPath $env:MAGI_ACL_PATH).Sddl"` |
| `sddl-dir-hardened.txt` | `icacls <dir> /inheritance:r /grant:r *<SID>:(OI)(CI)F`, then the same `Get-Acl` read |
| `sddl-inherited.txt` | the same `Get-Acl` read on a file created under `%TEMP%`, never hardened |

The only edit: the real user SID was replaced by `S-1-5-21-1111-2222-3333-1001`
and the real host and user name by `host\me`. Everything else, CRLF line
endings included, is as the tools printed it. With `core.autocrlf` a checkout
may store LF instead; the tests strip the line ending, so either works.
