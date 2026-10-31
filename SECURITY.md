# Security Policy

## Supported versions

Security fixes are applied to the latest released version.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/FahimFBA/lora-effsam/security/advisories/new),
or by email to fahimbinamin@gmail.com with:

* a description of the issue and its impact,
* steps or a minimal example to reproduce it,
* the affected version or commit.

You can expect an acknowledgement within 7 days and a status update within 30 days.
Once a fix is available, the issue will be disclosed in the release notes, with credit to
the reporter if desired.

## Scope

This is a research codebase. Relevant reports include unsafe deserialization of
checkpoints or data, command injection in scripts, and vulnerable dependencies.
Loading untrusted PyTorch checkpoints is inherently unsafe; only load files you trust.
