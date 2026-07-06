# Security Policy

## Supported versions

Zenith is pre-1.0 and under active development. Security fixes are applied to the
latest released minor version.

| Version | Supported |
| ------- | --------- |
| 0.7.x   | ✅        |
| < 0.7   | ❌        |

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Instead, report it privately through GitHub:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** (private vulnerability reporting).

Include a description, reproduction steps, affected version, and any relevant
logs. You can expect an acknowledgement within a few days. Once a fix is prepared
and released, the report will be disclosed with credit (unless you prefer to
remain anonymous).

## Scope

Zenith is a library that trains and serves models you provide. Note that:

- **Checkpoints are pickled (`torch.save`)** — only load checkpoints you trust,
  as with any PyTorch model file.
- The **serving app** (`zenith serve`) is a minimal FastAPI service with no
  authentication; run it behind your own auth/proxy for anything public.
