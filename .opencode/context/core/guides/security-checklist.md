# Security Checklist

**Purpose**: Security verification and incident response for IronClaw

**Last Updated**: 2026-03-22

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Core Concept

IronClaw security relies on file permissions, sandbox isolation, and secrets management. For personal use, plaintext config with `chmod 600` is acceptable; production requires encrypted secrets and disk encryption.

---

## Key Points

- **File permissions**: `~/.ironclaw/.env` must be `600` (`-rw-------`)
- **Sandbox policy**: Use `workspace_write` (not `full_access`) for Docker sandbox
- **Secrets storage**: Encrypted at rest in DB, master key in OS keychain or env var
- **Network isolation**: Docker sandbox uses network allowlist (not unrestricted)
- **Web gateway**: Binds to `127.0.0.1` (localhost only by default)

---

## Quick Verification

```bash
# File permissions
ls -la ~/.ironclaw/.env  # Expected: -rw-------

# Sandbox policy
grep SANDBOX_POLICY ~/.ironclaw/.env  # Expected: workspace_write

# Master key status
./target/release/ironclaw doctor | grep -A2 "Master Key"

# Running containers
docker ps --filter name=ironclaw

# Recent errors
tail -100 /tmp/ironclaw.log | grep -iE "error|warn"
```

---

## Incident Response

```bash
# 1. Stop Ironclaw
pkill -f ironclaw

# 2. Rotate Telegram token (@BotFather → Your Bot → Revoke Token)

# 3. Change database password
docker exec -it ironclaw-db psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'new';"

# 4. Review logs
grep "401\|403\|Unauthorized" /tmp/ironclaw.log

# 5. Check workspace modifications
find ~/.ironclaw/projects -mtime -1 -type f
```

---

## Reference

- Full checklist: `.tmp/SECURITY_CHECKLIST.md`
- Security patterns: `standards/security-patterns.md`

---

**Related**:
- architecture/concepts/docker-postgres.md
- standards/security-patterns.md
