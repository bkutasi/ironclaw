# IronClaw Startup Script Design Patterns

**Core Idea**: Startup scripts should NOT register webhooks (IronClaw handles that internally — duplicate registration causes HTTP 429 rate limits). Scripts should focus on pre-flight validation, auto-generating secrets, detecting config conflicts, and graceful shutdown. Let IronClaw own the webhook lifecycle.

**Key Points**:
- Never register webhook in scripts — IronClaw registers it on startup; duplicate calls hit Telegram rate limits (429)
- Auto-generate `HTTP_WEBHOOK_SECRET` with `openssl rand -hex 32` if not already set
- Pre-flight checks: verify binary exists, DB is reachable, required env vars are present — fail fast with actionable errors
- Graceful shutdown: use `trap` on EXIT/INT/TERM to clean up child processes (tunnel, IronClaw)
- Detect `TUNNEL_URL` conflicts: warn and unset when running in polling mode to avoid silent webhook fallback

**Quick Example**:
```bash
#!/bin/bash
trap 'echo "Shutting down..."; kill $(jobs -p) 2>/dev/null; exit' EXIT INT TERM

# Pre-flight: validate env
[[ -z "$NGC_KEY" ]] && { echo "ERROR: NGC_KEY not set"; exit 1; }

# Auto-generate secret
[[ -z "$HTTP_WEBHOOK_SECRET" ]] && export HTTP_WEBHOOK_SECRET=$(openssl rand -hex 32)

# Detect TUNNEL_URL conflict in polling mode
[[ -n "$TUNNEL_URL" ]] && { echo "WARN: Unsetting TUNNEL_URL for polling"; unset TUNNEL_URL; }

./target/release/ironclaw
```

**Reference**: `.tmp/SCRIPT_IMPROVEMENTS.md`, `.tmp/WEBHOOK_SCRIPT_FIXED.md`

**Related**: `core/architecture/errors/http-port-binding.md`, `core/telecom/errors/telegram-polling-mode-fix.md`
