# HTTP Port Binding Configuration

**Core Idea**: IronClaw's HTTP channel auto-enables when `HTTP_WEBHOOK_SECRET` is set, defaulting to port 8081. Missing `settings.json` causes fallback to 8080, which conflicts with services like qbittorrent. Config resolution order: `HTTP_PORT` env var → `HTTP_HOST` env var → `HTTP_WEBHOOK_SECRET` presence → settings file → default 8080.

**Key Points**:
- HTTP channel is enabled if ANY of `HTTP_PORT`, `HTTP_HOST`, or `HTTP_WEBHOOK_SECRET` env vars are set
- When `HTTP_WEBHOOK_SECRET` is set without explicit `HTTP_PORT`, default port is 8081 (webhook standard)
- `settings.json` missing → `Settings::default()` → `http_enabled: false` → channel not created unless env vars present
- Fallback in `src/main.rs` uses hardcoded `0.0.0.0:8080` when `config.channels.http` is `None`
- `optional_env()` checks real env vars, runtime overrides, and DB-injected secrets (INJECTED_VARS)

**Quick Example**:
```bash
# Simplest webhook setup — port defaults to 8081
export HTTP_WEBHOOK_SECRET="your-secret"
export TUNNEL_URL="https://your-tunnel.trycloudflare.com"
./target/release/ironclaw

# Explicit port control
export HTTP_PORT=8081
export HTTP_WEBHOOK_SECRET="your-secret"
./target/release/ironclaw
```

**Reference**: `.tmp/HTTP_PORT_FIX_SUMMARY.md`, `.tmp/HTTP_PORT_DEBUG_REPORT.md`

**Related**: `core/telecom/errors/telegram-polling-mode-fix.md`, `core/telecom/concepts/script-design-patterns.md`
