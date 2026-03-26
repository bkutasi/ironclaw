# Telegram Polling Mode Fix

**Core Idea**: IronClaw's mode selection follows a priority chain: database config → env var → default (polling). The `workspace_kv` table stores `tunnel_url` under key `channels/telegram/state/tunnel_url`, which overrides env vars and forces webhook mode even when polling is intended. Clearing this key from the database is required to switch to polling.

**Key Points**:
- `workspace_kv` key `channels/telegram/state/tunnel_url` persists across restarts and overrides `TUNNEL_URL` env var
- Mode selection order: (1) DB `tunnel_url` exists → webhook, (2) `TUNNEL_URL` env set → webhook, (3) neither → polling
- SQL fix: `DELETE FROM workspace_kv WHERE key LIKE '%telegram%tunnel%';`
- `TUNNEL_URL` env var in shell session from previous webhook setup also forces webhook mode — unset it explicitly
- Polling works because IronClaw initiates outbound connection (not blocked by Cloudflare)

**Quick Example**:
```sql
-- Clear stale tunnel URL to enable polling mode
DELETE FROM workspace_kv
WHERE key IN (
  'channels/telegram/state/tunnel_url',
  'channels/telegram/tunnel_url',
  'telegram/tunnel_url'
);
```

**Reference**: `.tmp/POLLING_MODE_FIX.md`

**Related**: `core/architecture/errors/http-port-binding.md`, `core/integrations/lookup/telegram-cloudflare-reference.md`
