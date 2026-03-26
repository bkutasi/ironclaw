# Telegram Integration Patterns

**Purpose**: Telegram bot integration with IronClaw agent via webhook or polling

**Last Updated**: 2026-03-22

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Core Concept

IronClaw integrates with Telegram using WASM channels that support two modes: **webhook** (instant, production) or **polling** (simple, reliable). Webhook requires Cloudflare tunnel; polling works outbound-only.

---

## Key Points

- **Two modes**: Webhook (instant) vs Polling (30s delay, no tunnel needed)
- **Cloudflare tunnel**: Required for webhook, ephemeral tunnels bypass Bot Fight Mode
- **Common failure**: Trailing newline in tunnel URL → "Failed to resolve host"
- **Bot Fight Mode**: Blocks Telegram webhook with 403, needs exception rule
- **Rate limits**: Telegram returns 429 on duplicate webhook registration

---

## Quick Example

```bash
# Polling mode (simple, no tunnel)
export NGC_KEY="key" && export TELEGRAM_BOT_TOKEN="token"
./target/release/ironclaw

# Webhook mode with ephemeral tunnel (testing)
./.tmp/start-ironclaw-ephemeral.sh

# Webhook mode with persistent tunnel (production)
# Add Cloudflare exception: Security → Bots → Skip for /webhook/telegram
./.tmp/start-ironclaw-webhook.sh
```

---

## Working Configuration (Polling Mode)

```bash
# Environment variables
export LLM_BACKEND=openai_compatible
export LLM_BASE_URL=https://integrate.api.nvidia.com/v1
export LLM_API_KEY=$NGC_KEY
export LLM_MODEL=z-ai/glm5
export GATEWAY_PORT=3004
export HTTP_PORT=8081
# DO NOT set TUNNEL_URL (forces polling mode)

# Start
export NGC_KEY="your-key"
./target/release/ironclaw
```

---

## Verification Commands

```bash
# Check polling mode active
tail -20 /tmp/ironclaw.log | grep -E "(Polling mode|on_poll)"

# Monitor polling activity (every 30s)
tail -f /tmp/ironclaw.log | grep -E "(getUpdates|emitted_count)"

# Check Telegram webhook status (should be empty for polling)
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo" | jq .result
```

---

## Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| `TUNNEL_URL` set | Webhook mode instead of polling | `unset TUNNEL_URL` or fresh shell |
| Multiple shell sessions | Vars not propagating | Check `env \| grep TUNNEL_URL` |
| Script can't unset vars | Env var persists | Start fresh shell explicitly |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `403 Forbidden` | Cloudflare Bot Fight Mode | Add exception rule |
| `429 Too Many Requests` | Duplicate webhook registration | Wait 60s, don't register twice |
| `Failed to resolve host` | Trailing newline in URL | `echo -n "$URL"` + `.trim()` in Rust |
| `HTTP 000` | Tunnel not connected | Check cloudflared logs |

---

## Reference

- Full setup: `docs/TELEGRAM_POLLING_VS_WEBHOOK.md`
- Debug history: `.tmp/TELEGRAM_DEBUG_HANDOFF.md`
- Scripts: `.tmp/start-ironclaw-*.sh`
- Polling summary: `.tmp/TELEGRAM_POLLING_FINAL_SUMMARY.md`

---

**Related**: 
- errors/telegram-webhook-errors.md
- guides/telegram-mode-selection.md
