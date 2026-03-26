<!-- Context: telecom/guides | Priority: critical | Version: 1.0 | Updated: 2026-03-19 -->
# Guide: Telegram Mode Selection

**Purpose**: Choose between webhook and polling mode for Telegram integration

**Last Updated**: 2026-03-19

---

## Quick Decision

| Mode | Use When | Delay | Setup | Stability |
|------|----------|-------|-------|-----------|
| **Polling** | Development, reliability | 5-30s | 2 env vars | ⭐⭐⭐⭐⭐ Never dies |
| **Webhook** | Production, instant | <1s | Tunnel + config | ⭐⭐⭐ Can die |

**Recommendation**: **Use polling mode** unless you need instant responses.

---

## Why Polling Mode is Recommended

### ✅ Advantages
- **No tunnel needed** - No Cloudflare, no DNS, no port forwarding
- **Never dies** - No 30-60 minute tunnel timeouts
- **Simple setup** - Just 2 environment variables
- **No 530 errors** - Outbound-only, firewall-friendly
- **No URL changes** - Stable connection

### ⚠️ Trade-offs
- **5-30 second delay** - Acceptable for most use cases
- **Continuous polling** - Minimal resource usage (~1% CPU)

---

## When to Use Webhook Mode

Only use webhook if:
- You need **instant responses** (<1s)
- You have **persistent tunnel** with your domain
- You can monitor tunnel health
- You've added Cloudflare Bot Fight Mode exception

**Don't use webhook** if:
- Using ephemeral tunnels (they die after 30-60 min)
- You can't monitor tunnel health
- Reliability > speed

---

## Setup Comparison

### Polling Mode (Recommended)
```bash
# Just 2 environment variables
export NGC_KEY="nvapi-your-key"
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."

# Start IronClaw
./target/release/ironclaw

# That's it - no tunnel, no webhook config
```

### Webhook Mode (Advanced)
```bash
# Requires tunnel setup
cloudflared tunnel --url http://localhost:8081 &
sleep 30  # Wait for DNS

# Get tunnel URL
TUNNEL_URL=$(cat /tmp/cloudflared-ephemeral-url.txt)

# Register webhook
curl -X POST "https://api.telegram.org/bot$TOKEN/setWebhook?url=$TUNNEL_URL/webhook/telegram"

# Start IronClaw
./target/release/ironclaw
```

---

## Migration: Webhook → Polling

```bash
# 1. Stop IronClaw
pkill ironclaw

# 2. Delete webhook (optional, Telegram auto-removes)
curl -X POST "https://api.telegram.org/bot$TOKEN/deleteWebhook"

# 3. Kill tunnel
pkill cloudflared

# 4. Start in polling mode
export NGC_KEY="..."
export TELEGRAM_BOT_TOKEN="..."
./target/release/ironclaw
```

---

## 📂 Codebase References

**Scripts**:
- `.tmp/start-ironclaw-webhook.sh` - Webhook mode startup
- `.tmp/start-ironclaw-ephemeral.sh` - Ephemeral tunnel setup

**Implementation**:
- `src/channels/telegram.rs` - Telegram channel (both modes)
- `src/channels/polling.rs` - Polling logic
- `src/http/server.rs` - Webhook server

**State**:
- `~/.ironclaw/channels/telegram/state/` - Channel state
- `/tmp/cloudflared-*.log` - Tunnel logs

---

## Related

- telegram-integration.md
- errors/tunnel-errors.md
- errors/bot-no-response.md
- guides/ephemeral-tunnel-setup.md
