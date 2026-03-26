# Telegram Mode Selection: Webhook vs Polling

This guide helps you choose between webhook and polling modes for Telegram bot integration.

## Overview

IronClaw supports two modes for receiving Telegram messages:

1. **Webhook Mode** (default, recommended for production)
2. **Polling Mode** (recommended for development)

## Mode Comparison

| Feature | Webhook | Polling |
|---------|---------|---------|
| **Latency** | Instant (<1s) | Delayed (30s interval) |
| **Reliability** | Depends on tunnel | Very high |
| **Resource Usage** | Low (event-driven) | Medium (periodic requests) |
| **Setup Complexity** | Medium (requires tunnel) | Low (just token) |
| **Production Ready** | ✅ Yes | ⚠️ Limited |
| **Development** | ⚠️ Ephemeral tunnels expire | ✅ Stable |
| **Rate Limits** | Telegram: 30/min | Telegram: 30/min |
| **Pending Updates** | Queued by Telegram | Fetched on poll |

## Webhook Mode

### How It Works

```
User Message → Telegram API → Cloudflare Tunnel → IronClaw Webhook
```

### Requirements

1. Cloudflare tunnel running (`cloudflared`)
2. Public HTTPS URL
3. IronClaw running on port 3000

### Configuration

```bash
# ~/.ironclaw/.env
TELEGRAM_BOT_TOKEN="123456789:AAHccDDeeFFggHHiiJJkkLLmmNNoo"

# No special config needed - webhook is default when tunnel detected
```

### Starting Tunnel

```bash
# Ephemeral tunnel (temporary URL)
cloudflared tunnel --url http://localhost:3000

# Output:
# +--------------------------------------------------------------------+
# |  Your quick Tunnel has been created!                               |
# |  Visit it at: https://abc123.trycloudflare.com                     |
# +--------------------------------------------------------------------+

# Copy the URL and restart IronClaw
```

### Persistent Tunnel (Production)

```bash
# Create named tunnel
cloudflared tunnel create ironclaw-prod

# Configure ~/.cloudflared/config.yml
tunnel: ironclaw-prod
credentials-file: /home/user/.cloudflared/ironclaw-prod.json
ingress:
  - hostname: bot.yourdomain.com
    service: http://localhost:3000
  - service: http_status:404

# Run tunnel
cloudflared tunnel run ironclaw-prod

# Or as systemd service
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

### Pros

- ✅ Instant message delivery
- ✅ Lower API usage (Telegram pushes to you)
- ✅ Better for high-volume bots
- ✅ Professional setup

### Cons

- ❌ Requires tunnel management
- ❌ Ephemeral tunnels expire (24hr)
- ❌ DNS propagation delays
- ❌ More failure points

### Troubleshooting Webhook

```bash
# Check webhook status
curl -s "https://api.telegram.org/bot$TOKEN/getWebhookInfo" | jq

# Check tunnel running
pgrep -af cloudflared

# Test endpoint
curl -I https://your-tunnel.trycloudflare.com/webhook/telegram

# Auto-fix
telegram_fix_common_issues
```

## Polling Mode

### How It Works

```
IronClaw Poll → Telegram API (getUpdates) → Receive Messages
     ↑                                        ↓
     └────────── Wait 30s ───────────────────┘
```

### Configuration

```bash
# ~/.ironclaw/.env
TELEGRAM_BOT_TOKEN="123456789:AAHccDDeeFFggHHiiJJkkLLmmNNoo"
TELEGRAM_POLLING=true

# Or runtime flag
cargo run -- --telegram-polling
```

### Starting in Polling Mode

```bash
# Method 1: Environment variable
export TELEGRAM_POLLING=true
cargo run

# Method 2: Command line flag
cargo run -- --telegram-polling

# Method 3: Config file
# Add to ~/.ironclaw/.env:
TELEGRAM_POLLING=true
```

### Pros

- ✅ No tunnel required
- ✅ Very reliable
- ✅ Simple setup
- ✅ Works behind NAT/firewall
- ✅ No DNS issues

### Cons

- ❌ 30-second delay (poll interval)
- ❌ Higher API usage
- ❌ Not suitable for high-volume
- ❌ Wastes resources when idle

### Troubleshooting Polling

```bash
# Check if polling enabled
grep TELEGRAM_POLLING ~/.ironclaw/.env

# Check logs for poll activity
RUST_LOG=debug cargo run 2>&1 | grep -i "poll\|getUpdates"

# Check for API errors
curl -s "https://api.telegram.org/bot$TOKEN/getUpdates?offset=-1" | jq
```

## Mode Selection Guide

### Choose Webhook If:

- ✅ Production deployment
- ✅ Need instant responses
- ✅ High message volume (>100/day)
- ✅ Have stable infrastructure
- ✅ Comfortable with tunnel management

### Choose Polling If:

- ✅ Local development
- ✅ Testing/debugging
- ✅ Low message volume (<50/day)
- ✅ Behind NAT/firewall
- ✅ Want simplest setup
- ✅ Don't mind 30s delay

## Switching Modes

### Webhook → Polling

```bash
# 1. Stop IronClaw
pkill ironclaw

# 2. Delete webhook (optional, IronClaw does this automatically)
curl -X POST "https://api.telegram.org/bot$TOKEN/deleteWebhook"

# 3. Enable polling in config
echo 'TELEGRAM_POLLING=true' >> ~/.ironclaw/.env

# 4. Start IronClaw
cargo run
```

### Polling → Webhook

```bash
# 1. Stop IronClaw
pkill ironclaw

# 2. Disable polling in config
sed -i '/TELEGRAM_POLLING/d' ~/.ironclaw/.env

# 3. Start tunnel
cloudflared tunnel --url http://localhost:3000
# Copy the URL

# 4. Start IronClaw (auto-registers webhook)
cargo run
```

## Hybrid Approach

For production with fallback:

```bash
# Primary: Webhook mode
cloudflared tunnel run ironclaw-prod &
cargo run

# Fallback: If tunnel dies, switch to polling
# Monitor tunnel health
*/5 * * * * pgrep -f cloudflared || (pkill ironclaw && TELEGRAM_POLLING=true cargo run &)
```

## Performance Comparison

### Webhook (Production)

```
Message → Bot Response: 1-3 seconds
API Calls: 1 per message (setWebhook once)
Resource Usage: Minimal (event-driven)
```

### Polling (Development)

```
Message → Bot Response: 30-60 seconds (worst case)
API Calls: 2 per minute (every 30s)
Resource Usage: Low (periodic)
```

## Best Practices

### Webhook Mode

1. **Use persistent tunnels** for production
2. **Monitor tunnel health** with health checks
3. **Enable webhook secret** validation
4. **Have polling fallback** for outages
5. **Auto-restart tunnel** on failure

```bash
# Example: Systemd service for cloudflared
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=ironclaw
ExecStart=/usr/local/bin/cloudflared tunnel run ironclaw-prod
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Polling Mode

1. **Increase poll interval** for low-traffic bots
2. **Handle long-polling timeouts** gracefully
3. **Track offset** to avoid duplicates
4. **Rate limit** your own requests
5. **Log poll activity** for debugging

```bash
# Custom poll interval (if supported)
TELEGRAM_POLL_INTERVAL_MS=60000  # 60 seconds
```

## Diagnostic Commands

```bash
# Check current mode
if grep -q TELEGRAM_POLLING ~/.ironclaw/.env; then
  echo "Mode: Polling"
else
  echo "Mode: Webhook"
fi

# Check webhook status (webhook mode)
curl -s "https://api.telegram.org/bot$TOKEN/getWebhookInfo" | jq '.result.url'

# Check last poll time (polling mode)
RUST_LOG=debug cargo run 2>&1 | grep "Polling getUpdates" | tail -1

# Test both modes
telegram_full_report
```

## Migration Examples

### Example 1: Development to Production

**Before (Development)**:
```bash
TELEGRAM_POLLING=true
cargo run
```

**After (Production)**:
```bash
# Remove polling config
sed -i '/TELEGRAM_POLLING/d' ~/.ironclaw/.env

# Create persistent tunnel
cloudflared tunnel create ironclaw-prod

# Configure and run tunnel
cloudflared tunnel run ironclaw-prod

# Start IronClaw
cargo run
```

### Example 2: Quick Testing

**Temporary webhook**:
```bash
# Start tunnel in background
cloudflared tunnel --url http://localhost:3000 &
TUNNEL_PID=$!

# Start IronClaw
cargo run

# Cleanup
kill $TUNNEL_PID
pkill ironclaw
```

## Related

- `integrations/errors/tunnel-errors.md` - Tunnel troubleshooting
- `skills/telegram-debug/SKILL.md` - Automated diagnostics
- `channels-src/telegram/README.md` - Telegram channel implementation
