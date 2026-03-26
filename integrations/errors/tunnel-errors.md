# Telegram Tunnel Errors

This document covers common tunnel-related errors when running Telegram bot in webhook mode.

## Overview

IronClaw uses Cloudflare tunnels (`cloudflared`) to expose a public HTTPS endpoint for Telegram webhooks. When the tunnel dies or becomes inaccessible, the bot stops receiving messages.

## Common Tunnel Errors

### 1. Tunnel Process Died

**Symptoms**:
- Bot stops responding
- Webhook shows as registered but no updates received
- `cloudflared` process not running

**Detection**:
```bash
pgrep -f cloudflared
# No output = process not running
```

**Causes**:
- System restart
- OOM killer terminated process
- Manual termination
- Cloudflare service outage

**Fix**:
```bash
# Restart tunnel
cloudflared tunnel --url http://localhost:3000

# Or use IronClaw auto-restart
telegram_fix_common_issues
```

### 2. DNS Resolution Failure

**Symptoms**:
- Tunnel URL doesn't resolve
- Telegram can't reach webhook
- DNS lookup fails

**Detection**:
```bash
dig +short abc123.trycloudflare.com
# No output or NXDOMAIN = DNS failure
```

**Causes**:
- Ephemeral tunnel expired (24hr lifetime)
- DNS cache stale
- Cloudflare DNS issues

**Fix**:
```bash
# Clear DNS cache
sudo systemd-resolve --flush-caches  # Linux
sudo dscacheutil -flushcache         # macOS

# Restart tunnel to get new URL
cloudflared tunnel --url http://localhost:3000

# Update webhook with new URL
telegram_fix_common_issues
```

### 3. HTTP Endpoint Unreachable

**Symptoms**:
- Tunnel running but HTTP requests fail
- Telegram returns "retry later" errors
- Connection timeout

**Detection**:
```bash
curl -I https://abc123.trycloudflare.com/webhook/telegram
# Should return HTTP 200 or 404
```

**Causes**:
- IronClaw not running on expected port
- Firewall blocking connections
- Tunnel misconfigured

**Fix**:
```bash
# Check IronClaw running
ps aux | grep ironclaw

# Check port listening
netstat -tlnp | grep 3000

# Restart IronClaw
cargo run

# Verify tunnel URL matches IronClaw port
cloudflared tunnel --url http://localhost:3000
```

### 4. Webhook URL Mismatch

**Symptoms**:
- Tunnel running with different URL than registered
- Bot receives some messages but not others
- Telegram sending to old URL

**Detection**:
```bash
# Get current webhook URL from Telegram
curl -s "https://api.telegram.org/bot$TOKEN/getWebhookInfo" | jq -r .result.url

# Compare with running tunnel URL
echo $TUNNEL_URL
```

**Causes**:
- Tunnel restarted with new ephemeral URL
- Webhook not updated after tunnel restart
- Manual webhook changes

**Fix**:
```bash
# Auto-fix
telegram_fix_common_issues

# Or manually re-register
curl -X POST "https://api.telegram.org/bot$TOKEN/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"https://new-url.trycloudflare.com/webhook/telegram\"}"
```

### 5. Rate Limiting by Cloudflare

**Symptoms**:
- Intermittent 429 Too Many Requests
- Messages delayed or dropped
- Tunnel returns errors under load

**Causes**:
- High message volume
- Free tier limits
- Bot spam

**Fix**:
```bash
# Reduce message frequency
# Upgrade to Cloudflare paid plan
# Switch to polling mode for high-volume bots
export TELEGRAM_POLLING=true
cargo run
```

## Prevention

### Use Persistent Tunnels

Instead of ephemeral tunnels, create a named tunnel:

```bash
# Create tunnel
cloudflared tunnel create my-ironclaw

# Configure config.yml
cat > ~/.cloudflared/config.yml <<EOF
tunnel: my-ironclaw
credentials-file: /home/user/.cloudflared/my-ironclaw.json
ingress:
  - hostname: ironclaw.example.com
    service: http://localhost:3000
  - service: http_status:404
EOF

# Run tunnel
cloudflared tunnel run my-ironclaw
```

### Monitor Tunnel Health

Add health check to cron:
```bash
# Check every 5 minutes
*/5 * * * * curl -s https://ironclaw.example.com/webhook/telegram || \
  systemctl restart cloudflared
```

### Use Polling Mode for Development

For local development, polling is more reliable:

```bash
# In ~/.ironclaw/.env
TELEGRAM_POLLING=true

# Or runtime flag
cargo run -- --telegram-polling
```

## Debugging Commands

```bash
# Check tunnel process
pgrep -af cloudflared

# View tunnel logs
journalctl -u cloudflared -f

# Test DNS resolution
dig +short tunnel-url.trycloudflare.com

# Test HTTP endpoint
curl -v https://tunnel-url.trycloudflare.com/webhook/telegram

# Check webhook status
curl -s "https://api.telegram.org/bot$TOKEN/getWebhookInfo" | jq

# Compare URLs
WEBHOOK=$(curl -s "https://api.telegram.org/bot$TOKEN/getWebhookInfo" | jq -r .result.url)
TUNNEL=$(cloudflared tunnel list | grep ironclaw | awk '{print $3}')
echo "Webhook: $WEBHOOK"
echo "Tunnel:  $TUNNEL"
```

## Related

- `integrations/errors/bot-no-response.md` - Bot response troubleshooting
- `telecom/guides/telegram-mode-selection.md` - Webhook vs polling selection
- `skills/telegram-debug/SKILL.md` - Automated diagnostics
