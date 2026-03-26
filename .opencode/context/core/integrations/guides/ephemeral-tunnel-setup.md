<!-- Context: integrations/guides | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Guide: Ephemeral Tunnel Setup

**Purpose**: Quick setup for testing with temporary Cloudflare tunnel

**Last Updated**: 2026-03-19

## Prerequisites

- Cloudflared binary installed
- IronClaw built and ready
- Port 8081 available

**Estimated time**: 2 min

## Steps

### 1. Install Cloudflared
```bash
# Linux (Debian/Ubuntu)
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# macOS
brew install cloudflared

# Verify
cloudflared --version
```
**Expected**: Version output

### 2. Clean Old Tunnel Files
```bash
rm -f /tmp/cloudflared-ephemeral-url.txt
rm -f /tmp/cloudflared-ephemeral.log
pkill cloudflared
```
**Expected**: Old files removed

### 3. Start Ephemeral Tunnel
```bash
# In background, forwarding to port 8081
cloudflared tunnel --url http://localhost:8081 > /tmp/cloudflared-ephemeral.log 2>&1 &

# Wait for tunnel to connect AND DNS propagation
sleep 30  # Critical: DNS needs 30-60 seconds

# Extract URL (no trailing newline)
TUNNEL_URL=$(grep "trycloudflare.com" /tmp/cloudflared-ephemeral.log | head -1 | sed 's/.*\(https:\/\/[^ ]*\).*/\1/')
echo -n "$TUNNEL_URL" > /tmp/cloudflared-ephemeral-url.txt
```
**Expected**: URL saved to file (no trailing newline)

**⚠️ Critical**: 
- Wait **30 seconds** for DNS propagation (not 5 seconds)
- Use `echo -n` to avoid trailing newline (causes 400 errors)
- Tunnel may die after 30-60 minutes - use polling for reliability

### 4. Verify Tunnel
```bash
# Check URL
cat /tmp/cloudflared-ephemeral-url.txt

# Test endpoint
curl -I "$(cat /tmp/cloudflared-ephemeral-url.txt)/webhook/telegram"
```
**Expected**: HTTP 200 or 404 (not 000, 530)

### 5. Start IronClaw with Auto-Detection
```bash
export NGC_KEY="your-key"
export TELEGRAM_BOT_TOKEN="your-token"

# Script auto-detects Docker PostgreSQL on port 5433
# If using system PostgreSQL, set DATABASE_URL explicitly:
# export DATABASE_URL="postgres://postgres:yourpass@localhost:5432/ironclaw"

./target/release/ironclaw
```
**Expected**: IronClaw starts and registers webhook

**Script Improvements** (in `.tmp/start-ironclaw-ephemeral.sh`):
- **30-second DNS wait** - Prevents 530 errors
- **Docker PostgreSQL auto-detection** - Checks port 5433 first
- **Rate limit recovery** - 60-second wait on 429 errors

## Verification

```bash
# Check tunnel running
ps aux | grep cloudflared

# Check URL accessible
curl -I "$(cat /tmp/cloudflared-ephemeral-url.txt)"

# Check webhook registered
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo" | jq .result.url
```

## 📂 Codebase References

**Scripts**:
- `.tmp/start-ironclaw-ephemeral.sh` - Automated ephemeral setup
- `scripts/test-tunnel.sh` - Tunnel verification script

**Logs**:
- `/tmp/cloudflared-ephemeral.log` - Tunnel logs
- `/tmp/ironclaw.log` - IronClaw logs

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `cloudflared: command not found` | Install cloudflared binary |
| URL has trailing newline | Use `echo -n "$URL"` |
| HTTP 530 error | Wait 30s for DNS propagation |
| Port 8081 in use | `pkill -f "localhost:8081"` |
| Tunnel died after 30-60 min | Restart tunnel or switch to polling mode |
| 429 rate limit | Wait 60 seconds, don't register webhook twice |

**Script Auto-Recovery**:
- Detects Docker PostgreSQL (port 5433) vs system PostgreSQL (5432)
- Waits 60 seconds on rate limit errors
- Verifies tunnel URL before starting IronClaw

## Related

- concepts/cloudflare-tunnels.md
- guides/persistent-tunnel-setup.md
- errors/tunnel-errors.md
