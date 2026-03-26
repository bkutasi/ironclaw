---
name: telegram-debug
version: 1.0.0
description: Diagnose and fix common Telegram integration issues automatically
activation:
  keywords:
    - telegram debug
    - telegram check
    - telegram health
    - telegram not working
    - bot not responding
    - telegram issues
    - webhook check
    - tunnel check
    - telegram doctor
    - fix telegram
  patterns:
    - "(?i)telegram.*not.*respond"
    - "(?i)bot.*not.*working"
    - "(?i)webhook.*issue"
    - "(?i)tunnel.*dead"
    - "(?i)telegram.*debug"
    - "(?i)check.*telegram"
  max_context_tokens: 4000
---

# Telegram Integration Debugger

Use this skill to automatically diagnose and fix common Telegram bot integration issues.

## Quick Start

```bash
# Run full health check
telegram_health_check

# Check only webhook status
telegram_check_webhook

# Verify tunnel is alive
telegram_check_tunnel

# Validate configuration
telegram_check_config

# Auto-fix common issues
telegram_fix_common_issues
```

## Tools

### telegram_check_webhook

Checks webhook status with Telegram API by calling `getWebhookInfo`:

**Checks**:
- ✅ Webhook URL is set (not empty)
- ✅ `pending_update_count` < 100 (not stuck)
- ✅ `last_error_message` is null or old
- ✅ `last_synchronization_error_date` is not recent
- ✅ Webhook URL matches expected tunnel URL

**Output Example**:
```
Webhook Status
==============
✅ URL: https://abc123.trycloudflare.com/webhook/telegram
✅ Pending Updates: 2
✅ Last Error: None
✅ Last Sync Error: Never
```

### telegram_check_tunnel

Verifies Cloudflare tunnel is alive and accessible:

**Checks**:
- ✅ `cloudflared` process is running
- ✅ Tunnel URL resolves via DNS
- ✅ Endpoint returns HTTP 200/404 (accessible)
- ✅ Tunnel URL matches Telegram webhook URL

**Output Example**:
```
Tunnel Health
=============
✅ Process: cloudflared running (PID 12345)
✅ DNS: abc123.trycloudflare.com resolves to 198.41.200.23
✅ HTTP: https://abc123.trycloudflare.com/webhook/telegram (HTTP 200)
✅ Match: Tunnel URL matches webhook URL
```

### telegram_check_config

Validates IronClaw configuration for Telegram:

**Checks**:
- ✅ `~/.ironclaw/.env` file exists
- ✅ `TELEGRAM_BOT_TOKEN` is set and valid format
- ✅ `LLM_MODEL` is correct (z-ai/glm5, not stepfun-ai/step-3.5-flash)
- ✅ `DATABASE_URL` has correct password
- ✅ `dm_policy` file exists and is readable
- ✅ Bot can authenticate with Telegram API

**Output Example**:
```
Configuration
=============
✅ Config File: ~/.ironclaw/.env (found)
✅ Bot Token: Configured (123456789:AAH...truncated)
✅ LLM Model: z-ai/glm5 (NVIDIA NIM)
✅ Database: postgres@localhost:5433 (connected)
⚠️  DM Policy: "pairing" - send /pair to activate
```

### telegram_full_report

Generates comprehensive health report combining all checks:

**Output Example**:
```
Telegram Integration Health Check
=================================

✅ Webhook: Registered (https://xxx.trycloudflare.com/webhook/telegram)
✅ Tunnel: Alive and accessible (HTTP 200)
⚠️  Config: dm_policy is "pairing" - send /pair first
✅ Database: Connected (postgres@localhost:5433)
✅ LLM Model: z-ai/glm5 (NVIDIA NIM)

Pending Updates: 2
Last Error: None

Recommendations:
1. Send /pair to activate bot for your user
2. Consider switching to polling mode for stability
```

### telegram_fix_common_issues

Auto-fixes common problems:

**Auto-Fixes**:
- 🔄 If tunnel dead → restart cloudflared
- 🔄 If webhook URL mismatch → re-register with Telegram
- 🔄 If dm_policy blocking → suggest /pair command
- 🔄 If LLM model wrong → update ~/.ironclaw/.env
- 🔄 If database password wrong → fix DATABASE_URL

**Output Example**:
```
Auto-Fix Telegram Issues
========================

Issue: Tunnel URL mismatch
  Webhook: https://old-url.trycloudflare.com/webhook/telegram
  Current: https://new-url.trycloudflare.com/webhook/telegram
  ✅ Fixed: Re-registered webhook with Telegram

Issue: LLM model invalid (stepfun-ai/step-3.5-flash)
  ✅ Fixed: Changed to z-ai/glm5 in ~/.ironclaw/.env

Issue: dm_policy is "pairing"
  ℹ️  Info: Send /pair to @YourBot to activate

Changes Applied: 2
Issues Resolved: 2
```

## Common Issues Detected

### 1. Webhook Not Registered

**Symptoms**: Bot doesn't respond to messages

**Detection**:
```
❌ Webhook URL: (empty)
❌ Pending Updates: 0
```

**Fix**:
```bash
telegram_fix_common_issues
# Or manually:
# 1. Get tunnel URL
# 2. Call Telegram setWebhook API
```

### 2. Tunnel Died / DNS Issues

**Symptoms**: Webhook registered but bot doesn't receive updates

**Detection**:
```
❌ Process: cloudflared not running
❌ DNS: xxx.trycloudflare.com does not resolve
```

**Fix**:
```bash
# Restart cloudflared
cloudflared tunnel --url http://localhost:3000

# Or switch to polling mode
export TELEGRAM_POLLING=true
cargo run
```

### 3. dm_policy Blocking Responses

**Symptoms**: Bot receives messages but doesn't respond to DMs

**Detection**:
```
⚠️  DM Policy: "pairing" (requires /pair command)
⚠️  User not in allowlist
```

**Fix**:
```bash
# Send /pair to bot on Telegram
# Or check pairing status
ironclaw pairing list telegram

# Or switch to open mode (not recommended)
# Edit ~/.ironclaw/.env:
# TELEGRAM_DM_POLICY=open
```

### 4. LLM Model Mismatch

**Symptoms**: Bot receives messages but fails to generate responses

**Detection**:
```
❌ LLM Model: stepfun-ai/step-3.5-flash (returns 404)
```

**Fix**:
```bash
# Edit ~/.ironclaw/.env
LLM_MODEL="z-ai/glm5"

# Or run auto-fix
telegram_fix_common_issues
```

### 5. Database Connection Issues

**Symptoms**: Bot crashes or fails to save state

**Detection**:
```
❌ Database: Connection failed
  Tried: postgres://localhost:5433/ironclaw
  Error: Password authentication failed
```

**Fix**:
```bash
# Check PostgreSQL running
docker ps | grep postgres

# Fix password in ~/.ironclaw/.env
DATABASE_URL="postgres://postgres:yourpass@localhost:5432/ironclaw"

# Or run auto-fix
telegram_fix_common_issues
```

### 6. Pending Updates Stuck

**Symptoms**: Bot responds slowly or with delay

**Detection**:
```
⚠️  Pending Updates: 150 (high)
⚠️  Last Sync Error: 2 hours ago
```

**Fix**:
```bash
# Check if bot is processing updates
# Restart IronClaw to clear backlog
# Or delete webhook and re-register
telegram_fix_common_issues
```

## Usage Examples

### Example 1: Full Health Check

```
telegram_full_report
```

Output:
```
Telegram Integration Health Check
=================================

✅ Webhook: Registered (https://abc123.trycloudflare.com/webhook/telegram)
✅ Tunnel: Alive and accessible (HTTP 200)
✅ Config: All settings valid
✅ Database: Connected (postgres@localhost:5433)
✅ LLM Model: z-ai/glm5 (NVIDIA NIM)

Pending Updates: 0
Last Error: None

Status: All systems operational
```

### Example 2: Bot Not Responding

```
telegram_check_webhook
```

Output:
```
Webhook Status
==============
❌ URL: (empty)
❌ Pending Updates: 0
❌ Last Error: "Webhook not set"

Recommendation: Run telegram_fix_common_issues to register webhook
```

### Example 3: Auto-Fix All Issues

```
telegram_fix_common_issues
```

Output:
```
Auto-Fix Telegram Issues
========================

Scanning for issues...

Issue: Webhook not registered
  ✅ Fixed: Registered https://abc123.trycloudflare.com/webhook/telegram

Issue: LLM model invalid
  ✅ Fixed: Changed LLM_MODEL to z-ai/glm5

Verifying fixes...
✅ Webhook: Registered
✅ LLM Model: Valid

All issues resolved!
```

### Example 4: Check Tunnel Only

```
telegram_check_tunnel
```

Output:
```
Tunnel Health
=============
✅ Process: cloudflared running (PID 12345)
✅ DNS: abc123.trycloudflare.com resolves to 198.41.200.23
✅ HTTP: https://abc123.trycloudflare.com/webhook/telegram (HTTP 200)
✅ Match: Tunnel URL matches webhook URL

Tunnel is healthy!
```

## Implementation Details

### Webhook Check

Calls Telegram Bot API:
```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo" | jq
```

Parses response:
```json
{
  "ok": true,
  "result": {
    "url": "https://xxx.trycloudflare.com/webhook/telegram",
    "has_custom_certificate": false,
    "pending_update_count": 2,
    "last_error_date": null,
    "last_error_message": null,
    "last_synchronization_error_date": null
  }
}
```

### Tunnel Check

1. **Process Check**: `pgrep -f cloudflared`
2. **DNS Resolution**: `dig +short xxx.trycloudflare.com` or `nslookup`
3. **HTTP Test**: `curl -I -s -o /dev/null -w "%{http_code}" https://xxx.trycloudflare.com/webhook/telegram`
4. **URL Match**: Compare tunnel URL with webhook URL from Telegram API

### Config Check

1. **File Existence**: `test -f ~/.ironclaw/.env`
2. **Token Format**: Regex `^[0-9]+:[A-Za-z0-9_-]+$`
3. **LLM Model**: Check against known valid models
4. **Database**: Attempt connection with `psql` or test via IronClaw
5. **dm_policy**: Read `~/.ironclaw/state/dm_policy` or check config

### Auto-Fix Logic

```rust
if webhook_url != tunnel_url {
    call_telegram_set_webhook(tunnel_url);
}

if llm_model == "stepfun-ai/step-3.5-flash" {
    update_env("LLM_MODEL", "z-ai/glm5");
}

if !cloudflared_running {
    suggest_restart_tunnel();
}

if dm_policy == "pairing" && user_not_paired {
    suggest_pair_command();
}
```

## Troubleshooting

### Bot Token Invalid

**Error**: `401 Unauthorized` from Telegram API

**Fix**:
1. Get new token from @BotFather
2. Update `TELEGRAM_BOT_TOKEN` in `~/.ironclaw/.env`
3. Restart IronClaw

### Tunnel Won't Start

**Error**: `cloudflared: command not found`

**Fix**:
```bash
# Install cloudflared
# macOS
brew install cloudflared

# Linux
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Start tunnel
cloudflared tunnel --url http://localhost:3000
```

### Webhook Returns 404

**Error**: Telegram webhook returns 404

**Fix**:
1. Ensure IronClaw is running
2. Check tunnel URL is correct
3. Verify `/webhook/telegram` endpoint exists
4. Re-register webhook: `telegram_fix_common_issues`

### dm_policy Confusion

**Question**: "Why doesn't bot respond to my messages?"

**Answer**: Check dm_policy:
- `pairing` (default): Must send `/pair` command first
- `allowlist`: Only responds to users in allowlist
- `open`: Responds to everyone (not recommended)

Check status:
```bash
ironclaw pairing list telegram
```

Approve user:
```bash
ironclaw pairing approve telegram <code>
```

## Related Documentation

- `channels-src/telegram/README.md` - Telegram channel implementation
- `channels-src/telegram/telegram.capabilities.json` - Channel capabilities
- `architecture/guides/env-config.md` - Environment configuration
- `integrations/errors/tunnel-errors.md` - Tunnel error handling
- `integrations/errors/bot-no-response.md` - Bot response troubleshooting

## Security Notes

- ⚠️  Never share your `TELEGRAM_BOT_TOKEN` publicly
- ⚠️  Use `dm_policy=pairing` or `allowlist` in production
- ⚠️  Enable webhook secret validation with `telegram_webhook_secret`
- ⚠️  Regularly rotate bot tokens if compromised

## Performance Tips

1. **Use webhooks** (not polling) for production - lower latency
2. **Monitor pending updates** - high count indicates processing issues
3. **Check tunnel stability** - ephemeral tunnels may expire
4. **Consider polling** for development - more reliable than ephemeral tunnels

## Support

If issues persist after running diagnostics:
1. Check IronClaw logs: `RUST_LOG=debug cargo run`
2. Enable Telegram debug logging in channel
3. Review Telegram Bot API logs for errors
4. Check cloudflared logs for tunnel issues
