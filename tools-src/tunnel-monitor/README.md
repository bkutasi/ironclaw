# Tunnel Monitor

Monitor Cloudflare tunnel health and auto-recover when it dies.

## Overview

Ephemeral Cloudflare tunnels die after 30-60 minutes, causing webhook failures. This tool provides:

- **Proactive monitoring** - Check tunnel health every 5 minutes
- **Auto-recovery** - Automatically restart dead tunnels
- **Webhook sync** - Update Telegram webhook when tunnel URL changes
- **Health reports** - Detailed status with actionable insights

## Installation

```bash
# Build for WASM
cd tools-src/tunnel-monitor
cargo build --release --target wasm32-wasip2

# Install to IronClaw skills directory
mkdir -p ~/.ironclaw/skills/tunnel-monitor
cp target/wasm32-wasip2/release/tunnel_monitor.wasm ~/.ironclaw/skills/tunnel-monitor/
cp tunnel-monitor.capabilities.json ~/.ironclaw/skills/tunnel-monitor/
```

## Tools

### 1. `tunnel_check_status`

Check if tunnel is alive and accessible.

**Parameters:**
```json
{
  "tool": "tunnel_check_status",
  "params": {
    "tunnel_url": "https://abc123.trycloudflare.com",  // optional, reads from file
    "check_telegram": true,                              // optional, default: false
    "telegram_bot_token": "123456:ABC-DEF1234",         // required if check_telegram
    "telegram_webhook_path": "/webhook/telegram",       // optional
    "timeout_seconds": 5                                 // optional, default: 5
  }
}
```

**Returns (Healthy):**
```json
{
  "status": "healthy",
  "checks": [
    {"name": "Process", "status": "pass", "message": "cloudflared running (PID: 123456)", "details": "123456"},
    {"name": "URL", "status": "pass", "message": "https://abc123.trycloudflare.com", "details": "https://abc123.trycloudflare.com"},
    {"name": "DNS", "status": "pass", "message": "Resolved (104.16.230.132)", "details": "104.16.230.132"},
    {"name": "Endpoint", "status": "pass", "message": "Accessible (HTTP 200)", "details": "200"},
    {"name": "Telegram Webhook", "status": "pass", "message": "Synced", "details": "URL: https://abc123.trycloudflare.com/webhook/telegram, Pending: 0"}
  ],
  "tunnel_url": "https://abc123.trycloudflare.com",
  "process_pid": 123456,
  "tunnel_age_minutes": 45,
  "recovery_actions": []
}
```

**Returns (Dead):**
```json
{
  "status": "process_not_running",
  "checks": [
    {"name": "Process", "status": "fail", "message": "cloudflared not running", "details": null}
  ],
  "tunnel_url": null,
  "process_pid": null,
  "tunnel_age_minutes": null,
  "recovery_actions": []
}
```

### 2. `tunnel_test_endpoint`

Test webhook endpoint accessibility and measure response time.

**Parameters:**
```json
{
  "tool": "tunnel_test_endpoint",
  "params": {
    "tunnel_url": "https://abc123.trycloudflare.com",  // optional
    "path": "/webhook/telegram",                        // optional
    "method": "GET",                                    // optional
    "timeout_seconds": 5                                // optional
  }
}
```

**Returns:**
```json
{
  "url": "https://abc123.trycloudflare.com/webhook/telegram",
  "status_code": 200,
  "response_time_ms": 145,
  "response_size_bytes": 52,
  "success": true,
  "error": null
}
```

### 3. `tunnel_restart_if_dead`

Auto-restart dead tunnel. Checks health first, only restarts if needed.

**Parameters:**
```json
{
  "tool": "tunnel_restart_if_dead",
  "params": {
    "local_port": 8081,                                  // optional, default: 8081
    "telegram_bot_token": "123456:ABC-DEF1234",         // optional
    "telegram_webhook_path": "/webhook/telegram"        // optional
  }
}
```

**Returns (Success):**
```json
{
  "success": true,
  "new_url": "https://xyz789.trycloudflare.com",
  "new_pid": 234567,
  "webhook_updated": true,
  "error": null,
  "actions": [
    "Killing old cloudflared process",
    "Starting new cloudflared tunnel",
    "New tunnel started (PID: 234567)",
    "New URL: https://xyz789.trycloudflare.com",
    "URL saved to file",
    "Updating Telegram webhook",
    "Webhook updated successfully"
  ]
}
```

**Returns (Already Healthy):**
```json
{
  "success": true,
  "new_url": "https://abc123.trycloudflare.com",
  "new_pid": 123456,
  "webhook_updated": false,
  "error": null,
  "actions": ["Tunnel already healthy, no restart needed"]
}
```

### 4. `tunnel_update_webhook`

Update Telegram webhook with current tunnel URL.

**Parameters:**
```json
{
  "tool": "tunnel_update_webhook",
  "params": {
    "bot_token": "123456:ABC-DEF1234",                  // required
    "tunnel_url": "https://abc123.trycloudflare.com",  // optional
    "webhook_path": "/webhook/telegram",                // optional
    "drop_pending_updates": false                       // optional
  }
}
```

**Returns (Updated):**
```json
{
  "updated": true,
  "current_url": "https://abc123.trycloudflare.com/webhook/telegram",
  "expected_url": "https://abc123.trycloudflare.com/webhook/telegram",
  "telegram_info": {
    "url": "https://abc123.trycloudflare.com/webhook/telegram",
    "has_custom_certificate": false,
    "pending_update_count": 0,
    "last_error_message": null,
    "max_connections": 40
  },
  "error": null
}
```

**Returns (No Update Needed):**
```json
{
  "updated": false,
  "current_url": "https://abc123.trycloudflare.com/webhook/telegram",
  "expected_url": "https://abc123.trycloudflare.com/webhook/telegram",
  "telegram_info": {...},
  "error": null
}
```

## Usage Examples

### Example 1: Quick Health Check

```json
{
  "tool": "tunnel_check_status",
  "params": {}
}
```

### Example 2: Full Health Check with Telegram

```json
{
  "tool": "tunnel_check_status",
  "params": {
    "check_telegram": true,
    "telegram_bot_token": "YOUR_BOT_TOKEN"
  }
}
```

### Example 3: Auto-Recover Dead Tunnel

```json
{
  "tool": "tunnel_restart_if_dead",
  "params": {
    "local_port": 8081,
    "telegram_bot_token": "YOUR_BOT_TOKEN",
    "telegram_webhook_path": "/webhook/telegram"
  }
}
```

### Example 4: Manual Webhook Update

```json
{
  "tool": "tunnel_update_webhook",
  "params": {
    "bot_token": "YOUR_BOT_TOKEN",
    "drop_pending_updates": true
  }
}
```

### Example 5: Endpoint Performance Test

```json
{
  "tool": "tunnel_test_endpoint",
  "params": {
    "path": "/health",
    "timeout_seconds": 10
  }
}
```

## Background Monitoring Mode

For continuous monitoring, run as a background task:

```bash
# Create monitoring script
cat > /tmp/tunnel-monitor-daemon.sh << 'EOF'
#!/bin/bash
LOG_FILE="/tmp/tunnel-monitor.log"
INTERVAL=300  # 5 minutes

while true; do
    echo "[$(date)] Checking tunnel health..." >> $LOG_FILE
    
    # Check status
    RESULT=$(ironclaw-tool tunnel_check_status --params '{"check_telegram":true}')
    
    # Parse status
    STATUS=$(echo $RESULT | jq -r '.status')
    
    if [ "$STATUS" != "healthy" ]; then
        echo "[$(date)] Tunnel unhealthy ($STATUS), restarting..." >> $LOG_FILE
        
        # Restart tunnel
        RECOVERY=$(ironclaw-tool tunnel_restart_if_dead --params '{
            "local_port": 8081,
            "telegram_bot_token": "YOUR_BOT_TOKEN"
        }')
        
        # Log recovery
        echo "[$(date)] Recovery: $RECOVERY" >> $LOG_FILE
        
        # Notify via Telegram
        if [ $(echo $RECOVERY | jq -r '.success') = "true" ]; then
            NEW_URL=$(echo $RECOVERY | jq -r '.new_url')
            curl -s "https://api.telegram.org/botYOUR_BOT_TOKEN/sendMessage" \
                -d "chat_id=YOUR_CHAT_ID&text=Tunnel recovered: $NEW_URL"
        fi
    fi
    
    sleep $INTERVAL
done
EOF

chmod +x /tmp/tunnel-monitor-daemon.sh

# Run in background
nohup /tmp/tunnel-monitor-daemon.sh &
```

## Error Codes

| Code | Meaning | Recovery Action |
|------|---------|-----------------|
| 530 | Origin DNS error | Restart tunnel |
| 522 | Connection timed out | Check tunnel process |
| 521 | Web server down | Check local service |
| 502 | Bad Gateway | Check service health |
| 524 | Timeout | Increase timeout or check service |
| 1033 | Cloudflare can't reach origin | Restart tunnel |

## System Requirements

- `cloudflared` installed and in PATH
- `curl` for HTTP requests
- `dig` for DNS resolution (optional)
- `pgrep`/`pkill` for process management
- Network access to Telegram API

## File Paths

| File | Purpose |
|------|---------|
| `/tmp/cloudflared-ephemeral-url.txt` | Stores current tunnel URL |
| `/tmp/tunnel-monitor.log` | Daemon log file |

## Best Practices

1. **Run health checks every 5 minutes** - Ephemeral tunnels typically last 30-60 minutes
2. **Always update webhook on URL change** - Prevents missed messages
3. **Monitor pending_update_count** - High values indicate delivery issues
4. **Use secret tokens** - Add `secret_token` parameter to webhook for security
5. **Log all recovery actions** - Helps diagnose recurring issues

## Troubleshooting

### Tunnel won't start
```bash
# Check if cloudflared is installed
which cloudflared

# Test manual start
cloudflared tunnel --url http://localhost:8081
```

### Webhook not updating
```bash
# Check bot token validity
curl "https://api.telegram.org/botYOUR_TOKEN/getMe"

# Check current webhook
curl "https://api.telegram.org/botYOUR_TOKEN/getWebhookInfo"
```

### DNS resolution failing
```bash
# Test DNS manually
dig +short abc123.trycloudflare.com

# Wait for propagation (can take 1-2 minutes)
```

## Security Considerations

- Bot tokens are passed as parameters, never stored
- Tunnel URLs are ephemeral and change on restart
- Use Telegram secret tokens for webhook verification
- Run with minimal privileges

## License

MIT OR Apache-2.0
