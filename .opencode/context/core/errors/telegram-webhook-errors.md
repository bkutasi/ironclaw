# Telegram Webhook Errors

**Purpose**: Common Telegram webhook integration errors and solutions

**Last Updated**: 2026-03-22

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Error: Failed to Resolve Host (400)

**Symptom**:
```json
{"ok":false,"error_code":400,
  "description":"Bad Request: bad webhook: Failed to resolve host: Name or service not known"}
```

**Cause**: Trailing newline character in tunnel URL file corrupts hostname sent to Telegram.

**Solution**:
1. Fix URL file creation: `echo -n "$TUNNEL_URL"` (no newline)
2. Add defensive trim in Rust: `let tunnel_url = tunnel_url.trim();`
3. Clean corrupted file: `rm /tmp/cloudflared-ephemeral-url.txt`

**Code**:
```bash
# ❌ Wrong (adds newline)
echo "$TUNNEL_URL" > /tmp/cloudflared-ephemeral-url.txt

# ✅ Correct (no newline)
echo -n "$TUNNEL_URL" > /tmp/cloudflared-ephemeral-url.txt
```

**Prevention**: Always use `echo -n` for URL files, trim before sending to APIs

**Frequency**: common

---

## Error: 403 Forbidden (Cloudflare)

**Symptom**:
```json
{"ok":false,"error_code":403,
  "description":"Wrong response from the webhook: 403 Forbidden"}
```

**Cause**: Cloudflare Bot Fight Mode blocking Telegram's IPs at edge.

**Solution**:
1. Cloudflare Dashboard → Security → Bots
2. Add exception: `URI Path contains "/webhook/telegram"`
3. Action: Skip Bot Fight Mode

**Alternative**: Use ephemeral tunnel (`*.trycloudflare.com`) - bypasses Bot Fight Mode

**Prevention**: Add exception rule before deploying webhook mode

**Frequency**: very common

---

## Error: 429 Too Many Requests

**Symptom**:
```json
{"ok":false,"error_code":429,
  "description":"Too Many Requests: retry after 1"}
```

**Cause**: Script + IronClaw both register webhook → duplicate API calls.

**Solution**:
1. Remove webhook registration from script
2. Let IronClaw handle registration on startup
3. Wait 60 seconds if rate limited

**Code**:
```bash
# Script should NOT register webhook:
# ❌ Don't do this in startup script
curl -X POST "https://api.telegram.org/bot$TOKEN/setWebhook" ...

# ✅ Let IronClaw handle it automatically
```

**Prevention**: Single registration point (IronClaw startup only)

**Frequency**: common

---

## Error: HTTP 000 (Connection Timeout)

**Symptom**:
```
⚠️  Unexpected response code: HTTP 000000
   Tunnel endpoint not accessible
```

**Cause**: Cloudflared tunnel not fully connected or port 8081 not listening.

**Solution**:
1. Check cloudflared running: `ps aux | grep cloudflared`
2. Check port listening: `netstat -tlnp | grep 8081`
3. Verify tunnel logs: `tail -f /tmp/cloudflared-*.log`
4. Restart if needed: `pkill cloudflared && restart_script`

**Prevention**: Wait 5 seconds after tunnel starts before testing endpoint

**Frequency**: occasional

---

## Error: 502 Bad Gateway

**Symptom**: Webhook returns 502 when Telegram tries to deliver message

**Cause**: IronClaw not running on port 8081 or tunnel config misconfigured

**Solution**:
1. Check IronClaw is running: `ps aux | grep ironclaw`
2. Check port binding: `netstat -tlnp | grep 8081`
3. Verify tunnel config routes correctly:
   ```yaml
   # /home/bkutasi/.cloudflared/ironclaw-config.yml
   ingress:
     - hostname: ironclaw.kutasi.dev
       path: /webhook/telegram
       service: http://localhost:8081
   ```
4. Restart: `pkill ironclaw && ./.tmp/start-ironclaw-webhook.sh`

**Prevention**: Use startup scripts that validate port binding

**Frequency**: common

---

## Related

- concepts/telegram-integration.md
- guides/telegram-mode-selection.md
