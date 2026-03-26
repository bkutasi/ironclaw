<!-- Context: integrations/errors | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Errors: Tunnel Issues

**Purpose**: Common Cloudflare tunnel errors and solutions

**Last Updated**: 2026-03-19

## Error: DNS Propagation Delay

**Symptom**: `HTTP 000` - Tunnel endpoint not accessible

**Cause**: New tunnel URL takes 30-60 seconds to propagate.

**Solution**:
```bash
sleep 60  # Wait for DNS propagation
dig +short your-tunnel.trycloudflare.com
curl -I https://your-tunnel.trycloudflare.com
```

**Prevention**: Wait 30-60 seconds after tunnel starts

**Frequency**: very common

---

## Error: HTTP 530 Origin DNS Error

**Symptom**:
```json
{
  "cf_ray": "...",
  "error": "Origin DNS error"
}
```

**Cause**: Cloudflare can't resolve your origin (localhost:8081).

**Solution**:
```bash
# Check if port 8081 listening
netstat -tlnp | grep 8081

# Start IronClaw first
./target/release/ironclaw &

# Then start tunnel
cloudflared tunnel --url http://localhost:8081
```

**Prevention**: Start IronClaw before tunnel, verify port listening

**Frequency**: common

---

## Error: 403 Forbidden (Bot Fight Mode)

**Symptom**:
```json
{"ok":false,"error_code":403,
  "description":"Wrong response from the webhook: 403 Forbidden"}
```

**Cause**: Cloudflare Bot Fight Mode blocking Telegram's IPs.

**Solution**:
```
1. Cloudflare Dashboard → Security → Bots
2. Add exception rule:
   - Field: URI Path
   - Operator: contains
   - Value: /webhook/telegram
3. Action: Skip Bot Fight Mode
4. Save
```

**Alternative**: Use ephemeral tunnel (bypasses Bot Fight Mode)

**Prevention**: Add exception rule before deploying webhook

**Frequency**: very common

---

## Error: Tunnel Already Running

**Symptom**: `Error: Another cloudflared process is already running`

**Cause**: Previous instance still active.

**Solution**: `pkill cloudflared && cloudflared tunnel run ironclaw-tunnel`

**Prevention**: Kill old tunnels first

**Frequency**: occasional

---

## Error: Tunnel Died (530 Errors)

**Symptom**: Telegram stops receiving messages, tunnel returns 530 errors, messages pile up as pending.

**Cause**: **Ephemeral tunnels DIE after 30-60 minutes**. Tunnel URL changes on restart, cloudflared silently exits.

**Critical**: Ephemeral tunnels NOT stable for production.

**Solution**:
```bash
pkill cloudflared
sleep 2
cloudflared tunnel --url http://localhost:8081 > /tmp/cloudflared-ephemeral.log 2>&1 &
sleep 30  # DNS propagation
TUNNEL_URL=$(grep "trycloudflare.com" /tmp/cloudflared-ephemeral.log | head -1 | sed 's/.*\(https:\/\/[^ ]*\).*/\1/')
curl -X POST "https://api.telegram.org/bot$TOKEN/setWebhook?url=$TUNNEL_URL/webhook/telegram"
```

**Prevention**: Use **polling mode** (reliable) or **persistent tunnel** (production).

**Frequency**: very common

---

## 📂 Codebase References

**Tunnel Integration**:
- `src/channels/telegram.rs` - Tunnel URL usage
- `src/http/server.rs` - Webhook server

**Scripts**:
- `.tmp/start-ironclaw-ephemeral.sh` - Ephemeral tunnel startup
- `.tmp/start-ironclaw-webhook.sh` - Persistent tunnel startup

**Logs**:
- `/tmp/cloudflared-ephemeral.log` - Ephemeral tunnel logs
- `/tmp/cloudflared-*.log` - Tunnel connection logs

---

## Multiple cloudflared Instances Conflict

**Symptom**: Webhook works with ephemeral tunnel but fails with persistent tunnel (403).

**Cause**: System-level cloudflared service (running as root) may intercept traffic before user-level tunnel. Multiple instances compete for the same hostname.

**Diagnosis**:
1. Check for system service: `systemctl status cloudflared` or `ps aux | grep cloudflared`
2. Compare behavior: ephemeral tunnel (429 = rate limited, working) vs persistent tunnel (403 = blocked)
3. If ephemeral works but persistent doesn't → Cloudflare security config issue, not tunnel issue

**Solution**:
1. Stop conflicting system service: `sudo systemctl stop cloudflared`
2. Or configure system service to use different hostname
3. Verify only one cloudflared instance handles your domain

**Reference**: TELEGRAM_DEBUG_HANDOFF.md investigation notes

---

## Related

- concepts/cloudflare-tunnels.md
- guides/ephemeral-tunnel-setup.md
- guides/telegram-mode-selection.md
