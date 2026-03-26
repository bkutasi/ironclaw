# Telegram + Cloudflare 403 Reference

**Core Idea**: Telegram webhook requests get 403 Forbidden at Cloudflare's edge before reaching the tunnel. The 403 is caused by Cloudflare security features blocking Telegram's IP ranges (ASN AS62041), not by tunnel or IronClaw misconfiguration. Manual curl tests succeed because they originate from trusted IPs.

**Key Points**:
- Telegram IP ranges: `149.154.160.0/20`, `149.154.164.0/22`, `149.154.168.0/21`, `149.154.170.0/23`, `149.154.172.0/22` (ASN: AS62041)
- Cloudflare features ranked by blocking probability: Bot Fight Mode (70%), WAF Custom Rules (15%), Access Applications (10%), Firewall Rules (4%), DDoS Protection (1%)
- Fix: Add exception rule for `/webhook/telegram` path or allow Telegram IP ranges in firewall
- Diagnostic: Check Cloudflare Dashboard → Security → Events for blocked requests from Telegram IPs
- Polling mode is a viable workaround since IronClaw initiates outbound (bypasses Cloudflare entirely)

**Quick Example**:
```bash
# Verify Telegram webhook status
curl -s "https://api.telegram.org/bot<TOKEN>/getWebhookInfo" | jq '.result | {error: .last_error_message, pending: .pending_update_count}'

# Expected after fix: {"error": null, "pending": 0}
```

**Reference**: `.tmp/TELEGRAM_403_DIAGNOSIS.md`, `.tmp/CLOUDFLARE_FIX_STEPS.md`

**Related**: `core/telecom/errors/telegram-polling-mode-fix.md`, `core/telecom/concepts/script-design-patterns.md`
