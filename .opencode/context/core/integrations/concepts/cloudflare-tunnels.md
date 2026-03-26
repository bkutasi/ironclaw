<!-- Context: integrations/concepts | Priority: critical | Version: 1.0 | Updated: 2026-03-19 -->
# Concept: Cloudflare Tunnels

**Purpose**: Expose local IronClaw instance to internet securely

**Last Updated**: 2026-03-19

## Core Idea

Cloudflare tunnels create secure, outbound-only connections from your local IronClaw instance to Cloudflare's edge network. No port forwarding or public IP needed. Two modes: **ephemeral** (temporary URL) and **persistent** (fixed domain).

## Key Points

- **Ephemeral tunnels**: `*.trycloudflare.com` URL, no account needed, URL changes each restart
- **Persistent tunnels**: Your domain (e.g., `ironclaw.kutasi.dev`), requires Cloudflare account
- **Outbound-only**: No inbound ports, firewall-friendly
- **Bot Fight Mode**: Cloudflare security can block Telegram webhooks (needs exception)
- **DNS propagation**: New tunnels take 30-60 seconds to propagate globally

## Quick Example

```bash
# Ephemeral tunnel (testing)
cloudflared tunnel --url http://localhost:8081

# Output:
# https://luther-grad-coffee-investigators.trycloudflare.com

# Persistent tunnel (production)
cloudflared tunnel run ironclaw-tunnel

# Output:
# https://ironclaw.kutasi.dev
```

## 📂 Codebase References

**Tunnel Integration**:
- `src/channels/telegram.rs` - Tunnel URL usage
- `src/http/server.rs` - Webhook server

**Scripts**:
- `.tmp/start-ironclaw-ephemeral.sh` - Ephemeral tunnel startup
- `.tmp/start-ironclaw-webhook.sh` - Persistent tunnel startup

**Configuration**:
- `~/.cloudflared/config.yml` - Persistent tunnel config
- `/tmp/cloudflared-ephemeral.log` - Ephemeral tunnel logs

## Deep Dive

**Reference**: See `guides/ephemeral-tunnel-setup.md` and `guides/persistent-tunnel-setup.md`

## Related

- guides/ephemeral-tunnel-setup.md
- errors/tunnel-errors.md
- telecom/telegram-integration.md
