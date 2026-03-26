<!-- Context: integrations/guides | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Guide: Persistent Tunnel Setup

**Purpose**: Production setup with fixed domain tunnel

**Last Updated**: 2026-03-19

## Prerequisites

- Cloudflare account
- Domain managed by Cloudflare
- Cloudflared binary installed
- Root/sudo access for initial setup

**Estimated time**: 15 min

## Steps

### 1. Create Tunnel in Cloudflare Dashboard
```
1. Login: https://dash.cloudflare.com
2. Zero Trust → Networks → Tunnels
3. Create tunnel → Name: ironclaw-tunnel
4. Save tunnel credentials (shown once)
```
**Expected**: Tunnel created, credentials shown

### 2. Install Tunnel Credentials
```bash
# Create config directory
sudo mkdir -p /etc/cloudflared

# Save credentials (from step 1)
sudo tee /etc/cloudflared/ironclaw-tunnel.json << EOF
{
  "AccountTag": "your-account-tag",
  "TunnelSecret": "your-tunnel-secret",
  "TunnelID": "your-tunnel-id"
}
EOF
```
**Expected**: Credentials file created

### 3. Configure Tunnel Routing
```bash
# Create config file
sudo tee /etc/cloudflared/config.yml << EOF
tunnel: ironclaw-tunnel
credentials-file: /etc/cloudflared/ironclaw-tunnel.json

ingress:
  - hostname: ironclaw.kutasi.dev
    service: http://localhost:8081
  - service: http_status:404
EOF
```
**Expected**: Config file created

### 4. Add DNS CNAME Record
```
Cloudflare Dashboard → DNS → Add record:
- Type: CNAME
- Name: ironclaw
- Target: your-tunnel-id.cfargotunnel.com
- Proxy: Enabled (orange cloud)
```
**Expected**: DNS record created

### 5. Start Tunnel as Service
```bash
# Linux (systemd)
sudo cloudflared service install

# Start service
sudo systemctl start cloudflared
sudo systemctl enable cloudflared

# Check status
sudo systemctl status cloudflared
```
**Expected**: Service running

### 6. Add Bot Fight Mode Exception
```
Cloudflare Dashboard → Security → Bots:
- Add rule: URI Path contains "/webhook/telegram"
- Action: Skip Bot Fight Mode
- Save
```
**Expected**: Exception rule created

## Verification

```bash
# Check tunnel status
cloudflared tunnel list

# Test endpoint
curl -I https://ironclaw.kutasi.dev/webhook/telegram

# Check webhook
curl -s "https://api.telegram.org/bot$TOKEN/getWebhookInfo" | jq .result
```

## 📂 Codebase References

**Scripts**:
- `.tmp/start-ironclaw-webhook.sh` - Webhook mode startup
- `scripts/setup-persistent-tunnel.sh` - Tunnel setup automation

**Configuration**:
- `/etc/cloudflared/config.yml` - Tunnel config
- `/etc/cloudflared/ironclaw-tunnel.json` - Tunnel credentials

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 403 Forbidden | Add Bot Fight Mode exception |
| DNS not resolving | Wait 60s, check CNAME record |
| Tunnel offline | `sudo systemctl restart cloudflared` |

## Related

- concepts/cloudflare-tunnels.md
- guides/ephemeral-tunnel-setup.md
- errors/tunnel-errors.md
