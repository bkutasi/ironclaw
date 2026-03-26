<!-- Context: integrations/lookup | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Lookup: Tunnel Commands

**Purpose**: Quick reference for Cloudflare tunnel commands

**Last Updated**: 2026-03-19

## Installation

```bash
# Linux (Debian/Ubuntu)
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# macOS
brew install cloudflared

# Verify installation
cloudflared --version
```

## Ephemeral Tunnels

```bash
# Start ephemeral tunnel
cloudflared tunnel --url http://localhost:8081

# Start in background
cloudflared tunnel --url http://localhost:8081 > /tmp/tunnel.log 2>&1 &

# Extract URL from logs
grep "trycloudflare.com" /tmp/tunnel.log | head -1

# Kill tunnel
pkill cloudflared
```

## Persistent Tunnels

```bash
# List tunnels
cloudflared tunnel list

# Show tunnel info
cloudflared tunnel info TUNNEL_ID

# Run tunnel
cloudflared tunnel run ironclaw-tunnel

# Run with config
cloudflared tunnel run --config /etc/cloudflared/config.yml

# Stop tunnel
pkill cloudflared
```

## Service Management

```bash
# Install as service (Linux)
sudo cloudflared service install

# Check status
sudo systemctl status cloudflared

# Start/Stop/Restart
sudo systemctl start cloudflared
sudo systemctl stop cloudflared
sudo systemctl restart cloudflared

# Enable on boot
sudo systemctl enable cloudflared

# View logs
sudo journalctl -u cloudflared -f
```

## 📂 Codebase References

**Scripts**:
- `.tmp/start-ironclaw-ephemeral.sh` - Ephemeral startup
- `.tmp/start-ironclaw-webhook.sh` - Persistent startup

**Logs**:
- `/tmp/cloudflared-ephemeral.log` - Ephemeral logs
- `/tmp/ironclaw.log` - IronClaw logs

## Related

- concepts/cloudflare-tunnels.md
- guides/ephemeral-tunnel-setup.md
- errors/tunnel-errors.md
