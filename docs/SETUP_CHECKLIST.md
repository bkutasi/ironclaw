# IronClaw Setup Checklist

Quick-reference verification checklist for secure IronClaw deployment.

---

## Pre-Installation

- [ ] Rust toolchain installed (`rustc --version`)
- [ ] PostgreSQL installed and running **OR** libSQL selected
- [ ] LLM provider account created (NEAR AI, OpenAI, Anthropic, or NVIDIA NIM)
- [ ] API key obtained from LLM provider

---

## Initial Configuration

- [ ] `.env` file created from `.env.secure.example`
- [ ] `DATABASE_URL` set and connection tested:
  ```bash
  psql "$DATABASE_URL" -c "SELECT 1"
  ```
- [ ] LLM backend configured in `.env`:
  ```bash
  # Example for NVIDIA NIM
  LLM_BACKEND=openai_compatible
  LLM_BASE_URL=https://integrate.api.nvidia.com/v1
  LLM_API_KEY=nvapi-your-key-here
  LLM_MODEL=meta/llama-3.1-70b-instruct
  ```
- [ ] `HTTP_WEBHOOK_SECRET` generated and set (REQUIRED):
  ```bash
  openssl rand -hex 32
  ```

---

## Security Verification

- [ ] Master key configured (OS keychain preferred over env var)
- [ ] Secrets store initialized
- [ ] Bot tokens stored in secrets store (NOT in `.env`):
  - `telegram_bot_token`
  - `telegram_webhook_secret`
  - `slack_bot_token`, `slack_app_token`, `slack_signing_secret`
- [ ] File permissions restricted:
  ```bash
  chmod 600 .env
  chmod 700 ~/.ironclaw
  ```
- [ ] `.env` added to `.gitignore`

---

## Channel Setup

### HTTP Channel
- [ ] `HTTP_HOST` configured (default: `0.0.0.0`)
- [ ] `HTTP_PORT` configured (default: `8080`, change if conflict)
- [ ] `HTTP_WEBHOOK_SECRET` set (channel won't start without it!)
- [ ] Webhook endpoint accessible: `/webhook/telegram`, `/webhook/http`

### Telegram Channel
- [ ] Bot token obtained from @BotFather
- [ ] Bot token stored in secrets store
- [ ] Tunnel running (ngrok/Cloudflare) for webhook mode
- [ ] `TUNNEL_URL` set (must be HTTPS)
- [ ] Webhook registered with Telegram:
  ```bash
  curl "https://api.telegram.org/bot$TOKEN/getWebhookInfo"
  ```
- [ ] DM pairing configured (`dm_policy`, `owner_id`)

### Gateway Channel
- [ ] `GATEWAY_HOST` set (default: `127.0.0.1` for local-only)
- [ ] `GATEWAY_PORT` set (default: `3000`)
- [ ] `GATEWAY_AUTH_TOKEN` set or auto-generated

### CLI Channel
- [ ] Enabled by default (no configuration needed)

---

## LLM Verification

- [ ] API key valid (test with curl):
  ```bash
  # For NVIDIA NIM
  curl -H "Authorization: Bearer $LLM_API_KEY" \
    "$LLM_BASE_URL/models"
  ```
- [ ] Model name correct (check provider docs)
- [ ] Test query successful:
  ```bash
  ironclaw run --test-llm
  ```

---

## Startup Tests

- [ ] `ironclaw run` starts without errors
- [ ] All configured channels start successfully
- [ ] No "secret leak detected" warnings in logs
- [ ] Health endpoints respond:
  ```bash
  curl http://localhost:8080/health
  curl http://localhost:3000/health
  ```
- [ ] No port conflict errors (8080, 3000, 50051)

---

## Production Hardening

- [ ] Ports changed from defaults if needed:
  - HTTP: `HTTP_PORT` (default 8080)
  - Gateway: `GATEWAY_PORT` (default 3000)
  - Orchestrator: 50051 (internal, document if conflicts)
- [ ] Host binding restricted:
  - `GATEWAY_HOST=127.0.0.1` (local-only, more secure)
  - `HTTP_HOST=0.0.0.0` (behind reverse proxy) or `127.0.0.1` (local-only)
- [ ] TLS/HTTPS configured for tunnels
- [ ] Rate limiting enabled (default: 60 req/min)
- [ ] Logging configured (`RUST_LOG=ironclaw=info`)
- [ ] Process management setup:
  - [ ] systemd service (Linux)
  - [ ] launchd agent (macOS)
  - [ ] Docker Compose (containerized)

---

## Ongoing Maintenance

- [ ] Backup strategy implemented:
  - [ ] Database backups scheduled
  - [ ] Secrets exported and backed up securely
- [ ] Monitoring configured:
  - [ ] Health check endpoint monitored
  - [ ] Log aggregation enabled
- [ ] Log rotation enabled (prevent disk fill)
- [ ] Secret rotation schedule planned (quarterly recommended)
- [ ] Security updates monitored (GitHub releases)

---

## Troubleshooting Quick Reference

### Common Errors

| Error | Solution |
|-------|----------|
| `error connecting to server` | Start PostgreSQL or set `DATABASE_BACKEND=libsql` |
| `Address already in use` | Change `HTTP_PORT` or `GATEWAY_PORT` |
| `HTTP webhook secret is required` | Generate and set `HTTP_WEBHOOK_SECRET` |
| `404 Not Found` (LLM) | Check `LLM_BASE_URL` format, verify model name |
| `Failed to register webhook` | Start tunnel before IronClaw, check `TUNNEL_URL` |

### Verification Commands

```bash
# Check running processes
ps aux | grep ironclaw

# Check ports in use
lsof -i :8080
lsof -i :3000
lsof -i :50051

# View logs
journalctl -u ironclaw -f  # systemd
tail -f ~/.ironclaw/ironclaw.log  # file logs

# Test database
psql "$DATABASE_URL" -c "SELECT count(*) FROM agents;"

# Test LLM connection
curl -H "Authorization: Bearer $LLM_API_KEY" "$LLM_BASE_URL/v1/models"
```

---

## Next Steps

After completing this checklist:

1. **Review**: [`docs/SECURE_SETUP_GUIDE.md`](SECURE_SETUP_GUIDE.md) for detailed guidance
2. **Template**: Refer to [`.env.secure.example`](../.env.secure.example) for all variables
3. **Telegram**: See [`docs/TELEGRAM_SETUP.md`](TELEGRAM_SETUP.md) for channel-specific setup
4. **Monitor**: Set up health check monitoring and alerting

---

**Last Updated**: 2026-02-20  
**Version**: 1.0
