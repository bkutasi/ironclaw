<!-- Context: architecture/lookup | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Lookup: Environment Variables

**Purpose**: Quick reference for IronClaw environment variables

**Last Updated**: 2026-03-19

## Required Variables

| Variable | Purpose | Example | Code |
|----------|---------|---------|------|
| `NGC_KEY` | NVIDIA API key | `nvapi-...` | `src/config/env.rs` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot auth | `123456:ABC-...` | `channels/telegram/` |
| `DATABASE_URL` | PostgreSQL connection | `postgres://...` | `src/database/pool.rs` |

## Optional Variables

| Variable | Purpose | Default | Code |
|----------|---------|---------|------|
| `TUNNEL_URL` | Webhook tunnel URL | (unset) | `src/channels/telegram.rs` |
| `HTTP_PORT` | Webhook server port | `8081` | `src/http/server.rs` |
| `GATEWAY_PORT` | Web UI port | `3004` | `src/gateway/` |
| `LLM_MODEL` | AI model to use | `minimax-m2.5` | `src/llm/config.rs` |
| `HTTP_WEBHOOK_SECRET` | Webhook validation | (auto-gen) | `src/http/middleware.rs` |
| `TELEGRAM_POLLING_ENABLED` | Force polling mode | `false` | `channels/telegram/` |

## Database Variables

| Variable | Purpose | Default | Code |
|----------|---------|---------|------|
| `POSTGRES_USER` | Docker DB user | `postgres` | `scripts/start-postgres.sh` |
| `POSTGRES_PASSWORD` | Docker DB password | (required) | `scripts/start-postgres.sh` |
| `POSTGRES_DB` | Database name | `ironclaw` | `scripts/start-postgres.sh` |
| `POSTGRES_PORT` | Docker port mapping | `5433` | `scripts/start-postgres.sh` |

## Commands

```bash
# Check current environment
env | grep -E "(NGC_KEY|TELEGRAM|DATABASE|TUNNEL)" | sort

# Validate required variables
test -n "$NGC_KEY" && echo "✓ NGC_KEY set" || echo "✗ NGC_KEY missing"
test -n "$TELEGRAM_BOT_TOKEN" && echo "✓ Token set" || echo "✗ Token missing"

# Test database connection
psql "$DATABASE_URL" -c "SELECT version();"
```

## Paths

```
.env - Local environment file (gitignored)
.env.example - Template with examples
src/config/env.rs - Environment parsing logic
```

## 📂 Codebase References

**Configuration**:
- `src/config/env.rs` - Environment variable parsing
- `src/config/mod.rs` - Configuration module
- `.env.example` - Environment template

**Validation**:
- `src/config/validation.rs` - Config validation

## Related

- guides/env-config.md
- concepts/docker-postgres.md
- errors/common-build-issues.md
