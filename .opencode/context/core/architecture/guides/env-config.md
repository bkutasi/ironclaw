<!-- Context: architecture/guides | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Guide: Environment Configuration

**Purpose**: Configure IronClaw environment variables

**Last Updated**: 2026-03-19

## Prerequisites

- NVIDIA NGC API key
- Telegram bot token (for Telegram integration)
- Database connection string

**Estimated time**: 10 min

## Steps

### 1. Get NVIDIA NGC Key
```bash
# Visit: https://org.ngc.nvidia.com/setup/personal-keys
# Generate personal API key
export NGC_KEY="nvapi-your-key-here"
```
**Expected**: Key saved to environment

### 2. Get Telegram Bot Token
```bash
# Message @BotFather on Telegram
# Send: /newbot
# Follow prompts, save token
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
```
**Expected**: Token saved to environment

### 3. Configure Database
```bash
# Docker PostgreSQL (recommended)
export DATABASE_URL="postgres://postgres:yourpass@localhost:5433/ironclaw"

# Or system PostgreSQL
export DATABASE_URL="postgres://postgres:yourpass@localhost:5432/ironclaw"
```
**Expected**: Database URL configured

**⚠️ Password Pattern**: Script expects `yourpass` but `.env` may have `ironclaw_pass`. Docker PostgreSQL defaults to `yourpass`. Sync `.env` with actual password:
```bash
# Check what Docker expects
docker inspect ironclaw-postgres | grep POSTGRES_PASSWORD

# Update .env to match
echo "DATABASE_URL=postgres://postgres:yourpass@localhost:5433/ironclaw" >> ~/.ironclaw/.env
```

### 4. Optional: Tunnel Configuration
```bash
# For webhook mode with persistent tunnel
export TUNNEL_URL="https://ironclaw.yourdomain.com"

# For ephemeral tunnel (testing)
# Don't set TUNNEL_URL - script generates automatically
```
**Expected**: Tunnel URL configured (or unset for ephemeral)

### 5. Create .env File (Optional)
```bash
cat > .env << EOF
NGC_KEY=nvapi-your-key
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
DATABASE_URL=postgres://postgres:pass@localhost:5433/ironclaw
EOF
```
**Expected**: `.env` file created

## Verification

```bash
# Check all required variables
env | grep -E "(NGC_KEY|TELEGRAM|DATABASE)" | sort

# Test database connection
psql "$DATABASE_URL" -c "SELECT 1;"
```

## 📂 Codebase References

**Environment Loading**:
- `src/config/env.rs` - Environment variable parsing
- `src/config/mod.rs` - Configuration module

**Validation**:
- `src/config/validation.rs` - Config validation logic

**Templates**:
- `.env.example` - Example environment file

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `NGC_KEY not set` | Export key or add to `.env` |
| Database connection failed | Check PostgreSQL running, port correct |
| Invalid bot token | Re-get from @BotFather |

## Related

- lookup/env-variables.md
- concepts/docker-postgres.md
- guides/build-process.md
