# Configuration Validation Rules

This document describes the validation rules used by the config-validator skill.

## Environment Variable Validation

### NGC_KEY

**Format**: Must match pattern `^nvapi-[A-Za-z0-9_-]+$`

**Validation Steps**:
1. Check if set in environment or .env file
2. Verify format matches NVIDIA NGC API key pattern
3. Truncate for display (show first 3 chars after "nvapi-")

**Common Issues**:
- Missing: User hasn't obtained API key yet
- Wrong format: User may have copied wrong value
- Expired: Key may have been revoked

**Fix**: Obtain key from https://org.ngc.nvidia.com/setup/personal-keys

### LLM_MODEL

**Valid Models** (as of 2026-03-19):
- `z-ai/glm5` ✅ (recommended)
- `z-ai/glm4` ✅
- `meta/llama3-70b-instruct` ✅
- `meta/llama3-8b-instruct` ✅
- `mistralai/mistral-large` ✅
- `mistralai/mixtral-8x7b-instruct-v0.1` ✅
- `nvidia/nemotron-4-340b-instruct` ✅

**Invalid Models**:
- `stepfun-ai/step-3.5-flash` ❌ (returns 404)

**Validation Steps**:
1. Check if in invalid models list (immediate warning)
2. Check if in valid models list (pass)
3. If unknown, warn and suggest verification

**API Test**:
- Send test request to NVIDIA NIM endpoint
- Check response code:
  - 200: Model available
  - 401: Invalid NGC_KEY
  - 404: Model not found

**Fix**: Change to `z-ai/glm5` in ~/.ironclaw/.env

### DATABASE_URL

**Expected Format**: `postgres://user:password@host:port/database`

**Validation Steps**:
1. Parse URL (must be valid postgres:// URL)
2. Extract components (host, port, database, password)
3. Check password against known values:
   - `yourpass` ✅ (Docker default)
   - `ironclaw_pass` ⚠️ (common mistake)

**Connection Test**:
1. Try configured URL
2. If fails, try alternatives:
   - Same URL, port 5432
   - Same URL, port 5433
   - Same URL, password "yourpass"
   - Same URL, password "ironclaw_pass"

**Common Issues**:
- Wrong port: Docker uses 5433 (mapped to 5432 inside container)
- Wrong password: Docker uses POSTGRES_PASSWORD env var value
- PostgreSQL not running: Check with `docker ps | grep postgres`

**Fix**: Update DATABASE_URL to match actual PostgreSQL setup

### TELEGRAM_BOT_TOKEN

**Format**: Must match pattern `^\d+:[A-Za-z0-9_-]+$`

**Validation Steps**:
1. Check if set (optional, skip if not present)
2. Verify format matches Telegram bot token pattern

**Common Issues**:
- Invalid format: Token may have been copied incorrectly
- Revoked: Token may have been revoked by @BotFather

**Fix**: Re-get token from @BotFather on Telegram

### TUNNEL_URL

**Format**: Must match pattern `^https://.+$`

**Validation Steps**:
1. Check if set (optional, skip if not present)
2. Verify URL uses HTTPS (not HTTP)

**Common Issues**:
- HTTP instead of HTTPS: Security risk
- Invalid URL: May not parse correctly

**Fix**: Use HTTPS URL or leave unset for ephemeral tunnel

## File Permission Validation

### ~/.ironclaw/ Directory

**Expected**: Directory exists and is readable

**Check**:
1. Verify directory exists
2. Check readable by current user

**Fix**: `mkdir -p ~/.ironclaw`

### ~/.ironclaw/.env File

**Expected Permissions**: 600 (owner read/write only) or 644 (owner read/write, others read)

**Check** (Unix only):
1. Read file permissions
2. Verify owner can read
3. Warn if writable by others

**Fix**: `chmod 600 ~/.ironclaw/.env`

### target/release/ironclaw Binary

**Expected**: Executable if exists

**Check** (Unix only):
1. Check if file exists
2. Verify executable bit is set

**Fix**: `chmod +x target/release/ironclaw`

## Auto-Fix Rules

### When to Auto-Fix

The `config_fix_all` tool will automatically fix:

1. **Invalid LLM_MODEL**:
   - If current model is in invalid models list
   - Changes to: `z-ai/glm5`

2. **Wrong DATABASE_URL Password**:
   - If password is "ironclaw_pass"
   - Changes to: "yourpass"

### Safety Measures

Before making changes:
1. Create backup at `~/.ironclaw/.env.backup`
2. Verify .env file is writable
3. Test fixes after applying

### When NOT to Auto-Fix

The skill will NOT automatically fix:
- Missing NGC_KEY (requires user action)
- Missing TELEGRAM_BOT_TOKEN (optional)
- Invalid TUNNEL_URL (requires user confirmation)
- File permissions (requires sudo in some cases)

## Error Codes and Meanings

### Database Connection

- **Connection refused**: PostgreSQL not running or wrong port
- **Password authentication failed**: Wrong password
- **Database does not exist**: Need to create or run migrations

### LLM API

- **200 OK**: Model available and key valid
- **401 Unauthorized**: Invalid or expired NGC_KEY
- **404 Not Found**: Model not available
- **503 Service Unavailable**: NVIDIA NIM temporarily down

### Environment File

- **File not found**: ~/.ironclaw/.env doesn't exist
- **Permission denied**: Can't read .env file
- **Invalid format**: Line doesn't match KEY=VALUE pattern

## Updating Validation Rules

To add new validation rules:

1. Add to `VALID_LLM_MODELS` or `INVALID_LLM_MODELS` constants
2. Add new check method to `ConfigValidator` struct
3. Update `check_env()` to call new check
4. Update `config_full_report()` to include new check
5. Add to `config_fix_all()` if auto-fixable
6. Update this documentation

## Testing Validation Rules

Test against these scenarios:

### Valid Configuration
```bash
NGC_KEY=nvapi-test123
LLM_MODEL=z-ai/glm5
DATABASE_URL=postgres://postgres:yourpass@localhost:5433/ironclaw
```

Expected: All checks pass

### Missing .env
```bash
rm ~/.ironclaw/.env
```

Expected: Environment File check fails

### Wrong LLM Model
```bash
LLM_MODEL=stepfun-ai/step-3.5-flash
```

Expected: LLM Model check warns, suggests fix

### Wrong Database Password
```bash
DATABASE_URL=postgres://postgres:ironclaw_pass@localhost:5433/ironclaw
```

Expected: Database URL check warns about password

## Related

- [env-config.md](../../../../.opencode/context/core/architecture/guides/env-config.md)
- [llm-model-mismatch.md](../../../../.opencode/context/core/architecture/errors/llm-model-mismatch.md)
- [docker-postgres.md](../../../../.opencode/context/core/architecture/concepts/docker-postgres.md)
