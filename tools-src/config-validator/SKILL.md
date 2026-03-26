---
name: config-validator
version: 1.0.0
description: Validate IronClaw configuration and catch common issues before runtime failures
activation:
  keywords:
    - config check
    - validate config
    - configuration test
    - doctor
    - ironclaw doctor
    - check env
    - test database
    - test llm
    - config report
  patterns:
    - "(?i)check.*config"
    - "(?i)validate.*configuration"
    - "(?i)test.*connection"
    - "(?i)config.*issue"
  max_context_tokens: 3000
---

# IronClaw Configuration Validator

Use this skill to validate IronClaw configuration and catch common issues before they cause runtime failures.

## Quick Start

```bash
# Run full configuration check
config_full_report

# Check only environment variables
config_check_env

# Test database connection
config_check_database

# Validate LLM model configuration
config_check_llm

# Auto-fix common issues
config_fix_all
```

## Tools

### config_check_env

Checks environment variables and ~/.ironclaw/.env file for:
- NGC_KEY format (must start with "nvapi-")
- LLM_MODEL validity (not stepfun-ai/step-3.5-flash which returns 404)
- DATABASE_URL format and password matching Docker setup
- TELEGRAM_BOT_TOKEN format (if present)
- TUNNEL_URL is HTTPS (if present)

**Common issues detected**:
- Wrong LLM model (stepfun-ai/step-3.5-flash → should be z-ai/glm5)
- Database password mismatch (ironclaw_pass vs yourpass)
- Missing required variables

### config_check_database

Tests database connection with DATABASE_URL:
- Attempts connection with configured URL
- If fails, tries common alternatives:
  - Port 5432 vs 5433
  - Password "yourpass" vs "ironclaw_pass"
- Reports which configuration works

### config_check_llm

Validates LLM model configuration:
- Checks LLM_MODEL against known valid models:
  - ✅ z-ai/glm5 (NVIDIA NIM)
  - ✅ meta/llama3-70b-instruct
  - ❌ stepfun-ai/step-3.5-flash (returns 404)
- Tests API endpoint if NGC_KEY is set
- Verifies 200 response (not 401 or 404)

### config_full_report

Generates complete configuration health report:
```
IronClaw Configuration Report
==============================

✅ Environment File: ~/.ironclaw/.env (found)
✅ NGC Key: Configured (nvapi-***...truncated)
⚠️  LLM Model: stepfun-ai/step-3.5-flash (INVALID - returns 404)
   → Fix: Change to "z-ai/glm5" in ~/.ironclaw/.env
✅ Database: postgres@localhost:5433 (connected)
⚠️  Database Password: ironclaw_pass (might be wrong)
   → Docker PostgreSQL uses "yourpass"
✅ Telegram Bot: Configured (@Tractor333_bot)
✅ Tunnel: Not configured (ephemeral will auto-generate)

Issues Found: 2
1. LLM model will cause 404 errors
2. Database password may not match Docker setup

Run: config_fix_all to auto-fix these issues
```

### config_fix_all

Auto-fixes common configuration issues:
- Backs up ~/.ironclaw/.env to ~/.ironclaw/.env.backup
- Fixes LLM_MODEL to "z-ai/glm5"
- Fixes DATABASE_URL password to "yourpass"
- Verifies fixes work
- Reports what changed

## Configuration Checks

### Environment Variables

Validates these variables in ~/.ironclaw/.env:

**Required**:
- `NGC_KEY` - NVIDIA NGC API key (format: nvapi-*)
- `LLM_MODEL` - Valid model name (z-ai/glm5 recommended)
- `DATABASE_URL` - PostgreSQL connection string

**Optional**:
- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `TUNNEL_URL` - Must be HTTPS if set

### Database Connection

Tests PostgreSQL connectivity:
1. Primary: DATABASE_URL as configured
2. Fallback 1: Same URL but port 5432
3. Fallback 2: Same URL but password "yourpass"
4. Fallback 3: Same URL but password "ironclaw_pass"

Reports which connection succeeds.

### LLM Model Validation

Checks model availability:
- Validates against NVIDIA NIM catalog
- Tests actual API endpoint with provided NGC_KEY
- Reports 401 (auth error) vs 404 (model not found)

### File Permissions

Checks:
- ~/.ironclaw/ directory exists and is readable
- .env file permissions (600 or 644 recommended)
- target/release/ironclaw is executable (if exists)

## Usage Examples

### Example 1: Check All Configuration

```
config_full_report
```

Output:
```
IronClaw Configuration Report
==============================

✅ Environment File: ~/.ironclaw/.env (found)
✅ NGC Key: Configured (nvapi-***abcd)
✅ LLM Model: z-ai/glm5 (valid)
✅ Database: postgres@localhost:5433 (connected)
✅ Telegram Bot: Not configured
✅ Tunnel: Not configured

Issues Found: 0
Configuration looks good!
```

### Example 2: Fix Common Issues

```
config_fix_all
```

Output:
```
Auto-Fix Configuration Issues
=============================

Backing up ~/.ironclaw/.env → ~/.ironclaw/.env.backup
✅ Fixed LLM_MODEL: stepfun-ai/step-3.5-flash → z-ai/glm5
✅ Fixed DATABASE_URL password: ironclaw_pass → yourpass

Verifying fixes...
✅ LLM model now valid
✅ Database connection successful

Changes applied successfully!
```

### Example 3: Test Database Only

```
config_check_database
```

Output:
```
Database Connection Test
========================

Testing: postgres://postgres:***@localhost:5433/ironclaw
❌ Failed: Connection refused

Trying alternatives...
✅ Success: postgres://postgres:yourpass@localhost:5432/ironclaw

Recommendation: Update DATABASE_URL to use port 5432 and password "yourpass"
```

## Troubleshooting

### NGC_KEY Invalid Format

**Error**: `NGC_KEY must start with "nvapi-"`

**Fix**: 
1. Visit https://org.ngc.nvidia.com/setup/personal-keys
2. Generate new API key
3. Update ~/.ironclaw/.env: `NGC_KEY=nvapi-your-key`

### LLM Model Returns 404

**Error**: `Model stepfun-ai/step-3.5-flash not available`

**Fix**:
```bash
# Edit ~/.ironclaw/.env
LLM_MODEL="z-ai/glm5"  # NVIDIA NIM (recommended)
```

Or run: `config_fix_all`

### Database Connection Failed

**Error**: `Connection refused` or `Password authentication failed`

**Fix**:
1. Check PostgreSQL running: `docker ps | grep postgres`
2. Verify port: 5432 (system) or 5433 (Docker)
3. Check password matches Docker: `yourpass` (default)

Or run: `config_check_database` to auto-detect working config

### Missing .env File

**Error**: `~/.ironclaw/.env not found`

**Fix**:
```bash
mkdir -p ~/.ironclaw
cp .env.example ~/.ironclaw/.env
# Edit with your values
nano ~/.ironclaw/.env
```

## Related

- architecture/guides/env-config.md
- architecture/errors/llm-model-mismatch.md
- architecture/errors/common-build-issues.md
- architecture/concepts/docker-postgres.md
