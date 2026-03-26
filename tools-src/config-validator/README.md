# IronClaw Configuration Validator Tool

A WASM-based configuration validation tool for IronClaw that catches common issues before they cause runtime failures.

## Overview

This tool validates your IronClaw configuration including:
- Environment variable validation (NGC_KEY, LLM_MODEL, DATABASE_URL)
- File existence checks (~/.ironclaw/.env)
- Format validation for URLs and tokens
- Detection of common misconfigurations

**Note**: This is a WASM tool that runs in a sandbox. For full functionality including database connection tests and LLM API tests, use the companion SKILL.md version which has host assistance.

## Installation

### Build from Source

```bash
# Navigate to the tool directory
cd tools-src/config-validator

# Build for WASM target
cargo build --release --target wasm32-wasip2

# The WASM module will be at:
# target/wasm32-wasip2/release/config_validator_tool.wasm
```

### Register with IronClaw

```bash
# Copy WASM module to IronClaw tools directory
cp target/wasm32-wasip2/release/config_validator_tool.wasm \
   ~/.ironclaw/tools/

# Or use the ironclaw CLI to install
ironclaw tool install ./target/wasm32-wasip2/release/config_validator_tool.wasm
```

## Usage

### Check Environment Variables

```json
{
  "tool": "config_validator",
  "params": {
    "check": "env"
  }
}
```

Returns JSON array of validation results:
```json
[
  {
    "status": "pass",
    "name": "Environment File",
    "message": "/home/user/.ironclaw/.env (found)",
    "fix_suggestion": null
  },
  {
    "status": "pass",
    "name": "NGC Key",
    "message": "Configured (nvapi-abc***)",
    "fix_suggestion": null
  },
  {
    "status": "warning",
    "name": "LLM Model",
    "message": "stepfun-ai/step-3.5-flash (INVALID - returns 404)",
    "fix_suggestion": "Change to \"z-ai/glm5\" in ~/.ironclaw/.env"
  }
]
```

### Generate Full Report

```json
{
  "tool": "config_validator",
  "params": {
    "check": "full_report"
  }
}
```

Returns formatted text report:
```
IronClaw Configuration Report
==============================

✅ Environment File: /home/user/.ironclaw/.env (found)
✅ NGC Key: Configured (nvapi-abc***)
⚠️  LLM Model: stepfun-ai/step-3.5-flash (INVALID - returns 404)
   → Fix: Change to "z-ai/glm5" in ~/.ironclaw/.env
✅ Database URL: postgres@localhost:5433 (format valid)
✅ Telegram Bot: Not configured
✅ Tunnel: Not configured

Issues Found: 1
1. LLM Model - stepfun-ai/step-3.5-flash (INVALID - returns 404)

Note: Run config_fix_all skill to auto-fix common issues
```

## Validation Rules

### NGC_KEY

**Format**: Must match pattern `^nvapi-[A-Za-z0-9_-]+$`

**Common Issues**:
- Missing: User hasn't obtained API key
- Wrong format: Copied wrong value

**Fix**: Get key from https://org.ngc.nvidia.com/setup/personal-keys

### LLM_MODEL

**Valid Models**:
- `z-ai/glm5` ✅ (recommended)
- `z-ai/glm4` ✅
- `meta/llama3-70b-instruct` ✅
- `meta/llama3-8b-instruct` ✅
- `mistralai/mistral-large` ✅
- `mistralai/mixtral-8x7b-instruct-v0.1` ✅
- `nvidia/nemotron-4-340b-instruct` ✅

**Invalid Models**:
- `stepfun-ai/step-3.5-flash` ❌ (returns 404)

### DATABASE_URL

**Expected Format**: `postgres://user:password@host:port/database`

**Password Check**:
- `yourpass` ✅ (Docker default)
- `ironclaw_pass` ⚠️ (common mistake)

### TELEGRAM_BOT_TOKEN

**Format**: Must match pattern `^\d+:[A-Za-z0-9_-]+$`

**Optional**: Only validated if present

### TUNNEL_URL

**Format**: Must match pattern `^https://.+$`

**Optional**: Only validated if present

## Common Issues Detected

### 1. Invalid LLM Model

**Problem**: `LLM_MODEL="stepfun-ai/step-3.5-flash"` returns 404

**Fix**:
```bash
echo 'LLM_MODEL="z-ai/glm5"' >> ~/.ironclaw/.env
```

### 2. Database Password Mismatch

**Problem**: `DATABASE_URL` uses `ironclaw_pass` but Docker uses `yourpass`

**Fix**:
```bash
sed -i 's/ironclaw_pass/yourpass/' ~/.ironclaw/.env
```

### 3. Missing NGC Key

**Problem**: `NGC_KEY` not set

**Fix**:
1. Visit https://org.ngc.nvidia.com/setup/personal-keys
2. Generate API key
3. Add to `~/.ironclaw/.env`:
   ```
   NGC_KEY=nvapi-your-key-here
   ```

## Limitations

As a WASM sandboxed tool, this validator has some limitations:

- ❌ Cannot test actual database connections
- ❌ Cannot test LLM API endpoints
- ❌ Cannot auto-fix configuration (no write access)
- ❌ Cannot check file permissions (platform-specific)

For full functionality, use the companion SKILL.md version which runs with host assistance.

## Building

```bash
# Add WASM target if not already installed
rustup target add wasm32-wasip2

# Build release version
cargo build --release --target wasm32-wasip2

# Check WASM module
file target/wasm32-wasip2/release/config_validator_tool.wasm
```

## Testing

```bash
# Run unit tests
cargo test

# Test with sample .env
mkdir -p ~/.ironclaw
cat > ~/.ironclaw/.env << EOF
NGC_KEY=nvapi-test123
LLM_MODEL=z-ai/glm5
DATABASE_URL=postgres://postgres:yourpass@localhost:5433/ironclaw
EOF
```

## Security

- Reads only from `~/.ironclaw/.env`
- No network access (WASM sandboxed)
- No write access (read-only validation)
- Credentials never leave the sandbox

## Troubleshooting

### Build Fails

**Error**: `error: package requires rustc 1.70 or newer`

**Solution**:
```bash
rustup update
```

### WASM Target Not Found

**Error**: `error: package cannot be built because target is not installed`

**Solution**:
```bash
rustup target add wasm32-wasip2
```

## Related Resources

- [Environment Configuration Guide](../../.opencode/context/core/architecture/guides/env-config.md)
- [LLM Model Mismatch Error](../../.opencode/context/core/architecture/errors/llm-model-mismatch.md)
- [Docker PostgreSQL Setup](../../.opencode/context/core/architecture/concepts/docker-postgres.md)

## License

MIT OR Apache-2.0
