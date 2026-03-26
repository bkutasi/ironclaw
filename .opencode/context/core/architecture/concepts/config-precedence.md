# Configuration Precedence

**Purpose**: How IronClaw resolves configuration values from multiple sources

**Last Updated**: 2026-03-22

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Core Concept

IronClaw uses a three-tier configuration precedence system: **environment variables** (highest) → **user config file** (`~/.ironclaw/.env`) → **script defaults** (lowest). Secrets should be exported in shell, persistent settings in the user config file.

---

## Key Points

- **Environment vars** override everything (export in shell for session secrets)
- **`~/.ironclaw/.env`** persists across sessions (sandbox policy, ports, database URL)
- **Script defaults** are fallbacks only (never edit startup scripts)
- **Secrets separation**: API keys in shell exports, not plaintext config files
- **Single source**: Consolidate settings to avoid conflicts

---

## Quick Example

```bash
# 1. Secrets → Shell export (highest priority)
export NGC_KEY="your-nvidia-api-key"

# 2. Persistent settings → ~/.ironclaw/.env
SANDBOX_POLICY=workspace_write
DATABASE_URL="postgres://postgres:pass@localhost:5433/ironclaw"

# 3. Script defaults → Fallback (don't edit)
# GATEWAY_PORT=3004 (default if not set elsewhere)
```

---

## Verification

```bash
# Check what's loaded from ~/.ironclaw/.env
./target/release/ironclaw 2>&1 | grep -E "(Loaded configuration|SANDBOX_POLICY)"

# Check for conflicts
grep SANDBOX_POLICY ~/.ironclaw/.env 2>/dev/null || echo "not set"
echo "Shell: SANDBOX_POLICY=$SANDBOX_POLICY"
```

---

## Reference

- Full guide: `.tmp/CONFIG_MIGRATION_GUIDE.md`
- Startup: `./target/release/ironclaw`

---

**Related**:
- architecture/guides/env-config.md
- architecture/lookup/env-variables.md
