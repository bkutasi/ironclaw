# Documentation Fixes - Secrets Management CLI

**Date**: 2026-02-20  
**Issue**: Documentation referenced non-existent `ironclaw secrets` CLI commands

---

## Summary

The `SECURE_SETUP_GUIDE.md` and related documentation incorrectly referenced `ironclaw secrets set/get/delete/list` commands that **do not exist** in IronClaw.

**Root Cause**: Assumptions made during documentation creation without verifying actual CLI implementation.

---

## What Was Fixed

### 1. Removed All `ironclaw secrets *` References

**Incorrect** (removed):
```bash
ironclaw secrets set http_webhook_secret <value>
ironclaw secrets get telegram_bot_token
ironclaw secrets delete my_secret
ironclaw secrets list
```

**Correct** (updated to):
```bash
# Via onboarding wizard
ironclaw onboard

# Via environment variables
export HTTP_WEBHOOK_SECRET=<value>
echo "HTTP_WEBHOOK_SECRET=<value>" >> .env

# Direct database query (advanced)
psql "$DATABASE_URL" -c "SELECT name FROM secrets WHERE user_id='default';"
```

### 2. Updated Sections

| Section | Lines Fixed | Change |
|---------|-------------|--------|
| Webhook Secret Setup | 711-728 | Removed `ironclaw secrets set`, added env var method |
| Telegram Bot Token | 805-813 | Updated to env var method |
| Slack Credentials | 930-941 | Updated to env var method |
| Gateway Auth Token | 970-986 | Clarified auto-generation, added env var method |
| Secret Lifecycle | 1155-1217 | Complete rewrite - removed all CLI commands |
| Troubleshooting | 1375-1397 | Updated "Secret not found" solution |
| Quick Reference | 1473-1484 | Removed secrets commands, added env var examples |
| Error Table | 2180 | Updated solution for "Secret not found" |

---

## How Secrets Actually Work in IronClaw

### Storage Methods

| Method | When Used | Security Level |
|--------|-----------|----------------|
| **Onboarding Wizard** | Initial setup | ✅ Recommended - encrypted in DB |
| **Environment Variables** | Manual config, Docker, CI | ⚠️ Development only |
| **OS Keychain** | Master key (automatic) | ✅ Best - hardware-backed |
| **Direct DB Insert** | Advanced users | ⚠️ Manual encryption required |

### Secret Resolution Priority

1. **Environment variables** (highest priority)
2. **Secrets store** (database, encrypted)
3. **Default values** (lowest priority)

### Available CLI Commands

```bash
# Setup
ironclaw onboard              # Full onboarding
ironclaw onboard --channels-only
ironclaw onboard --skip-auth

# Configuration (settings, NOT secrets)
ironclaw config list
ironclaw config get <key>
ironclaw config set <key> <value>

# MCP authentication (stores OAuth tokens in secrets)
ironclaw mcp auth <server-name>

# Other commands
ironclaw run
ironclaw tool list
ironclaw memory search
ironclaw pairing list telegram
ironclaw status
```

**Note**: No `ironclaw secrets` subcommand exists.

---

## Correct Ways to Set Secrets

### HTTP Webhook Secret

```bash
# Method 1: Environment variable (recommended for manual setup)
echo "HTTP_WEBHOOK_SECRET=$(openssl rand -hex 32)" >> .env

# Method 2: During onboarding
ironclaw onboard
# Step 6: Enable HTTP channel → auto-generates secret

# Method 3: Export for current session
export HTTP_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

### Telegram Bot Token

```bash
# Method 1: Environment variable
echo "TELEGRAM_BOT_TOKEN=123456789:ABCdef..." >> .env

# Method 2: During onboarding
ironclaw onboard
# Step 6: Enable Telegram → paste token

# Method 3: Export
export TELEGRAM_BOT_TOKEN=123456789:ABCdef...
```

### Slack Credentials

```bash
# All three required
echo "SLACK_BOT_TOKEN=xoxb-..." >> .env
echo "SLACK_APP_TOKEN=xapp-..." >> .env
echo "SLACK_SIGNING_SECRET=..." >> .env

# Or during onboarding
ironclaw onboard
```

### Gateway Auth Token

```bash
# Auto-generated on startup if not set
# Token printed to logs: "Web gateway enabled on 127.0.0.1:3003/?token=..."

# Or set manually
echo "GATEWAY_AUTH_TOKEN=$(openssl rand -hex 32)" >> .env
```

---

## Files Modified

- `docs/SECURE_SETUP_GUIDE.md` - 14 occurrences fixed
- `docs/SETUP_CHECKLIST.md` - No changes needed (didn't reference secrets CLI)
- `.env.secure.example` - No changes needed (already uses env vars)

---

## Verification

All references to `ironclaw secrets` have been removed or corrected:

```bash
# Search for remaining incorrect references
grep -r "ironclaw secrets" docs/
# Result: 0 matches (all fixed)
```

---

## Lessons Learned

1. **Verify CLI commands exist** before documenting them
2. **Read actual source code** (`src/main.rs`, `src/cli/`) to understand command structure
3. **Test commands** in a real environment before writing docs
4. **Assume nothing** - even if a feature "should" exist, verify it does

---

## Next Steps

1. ✅ All documentation corrected
2. ✅ `.env.secure.example` already correct (uses env vars)
3. ⏳ Consider adding `ironclaw secrets` CLI command in future (feature request)
4. ⏳ Update README if it references secrets CLI (check needed)

---

**Status**: ✅ Complete - All incorrect references fixed
