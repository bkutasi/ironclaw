# Telegram Bot No Response

This document covers troubleshooting when Telegram bot receives messages but doesn't respond.

## Symptom Checklist

- ✅ Bot receives messages (visible in logs)
- ❌ Bot doesn't send replies
- ✅ No errors in IronClaw logs
- ✅ Webhook registered correctly

## Common Causes

### 1. dm_policy Blocking

**Most Common Cause**

By default, IronClaw uses `dm_policy=pairing` which requires users to pair before the bot responds to DMs.

**Detection**:
```bash
# Check dm_policy setting
cat ~/.ironclaw/state/dm_policy
# Output: pairing

# Check if user is paired
ironclaw pairing list telegram
# User not in list = not paired
```

**Fix**:
```bash
# Option 1: Pair the user (recommended)
# Send /pair to bot on Telegram
# Then approve from IronClaw:
ironclaw pairing approve telegram <code>

# Option 2: Switch to open mode (NOT recommended for production)
# Edit ~/.ironclaw/.env:
TELEGRAM_DM_POLICY=open

# Restart IronClaw
cargo run
```

### 2. LLM Model Mismatch

**Second Most Common Cause**

Bot receives messages but fails to generate responses due to invalid LLM model.

**Detection**:
```bash
# Check LLM model
grep LLM_MODEL ~/.ironclaw/.env
# Output: LLM_MODEL="stepfun-ai/step-3.5-flash"  # WRONG - returns 404

# Test model availability
curl -X POST https://api.nvcf.nvidia.com/v2/nvcf/pfx/functions/invoke \
  -H "Authorization: Bearer $NGC_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"stepfun-ai/step-3.5-flash\"}"
# Returns 404 Not Found
```

**Fix**:
```bash
# Edit ~/.ironclaw/.env
LLM_MODEL="z-ai/glm5"  # NVIDIA NIM (recommended)

# Or use auto-fix
telegram_fix_common_issues

# Restart IronClaw
cargo run
```

### 3. Database Connection Failed

Bot can't save state or retrieve context.

**Detection**:
```bash
# Check database connection
psql "$DATABASE_URL" -c "SELECT 1"
# Error: password authentication failed

# Check IronClaw logs for database errors
RUST_LOG=debug cargo run 2>&1 | grep -i database
```

**Fix**:
```bash
# Check PostgreSQL running
docker ps | grep postgres

# Fix DATABASE_URL in ~/.ironclaw/.env
# Default Docker PostgreSQL:
DATABASE_URL="postgres://postgres:yourpass@localhost:5432/ironclaw"

# Or system PostgreSQL:
DATABASE_URL="postgres://postgres:ironclaw_pass@localhost:5433/ironclaw"

# Restart IronClaw
cargo run
```

### 4. Bot Token Invalid

Telegram API rejects requests.

**Detection**:
```bash
# Test bot token
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"
# Returns: {"ok":false,"error_code":401,"description":"Unauthorized"}
```

**Fix**:
```bash
# Get new token from @BotFather
# 1. Open Telegram
# 2. Search @BotFather
# 3. Send /token
# 4. Select your bot
# 5. Copy new token

# Update ~/.ironclaw/.env
TELEGRAM_BOT_TOKEN="123456789:AAHccDDeeFFggHHiiJJkkLLmmNNoo"

# Restart IronClaw
cargo run
```

### 5. Webhook Not Processing Updates

Webhook registered but updates not processed.

**Detection**:
```bash
# Check pending updates
curl -s "https://api.telegram.org/bot$TOKEN/getWebhookInfo" | jq .result.pending_update_count
# High number (>100) = updates stuck

# Check IronClaw logs
RUST_LOG=debug cargo run 2>&1 | grep -i "webhook\|update"
```

**Fix**:
```bash
# Restart IronClaw to clear backlog
pkill ironclaw
cargo run

# Or delete and re-register webhook
curl -X POST "https://api.telegram.org/bot$TOKEN/deleteWebhook"
telegram_fix_common_issues
```

### 6. Message Filtering

Bot ignores certain message types.

**Detection**:
```bash
# Check if bot ignores commands
# Send regular text (not /command)

# Check if bot ignores media
# Send text message instead of photo

# Check logs for filtered messages
RUST_LOG=debug cargo run 2>&1 | grep -i "filter\|ignore\|skip"
```

**Fix**:
```bash
# Check configuration
cat ~/.ironclaw/.env | grep -i "ignore\|filter"

# Adjust settings if needed
# SIGNAL_IGNORE_ATTACHMENTS=false
# SIGNAL_IGNORE_STORIES=true
```

## Debugging Workflow

### Step 1: Check dm_policy

```bash
cat ~/.ironclaw/state/dm_policy
# If "pairing", send /pair to bot
```

### Step 2: Check LLM Model

```bash
grep LLM_MODEL ~/.ironclaw/.env
# Should be "z-ai/glm5" not "stepfun-ai/step-3.5-flash"
```

### Step 3: Test Database

```bash
psql "$DATABASE_URL" -c "SELECT 1"
# Should return successfully
```

### Step 4: Check Logs

```bash
RUST_LOG=ironclaw=debug,telegram=debug cargo run 2>&1 | tee ironclaw.log
# Send message to bot
# Check logs for:
# - "Received update"
# - "Processing message"
# - "Sending response"
```

### Step 5: Run Diagnostics

```bash
telegram_full_report
telegram_fix_common_issues
```

## Response Flow Diagram

```
User Message
    ↓
Telegram API
    ↓
Webhook / Polling
    ↓
IronClaw Channel
    ↓
dm_policy Check ←── FAIL: "pairing" mode, user not paired
    ↓
LLM Processing ←── FAIL: Invalid model, API error
    ↓
Database Save ←── FAIL: Connection error
    ↓
Response Send ←── FAIL: Bot token invalid
    ↓
User Receives Reply ✅
```

## Quick Fix Checklist

Run in order:

1. ✅ Send `/pair` to bot (if dm_policy=pairing)
2. ✅ Check LLM_MODEL is valid
3. ✅ Verify database connection
4. ✅ Test bot token
5. ✅ Restart IronClaw
6. ✅ Run `telegram_fix_common_issues`

## Testing Response

After fixes, test with:

```bash
# Send simple message to bot
"Hello, are you working?"

# Expected response within 30 seconds
# If no response, check logs:
tail -f ironclaw.log | grep -A5 -B5 "Hello"
```

## Related

- `integrations/errors/tunnel-errors.md` - Tunnel troubleshooting
- `architecture/errors/llm-model-mismatch.md` - LLM model issues
- `skills/telegram-debug/SKILL.md` - Automated diagnostics
