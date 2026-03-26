<!-- Context: integrations/errors | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Error: Bot No Response to DMs

**Purpose**: Fix Telegram bot not responding to direct messages

**Last Updated**: 2026-03-19

---

## Symptom

- Bot doesn't respond to direct messages
- No errors in logs
- Bot responds to paired users but not unpaired users
- Commands like `/start` work but regular messages ignored

---

## Cause

Bot is in **pairing mode** by default. The `dm_policy` setting controls how bot handles DMs from unpaired users:

- `"pairing"` (default): Only responds to users who have sent `/pair` command
- `"open"`: Responds to all DMs

**Location**: `~/.ironclaw/channels/telegram/state/dm_policy`

---

## Solution

### Option 1: Send /pair Command (Quick Fix)
```
Message your bot on Telegram:
/pair

Bot will respond and you're now paired
```

### Option 2: Change dm_policy to Open
```bash
# Edit the dm_policy file
nano ~/.ironclaw/channels/telegram/state/dm_policy

# Change content from:
pairing

# To:
open

# Restart IronClaw
pkill ironclaw
./target/release/ironclaw
```

### Option 3: Set via Environment
```bash
# Add to ~/.ironclaw/.env
TELEGRAM_DM_POLICY="open"

# Restart IronClaw
```

---

## Verification

```bash
# Check current policy
cat ~/.ironclaw/channels/telegram/state/dm_policy

# Test with unpaired Telegram account
# Send message to bot - should get response
```

---

## 📂 Codebase References

**Configuration**:
- `~/.ironclaw/channels/telegram/state/dm_policy` - DM policy setting
- `src/channels/telegram.rs` - Telegram channel implementation

**State Management**:
- `src/channels/state.rs` - Channel state handling
- `~/.ironclaw/channels/telegram/state/` - Channel state directory

---

## Related

- concepts/telegram-integration.md
- guides/telegram-webhook-setup.md
- errors/telegram-webhook-errors.md
