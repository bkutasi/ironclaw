<!-- Context: core/errors | Priority: critical | Version: 1.0 | Created: 2026-03-25 -->

# Error Guide: Telegram Proactive Message Failure

## Symptom

Proactive Telegram messages fail to send even though routine executes successfully.

**Error indicators:**
- Logs show: `No stored owner routing target (have you messaged the bot first?)`
- Routine status: `Completed` but no message received
- Broadcast completes but proactive path returns early
- Works after sending `/start` to bot, fails otherwise

## Root Cause

**Proactive messaging requires prior conversation context.** The system routes proactive messages to the last user who received a broadcast (`last_broadcast_metadata.chat_id`), but this fails when:

1. **No broadcast history**: `last_broadcast_metadata` is empty/null
2. **Owner hasn't interacted**: `TELEGRAM_OWNER_ID` set but user never sent `/start` or any message
3. **Chat ID not persisted**: No conversation metadata stored in workspace KV

**Technical flow:**
```
routine_trigger.rs:223 → telegram.send_message(chat_id)
                          ↓
                      chat_id from last_broadcast_metadata OR TELEGRAM_OWNER_ID lookup
                          ↓
                      If no metadata exists → "No stored owner routing target" → return early
```

## Solution

### Immediate Fix (Development/Testing)

**Send `/start` to the bot first** to establish conversation context:

1. Open Telegram
2. Navigate to your bot
3. Send `/start` command
4. Verify bot responds (conversation metadata now stored)
5. Re-run routine or wait for next scheduled trigger

### Permanent Fix Options

#### Option 1: Store Owner Chat ID Explicitly (Recommended)

Add configuration to store owner's chat ID directly:

```toml
# .env
TELEGRAM_OWNER_CHAT_ID="-1001234567890"  # Optional: direct chat ID override
```

**Implementation:**
```rust
// In telegram channel send logic
let chat_id = env_var("TELEGRAM_OWNER_CHAT_ID")
    .or_else(|| last_broadcast_metadata.chat_id)
    .or_else(|| lookup_owner_chat_id())
    .ok_or("No routing target available")?;
```

#### Option 2: Initialize Chat ID on Bot Startup

Add startup routine to fetch bot info and store owner chat ID:

```rust
// On bot initialization
if let Ok(user) = telegram_api.get_me().await {
    // Store bot username/ID for reference
    workspace_kv.set("bot_username", user.username).await?;
}
```

#### Option 3: Fallback to User Lookup

If `TELEGRAM_OWNER_ID` is set but no chat context exists, query Telegram API:

```rust
// Fallback: try to resolve owner user to chat
if let Some(owner_id) = env_var("TELEGRAM_OWNER_ID") {
    if let Ok(chat) = telegram_api.get_user_chat(owner_id).await {
        workspace_kv.set("owner_chat_id", chat.id).await?;
        return Ok(chat.id);
    }
}
```

**Note**: Telegram API limitations may restrict this approach (privacy settings, bot-user chat requirements).

## Verification

After applying fix:

1. **Check workspace KV**:
   ```bash
   # Verify conversation metadata exists
   wrangler kv:key get --binding=WORKSPACE_KV last_broadcast_metadata
   ```

2. **Test proactive send**:
   ```bash
   # Trigger routine manually
   cargo run -- routine_trigger --name=your-routine
   ```

3. **Monitor logs**:
   ```
   ✓ Sending proactive Telegram message to chat_id: -1001234567890
   ✓ Message sent successfully (message_id: 12345)
   ```

## Prevention

### Development Workflow

1. **Always send `/start` after bot token refresh**
2. **Document chat ID requirement** in setup instructions
3. **Add startup check** that warns if no conversation context exists:
   ```rust
   if workspace_kv.get("last_broadcast_metadata").await?.is_none() {
       log::warn!("No conversation context - proactive messages will fail until user interacts");
   }
   ```

### Production Deployment

1. **Store owner chat ID in environment** (not derived from interaction)
2. **Add health check** for Telegram routing capability
3. **Implement fallback notification** (email, webhook) if Telegram unavailable

## Related Files

- `src/channels/telegram.rs` - Telegram channel implementation
- `src/worker/routine_trigger.rs:223` - Proactive message trigger point
- `src/context/workspace_kv.rs` - Workspace KV storage layer
- `.opencode/context/core/channels/telegram.md` - Telegram channel documentation

## Related Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `Telegram bot token invalid` | Token expired/revoked | Refresh token via @BotFather |
| `Webhook failed to resolve host` | Tunnel URL malformed | Check for trailing newlines in URL |
| `403 Forbidden (Bot Fight Mode)` | Cloudflare blocking Telegram | Whitelist Telegram IPs or disable Bot Fight Mode |
| `No stored owner routing target` | **This error** | Send `/start` or configure `TELEGRAM_OWNER_CHAT_ID` |

## Testing Checklist

- [ ] Bot responds to `/start` command
- [ ] `last_broadcast_metadata` populated in workspace KV
- [ ] Proactive message sends without prior broadcast
- [ ] Routine trigger logs show successful send
- [ ] Message received in Telegram within 5 seconds

## Recovery Procedure

If proactive messages suddenly stop working:

1. **Check workspace KV state**:
   ```bash
   wrangler kv:key get --binding=WORKSPACE_KV last_broadcast_metadata
   ```

2. **Verify bot token validity**:
   ```bash
   curl "https://api.telegram.org/bot<TOKEN>/getMe"
   ```

3. **Reset conversation context** (if corrupted):
   ```bash
   wrangler kv:key delete --binding=WORKSPACE_KV last_broadcast_metadata
   # Then send /start to bot again
   ```

4. **Check Cloudflare logs** for 403 errors:
   - Navigate to Cloudflare Dashboard → Security → Events
   - Filter by your worker domain
   - Look for blocked Telegram webhook requests

## Historical Context

**Discovered**: 2026-03-25 during routine self-evaluation testing

**Scenario**: Routine `self-evaluation-evolving` completed successfully but proactive Telegram notification never arrived. Investigation revealed:
- Bot token refreshed earlier in day
- No interaction with bot since refresh
- `last_broadcast_metadata` empty (no broadcast history)
- System had no chat ID to route proactive message

**Resolution**: Sent `/start` to bot, establishing conversation context. Proactive messages resumed.

**Lesson**: Conversation state persistence creates hidden dependency on user interaction. Production systems should decouple routing from interaction history.
