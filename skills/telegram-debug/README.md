# Telegram Debug Skill for IronClaw

Automated diagnostics and auto-fix for Telegram bot integration issues.

## Installation

The skill is located at `skills/telegram-debug/` in the IronClaw repository.

### Install to IronClaw

```bash
# Copy skill to IronClaw skills directory
cp -r skills/telegram-debug ~/.ironclaw/skills/

# Or symlink for development
ln -s $(pwd)/skills/telegram-debug ~/.ironclaw/skills/telegram-debug

# Verify installation
ironclaw skills list | grep telegram-debug
```

## Usage

### Quick Start

```bash
# Run full health check
telegram_full_report

# Auto-fix common issues
telegram_fix_common_issues
```

### Individual Checks

```bash
# Check webhook status with Telegram API
telegram_check_webhook

# Verify Cloudflare tunnel is alive
telegram_check_tunnel

# Validate IronClaw configuration
telegram_check_config
```

## What It Checks

### 1. Webhook Status

- ✅ URL is registered (not empty)
- ✅ Pending updates < 100
- ✅ No recent errors
- ✅ URL matches tunnel

### 2. Tunnel Health

- ✅ `cloudflared` process running
- ✅ DNS resolves correctly
- ✅ HTTP endpoint accessible
- ✅ URLs match

### 3. Configuration

- ✅ `~/.ironclaw/.env` exists
- ✅ Bot token valid format
- ✅ LLM model available
- ✅ Database connected
- ✅ dm_policy configured

### 4. Auto-Fixes

- 🔄 Restart dead tunnel
- 🔄 Re-register webhook
- 🔄 Fix LLM model
- 🔄 Fix database password
- 🔄 Suggest pairing

## Example Output

### Full Report

```
$ telegram_full_report

Telegram Integration Health Check
=================================

✅ Webhook: Registered (https://abc123.trycloudflare.com/webhook/telegram)
✅ Tunnel: Alive and accessible (HTTP 200)
⚠️  Config: dm_policy is "pairing" - send /pair first
✅ Database: Connected (postgres@localhost:5433)
✅ LLM Model: z-ai/glm5 (NVIDIA NIM)

Pending Updates: 2
Last Error: None

Recommendations:
1. Send /pair to activate bot for your user
2. Consider switching to polling mode for stability
```

### Auto-Fix

```
$ telegram_fix_common_issues

Auto-Fix Telegram Issues
========================

Issue: LLM model invalid (stepfun-ai/step-3.5-flash)
  ✅ Fixed: Changed to z-ai/glm5 in ~/.ironclaw/.env

Issue: Webhook URL mismatch
  ✅ Fixed: Re-registered with Telegram

Verifying fixes...
✅ LLM Model: Valid
✅ Webhook: Registered

Changes Applied: 2
Issues Resolved: 2
```

## Common Issues Fixed

### Issue: Bot Not Responding

**Symptoms**: Messages sent but no reply

**Diagnosis**:
```bash
telegram_check_webhook
# Output: ❌ Webhook URL: (empty)
```

**Fix**:
```bash
telegram_fix_common_issues
# Registers webhook automatically
```

### Issue: Tunnel Died

**Symptoms**: Webhook registered but no updates

**Diagnosis**:
```bash
telegram_check_tunnel
# Output: ❌ Process: cloudflared not running
```

**Fix**:
```bash
# Restart tunnel manually
cloudflared tunnel --url http://localhost:3000

# Or use auto-fix (suggests command)
telegram_fix_common_issues
```

### Issue: Wrong LLM Model

**Symptoms**: Bot receives but doesn't process

**Diagnosis**:
```bash
telegram_check_config
# Output: ❌ LLM Model: stepfun-ai/step-3.5-flash (INVALID)
```

**Fix**:
```bash
telegram_fix_common_issues
# Changes to z-ai/glm5 automatically
```

### Issue: dm_policy Blocking

**Symptoms**: Bot ignores DMs

**Diagnosis**:
```bash
telegram_check_config
# Output: ⚠️  DM Policy: "pairing" - send /pair first
```

**Fix**:
```bash
# Send /pair to bot on Telegram
# Then approve:
ironclaw pairing approve telegram <code>
```

## Extending the Skill

### Adding New Checks

Edit `SKILL.md` and add new tool section:

```markdown
### telegram_check_new_feature

Checks for new issue type:

**Checks**:
- ✅ Check 1
- ✅ Check 2

**Output Example**:
```
New Feature Check
=================
✅ Status: OK
```
```

### Adding Auto-Fix Logic

Add to `telegram_fix_common_issues` section:

```markdown
- 🔄 If new_issue → run fix command
```

## Testing

### Test Scenarios

1. **Working Setup**
   ```bash
   telegram_full_report
   # All checks should pass
   ```

2. **Dead Tunnel**
   ```bash
   pkill cloudflared
   telegram_check_tunnel
   # Should detect tunnel dead
   ```

3. **Wrong Model**
   ```bash
   echo 'LLM_MODEL="stepfun-ai/step-3.5-flash"' >> ~/.ironclaw/.env
   telegram_check_config
   # Should detect invalid model
   ```

4. **dm_policy Blocking**
   ```bash
   echo 'TELEGRAM_DM_POLICY=pairing' >> ~/.ironclaw/.env
   telegram_check_config
   # Should suggest /pair command
   ```

### Manual Testing

```bash
# 1. Start with clean config
rm -rf ~/.ironclaw/.env

# 2. Run diagnostics
telegram_full_report

# 3. Introduce issues one by one
# 4. Verify detection
# 5. Run auto-fix
# 6. Verify resolution
```

## Troubleshooting

### Skill Not Found

```bash
# Verify installation
ls -la ~/.ironclaw/skills/telegram-debug/

# Reload skills
ironclaw skills reload
```

### Checks Fail Silently

```bash
# Enable debug logging
RUST_LOG=debug ironclaw skills run telegram-debug

# Check logs
tail -f ~/.ironclaw/ironclaw.log | grep telegram
```

### Auto-Fix Doesn't Work

```bash
# Check permissions
ls -la ~/.ironclaw/.env

# Fix if needed
chmod 644 ~/.ironclaw/.env

# Try manual fix
# Then run diagnostics again
```

## Security Notes

- ⚠️  Skill reads `~/.ironclaw/.env` (contains secrets)
- ⚠️  Skill calls Telegram API (uses bot token)
- ⚠️  Skill can modify config (backs up first)
- ✅  Skill never uploads data externally
- ✅  Skill never shares credentials

## Performance

- **Full Report**: ~5 seconds
- **Individual Check**: ~1-2 seconds
- **Auto-Fix**: ~10 seconds
- **API Calls**: 2-4 per run

## Dependencies

- `curl` - HTTP requests
- `jq` - JSON parsing (optional, for pretty output)
- `pgrep` - Process checking
- `dig` or `nslookup` - DNS lookup
- `psql` - Database testing (optional)

## Version History

### 1.0.0 (Initial Release)

- ✅ Webhook status check
- ✅ Tunnel health check
- ✅ Configuration validation
- ✅ Auto-fix common issues
- ✅ Comprehensive documentation

## Contributing

1. Fork repository
2. Create feature branch
3. Add new check/fix
4. Test thoroughly
5. Update documentation
6. Submit PR

## Support

- **Issues**: GitHub Issues
- **Discussion**: IronClaw Discord
- **Docs**: `skills/telegram-debug/SKILL.md`

## Related

- `config-validator` skill - General config validation
- `integrations/errors/tunnel-errors.md` - Tunnel troubleshooting
- `channels-src/telegram/` - Telegram channel implementation

## License

MIT OR Apache-2.0 (same as IronClaw)
