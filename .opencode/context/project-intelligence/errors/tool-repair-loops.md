<!-- Context: project-intelligence/errors | Priority: high | Version: 1.0 | Updated: 2026-03-26 -->

# Error: Infinite Tool Repair Loop

**Symptom**: Recurring log messages every 60 seconds:
```
INFO Detected X broken tools needing repair
INFO Attempting to repair broken tool: {tool-name}
INFO Tool repair result: ManualRequired { message: "Builder not available..." }
```

## Root Cause

The self-repair system tracks tool failures in `tool_failures` table. After 5+ failures, tools are flagged as "broken" and repair is attempted every 60 seconds.

**Loop occurs when**:
1. Tool accumulates failures (usage errors or implementation bugs)
2. Builder not initialized (sandbox enabled + `allow_local_tools=false`)
3. Repair returns `ManualRequired` immediately
4. Failures continue accumulating → loop repeats

## Common Triggers

| Trigger | Example | Solution |
|---------|---------|----------|
| Usage errors | Invalid parameters, wrong API format | Add to `PROTECTED_TOOL_NAMES` |
| Missing config | API key not set, wrong endpoint | Fix configuration |
| Builder disabled | Sandbox mode without local tools | Enable builder or protect tool |
| Network issues | Rate limiting, connectivity | Add retry logic, protect tool |

## Diagnosis

**1. Check failure details**:
```sql
SELECT tool_name, error_message, error_count, last_failure 
FROM tool_failures 
WHERE repaired_at IS NULL 
ORDER BY last_failure DESC;
```

**2. Identify error type**:
- **Usage error**: Invalid parameters, validation failures → Protect tool
- **Implementation bug**: Crashes, panics, logic errors → Fix code
- **Configuration**: Missing keys, wrong URLs → Update config

## Solutions

### Option 1: Protect Tool (Usage Errors)

Add to `PROTECTED_TOOL_NAMES` in `src/tools/registry.rs`:

```rust
const PROTECTED_TOOL_NAMES: &[&str] = &[
    // ... existing tools ...
    "web-search",  // ← Add tool name here
];
```

**When**: Errors are from invalid usage, not implementation bugs.

### Option 2: Enable Builder (Want Auto-Repair)

```toml
# config.toml
[builder]
enabled = true

[agent]
allow_local_tools = true
```

**When**: You want WASM tools to auto-repair on failures.

### Option 3: Clear Failures (Temporary)

```sql
UPDATE tool_failures 
SET repaired_at = NOW(), error_count = 0 
WHERE tool_name = '{tool-name}';
```

**Warning**: Temporary fix - failures will reaccumulate if root cause persists.

## Prevention

1. **Validate parameters early** - Return clear errors before tool execution
2. **Protect usage-error tools** - Add to `PROTECTED_TOOL_NAMES` if failures are expected
3. **Initialize builder** - Enable if you want auto-repair for WASM tools
4. **Monitor failure counts** - Check `tool_failures` table periodically

## Reference

- **Table schema**: `migrations/V3__tool_failures.sql`
- **Protection list**: `src/tools/registry.rs` (line ~38-88)
- **Self-repair logic**: `src/agent/self_repair.rs` (line ~193-218)
- **Builder init**: `src/app.rs` (line ~375-385)

## Related

- [routine-job-errors.md](routine-job-errors.md) — Broken tools from builtin tool failures
- [decisions-log.md](../decisions-log.md) — ADR-002 (Builtin tool exclusion from failure tracking)
