<!-- Context: project-intelligence/concepts | Priority: high | Version: 1.0 | Updated: 2026-03-26 -->

# Concept: Tool Protection (PROTECTED_TOOL_NAMES)

**Core Idea**: Certain tools are excluded from self-repair tracking to prevent false "broken tool" alerts for usage errors and builtin tools that cannot be rebuilt.

## How It Works

The `PROTECTED_TOOL_NAMES` constant in `src/tools/registry.rs` defines tools that:
1. Are NOT tracked in `tool_failures` table
2. CANNOT be rebuilt by self-repair system
3. Are considered "builtin" or "protected" from repair attempts

## Location

```rust
// src/tools/registry.rs (line ~38-88)
const PROTECTED_TOOL_NAMES: &[&str] = &[
    "echo",
    "time",
    "json",
    "http",
    "shell",
    "read_file",
    "write_file",
    // ... 50+ more tools ...
];
```

## When to Add a Tool

**✅ Add to protected list when**:
- Tool is builtin (defined in code, not WASM)
- Failures are from usage errors (invalid parameters)
- Tool cannot be rebuilt by WASM builder
- Errors are expected/normal (validation failures)

**❌ Don't add when**:
- Tool has actual implementation bugs
- You want auto-repair for WASM tools
- Failures indicate real problems to fix

## Protected Tool Categories

| Category | Examples | Reason |
|----------|----------|--------|
| Builtin tools | `echo`, `time`, `shell` | Defined in code, not WASM |
| Memory tools | `memory_read`, `memory_write` | Core functionality |
| Job tools | `job_status`, `create_job` | System operations |
| Extension tools | `tool_install`, `tool_auth` | Extension manager |
| Usage-error tools | `web-search` | Validation failures expected |

## Effect on Self-Repair

**Protected tools**:
- `ToolRegistry::is_builtin(name)` returns `true`
- Failures NOT recorded in `tool_failures` table
- Never flagged as "broken"
- No repair attempts made

**Unprotected tools** (WASM):
- Failures tracked after each error
- Flagged as broken after 5+ failures
- Repair attempted every 60 seconds
- Can be rebuilt by WASM builder

## Adding a Tool

**Step 1**: Edit `src/tools/registry.rs`
```rust
const PROTECTED_TOOL_NAMES: &[&str] = &[
    // ... existing ...
    "web-search",  // ← Add alphabetically
];
```

**Step 2**: Rebuild
```bash
cargo build --release
```

**Step 3**: (Optional) Clear existing failures
```sql
UPDATE tool_failures 
SET repaired_at = NOW(), error_count = 0 
WHERE tool_name = 'web-search';
```

## 📂 Codebase References

**Implementation**:
- `src/tools/registry.rs` — PROTECTED_TOOL_NAMES constant (line ~38-88)
- `src/tools/registry.rs` — `is_builtin()` method (line ~198-200)

**Self-Repair Integration**:
- `src/worker/job.rs` — Failure recording skip logic (line ~805-818)
- `src/agent/self_repair.rs` — Broken tool detection (line ~193-218)

**Failure Tracking**:
- `src/history/store.rs` — `get_broken_tools()` method (line ~1946-1974)
- `src/history/store.rs` — `record_tool_failure()` method (line ~1922-1943)

## Related

- errors/tool-repair-loops.md — What happens without protection
- errors/web-search-errors.md — Example of protected tool
- decisions-log.md — ADR-002 (Builtin tool exclusion from failure tracking)
