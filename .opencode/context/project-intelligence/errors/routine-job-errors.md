<!-- Context: project-intelligence/errors | Priority: high | Version: 1.0 | Updated: 2026-03-25 -->
# Errors: Routine & Job System

**Purpose**: Common errors in routine/job execution and their fixes
**Last Updated**: 2026-03-25

## Error: Job "Maximum iterations exceeded" despite completing work

**Symptom**:
```
Maximum iterations exceeded: job hit the iteration cap
```

**Cause**: `JobDelegate` lacked the 3-phase force_text defense. LLM kept calling tools (e.g., `memory_write`) without producing a final text response, so `llm_signals_completion()` was never checked.

**Solution**:
1. Call the `done` tool to signal completion
2. The 3-phase defense now nudges at `max-1`, strips tools at `max`, and has a hard ceiling at `max+1`

**Prevention**: Always call `done` with a summary when your job is complete. Don't wait for the iteration limit.
**Frequency**: common

**Code References**:
- Fix: `src/worker/job.rs` — `JobDelegate::before_llm_call`, `execute_tool_calls`
- Pattern: `src/worker/container.rs` — `ContainerDelegate` reference implementation

---

## Error: "Detected 4 broken tools needing repair" every 60 seconds

**Symptom**:
```
Detected 4 broken tools needing repair... Builder not available for repairing tool 'memory_read'...
```

**Cause**: Builtin tools (`memory_read`, `message`, `job_status`, `routine_update`) accumulated failures from normal error conditions (invalid params, missing paths). After 5 failures, self-repair flagged them as broken but couldn't repair them (not WASM tools).

**Solution**: Builtin tools are now excluded from failure tracking via `ToolRegistry::is_builtin()`.

**Prevention**: Only WASM tools should be tracked for self-repair. Builtin tools fail naturally.
**Frequency**: common

**Code References**:
- Fix: `src/worker/job.rs` — `process_tool_result_job` with `is_builtin` guard
- Fix: `src/tools/registry.rs` — `is_builtin()` method

---

## Error: "Builder not available for repairing tool" - Complete Analysis

**Symptom**:
```
INFO Detected 6 broken tools needing repair
INFO Tool repair result: ManualRequired { message: "Builder not available for repairing tool 'job_events'" }
```

**Real Failure Data** (from production, 2026-03-25):
| Tool | Count | Error Type | Root Cause |
|------|-------|------------|------------|
| job_events | 7 | "no job found" | Invalid job prefix (user input) |
| job_status | 8 | "no job found" | Invalid job prefix (user input) |
| memory_read | 33 | "Document not found" | Missing memory document |
| memory_write | 13 | "rate limited" | Rate limiting working correctly |
| message | 13 | "Invalid chat_id" | WASM channel bug (fixed) |
| routine_update | 8 | "invalid timezone" | Empty string validation (fixed) |

**Key Insight**: ALL failures were expected (validation errors, rate limiting, old bugs). No crashes, no infrastructure issues.

**Complete Fix**:
1. Added 6 tools to `PROTECTED_TOOL_NAMES` in `src/tools/registry.rs`
2. Tools: job_events, job_prompt, tool_upgrade, extension_info, secret_list, secret_delete
3. Prevents failure tracking for builtin tools via `ToolRegistry::is_builtin()` check

**Cleanup SQL**:
```sql
DELETE FROM tool_failures WHERE tool_name IN (
    'job_events', 'job_prompt', 'tool_upgrade',
    'extension_info', 'secret_list', 'secret_delete'
);
```

**Prevention**: All tools registered via `register_sync()` MUST be in `PROTECTED_TOOL_NAMES`. CodeReviewer should verify this pattern.

---

## Error: "invalid IANA timezone: ''" in routine_update

**Symptom**:
```
ToolError::InvalidParameters("invalid IANA timezone: ''")
```

**Cause**: Empty timezone string `""` passed validation. `.and_then(|v| v.as_str())` returned `Some("")`, then `parse_timezone("")` failed.

**Solution**: Empty strings are now filtered with `.filter(|s| !s.is_empty())` before validation — treated as "not provided".

**Prevention**: Don't include timezone field in `routine_update` params unless you want to change it.
**Frequency**: occasional

**Code References**:
- Fix: `src/tools/builtin/routine.rs` — `RoutineUpdateTool::execute` timezone parsing

---

## Error: "PostgreSQL error: db error" on routine_create

**Symptom**:
```
PostgreSQL error: db error (unique constraint violation)
```

**Cause**: No pre-check for existing routine names before INSERT. The DB unique constraint `(user_id, name)` rejects duplicates with a raw error.

**Solution**: Pre-check via `get_routine_by_name` before INSERT. Now returns: "Routine 'X' already exists. Use routine_update to modify it."

**Prevention**: Check if routine exists before creating. Use `routine_update` to modify existing routines.
**Frequency**: occasional

**Code References**:
- Fix: `src/tools/builtin/routine.rs` — `RoutineCreateTool::execute` duplicate check

---

## 📂 Codebase References

**Error Handling**:
- `src/worker/job.rs` — Job iteration defense and tool result processing
- `src/tools/registry.rs` — Builtin tool identification

**Prevention Logic**:
- `src/tools/builtin/routine.rs` — Routine tool validation and duplicate checks

## Resolved Errors

### "Tool done not found" — RESOLVED

**Status**: ✅ Resolved (2026-03-25)  
**Resolution**: [ADR-003](../decisions-log.md#adr-003-always-register-done-tool) — `done` tool moved to builtin tools (always available)  
**Symptom**: Agents received "Tool done not found" errors when running with `allow_local_tools=false`  
**Fix**: `done` is now registered unconditionally in `register_builtin_tools()`, bypassing the `allow_local_tools` restriction

---

## Related
- decisions-log.md — ADR-001 (Job iteration defense), ADR-002 (Builtin tool exclusion), **ADR-003** (done tool always available)
- living-notes.md — Routine/job system notes
- [Navigation](../navigation.md#agent-tools) — done tool cross-references
