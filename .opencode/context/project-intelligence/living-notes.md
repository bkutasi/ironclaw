<!-- Context: project-intelligence/notes | Priority: high | Version: 2.0 | Updated: 2026-03-25 -->

# Living Notes

> Active issues, technical debt, open questions, and insights. Keep this alive.

## Quick Reference
- **Purpose**: Capture current state, problems, open questions
- **Update**: Weekly or when status changes
- **Archive**: Move resolved items to bottom with status

## What's Working

### Sandbox Job System ✅
Docker-based sandbox job system fully operational:
- **done tool** (`src/tools/builtin/done.rs`): LLM calls `done` → loop breaks → job completed. **Updated 2026-03-25**: ALWAYS available (builtin tool, bypasses `allow_local_tools`). No more "Tool done not found" errors.
- **DB status fix**: `Completed -> Completed` idempotent (`src/context/state.rs:383`). Prevents race conditions.
- **Container delegate** (`src/worker/container.rs`): Sequential tool exec, HTTP streaming, credential injection via `Command::envs()`.

### llama.cpp Local LLM Integration ✅
- **ensure_last_message_role** (`src/llm/provider.rs:479`): Strips trailing assistant prefill for thinking models (Qwen3, DeepSeek-R1, GLM-z1) — llama.cpp rejects prefill when `enable_thinking=true`.
- **reasoning_models module** (`src/llm/reasoning_models.rs`): Pattern-based detection. Covers Qwen3, QwQ, DeepSeek-R1, GLM-z1/4-plus/5, Nanbeige, Step-3.5, MiniMax-M2.
- **Thinking tag stripping**: Removes `<thinking>`, `<reflection>`, `<scratchpad>`, `<|think|>`, `<final>` from responses.

### Shared Agentic Loop Engine ✅
All paths use `run_agentic_loop()` (`src/agent/agentic_loop.rs`):
- **ChatDelegate** (`src/agent/dispatcher.rs`): Conversational turns, tool approval, skill context, cost tracking
- **JobDelegate** (`src/worker/job.rs`): Background scheduler jobs, planning support
- **ContainerDelegate** (`src/worker/container.rs`): Docker container worker, sequential tool exec

### force_text 3-Phase Defense ✅
`ChatDelegate` and `ContainerDelegate` implement anti-loop defense:
1. **Nudge** (`max_iterations - 1`): System message warns LLM to produce final answer
2. **Force text** (`max_iterations`): Swap to no-tools prompt, set `force_text = true`
3. **Hard ceiling** (`max_iterations + 1`): Loop terminates

### JobDelegate force_text + done Tool Parity ✅ (2026-03-25)
`JobDelegate` (`src/worker/job.rs`) has full parity with Container/Chat delegates:
- **3-phase defense**: Nudge at `max-1`, force_text at `max`, hard ceiling at `max+1`
- **`done` tool detection**: Calling `done` breaks loop immediately. **Guarantee**: always available (builtin).
- Fixed: Jobs like `self-evaluation-evolving` no longer hit iteration limits after completing work.

## Technical Debt

| Item | Impact | Priority | Mitigation |
|------|--------|----------|------------|
| Text-based completion detection fallback | `llm_signals_completion()` can false-positive on "tests passed" etc. | Medium | `done` tool is primary; text detection fallback only |
| No streaming support | LLM calls block until full response; UX delay | Low | Not critical for jobs; chat UX acceptable |
| NEAR AI session renewal interactive | Blocks terminal for OAuth; unsuitable for headless | Low | Use `NEARAI_SESSION_TOKEN` env var |

### Technical Debt Details

**Text-based completion detection fallback**
*Priority*: Medium | *Effort*: Small
*Impact*: `llm_signals_completion()` (`src/util.rs`) uses phrase matching. Can false-positive.
*Root Cause*: `done` tool newer; kept for backward compat.
*Solution*: Remove once all job paths use `done` consistently.
*Status*: Acknowledged — `done` is primary in Container/Job delegates.

**Tool intent nudge can loop**
*Priority*: Low | *Effort*: N/A
*Impact*: LLM says "let me search..." without calling tools; nudge fires up to 2 times.
*Root Cause*: Models express intent verbally instead of tool_calls.
*Status*: Managed — capped at 2 nudges.

## Open Questions

| Question | Stakeholders | Status | Next Action |
|----------|--------------|--------|-------------|
| Should `done` be required in job system prompts? | Core team | Open | Test if LLMs call it without explicit instruction |
| When to remove `llm_signals_completion` fallback? | Core team | Open | Track false-positives; remove at 100% `done` adoption |

## Known Issues

| Issue | Severity | Workaround | Status |
|-------|----------|------------|--------|
| LLM produces empty responses when entire output is `<suggestions>` block | Medium | Adjust prompt to require content before suggestions | Investigating |

## Insights & Lessons Learned

### What Works Well
- **Shared `LoopDelegate` trait** — Chat, job, container paths share one loop engine. Features benefit all paths.
- **Tool-based completion detection** — `done` tool more reliable than text regex. LLMs understand "call when done".
- **Idempotent state transitions** — `Completed -> Completed` no-op prevents race conditions.

### What Could Be Better
- **force_text threshold tuning** — Currently fixed at `max_iterations`. Could be adaptive based on job/model.
- **Container credential injection** — Uses `Command::envs()` per invocation. Could streamline if credentials stable.

### Lessons Learned
- **llama.cpp thinking model quirks** — Assistant prefill incompatible with `enable_thinking=true`. Test local LLMs with thinking models.
- **Idempotency in distributed systems** — `Completed -> Completed` race caught because loop and wrapper both marked completion. Design idempotency from start.

## Patterns & Conventions

### Code Patterns Worth Preserving
- **`LoopDelegate` trait** (`src/agent/agentic_loop.rs`) — Strategy pattern for shared loop with consumer hooks.
- **3-phase force_text defense** (`src/worker/container.rs:167-414`) — Nudge → Force text → Hard ceiling.
- **`ensure_last_message_role`** (`src/llm/provider.rs:479`) — Model-aware message normalization.

### Gotchas for Maintainers
- **Don't call `schedule()` directly** — Use `dispatch_job()` (persists to DB first; FK deps require valid job ID).
- **`CostGuard.check_allowed()` before LLM, `record_llm_call()` after** — Guard doesn't auto-record.
- **`force_text` strips tools AND sets `reason_ctx.force_text`** — Both needed for text-only mode.
- **AI agents must NOT access production database directly** — Database queries require explicit user permission. If investigation needs DB access, provide SQL queries for user to run and share results. This prevents accidental data modification or exposure.
- **`done` detection in `execute_tool_calls`, not `handle_text_response`** — Checks `tc.name == "done"`. **Classification**: orchestrator-domain tool (like `echo`, `time`, `json`, `http`) — always available.
- **WASM channel `on_respond` handles null metadata** — Internal messages have `metadata: Value::Null`. Use `match` not `map_err` to skip gracefully. All 5 channels had this bug.
- **Routines scoped by `user_id`** — `routines` table has `UNIQUE (user_id, name)`. CLI scoped, UI unscoped. Changing `TELEGRAM_OWNER_ID` doesn't migrate routines: `UPDATE routines SET user_id = 'new_id' WHERE user_id = 'old_id'`.
- **`<suggestions>` extraction can suppress responses** — If LLM output is ONLY `<suggestions>` block, extracted text is empty → suppressed. Fix: ensure content precedes suggestions tag.
- **Broken tools tracking: job worker path only** — Failures recorded in `tool_failures` table (UPSERT on `tool_name`). 5+ failures triggers self-repair. **UPDATE 2026-03-25**: Builtin tools excluded via `ToolRegistry::is_builtin()`. Cleanup: `DELETE FROM tool_failures WHERE tool_name IN ('tool1', 'tool2');`.
- **Telegram `chat_id` dynamic, not config** — From incoming message metadata, not env vars. Proactive messages use `last_broadcast_metadata`. `TELEGRAM_OWNER_ID` sets auth, not routing. No prior message = "No stored owner routing target".
- **Telegram proactive messages require conversation context** — Discovered 2026-03-25. Proactive sends fail if user never sent `/start` (no `last_broadcast_metadata`). **Fix**: Send `/start` first OR add `TELEGRAM_OWNER_CHAT_ID` env var for direct routing. See `.opencode/context/core/errors/telegram-proactive-message-fix.md`.

## Active Projects

| Project | Goal | Owner | Timeline |
|---------|------|-------|----------|
| Sandbox job system | Reliable container-based job execution | Core | Complete ✅ |
| Local LLM support | llama.cpp integration with thinking models | Core | Complete ✅ |

## Archive (Resolved Items)

### Resolved: Tool repair infinite loop (2026-03-25)
- **Problem**: 6 builtin tools falsely flagged as "broken" every 60 seconds, causing repair loop spam
- **Root Cause**: Tools missing from `PROTECTED_TOOL_NAMES` → `is_builtin()` returned false → failures recorded → self-repair couldn't fix (builder only for WASM)
- **Resolution**: Added 6 tools to `PROTECTED_TOOL_NAMES`: job_events, job_prompt, tool_upgrade, extension_info, secret_list, secret_delete
- **Impact**: Eliminates repair loop spam, prevents WASM shadowing attacks on critical tools
- **Files**: `src/tools/registry.rs` (lines 56-57, 65-66, 78-79)
- **Discovery**: CodeReviewer agent found 5 additional missing tools beyond original job_events

### Resolved: JobDelegate missing force_text defense (2026-03-25)
- **Resolution**: Added 3-phase defense + `done` tool detection to `JobDelegate` (parity with Container/Chat).
- **Impact**: `self-evaluation-evolving` and similar routines complete successfully.
- **Files**: `src/worker/job.rs`

### Resolved: Builtin tools falsely flagged as broken (2026-03-25)
- **Resolution**: `ToolRegistry::is_builtin()` check gates `record_tool_failure` — only WASM tools tracked.
- **Impact**: Eliminates log spam about broken builtin tools.
- **Files**: `src/worker/job.rs`, `src/tools/registry.rs`

### Resolved: done tool always available (2026-03-25)
- **Resolution**: Moved `done` from `register_dev_tools()` to `register_builtin_tools()`.
- **Impact**: ALWAYS available in ALL contexts. No more "Tool done not found" errors.
- **Files**: `src/tools/registry.rs`

### Resolved: Empty timezone blocks routine_update (2026-03-25)
- **Resolution**: Added `.filter(|s| !s.is_empty())` before validation.
- **Impact**: Updates changing only prompt/description no longer fail.
- **Files**: `src/tools/builtin/routine.rs`

### Resolved: routine_create duplicate name error (2026-03-25)
- **Resolution**: Pre-check with `get_routine_by_name`, returns `InvalidParameters`.
- **Impact**: Clear error: "Routine 'X' already exists. Use routine_update."
- **Files**: `src/tools/builtin/routine.rs`

### Resolved: Text-only completion detection (2026-03)
- **Resolution**: `done` tool as primary mechanism; `llm_signals_completion()` fallback.
- **Learning**: Tool signals more reliable than text parsing.

### Resolved: llama.cpp assistant prefill error (2026-02)
- **Resolution**: `ensure_last_message_role` strips prefill for thinking models.
- **Learning**: llama.cpp has undocumented model-specific constraints.

### Resolved: Completed -> Completed race (2026-02)
- **Resolution**: Idempotent transition (no-op).
- **Learning**: Design state transitions idempotent in distributed systems.

## Onboarding Checklist

- [ ] `done` tool replaces text-based detection. **Note**: ALWAYS available (builtin, 2026-03-25).
- [ ] 3-phase force_text defense (nudge → force text → hard ceiling)
- [ ] `ensure_last_message_role` and thinking model quirks
- [ ] Shared `LoopDelegate` trait and three implementations
- [ ] `llm_signals_completion()` is fallback (not primary)
- [ ] Idempotent state transitions for job completion
- [ ] WASM channel `on_respond` handles null metadata
- [ ] Routine user_id scoping — CLI scoped, UI unscoped
- [ ] `<suggestions>` extraction can suppress responses
- [ ] `tool_failures` tracks job worker path only
- [ ] Telegram `chat_id` from message metadata, not config
- [ ] `JobDelegate` has force_text + `done` detection (parity with Container/Chat)
- [ ] Builtin tools excluded from self-repair tracking

## Telegram Debug Session Lessons (2026-03-18)

**Key Learnings**:
- Cloudflare Bot Fight Mode blocks Telegram webhooks (70% 403 errors)
- `workspace_kv` stores `tunnel_url` persisting across restarts, overriding env vars
- Trailing newline in tunnel URL → "Failed to Resolve Host"
- Ephemeral vs persistent tunnel behavior differs (Cloudflare caching)
- `HTTP_WEBHOOK_SECRET` auto-enables HTTP channel

**Operational Gotchas**:
- Check `workspace_kv` when switching polling/webhook modes
- Cloudflare Security → Events log best for 403 diagnostics
- Telegram IP ranges: 149.154.160.0/20 (AS62041) — whitelist if using Bot Fight Mode

## Related Files

- `decisions-log.md` - Past decisions that inform current state
- `business-domain.md` - Business context for current priorities
- `technical-domain.md` - Technical context for current state
- `business-tech-bridge.md` - Context for current trade-offs
