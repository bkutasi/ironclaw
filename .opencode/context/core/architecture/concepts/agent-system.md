# Agent System

**Purpose**: Core agentic runtime — session/thread/turn model, submission parsing, tool loop, approval flows, skill context handling

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Quick Reference

**Session → Thread → Turn Model**:
```
Session (per user)
└── Thread (per conversation — can have many)
    └── Turn (per request/response pair)
```

**Agentic Loop** (all execution paths):
```
LLM call → Tool execution → Result processing → Repeat until text response
```

**Tool Approval**: Tools flagged `requires_approval` pause loop → store `PendingApproval` → send SSE event → user approval resumes

**Commands**: `/undo`, `/redo`, `/compact`, `/status`, `/cancel`, `/thread`, etc. parsed by `SubmissionParser` before agentic loop

---

## Core Concept

Ironclaw's agent system is a **turn-based agentic runtime** that manages multi-user conversations through sessions containing threads, which contain turns. All three execution paths (chat, background jobs, container workers) share the same `run_agentic_loop()` engine, customized via the `LoopDelegate` trait.

The system handles:
- **Session lifecycle**: Create/lookup sessions, map external thread IDs, prune stale sessions
- **Submission parsing**: Convert user input into typed commands before routing
- **Agentic execution**: LLM call → tool execution loop with approval checkpoints
- **Context management**: Compaction when approaching token limits, undo/redo checkpoints
- **Skill injection**: Deterministic skill selection with context blocks in system prompt
- **Parallel jobs**: Scheduler manages full LLM-driven jobs and lightweight subtasks

---

## Key Components

### Session Manager (`session_manager.rs`)

Maps external channel thread IDs to internal UUIDs. Uses double-checked locking for session creation.

```rust
// Fast path: read lock
{
    let sessions = self.sessions.read().await;
    if let Some(session) = sessions.get(user_id) {
        return Arc::clone(session);
    }
}

// Slow path: write lock with re-check
let mut sessions = self.sessions.write().await;
// Double-check after acquiring write lock
if let Some(session) = sessions.get(user_id) {
    return Arc::clone(session);
}
```

**Key behaviors**:
- Prunes idle sessions every 10 minutes (warns at 1000 sessions)
- Maps `(user_id, channel, external_thread_id)` → internal UUID
- Manages per-thread `UndoManager` instances (max 20 checkpoints)

### Submission Parser (`submission.rs`)

Parses user input into typed `Submission` variants **before** the agentic loop runs.

| Input | Variant | Notes |
|-------|---------|-------|
| `/undo`, `/redo` | `Undo`, `Redo` | Turn-based rollback |
| `/compact` | `Compact` | Context window management |
| `/thread <uuid>` | `SwitchThread` | Must be valid UUID |
| `/status [id]` | `JobStatus` | `/list` = all jobs |
| `yes/y/approve` | `ApprovalResponse { approved: true }` | Tool approval |
| JSON `{...}` | `ExecApproval` | From web gateway |
| Everything else | `UserInput` | Starts agentic turn |

**Processing order**: `SubmissionParser` → `Router` (for `/commands`) → `Dispatcher` (natural language)

### Agentic Loop Engine (`agentic_loop.rs`)

Shared engine used by all three execution paths via `LoopDelegate` trait:

```rust
pub async fn run_agentic_loop(
    delegate: &dyn LoopDelegate,
    reasoning: &Reasoning,
    reason_ctx: &mut ReasoningContext,
    config: &AgenticLoopConfig,
) -> Result<LoopOutcome, Error>
```

**Loop flow**:
1. Check signals (stop/cancel) via `delegate.check_signals()`
2. Pre-LLM hook via `delegate.before_llm_call()`
3. LLM call via `delegate.call_llm()`
4. If text → `delegate.handle_text_response()` → Continue or Return
5. If tool calls → `delegate.execute_tool_calls()` → Continue or Return
6. Post-iteration hook via `delegate.after_iteration()`
7. Repeat until `LoopOutcome` or max iterations

**Three delegates**:
- **`ChatDelegate`** (`dispatcher.rs`) — conversational turns, tool approval, skill context
- **`JobDelegate`** (`src/worker/job.rs`) — background scheduler jobs with planning
- **`ContainerDelegate`** (`src/worker/container.rs`) — Docker worker, HTTP event streaming

### Dispatcher (`dispatcher.rs`)

Runs the agentic loop for user-initiated conversational turns. Builds system prompts with skill context injection.

**Skill context injection**:
```xml
<skill name="skill-name" version="1.0.0" trust="TRUSTED">
  Skill prompt content here
</skill>
```

**Trust levels**:
- `TRUSTED` — Skill directives followed
- `INSTALLED` — Suggestions only, don't conflict with core instructions

**Tool attenuation**: Skills can restrict available tools via `attenuate_tools()`.

### Thread Operations (`thread_ops.rs`)

Handles session/thread mutations: `process_user_input`, undo/redo, approval flows, auth-mode interception, DB hydration.

**Auth mode**: When `tool_auth` returns `awaiting_token`, next user message is intercepted **before** turn creation/logging and routed to credential store. TTL: 5 minutes.

**Group chat detection**: If `metadata.chat_type` is `group`/`channel`/`supergroup`, `MEMORY.md` excluded from system prompt to prevent leaking personal context.

### Scheduler (`scheduler.rs`)

Manages parallel job execution with two maps under `Arc<RwLock<HashMap>>`:

- `jobs` — Full LLM-driven jobs with `Worker` and `mpsc` channel
- `subtasks` — Lightweight `ToolExec` or `Background` tasks

**Preferred entry point**: `dispatch_job()` — creates context, persists to DB, then schedules. Don't call `schedule()` directly unless already persisted.

```rust
// Check-insert under single write lock (prevents TOCTOU)
let mut jobs = self.jobs.write().await;
if jobs.contains_key(&job_id) {
    // Handle duplicate
}
jobs.insert(job_id, scheduled_job);
```

### Compaction (`compaction.rs`)

Triggered by `ContextMonitor` when token usage approaches model's context limit.

**Token estimation**: Word-count × 1.3 + 4 overhead per message. Default limit: 100,000 tokens. Threshold: 80%.

**Three strategies**:
1. **MoveToWorkspace** (80–85% usage) — Write full transcript to daily log, keep 10 turns
2. **Summarize** (85–95% usage) — LLM summary to daily log, remove old turns
3. **Truncate** (>95% usage) — Remove oldest turns without summary (fast path)

**Failure handling**: If LLM summarization fails, turns are **not** truncated — error propagates.

### Undo Manager (`undo.rs`)

Per-thread undo/redo with checkpoints storing message lists (not full snapshots).

```rust
pub struct UndoManager {
    undo_stack: VecDeque<Checkpoint>,  // Past states
    redo_stack: Vec<Checkpoint>,       // Future states
    max_checkpoints: usize,            // Default: 20
}
```

**Invariant**: `undo_count() + redo_count()` stays constant across undo/redo (only `checkpoint()` and `clear()` change total).

---

## Data Flow

### User Message Flow

```
IncomingMessage
    ↓
SessionManager.resolve_thread() — Map external ID → internal UUID
    ↓
SubmissionParser.parse() — Convert to typed Submission
    ↓
Agent.handle_message() — Route by variant
    ↓
├─ SystemCommand → commands.rs (bypass thread-state checks)
├─ Control (undo/interrupt) → thread_ops.rs
├─ ApprovalResponse → dispatcher.rs (resume loop)
└─ UserInput → dispatcher.rs (start agentic turn)
```

### Agentic Turn Flow

```
UserInput received
    ↓
Build ReasoningContext (messages + tools + system prompt)
    ↓
run_agentic_loop(ChatDelegate, ...)
    ↓
┌──────────────────────────────────────┐
│ 1. check_signals()                   │
│ 2. before_llm_call()                 │
│ 3. call_llm()                        │
│ 4. Handle text or tool calls         │
│ 5. after_iteration()                 │
│ 6. Repeat until LoopOutcome          │
└──────────────────────────────────────┘
    ↓
├─ Response(String) → Send to channel
├─ NeedApproval(PendingApproval) → Store on thread, send SSE
└─ Stopped/MaxIterations → Handle error
```

### Tool Approval Flow

```
Tool requires approval
    ↓
ChatDelegate.execute_tool_calls() returns LoopOutcome::NeedApproval(pending)
    ↓
Store PendingApproval on thread
    ↓
Send approval_needed SSE event
    ↓
User approves via /api/chat/approval or chat message
    ↓
SubmissionParser returns ApprovalResponse
    ↓
Restore context_messages from PendingApproval
    ↓
Resume agentic loop from checkpoint
```

---

## Code Patterns

### Session/Thread Access Pattern

```rust
// Get session (double-checked locking)
let session = self.session_manager.get_or_create_session(&user_id).await;

// Resolve thread (creates if doesn't exist)
let (session, thread_id) = self.session_manager
    .resolve_thread(&user_id, &channel, external_thread_id)
    .await;

// Mutate thread state (under lock)
{
    let mut sess = session.lock().await;
    let thread = sess.threads.get_mut(&thread_id).unwrap();
    thread.state = ThreadState::Processing;
}
```

### Building Reasoning Context

```rust
let mut reasoning = Reasoning::new(self.llm().clone())
    .with_channel(message.channel.clone())
    .with_model_name(self.llm().active_model_name())
    .with_group_chat(is_group_chat);

// Add system prompt (workspace identity files)
if let Some(prompt) = system_prompt {
    reasoning = reasoning.with_system_prompt(prompt);
}

// Add skill context
if let Some(ctx) = skill_context {
    reasoning = reasoning.with_skill_context(ctx);
}

// Build ReasoningContext
let mut reason_ctx = ReasoningContext::new()
    .with_messages(initial_messages)
    .with_tools(tool_definitions)
    .with_system_prompt(cached_prompt);
```

### Running Agentic Loop

```rust
let delegate = ChatDelegate {
    agent: self,
    session: session.clone(),
    thread_id,
    message,
    job_ctx,
    active_skills,
    cached_prompt,
    cached_prompt_no_tools,
    nudge_at,
    force_text_at,
    user_tz,
};

let loop_config = AgenticLoopConfig {
    max_iterations: max_tool_iterations + 1,
    enable_tool_intent_nudge: true,
    max_tool_intent_nudges: 2,
};

let outcome = run_agentic_loop(
    &delegate,
    &reasoning,
    &mut reason_ctx,
    &loop_config,
).await?;
```

### Tool Intent Nudge

Detect when LLM expresses tool intent without calling tool:

```rust
if config.enable_tool_intent_nudge
    && !reason_ctx.available_tools.is_empty()
    && !reason_ctx.force_text
    && consecutive_tool_intent_nudges < config.max_tool_intent_nudges
    && crate::llm::llm_signals_tool_intent(&text)
{
    consecutive_tool_intent_nudges += 1;
    tracing::info!("LLM expressed tool intent without calling a tool, nudging");
    
    reason_ctx.messages.push(ChatMessage::assistant(&text));
    reason_ctx.messages.push(ChatMessage::user(crate::llm::TOOL_INTENT_NUDGE));
}
```

### Cost Guard Pattern

```rust
// BEFORE LLM call
if let Err(e) = self.deps.cost_guard.check_allowed() {
    return Err(Error::CostLimitExceeded(e));
}

// Call LLM...
let output = llm.respond(request).await?;

// AFTER LLM call
self.deps.cost_guard.record_llm_call(
    &output.usage,
    &self.config.cost_tracking_window,
);
```

---

## Key Invariants

- **No unwrap/expect**: Use `?` with proper error mapping (except tests/infallible invariants)
- **Session lock**: All `Session`/`Thread` mutations under `Arc<Mutex<Session>>` lock
- **Single-threaded per thread**: Agent loop is single-threaded; parallel at job/scheduler level
- **Deterministic skill selection**: No LLM call — `skills/selector.rs` uses keyword matching
- **Safety layer**: Tool results pass through sanitizer → validator → policy → leak detector
- **Double-checked locking**: `SessionManager` uses read-then-write pattern for session creation
- **Scheduler lock**: `Scheduler.schedule()` holds write lock for entire check-insert
- **Cost guard separation**: `check_allowed()` before LLM, `record_llm_call()` after (separate calls)
- **Hooks fail-open**: `BeforeInbound`/`BeforeOutbound` errors logged but processing continues

---

## Related Files

**Core Implementation**:
- `src/agent/agent_loop.rs` — Main `Agent` struct, `AgentDeps`, event loop
- `src/agent/agentic_loop.rs` — Shared loop engine, `LoopDelegate` trait
- `src/agent/dispatcher.rs` — Chat delegate, skill injection, tool approval
- `src/agent/session.rs` — Session/Thread/Turn data models
- `src/agent/session_manager.rs` — Lifecycle, thread mapping, undo managers
- `src/agent/submission.rs` — Submission parser, command routing
- `src/agent/thread_ops.rs` — Thread operations, auth mode, DB hydration
- `src/agent/scheduler.rs` — Parallel job scheduling
- `src/agent/compaction.rs` — Context window management
- `src/agent/undo.rs` — Checkpoint-based undo/redo

**Supporting Components**:
- `src/agent/commands.rs` — System command handlers
- `src/agent/context_monitor.rs` — Memory pressure detection
- `src/agent/self_repair.rs` — Stuck job recovery
- `src/agent/cost_guard.rs` — Budget and rate limiting
- `src/agent/heartbeat.rs` — Proactive periodic execution
- `src/agent/routine_engine.rs` — Cron/event-triggered routines
- `src/agent/job_monitor.rs` — SSE injection for container output

**Documentation**:
- `src/agent/CLAUDE.md` — Module map, invariants, command reference
- `.opencode/context/core/architecture/concepts/config-precedence.md` — Configuration layers
- `src/workspace/README.md` — Workspace system (persistent memory)

---

## Common Pitfalls

### ❌ Calling schedule() directly without persisting

```rust
// WRONG: Don't call schedule() directly
scheduler.schedule(job_id, ...).await?;

// RIGHT: Use dispatch_job() which persists first
let job_id = scheduler.dispatch_job(&user_id, "Title", "Description", None).await?;
```

### ❌ Holding multiple locks during scheduler operations

```rust
// WRONG: Holding session lock while calling scheduler
{
    let sess = session.lock().await;
    scheduler.schedule(...).await?; // Deadlock risk
}

// RIGHT: Release session lock first
drop(sess);
scheduler.schedule(...).await?;
```

### ❌ Forgetting cost guard separation

```rust
// WRONG: Assuming auto-recording
cost_guard.check_allowed()?;
let output = llm.respond(request).await?;
// Missing: cost_guard.record_llm_call(...)

// RIGHT: Separate calls
cost_guard.check_allowed()?;
let output = llm.respond(request).await?;
cost_guard.record_llm_call(&output.usage, &config.cost_tracking_window);
```

### ❌ Truncating on summarization failure

```rust
// WRONG: Truncate even if LLM fails
match generate_summary(messages).await {
    Ok(summary) => truncate_turns(keep),
    Err(_) => truncate_turns(keep), // Data loss!
}

// RIGHT: Preserve turns on failure
match generate_summary(messages).await {
    Ok(summary) => truncate_turns(keep),
    Err(e) => {
        tracing::warn!("Summary failed, preserving turns: {}", e);
        // Turns NOT truncated
    }
}
```

---

## Testing Patterns

### Unit Test: Submission Parser

```rust
#[test]
fn test_parse_undo_command() {
    assert!(matches!(
        SubmissionParser::parse("/undo"),
        Submission::Undo
    ));
}

#[test]
fn test_parse_approval_yes() {
    assert!(matches!(
        SubmissionParser::parse("yes"),
        Submission::ApprovalResponse { approved: true, always: false }
    ));
}
```

### Integration Test: Session Lifecycle

```rust
#[tokio::test]
async fn test_session_manager_double_check_locking() {
    let manager = SessionManager::new();
    
    // Concurrent session creation should return same session
    let (s1, s2) = tokio::join!(
        manager.get_or_create_session("user1"),
        manager.get_or_create_session("user1")
    );
    
    assert_eq!(s1.lock().await.id, s2.lock().await.id);
}
```
