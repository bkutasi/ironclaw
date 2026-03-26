<!-- Context: project-intelligence/decisions | Priority: high | Version: 2.0 | Updated: 2026-03-25 -->

# Decisions Log

> Record major architectural and business decisions with full context.

## Quick Reference

- **Purpose**: Document decisions so future team members understand context
- **Format**: Each decision as a separate entry
- **Status**: Decided | Pending | Under Review | Deprecated

---

## ADR-001: Tool-Based Completion Detection (done)

**Date**: 2026-02
**Status**: Decided
**Owner**: Core team

### Context

The sandbox job system needed reliable completion detection. The original text-based pattern matching (`llm_signals_completion()` in `src/util.rs`) had two problems:

1. **False positives**: Phrases like "tests passed" appear naturally in development workflows
2. **False negatives**: LLM might complete work using phrasing not in the pattern list

### Decision

Add a `done` built-in tool (`src/tools/builtin/done.rs`) that the LLM must explicitly call to signal job completion. Forces structured summary and eliminates ambiguity.

> **Note**: Originally called `complete_job`, renamed to `done` for clarity. Orchestrator-side `ContainerJobManager::complete_job()` unchanged.

### Rationale

Tool-based completion is deterministic: the LLM must opt-in. No ambiguity about whether a phrase means "I'm done" or "a subtask finished."

### Alternatives Considered

| Alternative | Why Rejected? |
|-------------|---------------|
| Text pattern matching only | False positives/negatives, unreliable for production |
| Structured output (JSON response) | Requires provider support, not all support it |
| Separate completion endpoint | LLM should decide when done |
| Tool + text fallback | **Chosen** — tool is primary, text is fallback |

### Implementation

- **Primary** (tool-based): `ContainerDelegate::execute_tool_calls()` (`src/worker/container.rs:540-547`) and `JobDelegate::execute_tool_calls()` (`src/worker/job.rs:1415`) detect `done` tool and break loop
- **Fallback** (text-based): `handle_text_response()` in both delegates calls `llm_signals_completion()` for backward compat

### Impact

- **Positive**: Deterministic completion; structured summaries; no false positives
- **Negative**: Small token cost for tool definition; LLM must learn to call it
- **Risk**: None — fallback text detection prevents silent failure

### Related

- `src/tools/builtin/done.rs` — Tool implementation
- `src/util.rs:24` — `llm_signals_completion()` (fallback)
- `src/worker/container.rs:540`, `src/worker/job.rs:1415` — Delegate detection

---

## ADR-002: force_text 3-Phase Defense Against Infinite Tool Loops

**Date**: 2026-02
**Status**: Decided
**Owner**: Core team

### Context

LLMs can enter infinite tool-call loops, repeatedly calling tools without producing final text responses. Especially problematic with local models (llama.cpp) and complex jobs.

### Decision

Implement 3-phase defense in `ChatDelegate`, `ContainerDelegate`, and `JobDelegate`:

1. **Phase 1 — Nudge** (at `max_iterations - 1`): System message warning LLM to provide final answer
2. **Phase 2 — Force text** (at `max_iterations`): Swap to prompt variant without tool definitions, set `force_text = true`
3. **Phase 3 — Hard ceiling** (at `max_iterations + 1`): Loop terminates with `MaxIterations`

### Rationale

Graduated pressure is more effective than hard cutoff: nudge allows self-correction, force_text removes option entirely, hard ceiling is safety net. Pre-building two prompt variants avoids expensive reconstruction.

### Alternatives Considered

| Alternative | Why Rejected? |
|-------------|---------------|
| Hard cutoff only | Abrupt; LLM may not have useful answer ready |
| Decreasing tool set | Over-engineered, unclear which tools to remove |
| Temperature reduction | Doesn't prevent tool calls, just less likely |
| 3-phase (chosen) | Best balance of effectiveness and simplicity |

### Implementation

Configuration and execution in `src/worker/container.rs:167-421`, `src/agent/dispatcher.rs:165-311`:

- Pre-build two system prompt variants (with/without tools)
- `before_llm_call()`: Inject nudge message, swap prompts at thresholds
- `AgenticLoopConfig`: Set `max_iterations: max_tool_iterations + 1` for hard ceiling
- `JobDelegate` (`src/worker/job.rs`): Mirror implementation for parity (2026-03-25)

See source files for exact code — pattern is consistent across all three delegates.

### Impact

- **Positive**: Prevents infinite loops; graduated pressure; full parity across delegates (2026-03-25)
- **Negative**: Two prompt variants use slightly more memory; nudge adds tokens
- **Risk**: None — hard ceiling catches any provider bugs with force_text

### Related

- `src/agent/agentic_loop.rs` — Shared loop engine
- `src/worker/container.rs:167-421`, `src/agent/dispatcher.rs:165-311` — Implementations
- `src/llm/reasoning.rs` — `respond_with_tools()` respects `force_text` flag

---

## ADR-003: Always Register done Tool

**Date**: 2026-03-25
**Status**: Accepted
**Owner**: Core team

### Context

The `done` tool was registered conditionally in `register_dev_tools()` only when `allow_local_tools=true`. This caused "Tool done not found" errors in production deployments with `allow_local_tools=false`.

### Decision

Move `DoneTool::new()` from `register_dev_tools()` to `register_builtin_tools()` in `src/tools/registry.rs`. Ensures `done` is unconditionally available alongside `echo`, `time`, `json`, and `http`.

### Rationale

The `done` tool is fundamental to completion detection — an orchestrator-domain primitive, not a development convenience. Conditional registration created inconsistent behavior and unacceptable failure modes.

### Alternatives Considered

| Alternative | Why Rejected? |
|-------------|---------------|
| Keep conditional registration | Production failures, inconsistent behavior |
| Add done to both registries | Duplication, maintenance burden |
| Unconditional builtin (chosen) | Correct classification — done is orchestrator tool |

### Impact

- **Positive**: `done` always available; consistent behavior across deployments; eliminates errors
- **Negative**: None — superset behavior, no breaking changes
- **Trade-offs**: `done` now classified as orchestrator-domain tool (more accurate)

### Implementation

- **Before**: `DoneTool::new()` in `register_dev_tools()` only
- **After**: `DoneTool::new()` in `register_builtin_tools()`, removed from dev tools

### Related

- `src/tools/registry.rs` — Tool registry implementation
- `src/tools/builtin/done.rs` — Done tool implementation
- ADR-001 — Original decision to add `done` tool

---

## ADR-004: Idempotent Job State Transitions

**Date**: 2026-02
**Status**: Decided
**Owner**: Core team

### Context

Both the execution loop (`JobDelegate::execute_tool_calls`) and worker wrapper can independently call `mark_completed()` when a job finishes, creating a race condition for `JobState::InProgress → JobState::Completed` transitions.

### Decision

Make `Completed -> Completed` transitions a no-op in the state machine (`src/context/state.rs`). Second call succeeds silently without recording duplicate transition.

### Rationale

In distributed async systems, exactly-once delivery is hard. Idempotent operations are cheaper and more robust than preventing duplicate calls. The `done` tool, text detection, and worker wrapper can all race — idempotency handles all cases.

### Alternatives Considered

| Alternative | Why Rejected? |
|-------------|---------------|
| Mutex/lock around completion | Adds contention, deadlock risk, over-engineered |
| Deduplication key | Extra state to manage, unnecessary complexity |
| Idempotent transitions (chosen) | Simple, robust, no extra state |

### Impact

- **Positive**: Eliminates race condition errors; simplifies caller code
- **Negative**: State machine slightly less strict
- **Risk**: Minimal — transition is no-op, no state corruption possible

### Related

- `src/context/state.rs:383` — `test_completed_to_completed_is_idempotent` test
- `src/worker/job.rs:949` — `mark_completed()` implementation
- `src/orchestrator/job_manager.rs:523` — `complete_job()` orchestrator-side
- ADR-003 — `done` tool registration (prerequisite)

---

## ADR-005: Graceful Null Metadata Handling in WASM Channel on_respond

**Date**: 2026-03-24
**Status**: Decided
**Owner**: Core team

### Context

Internal messages (e.g., job monitor notifications) use `metadata: Value::Null`. When `WasmChannel::respond()` serializes null to `"null"`, channel WASM `on_respond` functions fail deserializing into channel-specific metadata structs (e.g., `TelegramMessageMetadata`), causing repeated errors for all 5 channels.

### Decision

Use `match` with graceful skip instead of `map_err` propagation in all channel `on_respond` functions, mirroring the existing `on_status` pattern.

### Rationale

Internal messages don't originate from real channel conversations — they have no `chat_id` to route responses to. Silently dropping the response is correct behavior.

### Alternatives Considered

| Alternative | Why Rejected? |
|-------------|---------------|
| Fix metadata at source | Requires broader changes; internal messages legitimately have no metadata |
| Return error from on_respond | Causes retry loops, log spam; no valid destination to report to |
| match with graceful skip (chosen) | Mirrors on_status, simple, correct behavior |

### Impact

- **Positive**: Eliminates deserialization errors; consistent pattern across `on_status` and `on_respond`
- **Negative**: None — silent skip is intended for internal messages
- **Risk**: Minimal — real channel messages always have valid metadata from inbound path

### Related

- All 5 WASM channel implementations (telegram, discord, slack, whatsapp, feishu)
- `WasmChannel::respond()` — serializes metadata to JSON string
- `IncomingMessage::into_internal()` — creates internal messages with null metadata

---

## Deprecated Decisions

| Decision | Date | Replaced By | Why |
|----------|------|-------------|-----|
| Text-only completion detection | Pre-2026 | ADR-001 | False positives/negatives in phrase matching |

## Onboarding Checklist

- [ ] Understand philosophy behind major architectural choices
- [ ] Know why certain technologies were chosen over alternatives
- [ ] Understand trade-offs that were made
- [ ] Know where to find decision context when questions arise

## Related Files

- `technical-domain.md` - Technical implementation affected by these decisions
- `business-tech-bridge.md` - How decisions connect business and technical
- `living-notes.md` - Current open questions that may become decisions
