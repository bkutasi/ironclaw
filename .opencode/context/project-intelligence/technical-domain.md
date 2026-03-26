<!-- Context: project-intelligence/technical | Priority: critical | Version: 2.0 | Updated: 2026-03-24 -->

# Technical Domain

**Purpose**: Tech stack, architecture, and development patterns for Ironclaw - a secure personal AI assistant.

**Last Updated**: 2026-03-24

## Quick Reference

**Update Triggers**: Tech stack changes | New patterns | Architecture decisions | Security updates

**Audience**: Developers, AI agents, security reviewers

## Primary Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Language | Rust | 1.85+ | Memory safety, performance, WASM compatibility |
| Edition | Rust 2024 | - | Latest language features |
| Async Runtime | Tokio | 1.x | Industry standard for async Rust |
| Web Framework | Axum + Tower | 0.8 / 0.5 | Modular, composable HTTP handling |
| Database | PostgreSQL + pgvector | 15+ | Relational + semantic search |
| Pool | deadpool-postgres | 0.14 | Async connection pooling |
| WASM Runtime | Wasmtime | 28.x | Component model, sandboxing |
| Docker | Bollard | 0.18 | Container management |
| Cryptography | AES-GCM, HKDF, Blake3 | Latest | Secrets encryption, hashing |
| CLI | Clap + Rustyline | 4.x / 17.x | Command parsing, interactive REPL |
| Terminal | Crossterm + Termimad | 0.28 / 0.34 | Cross-platform TUI |

## Architecture Pattern

**Type**: Modular Agent-Based with Orchestrator/Worker Pattern

```
┌─────────────────────────────────────────────────────────────┐
│ User Interaction Layer                                      │
│ ┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────────────┐   │
│ │  CLI   │ │ Slack  │ │ Telegram │ │  HTTP/Webhook   │   │
│ └───┬────┘ └───┬────┘ └────┬─────┘ └────────┬────────┘   │
│     └─────────┴────────────┴─────────────────┘              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Shared Agentic Loop (run_agentic_loop)                      │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ LoopDelegate trait — 3 implementations:              │   │
│ │  • ChatDelegate (dispatcher.rs) — conversational     │   │
│ │  • JobDelegate (worker/job.rs) — background jobs     │   │
│ │  • ContainerDelegate (worker/container.rs) — Docker  │   │
│ └──────────────────────────────────────────────────────┘   │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│ │ Msg Router   │──│ LLM Reasoning│──│ Action Exec  │     │
│ └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│   ┌────┴────┐      ┌────┴────┐       ┌────┴────┐       │
│   │ Safety  │      │ Repair  │       │Sandbox  │       │
│   │ Layer   │      │ System  │       │(Docker) │       │
│   └─────────┘      └─────────┘       └─────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Key Patterns**:
1. **Orchestrator/Worker**: Main process spawns isolated Docker containers per job
2. **Shared Agentic Loop**: `run_agentic_loop()` in `src/agent/agentic_loop.rs` — single engine, three consumers via `LoopDelegate` trait
3. **WASM Sandbox**: Untrusted tools run in Wasmtime with capability-based permissions
4. **Safety Layer**: Input sanitization, prompt injection detection, policy enforcement
5. **Self-Repair**: Automatic detection and recovery of stuck jobs
6. **force_text Defense**: 3-phase anti-loop protection (nudge → force text → hard ceiling)

## Worker/Container Architecture

### Orchestrator/Worker Communication

```
Orchestrator                          Worker Container
┌──────────────┐    HTTP/gRPC    ┌──────────────────┐
│ JobManager   │◄───────────────►│ WorkerRuntime    │
│              │  GET /job       │                  │
│              │  POST /events   │  ContainerDelegate│
│              │  POST /status   │  ─────────────── │
│              │  POST /complete │  LLM (proxied)   │
│              │  GET /prompt    │  ToolRegistry     │
│              │  GET /creds     │  SafetyLayer      │
└──────────────┘                 └──────────────────┘
```

**Communication Protocol**:
- Worker fetches job description via `GET /jobs/{id}`
- Worker fetches credentials via `GET /jobs/{id}/credentials` (injected into child processes via `Command::envs()`, not global env)
- Worker posts events (message, tool_use, tool_result, result) via `POST /jobs/{id}/events`
- Worker posts status updates via `POST /jobs/{id}/status`
- Worker reports completion via `POST /jobs/{id}/complete` with `CompletionReport { success, message, iterations }`
- Orchestrator can inject follow-up prompts via `GET /jobs/{id}/prompt` (polled by worker)

### Container Credential Injection

Credentials are fetched once at worker startup and stored in `Arc<HashMap<String, String>>`. They're passed to child processes via `Command::envs()` rather than mutating `std::env::set_var` (which is unsafe in multi-threaded Tokio runtimes).

### force_text 3-Phase Defense

Both `ChatDelegate` and `ContainerDelegate` implement graduated anti-loop protection:

```
Iteration N-1 (nudge_at):     Inject system warning → "Provide final answer, no more tools"
Iteration N   (force_text_at): Swap system prompt → no-tools variant, set force_text=true
Iteration N+1 (hard ceiling):  Loop terminates with MaxIterations
```

**Implementation** (`src/worker/container.rs:167-414`, `src/agent/dispatcher.rs:165-311`):
- `cached_prompt`: System prompt with tool definitions (normal iterations)
- `cached_prompt_no_tools`: System prompt without tools (force_text iteration)
- `reason_ctx.force_text = true`: Tells `respond_with_tools()` to use text-only mode
- `reason_ctx.available_tools`: Cleared or left populated (provider may ignore when force_text)

### done Tool

**Location**: `src/tools/builtin/done.rs`

**Purpose**: Explicit job completion signaling via tool call instead of text pattern matching.

**Description**: "Signal that you are done. Call this tool when you have finished all work and are ready to report your final results."

**Flow**:
1. LLM calls `done` with `summary` parameter
2. `ContainerDelegate::execute_tool_calls()` detects `tc.name == "done"` → returns `LoopOutcome::Response(output)`
3. `JobDelegate::execute_tool_calls()` detects `selection.tool_name == "done"` → calls `mark_completed()` → returns `LoopOutcome::Response(output)`
4. Agentic loop breaks, completion report sent to orchestrator

**Why tool-based over text-based**: `llm_signals_completion()` in `src/util.rs` uses phrase matching ("job is complete", "tests passed") which can false-positive. The tool approach is deterministic — the LLM must explicitly opt-in to completion.

### Tool Failure Tracking System

**Table**: `tool_failures` — tracks per-tool error counts with `UNIQUE` constraint on `tool_name`.

**Recording**: `record_tool_failure()` is called from `src/worker/job.rs` only (background job worker path). The chat dispatcher does NOT record tool failures. Each failure increments `error_count` via UPSERT on `tool_name`.

**Detection**: `get_broken_tools(threshold)` returns tools where `error_count >= threshold AND repaired_at IS NULL`. The self-repair loop in `src/self_repair.rs` detects these as "broken" and attempts repair.

**Repair limitation**: Self-repair attempts a WASM rebuild via `SoftwareBuilder` — this is the wrong approach for built-in tools (memory_read, time, message) which are not WASM. Built-in tool failures accumulate with no automatic repair path, requiring manual SQL cleanup:
```sql
-- Mark as repaired
UPDATE tool_failures SET repaired_at = NOW(), error_count = 0 WHERE tool_name = 'X';
-- Or delete stale records
DELETE FROM tool_failures WHERE tool_name IN ('tool1', 'tool2');
```

**LLM impact**: Broken tools do NOT affect the LLM — tool definitions are still sent to the model regardless of failure count. The `PROTECTED_TOOL_NAMES` list prevents dynamically built tools from shadowing built-in tool names.

**Builtin Tool Protection** (2026-03-25 fix):
Built-in tools registered via `register_sync()` MUST be in `PROTECTED_TOOL_NAMES` constant. This ensures:
1. `ToolRegistry::is_builtin()` returns `true` → excluded from failure tracking
2. Cannot be shadowed by malicious WASM tools (registration rejected)
3. Self-repair doesn't attempt WASM rebuild (impossible for builtin tools)

**Missing tools from protection** (fixed 2026-03-25): `job_events`, `job_prompt`, `tool_upgrade`, `extension_info`, `secret_list`, `secret_delete`

**Code Review Pattern**: When adding new builtin tools via `register_sync()`, always add to `PROTECTED_TOOL_NAMES` in same PR. CodeReviewer should verify this.

### LLM Response Suppression via `<suggestions>` Extraction

The system prompt instructs the LLM to "ALWAYS end your response with a `<suggestions>` tag". `extract_suggestions()` in `src/agent/dispatcher.rs` strips the `<suggestions>` block from the response text before returning it to the user.

**Failure mode**: If the LLM's entire response is JUST the `<suggestions>` block (no actual content), the extracted text becomes an empty string. `src/agent/agentic_loop.rs` checks `if !response.is_empty()` — empty responses are suppressed with "Suppressed empty response (not sent to channel)". This manifests as the bot appearing to not respond at all.

**Diagnosis**: Check logs for "Suppressed empty response" messages. If present, the LLM is producing suggestions-only responses. This is a prompting issue, not a Telegram or channel issue.

**Fix**: Adjust the system prompt or LLM instructions to ensure actual content precedes the suggestions tag. The `<suggestions>` block should be supplementary, not the entire response.

### Telegram chat_id Routing

`chat_id` flows from incoming Telegram message metadata, NOT from env vars or config files:

- **Responses**: `chat_id` is extracted from the original message's `TelegramMessageMetadata` (round-trip through the WASM boundary)
- **Proactive messages** (routines, broadcasts): `chat_id` comes from `last_broadcast_metadata`, which is populated when the owner first sends a message to the bot
- **`TELEGRAM_OWNER_ID`**: Sets authorization (who can talk to the bot), NOT routing. Changing this does not change where messages are sent.
- **Failure mode**: If no message was ever received from the owner, proactive messages fail with "No stored owner routing target"

### Thinking Model Support (llama.cpp)

**Problem**: llama.cpp rejects assistant prefill when `enable_thinking=true` (default for Qwen3, DeepSeek-R1, etc.) with error: `"Assistant response prefill is incompatible with enable_thinking"`.

**Solution**: `ensure_last_message_role()` in `src/llm/provider.rs:479`:
- Detects thinking models via `has_native_thinking()` in `src/llm/reasoning_models.rs`
- If last message is assistant role + thinking model → **strips** the prefill (pop)
- If last message is assistant role + non-thinking model → **appends** "Continue." user message

**Detection patterns** (`NATIVE_THINKING_PATTERNS`): qwen3, qwq, deepseek-r1, deepseek-reasoner, glm-z1, glm-4-plus, glm-5, nanbeige, step-3.5, minimax-m2

## Response Delivery Pipeline & Metadata Flow

Channel messages follow a round-trip metadata pipeline through the WASM boundary:

```
Inbound (channel → agent):
  Telegram API → TelegramMessageMetadata built → serialized to JSON
  → channel_host::emit_message → host applies metadata to IncomingMessage
  → agent loop processes message

Outbound (agent → channel):
  Agent produces response → WasmChannel::respond() serializes metadata back to JSON
  → WASM on_respond deserializes into channel-specific struct
  → send_response() delivers to channel API
```

**Internal messages break the inbound path**: Messages created via `IncomingMessage::new(...).into_internal()` (e.g., job monitor notifications) skip the inbound metadata construction entirely — they arrive with `metadata: Value::Null`. When the agent responds, `respond()` serializes null to `"null"`, which fails deserialization in `on_respond`.

**Fix pattern**: Channel WASM `on_respond` functions must use `match` (not `map_err`) to gracefully skip when metadata is invalid. This mirrors the existing `on_status` pattern. Internal messages have no `chat_id` to route to, so silent drop is correct behavior.

## Code Patterns

### Error Handling

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool {name} not found")]
    NotFound { name: String },
    #[error("Tool {name} execution failed: {reason}")]
    ExecutionFailed { name: String, reason: String },
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}
```

### Async Trait Pattern

```rust
use async_trait::async_trait;

#[async_trait]
pub trait Tool: Send + Sync {
    async fn execute(
        &self,
        params: serde_json::Value,
        context: &JobContext,
    ) -> Result<ToolOutput, ToolError>;
}
```

### LoopDelegate Pattern

```rust
#[async_trait]
pub trait LoopDelegate: Send + Sync {
    async fn check_signals(&self) -> LoopSignal;
    async fn before_llm_call(&self, ctx: &mut ReasoningContext, iter: usize) -> Option<LoopOutcome>;
    async fn call_llm(&self, reasoning: &Reasoning, ctx: &mut ReasoningContext, iter: usize) -> Result<RespondOutput, Error>;
    async fn handle_text_response(&self, text: &str, ctx: &mut ReasoningContext) -> TextAction;
    async fn execute_tool_calls(&self, calls: Vec<ToolCall>, content: Option<String>, ctx: &mut ReasoningContext) -> Result<Option<LoopOutcome>, Error>;
    async fn on_tool_intent_nudge(&self, text: &str, ctx: &mut ReasoningContext) {}
    async fn after_iteration(&self, iter: usize) {}
}
```

### Module Structure

```rust
// lib.rs - public interface
pub mod tools;
pub use tools::{Tool, ToolOutput};

// tools/mod.rs - module aggregation
pub mod registry;
pub mod tool;
pub mod wasm;

pub use tool::{Tool, ToolOutput, ToolError};
pub use registry::ToolRegistry;
```

## Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Files | `snake_case.rs` | `job_manager.rs` |
| Modules | `snake_case` | `mod job_manager;` |
| Types/Structs | `PascalCase` | `JobManager`, `ContainerConfig` |
| Functions | `snake_case` | `get_job_status()`, `execute_tool()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRY_ATTEMPTS` |
| Enums | `PascalCase` | `ToolDomain::Orchestrator` |
| Error Types | `PascalCase + Error` | `ToolError`, `SafetyError` |

## Code Standards

1. **Rust 2024 Edition** - Use latest language features
2. **Strict Compiler** - `#![deny(warnings)]` in production
3. **Error Types** - Use `thiserror` for structured errors per module
4. **Async Traits** - Use `async-trait` for interfaces
5. **Serde Derives** - Always derive `Serialize`/`Deserialize` for data types
6. **Documentation** - All public items must have doc comments (`//!`/`///`)
7. **Safety First** - Validate at boundaries, sanitize all external input
8. **Explicit Errors** - Return `Result<T, E>`, avoid panics in production code
9. **Pure Functions** - Prefer immutable data, explicit dependencies
10. **Small Functions** - Keep under 50 lines, single responsibility

## Security Requirements

1. **WASM Sandboxing** - All untrusted tools run in Wasmtime with capability restrictions
2. **Credential Protection** - Secrets injected at host boundary via `Command::envs()`, never exposed to tools
3. **Leak Detection** - Pattern matching to detect credential exfiltration attempts
4. **Prompt Injection Defense** - Multi-layer sanitization, policy enforcement
5. **Endpoint Allowlisting** - HTTP requests only to approved hosts/paths
6. **Docker Isolation** - Per-job containers with minimal privileges
7. **Token-Based Auth** - Per-job bearer tokens for orchestrator/worker communication
8. **Encryption at Rest** - AES-GCM for secrets storage in system keychain

## 📂 Codebase References

**Core Architecture**:
- `src/lib.rs` - Module exports and architecture diagram
- `src/error.rs` - Error type definitions (369 lines)
- `src/config.rs` - Configuration management

**Agent System**:
- `src/agent/agentic_loop.rs` - Shared agentic loop engine (`run_agentic_loop`, `LoopDelegate` trait)
- `src/agent/dispatcher.rs` - `ChatDelegate` implementation, tool approval, skill context
- `src/agent/session.rs` - Session/Thread/Turn data model
- `src/worker/job.rs` - `JobDelegate` implementation for background jobs
- `src/worker/container.rs` - `ContainerDelegate` implementation for Docker workers

**Orchestrator/Worker**:
- `src/orchestrator/job_manager.rs` - `complete_job()`, container lifecycle
- `src/orchestrator/api.rs` - HTTP endpoints for worker communication
- `src/worker/api.rs` - `WorkerHttpClient`, `CompletionReport`, `StatusUpdate`
- `src/worker/proxy_llm.rs` - `ProxyLlmProvider` (routes LLM calls through orchestrator)

**Tools**:
- `src/tools/builtin/done.rs` - Explicit job completion tool (originally `complete_job`, renamed for clarity)
- `src/tools/registry.rs` - Tool registration and container-safe tool set
- `src/tools/execute.rs` - `execute_tool_with_safety()`, `process_tool_result()`

**LLM Integration**:
- `src/llm/provider.rs` - `LlmProvider` trait, `ensure_last_message_role()`
- `src/llm/reasoning_models.rs` - `has_native_thinking()` detection
- `src/llm/reasoning.rs` - `Reasoning` engine, thinking tag stripping
- `src/llm/rig_adapter.rs` - Bridge for OpenAI/Anthropic/Ollama providers

**Safety & Security**:
- `src/safety/` - Sanitization, leak detection, policy enforcement
- `src/sandbox/` - Docker container management
- `src/secrets/` - Encryption and keychain integration

**Utilities**:
- `src/util.rs` - `llm_signals_completion()`, `floor_char_boundary()`

**Self-Repair & Tool Health**:
- `src/self_repair.rs` - Self-repair loop, `get_broken_tools()`, WASM rebuild attempts
- `src/worker/job.rs` - `record_tool_failure()` (tool failure recording, job worker path only)
- `migrations/` - `tool_failures` table schema (UNIQUE on `tool_name`, `error_count`, `repaired_at`)

**Config Files**:
- `Cargo.toml` - Dependencies and project metadata
- `Dockerfile.worker` - Worker container image
- `migrations/` - Database schema migrations

## Related Files

- `business-domain.md` - Business context and problem statement
- `business-tech-bridge.md` - How business needs map to technical solutions
- `decisions-log.md` - Architecture decision records
- `living-notes.md` - Active issues and technical debt
