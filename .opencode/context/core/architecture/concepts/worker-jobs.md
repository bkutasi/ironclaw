# Worker Jobs System

**Purpose**: Background job execution architecture using Docker containers and orchestrator coordination

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Quick Reference

**Worker Modes**:
- `ironclaw worker` — Standard agentic worker in Docker container
- `ironclaw claude-bridge` — Claude Code CLI bridge for sandboxed execution

**Architecture**: Orchestrator manages jobs → Workers execute in containers → Proxy LLM routes through orchestrator → Events streamed for UI

**Security Model**: Docker container isolation + bearer token auth + no direct API key access in containers

---

## Core Concept

IronClaw's worker system enables secure, isolated background job execution. Jobs run inside Docker containers with restricted capabilities, communicating with an orchestrator that holds API keys and manages authentication. Workers use a `ProxyLlmProvider` to route LLM calls through the orchestrator, ensuring secrets never touch the container.

```text
┌────────────────────────────────┐
│        Docker Container         │
│                                 │
│  ironclaw worker                │
│    ├─ ProxyLlmProvider ─────────┼──▶ Orchestrator /worker/{id}/llm/complete
│    ├─ SafetyLayer               │
│    ├─ ToolRegistry              │
│    │   ├─ shell                 │
│    │   ├─ read_file             │
│    │   ├─ write_file            │
│    │   ├─ list_dir              │
│    │   └─ apply_patch           │
│    └─ WorkerHttpClient ─────────┼──▶ Orchestrator /worker/{id}/status
│                                 │
└────────────────────────────────┘
```

---

## Architecture Overview

### Components

**Orchestrator** (`src/orchestrator/`):
- Manages job queue and scheduling
- Holds API keys and secrets
- Proxies LLM requests from workers
- Receives and broadcasts job events

**Worker Runtime** (`src/worker/`):
- `WorkerRuntime` — Main execution loop in container
- `ClaudeBridgeRuntime` — Claude Code CLI integration
- `ProxyLlmProvider` — LLM calls via orchestrator
- `WorkerHttpClient` — HTTP client for orchestrator communication

**Job Delegate** (`src/worker/job.rs`):
- `Worker` — Executes single job via `AgenticLoop`
- `WorkerDeps` — Shared dependencies bundle
- Implements `LoopDelegate` for agentic loop integration

### Execution Flow

1. **Job Submission**: User submits job → Orchestrator creates job record
2. **Container Launch**: Orchestrator spawns Docker container with worker binary
3. **Initialization**: Worker connects to orchestrator, fetches job description
4. **Credential Injection**: Orchestrator provides temporary credentials (env vars)
5. **Execution Loop**: Worker runs agentic loop with tools
6. **Event Streaming**: Real-time events posted to orchestrator for UI
7. **Completion**: Worker reports result → Container terminates

---

## Job Scheduler

### Worker Lifecycle

Jobs are managed by the orchestrator's scheduler (`src/agent/scheduler.rs`):

```rust
// Worker receives start signal via channel
match rx.recv().await {
    Some(WorkerMessage::Start) => { /* begin execution */ }
    Some(WorkerMessage::Stop) => { /* graceful shutdown */ }
    Some(WorkerMessage::UserMessage(content)) => { /* inject user message */ }
}
```

### Worker States

Jobs transition through states persisted to the database:

- `Submitted` → Job queued, awaiting execution
- `InProgress` → Worker actively executing
- `Completed` → Job finished successfully
- `Failed` → Job encountered unrecoverable error
- `Stuck` → Job hit iteration limit or deadlock (recoverable via self-repair)
- `Cancelled` → User terminated job

### Parallel Tool Execution

Workers can execute multiple tools in parallel using `JoinSet`:

```rust
// Parallel execution for independent tools
let mut join_set = JoinSet::new();
for (idx, selection) in selections.iter().enumerate() {
    join_set.spawn(async move {
        execute_tool_inner(&deps, job_id, &tool_name, &params).await
    });
}
// Results collected and reordered by original index
```

**Note**: Container workers execute tools sequentially; parallel execution is used in orchestrator-managed workers.

---

## Proxy LLM Provider

### Security Model

Containers **never** have direct access to:
- API keys (Anthropic, OpenAI, etc.)
- Session tokens
- Billing credentials

All LLM calls route through the orchestrator:

```rust
pub struct ProxyLlmProvider {
    client: Arc<WorkerHttpClient>,
    model_name: String,
}

#[async_trait]
impl LlmProvider for ProxyLlmProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.client.llm_complete(&request).await
            .map_err(|e| LlmError::RequestFailed {
                provider: "proxy".to_string(),
                reason: e.to_string(),
            })
    }
}
```

### API Endpoints

Worker → Orchestrator HTTP API:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/worker/{id}/job` | GET | Fetch job description |
| `/worker/{id}/llm/complete` | POST | Proxy text completion |
| `/worker/{id}/llm/complete_with_tools` | POST | Proxy tool-augmented completion |
| `/worker/{id}/status` | POST | Report job status |
| `/worker/{id}/event` | POST | Post job event (message, tool_use, etc.) |
| `/worker/{id}/prompt` | GET | Poll for follow-up prompts |
| `/worker/{id}/credentials` | GET | Fetch injected credentials |
| `/worker/{id}/complete` | POST | Report job completion |

### Authentication

Every request includes bearer token from `IRONCLAW_WORKER_TOKEN` env var:

```rust
fn get_json<T>(&self, path: &str, context: &str) -> Result<T, WorkerError> {
    let resp = self.client
        .get(self.url(path))
        .bearer_auth(&self.token)
        .send()
        .await?;
    // ...
}
```

---

## Worker API

### WorkerHttpClient

The HTTP client for worker-to-orchestrator communication:

```rust
pub struct WorkerHttpClient {
    client: reqwest::Client,
    orchestrator_url: String,
    job_id: Uuid,
    token: String,
}
```

**Key Methods**:
- `from_env(orchestrator_url, job_id)` — Create from environment (reads `IRONCLAW_WORKER_TOKEN`)
- `get_job()` — Fetch job description
- `llm_complete(request)` — Proxy completion request
- `llm_complete_with_tools(request)` — Proxy tool completion
- `report_status(update)` — Send status update
- `post_event(payload)` — Broadcast event (fire-and-forget)
- `poll_prompt()` — Check for follow-up prompts
- `fetch_credentials()` — Get injected credentials
- `report_complete(report)` — Signal job completion

### Event Payloads

Workers stream events for real-time UI updates:

```rust
pub struct JobEventPayload {
    pub event_type: String,
    pub data: serde_json::Value,
}
```

**Event Types**:
- `message` — Assistant or user message
- `tool_use` — Tool invocation with input
- `tool_result` — Tool execution output
- `status` — Status update (iteration, state)
- `result` — Final job result
- `reasoning` — LLM reasoning with decisions

---

## Self-Repair Loop

### Iteration Limits

Workers defend against infinite loops with a 3-phase strategy:

```rust
let max_iterations = 50; // Configurable per job
let nudge_at = max_iterations.saturating_sub(1); // Phase 1
let force_text_at = max_iterations; // Phase 2
```

**Phase 1 — Nudge** (iteration 49/50):
- Inject system message: "You are approaching the tool call limit..."
- LLM warned to provide final answer without more tools

**Phase 2 — Force Text** (iteration 50):
- Swap system prompt to version without tool definitions
- Set `reason_ctx.force_text = true`
- LLM can only respond with text, no tool calls

**Phase 3 — Hard Ceiling** (iteration 51):
- Loop terminates with `MaxIterations` outcome
- Job marked as `Failed` or `Stuck` for potential self-repair

### Stuck Job Recovery

Jobs marked `Stuck` can be recovered:

1. **Detection**: Orchestrator identifies stuck jobs (no progress, hit limits)
2. **State Transition**: `Stuck` → `InProgress` (via repair trigger)
3. **Context Preservation**: Memory and action history retained
4. **Fresh Worker**: New container spawned with same context
5. **Continuation**: Worker resumes from last known state

**Note**: The loop explicitly does NOT stop on `Stuck` state to allow recovery:

```rust
// From JobDelegate::check_signals
if let Ok(ctx) = self.context_manager().get_context(self.job_id).await
    && matches!(ctx.state, JobState::Cancelled | JobState::Failed | JobState::Completed)
{
    return LoopSignal::Stop; // But NOT Stuck — allows recovery
}
```

### Rate Limit Handling

Workers implement exponential backoff for LLM rate limits:

```rust
async fn handle_rate_limit(&self, retry_after: Option<Duration>) -> Result<..., Error> {
    let count = self.consecutive_rate_limits.fetch_add(1, Relaxed) + 1;
    let wait = retry_after.unwrap_or(Duration::from_secs(5));
    
    if count >= MAX_CONSECUTIVE_RATE_LIMITS {
        self.worker.mark_failed("Persistent rate limiting").await?;
        return Err(LlmError::RateLimited { ... });
    }
    
    tokio::time::sleep(wait).await;
    // Retry...
}
```

**Fail-Fast**: After 10 consecutive rate limits, job fails instead of burning iterations.

---

## Code Patterns

### Creating a Worker

```rust
use crate::worker::{Worker, WorkerDeps};
use crate::context::ContextManager;
use crate::llm::LlmProvider;
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;

let deps = WorkerDeps {
    context_manager: Arc::new(ContextManager::new(...)),
    llm: Arc::new(MyLlmProvider::new(...)),
    safety: Arc::new(SafetyLayer::new(&config)),
    tools: Arc::new(ToolRegistry::new()),
    store: Some(db),
    hooks: Arc::new(HookRegistry::new()),
    timeout: Duration::from_secs(600),
    use_planning: true,
    sse_tx: Some(sse_manager),
    approval_context: Some(approval_ctx),
    http_interceptor: None,
};

let worker = Worker::new(job_id, deps);
```

### Running Worker Mode

```rust
// From src/worker/mod.rs
pub async fn run_worker(
    job_id: uuid::Uuid,
    orchestrator_url: &str,
    max_iterations: u32,
) -> anyhow::Result<()> {
    let config = container::WorkerConfig {
        job_id,
        orchestrator_url: orchestrator_url.to_string(),
        max_iterations,
        timeout: std::time::Duration::from_secs(600),
    };

    let rt = WorkerRuntime::new(config)?;
    rt.run().await
}
```

### Claude Bridge Pattern

```rust
pub async fn run_claude_bridge(
    job_id: uuid::Uuid,
    orchestrator_url: &str,
    max_turns: u32,
    model: &str,
) -> anyhow::Result<()> {
    let config = claude_bridge::ClaudeBridgeConfig {
        job_id,
        orchestrator_url: orchestrator_url.to_string(),
        max_turns,
        model: model.to_string(),
        timeout: std::time::Duration::from_secs(1800),
        allowed_tools: vec!["Bash(*)".into(), "Read".into(), "Edit(*)".into()],
    };

    let rt = ClaudeBridgeRuntime::new(config)?;
    rt.run().await
}
```

### Tool Execution with Approval

```rust
// Check if tool is blocked based on approval context
let requirement = tool.requires_approval(&normalized_params);
let blocked = ApprovalContext::is_blocked_or_default(
    &deps.approval_context, 
    tool_name, 
    requirement
);
if blocked {
    return Err(autonomous_unavailable_error(tool_name, &user_id).into());
}

// Execute tool
let result = tool.execute(effective_params, &job_ctx).await;
```

### Event Broadcasting

```rust
// Log event to DB and broadcast via SSE
self.log_event("tool_use", serde_json::json!({
    "tool_name": selection.tool_name,
    "input": truncate_for_preview(&selection.parameters.to_string(), 500),
}));

// Internally handles both DB persistence and SSE broadcast
fn log_event(&self, event_type: &str, data: serde_json::Value) {
    // DB persistence (fire-and-forget)
    if let Some(store) = self.store() {
        tokio::spawn(async move {
            store.save_job_event(job_id, event_type, &data).await.ok();
        });
    }
    
    // SSE broadcast for live UI
    if let Some(ref sse) = self.deps.sse_tx {
        sse.broadcast(AppEvent::JobToolUse { ... });
    }
}
```

---

## Related

- `src/worker/` — Worker runtime implementation
- `src/orchestrator/` — Job scheduler and management
- `src/agent/agentic_loop.rs` — Shared agentic loop engine
- `src/context/` — Job context and memory management
- architecture/concepts/docker-postgres.md — Container isolation patterns
- architecture/concepts/config-precedence.md — Environment configuration

---

**Source Files**:
- `src/worker/mod.rs` — Worker mode entry points
- `src/worker/job.rs` — Worker execution via AgenticLoop
- `src/worker/container.rs` — Container runtime delegate
- `src/worker/api.rs` — HTTP client for orchestrator communication
- `src/worker/proxy_llm.rs` — Proxy LLM provider
- `src/worker/claude_bridge.rs` — Claude Code integration
