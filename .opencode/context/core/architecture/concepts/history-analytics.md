# History and Analytics

**Purpose**: PostgreSQL-backed persistence layer for job history, conversations, tool actions, and analytics for audit trails, learning from past executions, and metrics.

**Last Updated**: 2026-03-26

---

## Quick Reference

```rust
use crate::history::Store;

// Create store with database config
let store = Store::new(&db_config).await?;

// Run migrations on startup
store.run_migrations().await?;

// Record job, actions, LLM calls
store.save_job(&job_context).await?;
store.save_action(job_id, &action).await?;
store.record_llm_call(&llm_record).await?;

// Get analytics
let job_stats = store.get_job_stats().await?;
let tool_stats = store.get_tool_stats().await?;
```

**Feature Flag**: `postgres` — All history/analytics features require PostgreSQL backend.

**Key Types**: `Store`, `JobStats`, `ToolStats`, `LlmCallRecord`, `JobEventRecord`, `AgentJobRecord`, `SandboxJobRecord`, `Routine`, `RoutineRun`

---

## Architecture Overview

The history module provides **persistent storage and analytics** for the agent system:

```
┌─────────────────────────────────────────────────────────────┐
│ Agent System                                                │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│ │ Sessions │  │   Jobs   │  │  Tools   │  │  LLMs    │   │
│ └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│      │             │             │             │           │
│      └─────────────┴─────────────┴─────────────┘           │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ History Module (src/history/)                               │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Store — PostgreSQL persistence layer                 │   │
│ │  • Conversations & messages                          │   │
│ │  • Agent jobs & sandbox jobs                         │   │
│ │  • Tool actions & LLM calls                          │   │
│ │  • Job events (streaming)                            │   │
│ │  • Routines & routine runs                           │   │
│ │  • Estimation snapshots                              │   │
│ └──────────────────────────────────────────────────────┘   │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Analytics — Aggregation & metrics                    │   │
│ │  • Job statistics (success rate, avg duration)       │   │
│ │  • Tool statistics (calls, success rate, cost)       │   │
│ │  • Estimation accuracy (learning)                    │   │
│ │  • Category history (for learning)                   │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PostgreSQL Database                                         │
│ Tables: agent_jobs, job_actions, llm_calls, conversations, │
│         conversation_messages, job_events, routines,        │
│         routine_runs, estimation_snapshots, settings        │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/history/
├── mod.rs          # Module exports, conditional compilation
├── store.rs        # Store struct, persistence methods (~1600 lines)
└── analytics.rs    # Analytics methods on Store (~226 lines)
```

### Feature Gating

**Critical**: All analytics methods are gated behind `#[cfg(feature = "postgres")]`:

```rust
#[cfg(feature = "postgres")]
mod analytics;

#[cfg(feature = "postgres")]
pub use analytics::{JobStats, ToolStats};
#[cfg(feature = "postgres")]
pub use store::Store;
```

**Implication**: History/analytics features are **PostgreSQL-only**. libSQL backend does not support analytics.

---

## Session History

### Conversations

Conversations track user interactions across channels (CLI, Slack, Telegram, HTTP):

```rust
// Create conversation
let conv_id = store.create_conversation(
    "telegram",      // channel
    "user_123",      // user_id
    Some("chat_456") // thread_id (optional)
).await?;

// Add messages
store.add_conversation_message(
    conv_id,
    "user",          // role: "user" | "assistant" | "system"
    "What's the weather?"
).await?;

store.add_conversation_message(
    conv_id,
    "assistant",
    "I don't have access to weather data."
).await?;

// Touch (update last_activity)
store.touch_conversation(conv_id).await?;
```

**Schema**:
- `conversations`: `id`, `channel`, `user_id`, `thread_id`, `last_activity`
- `conversation_messages`: `id`, `conversation_id`, `role`, `content`

**Auto-update**: `add_conversation_message()` automatically calls `touch_conversation()`.

---

## Job Storage

### Agent Jobs (Direct)

Jobs created directly via CLI or web interface (`source = 'direct'`):

```rust
// Save job context
store.save_job(&job_context).await?;

// Get job by ID
let job = store.get_job(job_id).await?;

// Update status
store.update_job_status(job_id, JobState::Completed, None).await?;

// Mark as stuck
store.mark_job_stuck(job_id).await?;

// Get stuck jobs
let stuck = store.get_stuck_jobs().await?;
```

### Sandbox Jobs

Jobs running in isolated Docker containers (`source = 'sandbox'`):

```rust
// Save sandbox job
store.save_sandbox_job(&SandboxJobRecord {
    id: Uuid::new_v4(),
    task: "Run tests".to_string(),
    status: "running".to_string(),
    user_id: "user_123".to_string(),
    project_dir: "/app".to_string(),
    success: None,
    failure_reason: None,
    created_at: Utc::now(),
    started_at: Some(Utc::now()),
    completed_at: None,
    credential_grants_json: "[]".to_string(),
}).await?;

// Update status
store.update_sandbox_job_status(
    job_id, "completed", Some(true), None,
    Some(started_at), Some(completed_at)
).await?;

// List jobs
let jobs = store.list_sandbox_jobs().await?;
let user_jobs = store.list_sandbox_jobs_for_user("user_123").await?;

// Summary by status
let summary = store.sandbox_job_summary().await?;
// SandboxJobSummary { total, creating, running, completed, failed, interrupted }

// Cleanup stale jobs (on startup)
let count = store.cleanup_stale_sandbox_jobs().await?;
// Marks 'running'/'creating' jobs as 'interrupted'
```

### Job Ownership

Jobs are associated with users via `user_id`:

```rust
// List agent jobs for user
let jobs = store.list_agent_jobs_for_user("user_123").await?;

// Summary for user
let summary = store.agent_job_summary_for_user("user_123").await?;

// Verify ownership (sandbox jobs)
let belongs = store.sandbox_job_belongs_to_user(job_id, "user_123").await?;
```

### Job Mode (Sandbox)

Sandbox jobs can operate in different modes:

```rust
// Update mode
store.update_sandbox_job_mode(job_id, "full_job").await?;

// Get mode
let mode = store.get_sandbox_job_mode(job_id).await?;
// Returns Option<String> — "full_job", "container", etc.
```

---

## Event Storage

### Job Events

Streaming events from workers or Claude Code bridge:

```rust
// Save event (fire-and-forget)
store.save_job_event(
    job_id,
    "tool_result",           // event_type
    &json!({"output": "..."}) // data (serde_json::Value)
).await?;

// List events (all)
let events = store.list_job_events(job_id, None).await?;

// List last N events (most recent, ordered ascending)
let last_50 = store.list_job_events(job_id, Some(50)).await?;
```

**Schema**: `job_events`: `id`, `job_id`, `event_type`, `data` (JSONB), `created_at`

**Use Cases**:
- Real-time job progress streaming
- Audit trail for job execution
- Debugging stuck/failed jobs

---

## Tool Actions

### Recording Actions

Every tool execution is recorded:

```rust
store.save_action(job_id, &ActionRecord {
    id: Uuid::new_v4(),
    sequence: 1,
    tool_name: "file_write".to_string(),
    input: r#"{"path": "test.txt", "content": "hello"}"#.to_string(),
    output_raw: "File written".to_string(),
    output_sanitized: "File written".to_string(),
    sanitization_warnings: vec![],
    cost: rust_decimal::Decimal::ZERO,
    duration: Duration::from_millis(50),
    success: true,
    error: None,
    executed_at: Utc::now(),
}).await?;

// Get actions for job (ordered by sequence)
let actions = store.get_job_actions(job_id).await?;
```

**Schema**: `job_actions`: `id`, `job_id`, `sequence_num`, `tool_name`, `input`, `output_raw`, `output_sanitized`, `sanitization_warnings` (JSON), `cost`, `duration_ms`, `success`, `error_message`, `created_at`

**Sanitization**: `output_raw` is the unfiltered tool output; `output_sanitized` is cleaned for user display; `sanitization_warnings` lists potential credential leaks detected.

---

## LLM Call Tracking

### Recording LLM Calls

Track token usage and costs:

```rust
use crate::history::LlmCallRecord;

store.record_llm_call(&LlmCallRecord {
    job_id: Some(job_id),
    conversation_id: Some(conv_id),
    provider: "anthropic",
    model: "claude-sonnet-4-20250514",
    input_tokens: 1500,
    output_tokens: 300,
    cost: rust_decimal::Decimal::from_str("0.012").unwrap(),
    purpose: Some("planning"),
}).await?;
```

**Schema**: `llm_calls`: `id`, `job_id`, `conversation_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `cost`, `purpose`

**Use Cases**:
- Cost tracking per job/user
- Token usage analytics
- Model performance comparison

---

## Analytics

### Job Statistics

Aggregate metrics across all jobs:

```rust
use crate::history::JobStats;

let stats = store.get_job_stats().await?;
// JobStats {
//     total_jobs: 150,
//     completed_jobs: 142,
//     failed_jobs: 8,
//     success_rate: 0.947,
//     avg_duration_secs: 45.3,
//     avg_cost: 0.0234,
//     total_cost: 3.51,
// }
```

**Query**: Aggregates from `agent_jobs` table with filters for status.

### Tool Statistics

Usage metrics per tool:

```rust
use crate::history::ToolStats;

let stats = store.get_tool_stats().await?;
// Vec<ToolStats> ordered by total_calls DESC
// ToolStats {
//     tool_name: "file_read",
//     total_calls: 523,
//     successful_calls: 518,
//     failed_calls: 5,
//     success_rate: 0.990,
//     avg_duration_ms: 12.4,
//     total_cost: 0.0,
// }
```

**Query**: Groups `job_actions` by `tool_name` with success/failure counts.

### Estimation Accuracy

Track estimation vs actual for learning:

```rust
use crate::history::EstimationAccuracy;

// Overall accuracy
let accuracy = store.get_estimation_accuracy(None).await?;

// By category
let accuracy = store.get_estimation_accuracy(Some("bug_fix")).await?;
// EstimationAccuracy {
//     cost_error_rate: 0.15,  // 15% average error
//     time_error_rate: 0.23,  // 23% average error
//     sample_count: 47,
// }
```

**Query**: Calculates mean absolute percentage error (MAPE) from `estimation_snapshots`.

### Category History

Historical data for learning patterns:

```rust
use crate::history::CategoryHistoryEntry;

let history = store.get_category_history("feature", 100).await?;
// Vec<CategoryHistoryEntry> {
//     tool_names: vec!["file_read", "file_write", "bash"],
//     estimated_cost: 0.05,
//     actual_cost: Some(0.062),
//     estimated_time_secs: 120,
//     actual_time_secs: Some(145),
//     created_at: DateTime<Utc>,
// }
```

**Use Case**: Feed into estimation system for improved predictions based on historical patterns.

---

## Routines

### Creating Routines

Automated background tasks with triggers:

```rust
use crate::agent::routine::{Routine, Trigger, RoutineAction, RoutineGuardrails, NotifyConfig};

let routine = Routine {
    id: Uuid::new_v4(),
    name: "daily_backup".to_string(),
    description: "Backup database daily".to_string(),
    user_id: "user_123".to_string(),
    enabled: true,
    trigger: Trigger::Cron { schedule: "0 0 * * *".to_string() },
    action: RoutineAction::FullJob { task: "Run backup".to_string() },
    guardrails: RoutineGuardrails {
        cooldown: Duration::from_secs(3600),
        max_concurrent: 1,
        dedup_window: Some(Duration::from_secs(300)),
    },
    notify: NotifyConfig {
        channel: "telegram".to_string(),
        user: Some("user_123".to_string()),
        on_success: true,
        on_failure: true,
        on_attention: false,
    },
    state: serde_json::Value::Null,
    next_fire_at: Some(Utc::now() + Duration::from_secs(86400)),
    created_at: Utc::now(),
    updated_at: Utc::now(),
    last_run_at: None,
    run_count: 0,
    consecutive_failures: 0,
};

store.create_routine(&routine).await?;
```

### Trigger Types

```rust
// Cron trigger
Trigger::Cron { schedule: "0 0 * * *".to_string() }

// Event trigger
Trigger::Event { event_pattern: "job.completed".to_string() }

// System event trigger
Trigger::SystemEvent { event: "startup".to_string() }

// Webhook trigger
Trigger::Webhook { path: "/hooks/backup".to_string() }
```

### Listing Routines

```rust
// By user
let routines = store.list_routines("user_123").await?;

// All routines (admin)
let all = store.list_all_routines().await?;

// Enabled event triggers (for event matching)
let event_routines = store.list_event_routines().await?;

// Due cron routines (scheduler polls this)
let due = store.list_due_cron_routines().await?;

// Webhook by path
let webhook = store.get_webhook_routine_by_path("/hooks/backup").await?;
```

### Updating Routines

```rust
// Full update
routine.enabled = false;
store.update_routine(&routine).await?;

// Runtime state update (after firing)
store.update_routine_runtime(
    routine_id,
    Utc::now(),            // last_run_at
    Some(next_fire),       // next_fire_at
    42,                    // run_count
    0,                     // consecutive_failures
    &serde_json::json!({}), // state
).await?;
```

### Routine Runs

Track execution history:

```rust
use crate::agent::routine::RoutineRun;

// Record run starting
let run = RoutineRun {
    id: Uuid::new_v4(),
    routine_id,
    trigger_type: "cron".to_string(),
    trigger_detail: "0 0 * * *".to_string(),
    started_at: Utc::now(),
    status: RunStatus::Running,
    job_id: Some(job_id),
    completed_at: None,
    result_summary: None,
    tokens_used: None,
};
store.create_routine_run(&run).await?;

// Complete run
store.complete_routine_run(
    run_id,
    RunStatus::Success,
    Some("Backup completed successfully"),
    Some(1500), // tokens_used
).await?;

// List recent runs
let runs = store.list_routine_runs(routine_id, 10).await?;

// Count running (for guardrails)
let running = store.count_running_routine_runs(routine_id).await?;

// Batch load concurrent counts (multiple routines)
let counts = store.count_running_routine_runs_batch(&[id1, id2, id3]).await?;
// HashMap<Uuid, i64> — missing IDs default to 0

// Batch load last run status
let statuses = store.batch_get_last_run_status(&[id1, id2, id3]).await?;
// HashMap<Uuid, RunStatus>
```

---

## Estimation Snapshots

### Saving Snapshots

Capture estimation context for learning:

```rust
store.save_estimation_snapshot(
    job_id,
    "bug_fix",                    // category
    &["file_read", "file_write"], // tool_names
    estimated_cost,                // Decimal
    120,                           // estimated_time_secs
    estimated_value,               // Decimal
).await?;
```

### Updating with Actuals

```rust
store.update_estimation_actuals(
    snapshot_id,
    actual_cost,    // Decimal
    145,            // actual_time_secs
    Some(actual_value), // Optional<Decimal>
).await?;
```

**Schema**: `estimation_snapshots`: `id`, `job_id`, `category`, `tool_names` (TEXT[]), `estimated_cost`, `estimated_time_secs`, `estimated_value`, `actual_cost`, `actual_time_secs`, `actual_value`, `created_at`

**Use Case**: Feed into estimation system for continuous improvement based on historical accuracy.

---

## Query Patterns

### Job Queries

```rust
// Get job by ID
let job = store.get_job(job_id).await?;

// Get failure reason
let reason = store.get_agent_job_failure_reason(job_id).await?;

// List jobs (all)
let jobs = store.list_agent_jobs().await?;

// List by user
let jobs = store.list_agent_jobs_for_user("user_123").await?;

// Summary (all)
let summary = store.agent_job_summary().await?;

// Summary by user
let summary = store.agent_job_summary_for_user("user_123").await?;
// AgentJobSummary { total, pending, in_progress, completed, failed, stuck }
```

### Sandbox Job Queries

```rust
// Get by ID
let job = store.get_sandbox_job(job_id).await?;

// List (all)
let jobs = store.list_sandbox_jobs().await?;

// List by user
let jobs = store.list_sandbox_jobs_for_user("user_123").await?;

// Summary (all)
let summary = store.sandbox_job_summary().await?;

// Summary by user
let summary = store.sandbox_job_summary_for_user("user_123").await?;
// SandboxJobSummary { total, creating, running, completed, failed, interrupted }
```

### Event Queries

```rust
// All events for job
let events = store.list_job_events(job_id, None).await?;

// Last N events (most recent, ascending order)
let events = store.list_job_events(job_id, Some(50)).await?;
```

### Action Queries

```rust
// All actions for job (ordered by sequence)
let actions = store.get_job_actions(job_id).await?;
```

---

## Code Patterns

### Pattern 1: Store Initialization

```rust
use crate::history::Store;
use crate::config::DatabaseConfig;

// Create from config
let config = DatabaseConfig {
    url: "postgresql://...".to_string(),
    pool_size: 10,
    ssl_mode: "prefer".to_string(),
    backend: DatabaseBackend::Postgres,
};

let store = Store::new(&config).await?;

// Or from existing pool
let store = Store::from_pool(existing_pool);

// Run migrations
store.run_migrations().await?;
```

### Pattern 2: Transaction-Safe Job Updates

```rust
// Save job context (upsert by id)
store.save_job(&job_context).await?;

// Update status atomically
store.update_job_status(job_id, JobState::Completed, None).await?;

// Save actions (append-only)
for action in &actions {
    store.save_action(job_id, action).await?;
}

// Record LLM calls
store.record_llm_call(&LlmCallRecord {
    job_id: Some(job_id),
    conversation_id: None,
    provider: "anthropic",
    model: "claude-sonnet-4-20250514",
    input_tokens: 1000,
    output_tokens: 200,
    cost: Decimal::from_str("0.008").unwrap(),
    purpose: Some("execution"),
}).await?;
```

### Pattern 3: Analytics Dashboard

```rust
// Job metrics
let job_stats = store.get_job_stats().await?;
println!("Success rate: {:.1}%", job_stats.success_rate * 100.0);
println!("Avg duration: {:.1}s", job_stats.avg_duration_secs);
println!("Total cost: ${:.2}", job_stats.total_cost);

// Tool metrics
let tool_stats = store.get_tool_stats().await?;
for stat in &tool_stats {
    println!("{}: {} calls ({:.1}% success)",
        stat.tool_name,
        stat.total_calls,
        stat.success_rate * 100.0
    );
}

// Estimation accuracy
let accuracy = store.get_estimation_accuracy(None).await?;
println!("Cost error: {:.1}%", accuracy.cost_error_rate * 100.0);
println!("Time error: {:.1}%", accuracy.time_error_rate * 100.0);
println!("Samples: {}", accuracy.sample_count);
```

### Pattern 4: Routine Scheduler

```rust
// Poll for due cron routines
let due_routines = store.list_due_cron_routines().await?;

for routine in due_routines {
    // Check guardrails
    let running = store.count_running_routine_runs(routine.id).await?;
    if running >= routine.guardrails.max_concurrent as i64 {
        continue; // Skip — at max concurrency
    }

    // Execute routine...
    
    // Update runtime state
    store.update_routine_runtime(
        routine.id,
        Utc::now(),
        Some(calculate_next_fire(&routine.trigger)),
        routine.run_count + 1,
        if success { 0 } else { routine.consecutive_failures + 1 },
        &routine.state,
    ).await?;
}
```

### Pattern 5: Stale Job Cleanup

```rust
// On startup — mark orphaned sandbox jobs as interrupted
let count = store.cleanup_stale_sandbox_jobs().await?;
if count > 0 {
    tracing::info!("Marked {} stale sandbox jobs as interrupted", count);
}

// Get stuck agent jobs
let stuck_ids = store.get_stuck_jobs().await?;
for job_id in stuck_ids {
    // Attempt recovery or notification
    tracing::warn!("Job {} is stuck", job_id);
}
```

### Pattern 6: Batch Loading for Performance

```rust
// Batch load concurrent run counts (single query)
let routine_ids = vec![id1, id2, id3, id4, id5];
let counts = store.count_running_routine_runs_batch(&routine_ids).await?;
// More efficient than N individual queries

// Batch load last run statuses (single query with window function)
let statuses = store.batch_get_last_run_status(&routine_ids).await?;
// Uses: DISTINCT ON (routine_id) ... ORDER BY routine_id, started_at DESC
```

---

## Current Limitations

### PostgreSQL-Only

- **No libSQL support**: Analytics module is gated behind `#[cfg(feature = "postgres")]`
- **No hybrid search**: Unlike WorkspaceStore, history analytics don't have libSQL fallback
- **Migration dependency**: Requires `refinery` crate and `migrations/` directory

### Schema Constraints

- **JobContext transitions**: `transitions` field not persisted (loaded as empty Vec)
- **user_timezone**: TODO comment in `get_job()` — not persisted in `agent_jobs` table
- **Routine state**: Stored as JSON — no type safety or validation

### Performance

- **No pagination**: List methods return all rows (could be large for busy systems)
- **No indexes documented**: Query performance depends on database indexes (not shown in code)
- **Event streaming**: `list_job_events()` loads all into memory — no cursor-based pagination

---

## Key Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `conversations` | User sessions | `id`, `channel`, `user_id`, `thread_id`, `last_activity` |
| `conversation_messages` | Message history | `id`, `conversation_id`, `role`, `content` |
| `agent_jobs` | All jobs (agent + sandbox) | `id`, `title`, `status`, `source`, `user_id`, `created_at`, `started_at`, `completed_at`, `failure_reason`, `job_mode` |
| `job_actions` | Tool executions | `id`, `job_id`, `sequence_num`, `tool_name`, `input`, `output_raw`, `output_sanitized`, `success`, `duration_ms`, `cost` |
| `job_events` | Streaming events | `id`, `job_id`, `event_type`, `data` (JSONB) |
| `llm_calls` | LLM usage tracking | `id`, `job_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `cost` |
| `estimation_snapshots` | Learning data | `id`, `job_id`, `category`, `tool_names`, `estimated_*`, `actual_*` |
| `routines` | Automated tasks | `id`, `name`, `user_id`, `enabled`, `trigger_type`, `action_type`, `next_fire_at`, `run_count` |
| `routine_runs` | Execution history | `id`, `routine_id`, `trigger_type`, `status`, `started_at`, `completed_at`, `job_id` |
| `tool_failures` | Self-repair tracking | `tool_name`, `error_count`, `repaired_at` |
| `settings` | Per-user config | `user_id`, `key`, `value` |

---

## Related

- `src/history/mod.rs` — Module exports and type definitions
- `src/history/store.rs` — Store implementation (~1600 lines)
- `src/history/analytics.rs` — Analytics methods (~226 lines)
- `src/agent/routine.rs` — Routine data model
- `src/context/mod.rs` — `JobContext`, `ActionRecord`, `JobState`
- `migrations/` — PostgreSQL schema migrations
- `database-abstraction.md` — DB trait architecture
- `worker-jobs.md` — Worker/Container job execution

---

**Quality Note**: This document reflects the actual implementation as of 2026-03-26. Always verify against source code for critical changes.
