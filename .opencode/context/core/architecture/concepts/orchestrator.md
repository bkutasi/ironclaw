# Orchestrator System

**Purpose**: Manage sandboxed worker containers with secure LLM proxying, lifecycle management, and job isolation

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Quick Reference

**Default Port**: `50051` (configurable via `ORCHESTRATOR_PORT` env var)

**Core Components**:
- `ContainerJobManager` — Container lifecycle (create, stop, monitor)
- `TokenStore` — Per-job bearer tokens (ephemeral, in-memory)
- `OrchestratorApi` — Internal HTTP API for worker communication
- `SandboxReaper` — Orphaned container cleanup

**Security Model**:
- Per-job bearer tokens (64-char hex, cryptographically random)
- Token scoped to specific `job_id` (Job A token cannot access Job B endpoints)
- Credential grants isolated per-job (revoked on container cleanup)
- Constant-time token comparison (prevents timing attacks)

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                              │
│                                                              │
│  Internal API (default :50051, configurable via env var)     │
│    POST /worker/{id}/llm/complete                           │
│    POST /worker/{id}/llm/complete_with_tools                │
│    GET  /worker/{id}/job                                    │
│    GET  /worker/{id}/credentials                            │
│    POST /worker/{id}/status                                 │
│    POST /worker/{id}/complete                               │
│    POST /worker/{id}/event                                  │
│    GET  /worker/{id}/prompt                                 │
│                                                              │
│  ContainerJobManager                                         │
│    create_job() -> container + token                         │
│    stop_job()                                                │
│    list_jobs()                                               │
│    complete_job()                                            │
│                                                              │
│  TokenStore                                                  │
│    per-job bearer tokens (in-memory only)                   │
│    per-job credential grants (in-memory only)               │
│                                                              │
│  SandboxReaper (background task)                            │
│    scans every 5 minutes                                     │
│    reaps containers older than 10 minutes with no active job │
└─────────────────────────────────────────────────────────────┘
```

**Network Binding**:
- **Linux**: Binds to `0.0.0.0:50051` (containers reach host via `172.17.0.1`)
- **macOS/Windows**: Binds to `127.0.0.1:50051` (Docker Desktop routes via VM)

---

## Container Lifecycle

### States

```rust
pub enum ContainerState {
    Creating,   // Container being created
    Running,    // Container is running
    Stopped,    // Container stopped (normal completion)
    Failed,     // Container failed
}
```

### Job Modes

```rust
pub enum JobMode {
    Worker,     // Standard IronClaw worker with proxied LLM calls
    ClaudeCode, // Claude Code bridge (spawns `claude` CLI directly)
}
```

### Creation Flow

1. **Generate Token**: `TokenStore::create_token(job_id)` creates 64-char hex token
2. **Store Grants**: `TokenStore::store_grants(job_id, grants)` stores credential mappings
3. **Create Handle**: `ContainerHandle` inserted into manager's HashMap
4. **Build Config**: Docker container config with:
   - Environment: `IRONCLAW_WORKER_TOKEN`, `IRONCLAW_JOB_ID`, `IRONCLAW_ORCHESTRATOR_URL`
   - Volume mounts: Validated to stay within `~/.ironclaw/projects/`
   - Resource limits: Memory (2GB worker / 4GB Claude Code), CPU shares
   - Security: `no-new-privileges:true`, `cap_drop: ALL`, `cap_add: CHOWN`
5. **Start Container**: Docker container created and started
6. **Update Handle**: Container ID stored, state set to `Running`

### Cleanup Flow

**Normal Completion** (`complete_job`):
1. Store `CompletionResult` in handle
2. Stop container (5 second grace)
3. Remove container (force)
4. Revoke token (also revokes credential grants)
5. Keep handle in memory (for result retrieval)
6. Caller invokes `cleanup_job()` to remove handle from memory

**Manual Stop** (`stop_job`):
1. Stop container (10 second grace)
2. Remove container (force)
3. Revoke token
4. Update state to `Stopped`

**Reaper Cleanup** (orphaned containers):
1. Scan Docker for containers with `ironclaw.job_id` label
2. Check if job is active in `ContextManager`
3. If job is terminal/missing AND container age > threshold (10 min):
   - Try `job_manager.stop_job()` (handles token revocation)
   - Fall back to direct Docker API if handle not in memory
   - Log cleanup action

### Bind Mount Security

Project directories are validated to prevent directory traversal:

```rust
fn validate_bind_mount_path(
    dir: &Path,
    job_id: Uuid,
) -> Result<PathBuf, OrchestratorError> {
    // Canonicalize path
    // Verify it starts with ~/.ironclaw/projects/
    // Return canonical path or error
}
```

**TOCTOU Note**: Time-of-check/time-of-use gap exists between validation and Docker bind mount. Acceptable in single-tenant design where user controls filesystem.

---

## Worker Communication Protocol

### Authentication

All `/worker/{job_id}/...` endpoints require:
```
Authorization: Bearer <token>
```

Token validation:
- Extracted from path (`/worker/{uuid}/...`)
- Constant-time comparison via `subtle::ConstantTimeEq`
- Returns `401 Unauthorized` if invalid or job mismatch

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/worker/{id}/job` | GET | Get job description (title, task, project_dir) |
| `/worker/{id}/llm/complete` | POST | Proxy LLM completion request |
| `/worker/{id}/llm/complete_with_tools` | POST | Proxy LLM completion with tools |
| `/worker/{id}/status` | POST | Report worker status (iteration, message) |
| `/worker/{id}/complete` | POST | Report job completion (success/failure) |
| `/worker/{id}/event` | POST | Broadcast job event (message, tool_use, result, reasoning) |
| `/worker/{id}/prompt` | GET | Get queued follow-up prompt (Claude Code mode) |
| `/worker/{id}/credentials` | GET | Get decrypted credentials for granted secrets |
| `/health` | GET | Health check (no auth required) |

### Event Types

Workers broadcast events via `/worker/{id}/event`:

```json
{
  "event_type": "message",
  "data": { "role": "assistant", "content": "..." }
}
```

Supported event types:
- `message` — Chat message from worker
- `tool_use` — Tool invocation (includes tool_name, input)
- `tool_result` — Tool output (includes tool_name, output)
- `result` — Job completion status
- `reasoning` — Reasoning trace (narrative, decisions)
- Unknown types → fallback to `JobStatus`

### Credential Injection

Workers request credentials via `/worker/{id}/credentials`:

1. Worker authenticates with job token
2. Orchestrator looks up credential grants for job
3. Decrypts each secret via `SecretsStore::get_decrypted()`
4. Returns array of `{ env_var, value }` pairs
5. Records usage audit trail

**Security**: Secrets never logged, never persisted, only served to authenticated container.

---

## API Patterns

### Setup and Initialization

```rust
// In app initialization
let orchestrator_setup = setup_orchestrator(
    &config,
    &llm,
    Some(&db),
    Some(&secrets_store),
).await;

// Returns:
// - container_job_manager: Option<Arc<ContainerJobManager>>
// - job_event_tx: Option<broadcast::Sender<(Uuid, String, AppEvent)>>
// - prompt_queue: Arc<Mutex<HashMap<Uuid, VecDeque<PendingPrompt>>>>
// - docker_status: DockerStatus
```

### Creating a Job

```rust
let token = job_manager
    .create_job(
        job_id,
        "Build feature X",
        Some(PathBuf::from("/path/to/project")),
        JobMode::Worker,
        vec![CredentialGrant {
            secret_name: "github_token".to_string(),
            env_var: "GITHUB_TOKEN".to_string(),
        }],
    )
    .await?;

// Pass token to container via IRONCLAW_WORKER_TOKEN env var
```

### Broadcasting Job Events

```rust
// Orchestrator receives event from worker
// Converts to AppEvent and broadcasts via job_event_tx
// Web gateway subscribes to broadcast channel for SSE

let app_event = match payload.event_type.as_str() {
    "message" => AppEvent::JobMessage { job_id, role, content },
    "tool_use" => AppEvent::JobToolUse { job_id, tool_name, input },
    "result" => AppEvent::JobResult { job_id, status, session_id, .. },
    // ... etc
};

let _ = tx.send((job_id, user_id, app_event));
```

### Reaper Configuration

```rust
let reaper_config = ReaperConfig {
    scan_interval: Duration::from_secs(300),      // 5 minutes
    orphan_threshold: Duration::from_secs(600),   // 10 minutes
    container_label: "ironclaw.job_id".to_string(),
};

let reaper = SandboxReaper::new(
    job_manager.clone(),
    context_manager.clone(),
    reaper_config,
).await?;

tokio::spawn(reaper.run());
```

---

## Code Patterns

### Job Mode Detection

```rust
let memory_mb = match mode {
    JobMode::ClaudeCode => self.config.claude_code_memory_limit_mb, // 4096
    JobMode::Worker => self.config.memory_limit_mb,                  // 2048
};
```

### Token Validation Middleware

```rust
pub async fn worker_auth_middleware(
    State(token_store): State<TokenStore>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let job_id = extract_job_id_from_path(&path).ok_or(StatusCode::BAD_REQUEST)?;
    let token = extract_bearer_token(&request).ok_or(StatusCode::UNAUTHORIZED)?;
    
    if !token_store.validate(job_id, token).await {
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    Ok(next.run(request).await)
}
```

### Container Configuration

```rust
let host_config = HostConfig {
    binds: Some(binds),  // Validated bind mounts
    memory: Some((memory_mb * 1024 * 1024) as i64),
    cpu_shares: Some(cpu_shares as i64),
    network_mode: Some("bridge".to_string()),
    extra_hosts: Some(vec!["host.docker.internal:host-gateway".to_string()]),
    cap_drop: Some(vec!["ALL".to_string()]),
    cap_add: Some(vec!["CHOWN".to_string()]),
    security_opt: Some(vec!["no-new-privileges:true".to_string()]),
    tmpfs: Some([("/tmp".to_string(), "size=512M".to_string())].into_iter().collect()),
    ..Default::default()
};
```

### Job Owner Caching (for SSE scoping)

```rust
// At job creation: populate cache
state.job_owner_cache
    .write()
    .unwrap()
    .insert(job_id, user_id.clone());

// At event broadcast: check cache first
let cached_uid = state.job_owner_cache.read().unwrap().get(&job_id).cloned();
let user_id = match cached_uid {
    Some(uid) => uid,
    None => {
        // Cache miss: fall back to DB lookup
        let uid = store.get_sandbox_job(job_id).await?.map(|j| j.user_id);
        // Populate cache
        state.job_owner_cache.write().unwrap().insert(job_id, uid.clone());
        uid.unwrap_or_default()
    }
};
```

---

## Related

- `core/architecture/concepts/docker-postgres.md` — Docker infrastructure
- `core/standards/code-quality.md` — Error handling patterns
- `project-intelligence/technical-domain.md` — Overall architecture

---

**Source Files**:
- `src/orchestrator/mod.rs` — Module entry point, setup
- `src/orchestrator/job_manager.rs` — Container lifecycle
- `src/orchestrator/api.rs` — HTTP API handlers
- `src/orchestrator/auth.rs` — Token authentication
- `src/orchestrator/reaper.rs` — Orphan cleanup
