<!-- Context: architecture/concepts | Priority: high | Version: 1.0 | Updated: 2026-03-26 -->

# Hooks System

**Purpose**: Lifecycle interception and transformation system for agent operations

**Last Updated**: 2026-03-26

---

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Hook trait | `src/hooks/hook.rs` | Core trait for implementing hooks |
| Registry | `src/hooks/registry.rs` | Manages hook registration and execution |
| Bootstrap | `src/hooks/bootstrap.rs` | Loads bundled, plugin, and workspace hooks |
| Bundled hooks | `src/hooks/bundled.rs` | Declarative hook configs and built-in hooks |

**Hook Points** (6 total):
- `beforeInbound` — Before processing user message
- `beforeToolCall` — Before executing tool call
- `beforeOutbound` — Before sending response
- `onSessionStart` — When session starts
- `onSessionEnd` — When session ends
- `transformResponse` — Transform final response

**Priority**: Lower number = runs first (default: 100)

---

## Core Concept

The hooks system provides 6 well-defined interception points in the agent lifecycle where custom logic can:
- **Pass through** — Continue processing unchanged
- **Modify** — Transform content before continuing
- **Reject** — Stop processing entirely with a reason

Hooks are executed in **priority order** (lower number = higher priority). A `Reject` outcome stops the chain immediately. A `Modify` outcome chains the modification through subsequent hooks.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HookRegistry                              │
│  - Manages all registered hooks                              │
│  - Executes hooks in priority order                          │
│  - Handles timeout/failure modes                             │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼───────┐  ┌───────▼────────┐
│  Built-in      │  │  Plugin      │  │  Workspace     │
│  Hooks         │  │  Hooks       │  │  Hooks         │
│  (audit_log)   │  │  (WASM)      │  │  (JSON rules)  │
└────────────────┘  └──────────────┘  └────────────────┘
```

**Three hook sources**:
1. **Built-in hooks** — Ship with Ironclaw (e.g., `builtin.audit_log`)
2. **Plugin hooks** — Loaded from WASM tools/channels capabilities files
3. **Workspace hooks** — Declarative rules from `hooks/*.hook.json` files

---

## Lifecycle Hook Points

### BeforeInbound
**Trigger**: User message received, before processing

**Event data**:
```rust
HookEvent::Inbound {
    user_id: String,
    channel: String,
    content: String,      // ← Can be modified
    thread_id: Option<String>,
}
```

**Use cases**: Content filtering, redaction, validation, enrichment

---

### BeforeToolCall
**Trigger**: Tool about to execute, before execution

**Event data**:
```rust
HookEvent::ToolCall {
    tool_name: String,
    parameters: Value,    // ← Can be modified (JSON string)
    user_id: String,
    context: String,      // "chat" or job ID
}
```

**Use cases**: Parameter validation, tool access control, auditing

---

### BeforeOutbound
**Trigger**: Response ready, before sending to user

**Event data**:
```rust
HookEvent::Outbound {
    user_id: String,
    channel: String,
    content: String,      // ← Can be modified
    thread_id: Option<String>,
}
```

**Use cases**: Response formatting, redaction, compliance checks

---

### OnSessionStart
**Trigger**: New session created

**Event data**:
```rust
HookEvent::SessionStart {
    user_id: String,
    session_id: String,
}
```

**Use cases**: Session initialization, analytics, welcome messages

---

### OnSessionEnd
**Trigger**: Session pruned or expired

**Event data**:
```rust
HookEvent::SessionEnd {
    user_id: String,
    session_id: String,
}
```

**Use cases**: Cleanup, analytics, archival

---

### TransformResponse
**Trigger**: Final response transformation before completing turn

**Event data**:
```rust
HookEvent::ResponseTransform {
    user_id: String,
    thread_id: String,
    response: String,     // ← Can be modified
}
```

**Use cases**: Final formatting, branding, disclaimers

---

## Hook Execution Order

### Priority System
- **Lower number = runs first** (priority 10 runs before priority 100)
- **Default priority**: 100 for rule hooks, 300 for webhooks
- **Built-in audit hook**: Priority 25 (runs very early)

### Execution Flow
```
1. Registry receives event
2. Filters hooks matching event.hook_point()
3. Sorts by priority (ascending)
4. For each hook:
   a. Execute with timeout (default: 5s, max: 30s)
   b. If Reject → Stop chain, return error
   c. If Modify → Update event content, continue
   d. If error/timeout → Respect failure_mode
5. Return final outcome (ok/modified)
```

### Example Chain
```
Event: "hello" (priority order)
1. builtin.audit_log (prio 25) → ok()
2. high-prio hook (prio 10) → modify("hello-HIGH")
3. low-prio hook (prio 200) → modify("hello-HIGH-LOW")
Final: "hello-HIGH-LOW"
```

---

## Bootstrap Process

Hook bootstrapping happens at startup via `bootstrap_hooks()`:

```rust
pub async fn bootstrap_hooks(
    registry: &Arc<HookRegistry>,
    workspace: Option<&Arc<Workspace>>,
    wasm_tools_dir: &Path,
    wasm_channels_dir: &Path,
    active_tool_names: &[String],
    active_channel_names: &[String],
    dev_loaded_tool_names: &[String],
) -> HookBootstrapSummary
```

### Bootstrap Sequence

1. **Register built-in hooks**
   - `builtin.audit_log` (priority 25)
   - Logs all lifecycle events to `hooks::audit` trace target

2. **Load plugin hooks**
   - Scans WASM tools directory for capabilities files
   - Scans WASM channels directory for capabilities files
   - Extracts `hooks` section from each capabilities JSON
   - Registers declarative rules and webhooks

3. **Load workspace hooks**
   - Reads `hooks/hooks.json` and `hooks/*.hook.json` from workspace
   - Parses declarative hook bundles
   - Registers rules and webhooks

### Bootstrap Summary
```rust
HookBootstrapSummary {
    bundled_hooks: usize,      // Built-in hooks registered
    plugin_hooks: usize,       // Plugin-provided hooks
    workspace_hooks: usize,    // Workspace-provided hooks
    outbound_webhooks: usize,  // Webhook hooks
    errors: usize,             // Invalid configs skipped
}
```

---

## Hook Registry

### Core API

```rust
pub struct HookRegistry {
    hooks: RwLock<Vec<HookEntry>>,
}

impl HookRegistry {
    pub fn new() -> Self;
    
    pub async fn register(&self, hook: Arc<dyn Hook>);
    
    pub async fn register_with_priority(
        &self, 
        hook: Arc<dyn Hook>, 
        priority: u32
    );
    
    pub async fn unregister(&self, name: &str) -> bool;
    
    pub async fn list(&self) -> Vec<String>;
    
    pub async fn run(&self, event: &HookEvent) 
        -> Result<HookOutcome, HookError>;
}
```

### Thread Safety
- Uses `tokio::sync::RwLock` for async-safe concurrent access
- Clones hooks before execution to avoid holding lock during timeout
- Supports concurrent `register`/`unregister`/`run` operations

---

## Code Patterns

### Implementing a Custom Hook

```rust
use async_trait::async_trait;
use ironclaw::hooks::{
    Hook, HookContext, HookError, HookEvent, HookOutcome, HookPoint,
    HookFailureMode,
};
use std::time::Duration;

struct MyCustomHook;

#[async_trait]
impl Hook for MyCustomHook {
    fn name(&self) -> &str {
        "my-custom-hook"
    }

    fn hook_points(&self) -> &[HookPoint] {
        &[HookPoint::BeforeInbound, HookPoint::BeforeOutbound]
    }

    fn failure_mode(&self) -> HookFailureMode {
        HookFailureMode::FailOpen  // Continue on error
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(3)
    }

    async fn execute(
        &self,
        event: &HookEvent,
        _ctx: &HookContext,
    ) -> Result<HookOutcome, HookError> {
        // Extract content
        let content = match event {
            HookEvent::Inbound { content, .. } => content,
            HookEvent::Outbound { content, .. } => content,
            _ => return Ok(HookOutcome::ok()),
        };

        // Apply transformation
        if content.contains("secret") {
            return Ok(HookOutcome::modify(
                content.replace("secret", "[REDACTED]")
            ));
        }

        Ok(HookOutcome::ok())
    }
}

// Register with registry
let registry = Arc::new(HookRegistry::new());
registry
    .register_with_priority(Arc::new(MyCustomHook), 50)
    .await;
```

### Declarative Rule Hook (JSON)

```json
{
  "name": "redact-secrets",
  "points": ["beforeInbound", "beforeOutbound"],
  "priority": 50,
  "failure_mode": "fail_open",
  "timeout_ms": 2000,
  "when_regex": "secret|password|token",
  "replacements": [
    {
      "pattern": "(secret|password|token)",
      "replacement": "[REDACTED]"
    }
  ],
  "prepend": null,
  "append": null
}
```

### Declarative Webhook Hook (JSON)

```json
{
  "name": "slack-notify",
  "points": ["onSessionStart", "onSessionEnd"],
  "url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
  "headers": {
    "Content-Type": "application/json"
  },
  "timeout_ms": 2000,
  "priority": 300,
  "max_in_flight": 32
}
```

### Using Hook Outcomes

```rust
// Pass through unchanged
Ok(HookOutcome::ok())

// Modify content
Ok(HookOutcome::modify("new content".to_string()))

// Reject with reason
Ok(HookOutcome::reject("Content violates policy"))

// Error handling
Err(HookError::ExecutionFailed { reason: "DB connection failed".into() })
Err(HookError::Timeout { timeout: Duration::from_secs(5) })
Err(HookError::Rejected { reason: "Blocked by policy".into() })
```

---

## Failure Modes

### FailOpen (Default)
On error or timeout, continue processing as if hook returned `ok()`.

**Use when**: Hook is non-critical (logging, analytics, optional enrichment)

```rust
fn failure_mode(&self) -> HookFailureMode {
    HookFailureMode::FailOpen
}
```

### FailClosed
On error or timeout, reject the event entirely.

**Use when**: Hook is critical (security checks, compliance validation)

```rust
fn failure_mode(&self) -> HookFailureMode {
    HookFailureMode::FailClosed
}
```

---

## Timeout Handling

- **Default timeout**: 5 seconds
- **Maximum timeout**: 30,000 ms (30 seconds)
- **Timeout behavior**: Respects `failure_mode` (fail-open or fail-closed)
- **Webhook default**: 2,000 ms (2 seconds)

```rust
fn timeout(&self) -> Duration {
    Duration::from_secs(5)  // Default
}

// Or in declarative config:
"timeout_ms": 2000
```

---

## Security Considerations

### Webhook URL Validation
- **HTTPS required** — HTTP URLs rejected
- **No credentials in URL** — Username/password not allowed
- **Private IPs blocked** — localhost, 127.x.x.x, 10.x.x.x, etc.
- **Internal hosts blocked** — `.localhost`, `host.docker.internal`, cloud metadata endpoints
- **DNS rebinding protection** — Runtime validation of resolved IPs

### Restricted Headers
Webhooks cannot set these headers:
- `Host`, `Authorization`, `Cookie`, `Proxy-Authorization`
- `Forwarded`, `X-Real-IP`, `Transfer-Encoding`, `Connection`
- Any `X-Forwarded-*` header

### Concurrency Limits
- **Default max in-flight**: 32 concurrent webhook deliveries
- Excess deliveries are dropped with warning log

---

## Testing Patterns

### Unit Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::{HookRegistry, HookEvent, HookOutcome};

    fn test_event() -> HookEvent {
        HookEvent::Inbound {
            user_id: "user-1".into(),
            channel: "test".into(),
            content: "hello".into(),
            thread_id: None,
        }
    }

    #[tokio::test]
    async fn test_hook_modifies_content() {
        let registry = HookRegistry::new();
        
        // Register test hook
        registry
            .register_with_priority(
                Arc::new(ModifyHook { suffix: "-MODIFIED".into() }),
                100,
            )
            .await;

        let result = registry.run(&test_event()).await.unwrap();
        match result {
            HookOutcome::Continue { modified: Some(m) } => {
                assert_eq!(m, "hello-MODIFIED");
            }
            other => panic!("Expected modification, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_reject_stops_chain() {
        let registry = HookRegistry::new();
        
        registry
            .register(Arc::new(RejectHook {
                name: "blocker".into(),
                reason: "blocked".into(),
            }))
            .await;

        let result = registry.run(&test_event()).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            HookError::Rejected { reason } if reason == "blocked"
        ));
    }
}
```

---

## Related

- `src/hooks/hook.rs` — Core trait and types
- `src/hooks/registry.rs` — Registry implementation with tests
- `src/hooks/bundled.rs` — Declarative hook configs
- `src/hooks/bootstrap.rs` — Startup bootstrap process
- `.opencode/context/core/architecture/concepts/config-precedence.md`

---

## Quality Notes

> ⚠️ **Verification Required**: This document was generated from source code analysis. Verify against actual runtime behavior before treating as authoritative reference.

**Known implementation details verified**:
- ✅ 6 hook points defined in `hook.rs`
- ✅ Priority ordering (lower = first) in `registry.rs`
- ✅ Fail-open/fail-closed modes in `hook.rs`
- ✅ Bootstrap sequence in `bootstrap.rs`
- ✅ Webhook security validation in `bundled.rs`
- ✅ Timeout handling (5s default, 30s max) in `bundled.rs`
