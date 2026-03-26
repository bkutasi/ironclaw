# Observability System

**Purpose**: Trait-based event and metric recording with pluggable backends for agent lifecycle observability

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Quick Reference

**Observer Trait** (core abstraction):
```rust
pub trait Observer: Send + Sync {
    fn record_event(&self, event: &ObserverEvent);
    fn record_metric(&self, metric: &ObserverMetric);
    fn flush(&self) {}  // Optional batch flush
    fn name(&self) -> &str;  // Backend identifier
}
```

**Available Backends**:
| Backend | Description | Use Case |
|---------|-------------|----------|
| `noop` | Zero overhead, discards everything | Default, production (disabled) |
| `log` | Emits structured events via `tracing` | Development, debugging |
| `multi` | Fan-out to multiple backends | Combined logging + telemetry |

**Configuration** (`ObservabilityConfig`):
```rust
ObservabilityConfig {
    backend: "none" | "noop" | "log"  // Unknown values → noop
}
```

---

## Architecture Overview

Ironclaw's observability system is a **trait-based, pluggable observer pattern** designed for recording agent lifecycle events and metrics with minimal overhead. The system decouples event emission from event consumption, allowing multiple backends to be swapped without changing the agent code.

```
┌─────────────────────────────────────────────────────────┐
│ Agent Runtime                                           │
│ (agent/, worker/, channels/)                            │
│                                                         │
│  record_event(ObserverEvent)                           │
│  record_metric(ObserverMetric)                         │
└────────────────────┬────────────────────────────────────┘
                     │ Arc<dyn Observer>
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Observer Trait (src/observability/traits.rs)            │
│ - record_event()                                        │
│ - record_metric()                                       │
│ - flush()                                               │
│ - name()                                                │
└────────────────────┬────────────────────────────────────┘
                     │ Implementation
         ┌───────────┼───────────┬────────────┐
         ▼           ▼           ▼            ▼
    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
    │ Noop   │  │ Log    │  │ Multi  │  │ Future │
    │Observer│  │Observer│  │Observer│  │ OTLP?  │
    └────────┘  └────────┘  └────────┘  └────────┘
        │           │           │
        │           ▼           │
        │    tracing::info!()  │
        │    tracing::debug!() │
        │                      │
        └───────────┬──────────┘
                    ▼
         ┌──────────────────┐
         │  Log Aggregator  │
         │  (stdout/files)  │
         └──────────────────┘
```

**Design Goals**:
1. **Zero overhead when disabled**: `NoopObserver` compiles to nothing with `#[inline(always)]`
2. **Pluggable backends**: Add new exporters (OpenTelemetry, Prometheus) by implementing `Observer`
3. **Thread-safe**: `Arc<dyn Observer>` allows cheap cloning across async tasks
4. **Structured data**: Events and metrics carry typed fields for downstream querying

---

## Tracing

### ObserverEvent Types

The system defines discrete lifecycle events that agents record at key points:

| Event | When Recorded | Fields |
|-------|---------------|--------|
| `AgentStart` | Agent begins processing | `provider`, `model` |
| `LlmRequest` | LLM request sent | `provider`, `model`, `message_count` |
| `LlmResponse` | LLM response received | `provider`, `model`, `duration`, `success`, `error_message` |
| `ToolCallStart` | Tool execution begins | `tool` |
| `ToolCallEnd` | Tool execution completes | `tool`, `duration`, `success` |
| `TurnComplete` | One reasoning turn finishes | (none) |
| `ChannelMessage` | Message sent/received on channel | `channel`, `direction` |
| `HeartbeatTick` | Heartbeat system tick | (none) |
| `AgentEnd` | Agent finishes processing | `duration`, `tokens_used` |
| `Error` | Error in component | `component`, `message` |

**Usage Pattern**:
```rust
// At agent start
observer.record_event(&ObserverEvent::AgentStart {
    provider: "openai".into(),
    model: "gpt-4".into(),
});

// Around LLM calls
let start = Instant::now();
observer.record_event(&ObserverEvent::LlmRequest {
    provider: "openai".into(),
    model: "gpt-4".into(),
    message_count: messages.len(),
});

let result = llm.chat(messages).await;
let duration = start.elapsed();

observer.record_event(&ObserverEvent::LlmResponse {
    provider: "openai".into(),
    model: "gpt-4".into(),
    duration,
    success: result.is_ok(),
    error_message: result.as_ref().err().map(|e| e.to_string()),
});
```

### LogObserver Implementation

The `LogObserver` emits events via the `tracing` crate:

```rust
impl Observer for LogObserver {
    fn record_event(&self, event: &ObserverEvent) {
        match event {
            ObserverEvent::LlmResponse {
                provider,
                model,
                duration,
                success,
                error_message,
            } => {
                tracing::info!(
                    provider,
                    model,
                    duration_ms = duration.as_millis() as u64,
                    success,
                    error = error_message.as_deref().unwrap_or(""),
                    "observer: llm.response"
                );
            }
            // ... other events
        }
    }
}
```

**Log Levels**:
- `info`: Normal lifecycle events (start, request, response, turn complete)
- `debug`: Metrics and heartbeat ticks
- `warn`: Errors

---

## Metrics

### ObserverMetric Types

Numeric measurements sampled during agent execution:

| Metric | Type | Description |
|--------|------|-------------|
| `RequestLatency(Duration)` | Histogram | Latency of individual requests |
| `TokensUsed(u64)` | Counter | Cumulative token consumption |
| `ActiveJobs(u64)` | Gauge | Current number of running jobs |
| `QueueDepth(u64)` | Gauge | Current message queue depth |

**Usage Pattern**:
```rust
// Record latency
observer.record_metric(&ObserverMetric::RequestLatency(duration));

// Track token usage
observer.record_metric(&ObserverMetric::TokensUsed(tokens));

// Monitor job system
observer.record_metric(&ObserverMetric::ActiveJobs(active_count));
observer.record_metric(&ObserverMetric::QueueDepth(queue_len));
```

### Metric Collection Strategy

Current implementation logs metrics via `tracing::debug!()`. Future backends could:
- Export to **Prometheus** via `prometheus-client` crate
- Export to **OpenTelemetry** via OTLP protocol
- Buffer and batch-export to reduce overhead

**Gauge vs Counter vs Histogram**:
- **Counter** (`TokensUsed`): Monotonically increasing, cumulative
- **Gauge** (`ActiveJobs`, `QueueDepth`): Point-in-time value, can go up/down
- **Histogram** (`RequestLatency`): Distribution of values, useful for percentiles

---

## Logging

### Structured Logging with tracing

The `LogObserver` leverages Rust's `tracing` ecosystem for structured logging:

```rust
// Fields become structured data in log output
tracing::info!(
    provider = "openai",
    model = "gpt-4",
    duration_ms = 150,
    success = true,
    "observer: llm.response"
);
```

**Output Format** (with tracing-subscriber):
```json
{
  "timestamp": "2026-03-26T18:00:00.000Z",
  "level": "INFO",
  "target": "ironclaw::observability::log",
  "fields": {
    "message": "observer: llm.response",
    "provider": "openai",
    "model": "gpt-4",
    "duration_ms": 150,
    "success": true
  }
}
```

### LogObserver Event Mapping

| Event | Log Level | Message Pattern |
|-------|-----------|-----------------|
| `AgentStart` | INFO | `observer: agent.start` |
| `LlmRequest` | INFO | `observer: llm.request` |
| `LlmResponse` | INFO | `observer: llm.response` |
| `ToolCallStart` | INFO | `observer: tool.start` |
| `ToolCallEnd` | INFO | `observer: tool.end` |
| `TurnComplete` | INFO | `observer: turn.complete` |
| `ChannelMessage` | INFO | `observer: channel.message` |
| `HeartbeatTick` | DEBUG | `observer: heartbeat.tick` |
| `AgentEnd` | INFO | `observer: agent.end` |
| `Error` | WARN | `observer: error` |
| Metrics | DEBUG | `observer: metric.*` |

---

## Dashboards

### Current State

As of 2026-03-26, Ironclaw does **not** have built-in dashboards or visualization. The observability system provides the foundation for future telemetry export.

### Future Dashboard Opportunities

With an OpenTelemetry or Prometheus backend, dashboards could show:

**Agent Performance**:
- P50/P95/P99 LLM response latency
- Token consumption over time
- Success/failure rates by provider

**Job System Health**:
- Active jobs over time
- Queue depth trends
- Job completion rate

**Tool Execution**:
- Tool call frequency
- Tool success rates
- Average tool execution time

**Error Tracking**:
- Error rate by component
- Error message clustering
- Mean time to recovery

### Recommended Stack

For production observability:
1. **OpenTelemetry Collector** → OTLP export from Rust
2. **Tempo/Jaeger** → Distributed tracing
3. **Prometheus** → Metrics storage
4. **Grafana** → Dashboards
5. **Loki** → Log aggregation (if not using tracing)

---

## Code Patterns

### Creating an Observer

```rust
use ironclaw::observability::{create_observer, ObservabilityConfig};

// From configuration
let config = ObservabilityConfig {
    backend: "log".into(),
};
let observer = create_observer(&config);

// observer: Box<dyn Observer>
```

### Custom Observer Implementation

```rust
use std::sync::Arc;
use ironclaw::observability::traits::{Observer, ObserverEvent, ObserverMetric};

struct MyCustomObserver {
    // Your state here
    endpoint: String,
}

impl Observer for MyCustomObserver {
    fn record_event(&self, event: &ObserverEvent) {
        // Handle event
        match event {
            ObserverEvent::LlmResponse { duration, success, .. } => {
                // Export to your telemetry system
            }
            // ...
        }
    }

    fn record_metric(&self, metric: &ObserverMetric) {
        // Handle metric
        match metric {
            ObserverMetric::TokensUsed(n) => {
                // Update counter
            }
            // ...
        }
    }

    fn flush(&self) {
        // Flush buffered data (for batch exporters)
    }

    fn name(&self) -> &str {
        "my-custom"
    }
}
```

### MultiObserver for Multiple Backends

```rust
use ironclaw::observability::{MultiObserver, LogObserver, NoopObserver};

let multi = MultiObserver::new(vec![
    Box::new(LogObserver),
    // Box::new(MyCustomObserver { ... }),
]);

// Events are dispatched to all observers
multi.record_event(&ObserverEvent::TurnComplete);
```

### Arc<dyn Observer> for Shared Access

```rust
use std::sync::Arc;
use ironclaw::observability::{Observer, LogObserver};

let observer: Arc<dyn Observer> = Arc::new(LogObserver);

// Cheaply clone for async tasks
let observer_clone = Arc::clone(&observer);
tokio::spawn(async move {
    observer_clone.record_event(&ObserverEvent::HeartbeatTick);
});
```

### Factory Pattern

```rust
// src/observability/mod.rs
pub fn create_observer(config: &ObservabilityConfig) -> Box<dyn Observer> {
    match config.backend.as_str() {
        "log" => Box::new(LogObserver),
        _ => Box::new(NoopObserver),  // Default fallback
    }
}
```

---

## 📂 Codebase References

**Core Observer Trait**:
- `src/observability/traits.rs` - `Observer` trait, `ObserverEvent`, `ObserverMetric` (143 lines)

**Backend Implementations**:
- `src/observability/noop.rs` - `NoopObserver` (zero overhead, default)
- `src/observability/log.rs` - `LogObserver` (tracing-based, dev/debug)
- `src/observability/multi.rs` - `MultiObserver` (fan-out to multiple backends)

**Module Entry Point**:
- `src/observability/mod.rs` - `ObservabilityConfig`, `create_observer()` factory (105 lines)

**Integration Points**:
- `src/agent/` - Agent lifecycle event recording
- `src/worker/` - Job execution metrics
- `src/channels/` - Channel message tracking

---

## Related Files

- `worker-jobs.md` - Background job system (emits job metrics)
- `agent-system.md` - Agent lifecycle (emits agent events)
- `channels-system.md` - Channel message routing (emits channel events)
