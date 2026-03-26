# Evaluation Framework

**Purpose**: Quality assessment and success evaluation for completed jobs using metrics tracking and rule-based scoring

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge is harvested from source code analysis. Requires validation against actual behavior. Do not treat as authoritative reference without verification.

---

## Quick Reference

**Core Components**:
- `MetricsCollector` — Tracks action success/failure rates, timing, costs per tool
- `SuccessEvaluator` — Evaluates job completion quality via trait interface
- `RuleBasedEvaluator` — Reference implementation with configurable thresholds
- `QualityMetrics` — Aggregated metrics data structure
- `EvaluationResult` — Structured evaluation output with scoring

**Evaluation Flow**: Collect metrics during execution → Apply success criteria → Generate scored result with issues/suggestions

**Key Metrics**: Success rate, total actions, execution time, cost, error categorization, per-tool breakdown

---

## Core Concept

Ironclaw's evaluation framework provides systematic quality assessment for completed jobs. It operates in two phases:

1. **Metrics Collection** — During job execution, track every action's success/failure, duration, cost, and error type
2. **Success Evaluation** — After completion, analyze collected metrics against configurable thresholds to determine success/failure with quality scoring

The framework separates concerns: `MetricsCollector` handles data gathering, while `SuccessEvaluator` (trait) defines evaluation strategies. The default `RuleBasedEvaluator` uses configurable thresholds for action success rates and failure counts.

```text
┌─────────────────────────────────────────────────────────────┐
│ Job Execution                                               │
│                                                             │
│  Action 1 (success) ──┐                                     │
│  Action 2 (failure) ──┼──► MetricsCollector.record_*()      │
│  Action 3 (success) ──┘                                     │
│                                                             │
│  Tracks:                                                    │
│  • Total/successful/failed actions                          │
│  • Duration per action                                      │
│  • Cost per action (optional)                               │
│  • Error categorization                                     │
│  • Per-tool breakdown                                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Post-Job Evaluation                                         │
│                                                             │
│  SuccessEvaluator.evaluate(job, actions, output)            │
│                                                             │
│  Checks:                                                    │
│  • Action success rate vs threshold (default 80%)           │
│  • Failure count vs max (default 3)                         │
│  • Critical/fatal error detection                           │
│  • Job state validation (Completed/Submitted)               │
│                                                             │
│  Output: EvaluationResult                                   │
│  • success: bool                                            │
│  • confidence: f64 (0-1)                                    │
│  • quality_score: u32 (0-100)                               │
│  • issues: Vec<String>                                      │
│  • suggestions: Vec<String>                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### Components

**Metrics Module** (`src/evaluation/metrics.rs`):
- `QualityMetrics` — Aggregated metrics data structure
- `ToolMetrics` — Per-tool breakdown (calls, successes, failures, avg time, cost)
- `MetricsCollector` — Records actions, categorizes errors, generates summaries
- `MetricsSummary` — High-level report (most used/failed tools, top errors)

**Success Module** (`src/evaluation/success.rs`):
- `EvaluationResult` — Structured evaluation output
- `SuccessEvaluator` — Trait for evaluation strategies
- `RuleBasedEvaluator` — Reference implementation with configurable thresholds

### Module Structure

```rust
// src/evaluation/mod.rs
mod metrics;
mod success;

pub use metrics::{MetricsCollector, QualityMetrics};
pub use success::{EvaluationResult, SuccessEvaluator};
```

---

## Metrics System

### QualityMetrics Structure

```rust
pub struct QualityMetrics {
    pub total_actions: u64,                    // Total actions taken
    pub successful_actions: u64,               // Successful actions
    pub failed_actions: u64,                   // Failed actions
    pub total_time: Duration,                  // Total execution time
    pub total_cost: Decimal,                   // Total cost (token usage)
    pub tool_metrics: HashMap<String, ToolMetrics>,  // Per-tool breakdown
    pub error_types: HashMap<String, u64>,     // Error categorization counts
}
```

### ToolMetrics Structure

```rust
pub struct ToolMetrics {
    pub calls: u64,           // Total calls to this tool
    pub successes: u64,       // Successful calls
    pub failures: u64,        // Failed calls
    pub total_time: Duration, // Cumulative execution time
    pub avg_time: Duration,   // Average time per call
    pub total_cost: Decimal,  // Cumulative cost
}
```

**Methods**:
- `success_rate()` — Returns `successes / calls` as f64 (0.0 if no calls)

### MetricsCollector API

```rust
pub struct MetricsCollector {
    metrics: QualityMetrics,
}

impl MetricsCollector {
    pub fn new() -> Self;                              // Create new collector
    pub fn record_success(&mut self, tool_name: &str, duration: Duration, cost: Option<Decimal>);
    pub fn record_failure(&mut self, tool_name: &str, error: &str, duration: Duration);
    pub fn metrics(&self) -> &QualityMetrics;          // Get current metrics
    pub fn success_rate(&self) -> f64;                 // Overall success rate
    pub fn tool_metrics(&self, tool_name: &str) -> Option<&ToolMetrics>;
    pub fn reset(&mut self);                           // Clear all metrics
    pub fn summary(&self) -> MetricsSummary;           // Generate summary report
}
```

### Error Categorization

Errors are automatically categorized into types for analysis:

| Category | Trigger Patterns |
|----------|------------------|
| `timeout` | "timeout", "TIMEOUT" |
| `rate_limit` | "rate limit", "Rate Limit" |
| `auth` | "auth", "unauthorized", "Unauthorized" |
| `not_found` | "not found", "404", "HTTP 404" |
| `invalid_input` | "invalid", "parameter", "Invalid JSON" |
| `network` | "network", "connection", "connection refused" |
| `unknown` | Anything else |

**Implementation**: `categorize_error()` function performs case-insensitive pattern matching.

### MetricsSummary Structure

```rust
pub struct MetricsSummary {
    pub total_actions: u64,
    pub success_rate: f64,
    pub total_time: Duration,
    pub total_cost: Decimal,
    pub most_used_tool: Option<String>,      // Tool with most calls
    pub most_failed_tool: Option<String>,    // Tool with most failures
    pub top_errors: Vec<(String, u64)>,      // Top 3 error types with counts
}
```

---

## Success Evaluation

### EvaluationResult Structure

```rust
pub struct EvaluationResult {
    pub success: bool,              // Pass/fail determination
    pub confidence: f64,            // Confidence in evaluation (0-1)
    pub reasoning: String,          // Detailed explanation
    pub issues: Vec<String>,        // Specific problems found
    pub suggestions: Vec<String>,   // Improvement recommendations
    pub quality_score: u32,         // Score 0-100
}
```

**Constructor Methods**:
- `EvaluationResult::success(reasoning, quality_score)` — Create passing result (confidence: 0.9, issues: [])
- `EvaluationResult::failure(reasoning, issues)` — Create failing result (confidence: 0.9, quality_score: 0)

### SuccessEvaluator Trait

```rust
#[async_trait]
pub trait SuccessEvaluator: Send + Sync {
    async fn evaluate(
        &self,
        job: &JobContext,
        actions: &[ActionRecord],
        output: Option<&str>,
    ) -> Result<EvaluationResult, EvaluationError>;
}
```

**Parameters**:
- `job` — JobContext with state and metadata
- `actions` — Slice of ActionRecord with success/failure status
- `output` — Optional job output text for analysis

---

## RuleBasedEvaluator

### Configuration

```rust
struct RuleBasedEvaluator {
    min_action_success_rate: f64,   // Default: 0.8 (80%)
    max_failures: u32,               // Default: 3
}
```

**Builder Methods**:
- `RuleBasedEvaluator::new()` — Create with defaults
- `.with_min_success_rate(rate: f64)` — Customize success rate threshold
- `.with_max_failures(max: u32)` — Customize max failure count

### Evaluation Logic

The evaluator checks these criteria in order:

1. **Empty Actions Check**
   - If `actions.is_empty()` → Immediate failure with issue "No actions were taken"

2. **Success Rate Threshold**
   - Calculate: `successful / total` as percentage
   - If below `min_action_success_rate` → Add issue: "Action success rate X% below threshold Y%"

3. **Failure Count Limit**
   - Count failures in actions
   - If exceeds `max_failures` → Add issue: "Too many failures: X (max Y)"

4. **Critical Error Detection**
   - Scan failed actions for "critical" or "fatal" in error messages
   - If found → Add issue: "Critical error in {tool}: {error}"

5. **Job State Validation**
   - Check `job.state` is `Completed` or `Submitted`
   - If neither → Add issue: "Job not in completed state: {state}"

### Quality Score Calculation

**With No Issues** (success case):
```rust
let base_score = (success_rate * 80.0) as u32;
let completion_bonus = if job.state == JobState::Completed { 20 } else { 0 };
quality_score = (base_score + completion_bonus).min(100);
```

- 100% success rate + Completed state = 100 points (80 + 20 bonus)
- 100% success rate + Submitted state = 80 points (no bonus)
- 80% success rate + Completed state = 84 points (64 + 20)

**With Issues** (failure case):
```rust
quality_score = ((success_rate * 50.0) as u32).min(50);
```

- Capped at 50 maximum when issues exist
- Scales with success rate but penalized

### Default Suggestions for Failures

When evaluation fails, these suggestions are automatically included:
- "Review failed actions for common patterns"
- "Consider adjusting retry logic"

---

## Running Evaluations

### Basic Usage

```rust
use crate::evaluation::{MetricsCollector, SuccessEvaluator};
use crate::evaluation::success::RuleBasedEvaluator;
use std::time::Duration;
use rust_decimal_macros::dec;

// 1. Create metrics collector
let mut collector = MetricsCollector::new();

// 2. Record actions during execution
collector.record_success("read_file", Duration::from_millis(50), Some(dec!(0.001)));
collector.record_success("write_file", Duration::from_millis(120), Some(dec!(0.002)));
collector.record_failure("shell", "timeout error", Duration::from_secs(30));

// 3. Get summary
let summary = collector.summary();
println!("Success rate: {:.1}%", summary.success_rate * 100.0);
println!("Most used tool: {:?}", summary.most_used_tool);
println!("Top errors: {:?}", summary.top_errors);

// 4. Evaluate job success
let evaluator = RuleBasedEvaluator::new();
let result = evaluator.evaluate(&job_context, &actions, None).await?;

if result.success {
    println!("Job passed evaluation (score: {})", result.quality_score);
} else {
    println!("Job failed: {}", result.reasoning);
    for issue in &result.issues {
        println!("  - {}", issue);
    }
}
```

### Custom Thresholds

```rust
// Stricter evaluation: 90% success rate required, max 1 failure
let evaluator = RuleBasedEvaluator::new()
    .with_min_success_rate(0.9)
    .with_max_failures(1);
```

### Integration Points

**Where Metrics Are Collected**:
- Tool execution wrappers record success/failure with timing
- Cost tracking from LLM provider responses
- Error messages passed to `record_failure()` for categorization

**Where Evaluation Occurs**:
- Post-job completion analysis
- Self-repair loop decision making (determine if job needs retry)
- Analytics and reporting pipelines

---

## Code Patterns

### Recording Tool Execution

```rust
use crate::evaluation::MetricsCollector;
use std::time::Duration;
use rust_decimal::Decimal;

async fn execute_tool_with_metrics(
    collector: &mut MetricsCollector,
    tool_name: &str,
    // ... other params
) -> Result<ToolOutput, ToolError> {
    let start = Instant::now();
    
    match tool.execute(params, context).await {
        Ok(output) => {
            let duration = start.elapsed();
            let cost = output.token_cost; // Optional
            collector.record_success(tool_name, duration, cost);
            Ok(output)
        }
        Err(e) => {
            let duration = start.elapsed();
            collector.record_failure(tool_name, &e.to_string(), duration);
            Err(e)
        }
    }
}
```

### Custom Evaluator Implementation

```rust
use async_trait::async_trait;
use crate::evaluation::{EvaluationResult, SuccessEvaluator};
use crate::error::EvaluationError;

struct CustomEvaluator {
    custom_threshold: f64,
}

#[async_trait]
impl SuccessEvaluator for CustomEvaluator {
    async fn evaluate(
        &self,
        job: &JobContext,
        actions: &[ActionRecord],
        output: Option<&str>,
    ) -> Result<EvaluationResult, EvaluationError> {
        // Custom evaluation logic
        let success = /* your logic */;
        
        if success {
            Ok(EvaluationResult::success("Custom evaluation passed", 85))
        } else {
            Ok(EvaluationResult::failure("Custom evaluation failed", vec!["Issue 1".into()]))
        }
    }
}
```

### Analyzing Error Patterns

```rust
let summary = collector.summary();

// Identify problematic tools
if let Some(problematic_tool) = &summary.most_failed_tool {
    println!("Tool '{}' has the most failures", problematic_tool);
    
    if let Some(metrics) = collector.tool_metrics(problematic_tool) {
        println!("  Success rate: {:.1}%", metrics.success_rate() * 100.0);
        println!("  Avg execution time: {:?}", metrics.avg_time);
    }
}

// Analyze error distribution
for (error_type, count) in &summary.top_errors {
    println!("{} errors: {} occurrences", error_type, count);
}
```

### Generating Reports

```rust
fn generate_job_report(collector: &MetricsCollector, result: &EvaluationResult) -> String {
    let summary = collector.summary();
    
    format!(
        r#"Job Evaluation Report
=====================

Overall Result: {}
Quality Score: {}/100
Confidence: {:.0}%

Performance Metrics:
- Total Actions: {}
- Success Rate: {:.1}%
- Total Time: {:?}
- Total Cost: ${}

Tool Performance:
- Most Used: {:?}
- Most Failed: {:?}

Top Errors:
{}

Issues Found:
{}

Suggestions:
{}"#,
        if result.success { "✅ PASSED" } else { "❌ FAILED" },
        result.quality_score,
        result.confidence * 100.0,
        summary.total_actions,
        summary.success_rate * 100.0,
        summary.total_time,
        summary.total_cost,
        summary.most_used_tool.as_deref().unwrap_or("N/A"),
        summary.most_failed_tool.as_deref().unwrap_or("N/A"),
        summary.top_errors.iter()
            .map(|(e, c)| format!("  - {}: {} occurrences", e, c))
            .collect::<Vec<_>>()
            .join("\n"),
        if result.issues.is_empty() { "  None".into() } 
        else { result.issues.iter().map(|i| format!("  - {}", i)).collect::<Vec<_>>().join("\n") },
        if result.suggestions.is_empty() { "  None".into() } 
        else { result.suggestions.iter().map(|s| format!("  - {}", s)).collect::<Vec<_>>().join("\n") }
    )
}
```

---

## Related

- `src/evaluation/` — Evaluation framework implementation
- `src/context/` — JobContext and ActionRecord types
- `src/error.rs` — EvaluationError type definition
- `src/worker/job.rs` — Job execution where metrics are collected
- architecture/concepts/worker-jobs.md — Background job architecture
- architecture/concepts/orchestrator.md — Job orchestration and management

---

**Source Files**:
- `src/evaluation/mod.rs` — Module exports (13 lines)
- `src/evaluation/metrics.rs` — Metrics collection and error categorization (457 lines)
- `src/evaluation/success.rs` — Success evaluation trait and RuleBasedEvaluator (495 lines)
