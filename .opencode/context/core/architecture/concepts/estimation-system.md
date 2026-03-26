# Estimation System

**Purpose**: Cost, time, and value estimation with continuous learning from historical data

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Quick Reference

**Combined Job Estimate**:
```rust
JobEstimate {
    cost: Decimal,           // Estimated cost to complete
    duration: Duration,      // Estimated time to complete
    value: Decimal,          // Estimated value/earnings
    confidence: f64,         // Confidence (0-1)
    tool_breakdown: Vec<ToolEstimate>,
}
```

**Estimator Usage**:
```rust
let estimator = Estimator::new();
let estimate = estimator.estimate_job("Build API", "backend", &["http", "json"]);
estimator.record_actuals("backend", estimated_cost, actual_cost, estimated_time, actual_time);
```

**Default Tool Costs**: `http`=$0.0001, `echo`/`time`/`json`=$0.0, unknown=$0.001

**Default Tool Durations**: `http`=5s, `echo`=10ms, `time`=1ms, `json`=5ms

**Learning**: Exponential moving average (α=0.1), minimum 5 samples before adjustment

---

## Core Concept

Ironclaw's estimation system provides **continuous learning estimates** for job costs, durations, and value potential. It combines:

- **Base estimates** from tool cost/duration characteristics
- **Statistical learning** that adjusts estimates based on historical accuracy
- **Value estimation** for pricing decisions with configurable profit margins

The system improves over time by recording actual results and using exponential moving averages to adjust future estimates per job category.

---

## Key Components

### Cost Estimator (`cost.rs`)

Estimates monetary costs for tools and LLM token usage.

```rust
pub struct CostEstimator {
    tool_costs: HashMap<String, Decimal>,    // Base costs per tool
    llm_cost_per_1k: Decimal,                // LLM cost per 1K tokens
}
```

**Default Configuration**:
| Tool | Cost (USD) | Notes |
|------|------------|-------|
| `http` | $0.0001 | API calls |
| `echo` | $0.0 | Free operation |
| `time` | $0.0 | Free operation |
| `json` | $0.0 | Free operation |
| Unknown | $0.001 | Default fallback |
| LLM | $0.01/1K tokens | Approximate |

**Key Methods**:
- `estimate_tool(tool_name)` — Get cost for a tool call
- `estimate_llm_tokens(input, output)` — Calculate LLM token costs
- `set_tool_cost(name, cost)` — Override default cost
- `all_tool_costs()` — Get all configured costs

### Time Estimator (`time.rs`)

Estimates execution durations for tools and LLM responses.

```rust
pub struct TimeEstimator {
    tool_durations: HashMap<String, Duration>,  // Base durations per tool
}
```

**Default Configuration**:
| Tool | Duration | Notes |
|------|----------|-------|
| `http` | 5 seconds | API calls |
| `echo` | 10 milliseconds | Free operation |
| `time` | 1 millisecond | Free operation |
| `json` | 5 milliseconds | Free operation |
| Unknown | 5 seconds | Default fallback |

**LLM Response Time**: ~50 tokens/second (rough estimate)

**Key Methods**:
- `estimate_tool(tool_name)` — Get duration for a tool call
- `estimate_llm_response(tokens)` — Estimate LLM response time
- `set_tool_duration(name, duration)` — Override default duration
- `all_tool_durations()` — Get all configured durations

### Value Estimator (`value.rs`)

Estimates job value/earnings potential and calculates profitability.

```rust
pub struct ValueEstimator {
    min_margin: Decimal,    // Minimum profit margin (default: 10%)
    target_margin: Decimal, // Target profit margin (default: 30%)
}
```

**Value Formula**: `value = cost + (cost × target_margin)`

**Key Methods**:
- `estimate(description, cost)` — Estimate value from cost
- `minimum_bid(cost)` — Calculate minimum acceptable bid (cost + min_margin)
- `ideal_bid(cost)` — Calculate ideal bid (cost + target_margin)
- `is_profitable(price, cost)` — Check if job is profitable at given price
- `calculate_profit(earnings, cost)` — Calculate actual profit
- `calculate_margin(earnings, cost)` — Calculate profit margin

**Profitability Logic**:
```rust
// Margin = (price - cost) / price
// Profitable if margin >= min_margin (10% default)

// Special case: zero price
if price.is_zero() {
    return estimated_cost < Decimal::ZERO; // Only profitable if negative cost
}
```

**Boundary Handling**:
- Zero earnings in `calculate_margin()` → returns `Decimal::ZERO` (no panic)
- Negative costs handled correctly (means getting paid to do work)
- Large values use `rust_decimal` to avoid overflow

### Estimation Learner (`learner.rs`)

Improves estimates over time using statistical learning.

```rust
pub struct EstimationLearner {
    models: HashMap<String, LearningModel>,  // Per-category models
    alpha: f64,                               // EMA smoothing (default: 0.1)
    min_samples: u64,                         // Min samples before adjustment (default: 5)
}

pub struct LearningModel {
    cost_factor: f64,      // Cost adjustment multiplier
    time_factor: f64,      // Time adjustment multiplier
    sample_count: u64,     // Number of samples
    cost_error_rate: f64,  // Running cost error rate
    time_error_rate: f64,  // Running time error rate
}
```

**Learning Algorithm**:
1. Record actual vs. estimated results
2. Calculate ratios: `actual / estimated`
3. Update factors using exponential moving average: `factor = factor × (1-α) + ratio × α`
4. Track error rates: `|ratio - 1.0|`

**Confidence Calculation**:
```rust
confidence = 0.5 + (sample_factor × 0.3) + (error_factor × 0.2)

sample_factor = min(sample_count / 100, 1.0)
error_factor = 1.0 - avg(cost_error_rate, time_error_rate)
```

| Samples | Accuracy | Confidence |
|---------|----------|------------|
| 0 | N/A | 0.2 (no data) |
| 1-4 | N/A | 0.3 (some data) |
| 5+ | High | 0.5-1.0 (learned) |

**Key Methods**:
- `record(category, est_cost, act_cost, est_time, act_time)` — Record results
- `adjust(category, cost, time)` — Apply learned adjustments
- `confidence(category)` — Get confidence score (0-1)
- `get_model(category)` — Get learning model for inspection
- `set_alpha(alpha)` — Configure EMA smoothing (clamped to 0.01-0.5)
- `set_min_samples(min)` — Configure minimum samples threshold

### Combined Estimator (`mod.rs`)

Unifies cost, time, value, and learning into a single interface.

```rust
pub struct Estimator {
    cost: CostEstimator,
    time: TimeEstimator,
    value: ValueEstimator,
    learner: EstimationLearner,
}

pub struct JobEstimate {
    pub cost: Decimal,
    pub duration: Duration,
    pub value: Decimal,
    pub confidence: f64,
    pub tool_breakdown: Vec<ToolEstimate>,
}

pub struct ToolEstimate {
    pub tool_name: String,
    pub cost: Decimal,
    pub duration: Duration,
    pub confidence: f64,
}
```

**Key Methods**:
- `estimate_job(description, category, tools)` — Get combined estimate
- `record_actuals(category, est_cost, act_cost, est_time, act_time)` — Learn from results
- `cost()` / `time()` / `value()` — Access individual estimators

---

## Data Flow

### Job Estimation Flow

```
Estimator.estimate_job(description, category, tools)
    ↓
For each tool in tools[]:
    ├─ cost.estimate_tool(tool_name)
    ├─ time.estimate_tool(tool_name)
    └─ Create ToolEstimate { tool_name, cost, duration, confidence: 0.7 }
    ↓
Sum all tool costs and durations
    ↓
learner.adjust(category, total_cost, total_time)
    ├─ If samples >= min_samples: apply learned factors
    └─ Else: use original estimates
    ↓
value.estimate(description, adjusted_cost)
    ↓
confidence = learner.confidence(category)
    ↓
Return JobEstimate { cost, duration, value, confidence, tool_breakdown }
```

### Learning Feedback Loop

```
Job completes with actual results
    ↓
Estimator.record_actuals(category, est_cost, act_cost, est_time, act_time)
    ↓
learner.record(category, est_cost, act_cost, est_time, act_time)
    ↓
Update LearningModel for category:
    ├─ sample_count += 1
    ├─ cost_ratio = act_cost / est_cost
    ├─ time_ratio = act_time / est_time
    ├─ cost_factor = EMA(cost_factor, cost_ratio, alpha)
    ├─ time_factor = EMA(time_factor, time_ratio, alpha)
    ├─ cost_error_rate = EMA(error_rate, |cost_ratio - 1|, alpha)
    └─ time_error_rate = EMA(error_rate, |time_ratio - 1|, alpha)
    ↓
Future estimates for this category will be adjusted
```

### Value Estimation Flow

```
ValueEstimator.estimate(description, cost)
    ↓
margin = cost × target_margin (default: 30%)
    ↓
value = cost + margin
    ↓
Return value

// For pricing decisions:
minimum_bid = cost + (cost × min_margin)    // 10% minimum
ideal_bid = cost + (cost × target_margin)   // 30% target

// For profitability check:
margin = (price - cost) / price
profitable = margin >= min_margin
```

---

## Code Patterns

### Basic Estimation

```rust
use crate::estimation::Estimator;

let estimator = Estimator::new();

// Estimate a job
let estimate = estimator.estimate_job(
    "Build REST API endpoint",
    Some("backend"),
    &["http".to_string(), "json".to_string()],
);

println!("Cost: ${}", estimate.cost);
println!("Duration: {:?}", estimate.duration);
println!("Value: ${}", estimate.value);
println!("Confidence: {:.2}", estimate confidence);

// Inspect tool breakdown
for tool in &estimate.tool_breakdown {
    println!("  {}: ${} ({:?})", tool.tool_name, tool.cost, tool.duration);
}
```

### Recording Actuals for Learning

```rust
use rust_decimal::Decimal;
use std::time::Duration;

// After job completes, record actual results
estimator.record_actuals(
    "backend",
    estimate.cost,        // Estimated cost
    actual_cost,          // Actual cost incurred
    estimate.duration,    // Estimated time
    actual_duration,      // Actual time taken
);

// Future estimates for "backend" category will be adjusted
```

### Customizing Estimator Configuration

```rust
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

// Customize cost estimator
let mut estimator = Estimator::new();
estimator.cost().set_tool_cost("custom_tool", dec!(0.005));
estimator.cost().set_tool_cost("expensive_api", dec!(0.01));

// Customize time estimator
estimator.time().set_tool_duration("slow_tool", Duration::from_secs(30));

// Customize value estimator margins
estimator.value().set_min_margin(dec!(0.15));    // 15% minimum
estimator.value().set_target_margin(dec!(0.40)); // 40% target

// Customize learner
estimator.learner().set_alpha(0.2);       // Faster learning (range: 0.01-0.5)
estimator.learner().set_min_samples(3);   // Adjust after 3 samples (default: 5)
```

### Pricing Decisions

```rust
use rust_decimal::Decimal;

let cost = dec!(100.0);

// Calculate bids
let min_bid = estimator.value().minimum_bid(cost);    // $110 (10% margin)
let ideal_bid = estimator.value().ideal_bid(cost);    // $130 (30% margin)

// Check profitability
let customer_offer = dec!(120.0);
if estimator.value().is_profitable(customer_offer, cost) {
    println!("Acceptable offer with {:.2}% margin", 
             estimator.value().calculate_margin(customer_offer, cost) * 100.0);
}

// Calculate actual profit after completion
let earnings = dec!(120.0);
let actual_cost = dec!(95.0);
let profit = estimator.value().calculate_profit(earnings, actual_cost);
let margin = estimator.value().calculate_margin(earnings, actual_cost);
```

### Inspecting Learning Models

```rust
// Check learning progress for a category
if let Some(model) = estimator.learner().get_model("backend") {
    println!("Samples: {}", model.sample_count);
    println!("Cost factor: {:.2} (×{})", model.cost_factor, model.cost_factor);
    println!("Time factor: {:.2} (×{})", model.time_factor, model.time_factor);
    println!("Cost error rate: {:.2}%", model.cost_error_rate * 100.0);
    println!("Time error rate: {:.2}%", model.time_error_rate * 100.0);
    println!("Confidence: {:.2}", estimator.learner().confidence("backend"));
}

// View all learned models
for (category, model) in estimator.learner().all_models() {
    println!("{}: {} samples, confidence {:.2}", 
             category, model.sample_count, estimator.learner().confidence(category));
}
```

---

## Key Invariants

- **No unwrap/expect in production**: Use `?` with proper error handling (except tests/infallible invariants)
- **Decimal precision**: All monetary values use `rust_decimal::Decimal` to avoid floating-point errors
- **Zero handling**: Division-by-zero protected in `is_profitable()` and `calculate_margin()`
- **EMA bounds**: Alpha clamped to 0.01-0.5 to prevent unstable learning
- **Minimum samples**: Adjustments only applied after `min_samples` (default: 5) to prevent overfitting
- **Confidence range**: Always 0.0-1.0, with floor of 0.2 for unknown categories
- **Negative cost handling**: Correctly handled (means getting paid to do work)
- **Large value safety**: Uses `rust_decimal` to avoid overflow with large values

---

## Related Files

**Core Implementation**:
- `src/estimation/mod.rs` — Combined `Estimator`, `JobEstimate`, `ToolEstimate`
- `src/estimation/cost.rs` — `CostEstimator` for tool and LLM costs
- `src/estimation/time.rs` — `TimeEstimator` for durations
- `src/estimation/value.rs` — `ValueEstimator` for pricing and profitability
- `src/estimation/learner.rs` — `EstimationLearner` with statistical learning

**Dependencies**:
- `rust_decimal` — Precise decimal arithmetic for monetary values
- `rust_decimal_macros` — `dec!()` macro for decimal literals

**Documentation**:
- `.opencode/context/core/standards/code-quality.md` — Code quality standards
- `.opencode/context/core/standards/documentation.md` — Documentation standards

---

## Common Pitfalls

### ❌ Using floating-point for monetary values

```rust
// WRONG: Floating-point precision issues
let cost: f64 = 0.1 + 0.2; // 0.30000000000000004

// RIGHT: Use rust_decimal
use rust_decimal::Decimal;
let cost: Decimal = dec!(0.1) + dec!(0.2); // 0.3 exactly
```

### ❌ Not recording actuals for learning

```rust
// WRONG: Estimate but never learn
let estimate = estimator.estimate_job(...);
// ... job executes ...
// Missing: estimator.record_actuals(...)

// RIGHT: Always record actuals after job completes
let estimate = estimator.estimate_job(...);
// ... job executes ...
estimator.record_actuals(
    category,
    estimate.cost,
    actual_cost,
    estimate.duration,
    actual_duration,
);
```

### ❌ Ignoring confidence scores

```rust
// WRONG: Trust all estimates equally
let estimate = estimator.estimate_job("new_task_type", ...);
println!("Will take {:?}", estimate.duration); // Low confidence!

// RIGHT: Check confidence before relying on estimate
let estimate = estimator.estimate_job("new_task_type", ...);
if estimate.confidence < 0.5 {
    println!("Low confidence estimate - actual may vary significantly");
}
```

### ❌ Setting alpha too high

```rust
// WRONG: Unstable learning with high alpha
learner.set_alpha(0.9); // Overreacts to recent samples

// RIGHT: Moderate alpha for stable learning
learner.set_alpha(0.1); // Default, good balance
learner.set_alpha(0.2); // Faster learning, still stable
```

### ❌ Not handling zero price edge case

```rust
// WRONG: Assume price is always positive
let margin = (price - cost) / price; // Panics if price = 0

// RIGHT: ValueEstimator handles this correctly
if estimator.is_profitable(Decimal::ZERO, dec!(10.0)) {
    // Only true if cost is negative (getting paid to do it)
}
```

---

## Testing Patterns

### Unit Test: Cost Estimation

```rust
#[test]
fn test_tool_cost_estimation() {
    let estimator = CostEstimator::new();

    assert_eq!(estimator.estimate_tool("echo"), dec!(0.0));
    assert_eq!(estimator.estimate_tool("http"), dec!(0.0001));
    assert!(estimator.estimate_tool("unknown") > dec!(0.0));
}

#[test]
fn test_llm_cost_estimation() {
    let estimator = CostEstimator::new();

    let cost = estimator.estimate_llm_tokens(1000, 500);
    assert!(cost > dec!(0.0));
}
```

### Unit Test: Learning Model Update

```rust
#[test]
fn test_learning_model_update() {
    let mut learner = EstimationLearner::new();
    learner.set_min_samples(2);

    // Record results where actuals are 20% higher than estimates
    for _ in 0..5 {
        learner.record(
            "test",
            dec!(100.0),
            dec!(120.0),
            Duration::from_secs(60),
            Duration::from_secs(72),
        );
    }

    let model = learner.get_model("test").unwrap();
    assert!(model.cost_factor > 1.0);
    assert!(model.time_factor > 1.0);
}
```

### Unit Test: Value Estimator Boundaries

```rust
#[test]
fn test_profitability_zero_price() {
    let estimator = ValueEstimator::new();

    // Zero price should not panic
    assert!(!estimator.is_profitable(Decimal::ZERO, dec!(10.0)));
    assert!(!estimator.is_profitable(Decimal::ZERO, Decimal::ZERO));
    // Negative cost with zero price is profitable
    assert!(estimator.is_profitable(Decimal::ZERO, dec!(-10.0)));
}

#[test]
fn test_margin_zero_earnings() {
    let estimator = ValueEstimator::new();

    // Zero earnings → margin should be zero, not panic
    assert_eq!(
        estimator.calculate_margin(Decimal::ZERO, dec!(50.0)),
        Decimal::ZERO
    );
}
```

### Integration Test: End-to-End Learning

```rust
#[test]
fn test_end_to_end_learning() {
    let mut estimator = Estimator::new();

    // Initial estimate (no learning yet)
    let estimate1 = estimator.estimate_job("task", Some("test"), &["http"]);
    let initial_cost = estimate1.cost;

    // Simulate consistent underestimation (actuals 50% higher)
    for _ in 0..10 {
        let estimate = estimator.estimate_job("task", Some("test"), &["http"]);
        estimator.record_actuals(
            "test",
            estimate.cost,
            estimate.cost * dec!(1.5), // 50% higher
            estimate.duration,
            estimate.duration.mul_f64(1.5),
        );
    }

    // New estimate should be adjusted upward
    let estimate2 = estimator.estimate_job("task", Some("test"), &["http"]);
    assert!(estimate2.cost > initial_cost);
    assert!(estimate2.confidence > 0.5); // Higher confidence after learning
}
```
