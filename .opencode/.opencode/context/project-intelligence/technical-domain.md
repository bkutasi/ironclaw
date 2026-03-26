<!-- Context: project-intelligence/technical | Priority: critical | Version: 1.0 | Updated: 2026-02-11 -->

# Technical Domain

**Purpose**: Tech stack, architecture, and development patterns for Ironclaw - a secure personal AI assistant.

**Last Updated**: 2026-02-11

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
│ Main Agent Loop                                             │
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
2. **WASM Sandbox**: Untrusted tools run in Wasmtime with capability-based permissions
3. **Safety Layer**: Input sanitization, prompt injection detection, policy enforcement
4. **Self-Repair**: Automatic detection and recovery of stuck jobs

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
2. **Credential Protection** - Secrets injected at host boundary, never exposed to tools
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
- `src/agent/` - Agent lifecycle and session management
- `src/orchestrator/` - Job management and container orchestration
- `src/worker/` - Container-side execution and LLM proxy

**Safety & Security**:
- `src/safety/` - Sanitization, leak detection, policy enforcement
- `src/sandbox/` - Docker container management
- `src/secrets/` - Encryption and keychain integration

**Tools & Extensions**:
- `src/tools/` - Tool trait, registry, WASM runtime, MCP client
- `src/extensions/` - Extension discovery and management
- `tools-src/` - WASM tool implementations (9 tools)

**Channels**:
- `src/channels/` - Multi-channel support (REPL, HTTP, webhooks)
- `channels-src/` - WASM channel implementations (WhatsApp, Slack, Telegram)

**LLM Integration**:
- `src/llm/` - Multi-provider LLM support (OpenAI, NearAI, Claude)

**Workspace**:
- `src/workspace/` - Embeddings, semantic search, repository management

**Config Files**:
- `Cargo.toml` - Dependencies and project metadata
- `Dockerfile.worker` - Worker container image
- `migrations/` - Database schema migrations

## Related Files

- `business-domain.md` - Business context and problem statement
- `business-tech-bridge.md` - How business needs map to technical solutions
- `decisions-log.md` - Architecture decision records
- `living-notes.md` - Active issues and technical debt
