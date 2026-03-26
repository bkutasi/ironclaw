<!-- Context: architecture/workspace | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->

# Workspace Memory System

**Purpose**: Persistent agent memory with filesystem-like structure, hybrid search, and privacy-aware multi-layer storage

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Core Concept

The workspace provides persistent memory for agents with a flexible filesystem-like structure. Agents can create arbitrary markdown file hierarchies that get indexed for full-text and semantic search. Memory is database-backed (PostgreSQL or libSQL), not in-memory — if you want to remember something, write it explicitly.

---

## Quick Reference

**Key Operations**:
- `workspace.read(path)` - Read a file
- `workspace.write(path, content)` - Create or update a file
- `workspace.append(path, content)` - Append to a file
- `workspace.list(dir)` - List directory contents
- `workspace.search(query)` - Hybrid full-text + semantic search
- `workspace.append_memory(entry)` - Append to long-term MEMORY.md
- `workspace.append_daily_log(entry)` - Append to today's daily log

**Well-Known Paths**:
- `MEMORY.md` - Long-term curated memory
- `HEARTBEAT.md` - Periodic checklist
- `IDENTITY.md` - Agent name, nature, vibe
- `SOUL.md` - Core values
- `AGENTS.md` - Behavior instructions
- `USER.md` - User context
- `TOOLS.md` - Environment-specific tool notes
- `BOOTSTRAP.md` - First-run ritual (self-deletes after onboarding)
- `context/profile.json` - User psychographic profile
- `daily/YYYY-MM-DD.md` - Daily logs

**Memory Layers**:
- **Private** (default) - Isolated to user scope
- **Shared** - Cross-user readable (opt-in)
- Writes to shared layers can be redirected to private if sensitive content detected

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Workspace                               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Private    │  │   Shared     │  │  Read-Only   │      │
│  │   Layer      │  │   Layer      │  │   Layer      │      │
│  │  (user_123)  │  │  (shared)    │  │  (reports)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                           │                                  │
│                  ┌────────▼────────┐                        │
│                  │ WorkspaceStorage │                        │
│                  │  (Repo or DB)   │                        │
│                  └────────┬────────┘                        │
└───────────────────────────┼─────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
┌────────▼────────┐ ┌──────▼───────┐ ┌────────▼────────┐
│   Documents     │ │    Chunks    │ │   Embeddings    │
│   (memory_      │ │   (memory_   │ │   (cached)      │
│    documents)   │ │    chunks)   │ │                 │
└─────────────────┘ └──────────────┘ └─────────────────┘
```

**Source**: `src/workspace/mod.rs`, `src/workspace/layer.rs`, `src/workspace/repository.rs`

---

## Filesystem Structure

```
workspace/
├── README.md              # Root runbook/index
├── MEMORY.md              # Long-term curated memory
├── HEARTBEAT.md           # Periodic checklist
├── IDENTITY.md            # Agent name, nature, vibe
├── SOUL.md                # Core values
├── AGENTS.md              # Behavior instructions
├── USER.md                # User context
├── TOOLS.md               # Environment-specific tool notes
├── BOOTSTRAP.md           # First-run ritual (deleted after onboarding)
├── context/               # Identity-related docs
│   ├── vision.md
│   ├── priorities.md
│   ├── profile.json       # Psychographic profile (JSON)
│   └── assistant-directives.md
├── daily/                 # Daily logs
│   ├── 2024-01-15.md
│   └── 2024-01-16.md
└── projects/              # Arbitrary structure
    └── alpha/
        ├── README.md
        └── notes.md
```

---

## Embeddings

**Source**: `src/workspace/embeddings.rs`, `src/workspace/embedding_cache.rs`

### Providers

Ironclaw supports multiple embedding providers:

| Provider | Model | Dimensions | Max Input |
|----------|-------|------------|-----------|
| `OpenAiEmbeddings` | text-embedding-3-small | 1536 | 32k chars |
| `OpenAiEmbeddings::large` | text-embedding-3-large | 3072 | 32k chars |
| `OpenAiEmbeddings::ada_002` | text-embedding-ada-002 | 1536 | 32k chars |
| `NearAiEmbeddings` | text-embedding-3-small | 1536 | 32k chars |
| `OllamaEmbeddings` | nomic-embed-text | 768 | 32k chars |
| `MockEmbeddings` | mock-embedding | configurable | 10k chars |

### Caching

All embedding providers are wrapped in `CachedEmbeddingProvider` by default:

```rust
// Default cache: 10,000 entries (~58 MB raw payload for 1536-dim)
let workspace = Workspace::new("user_123", pool)
    .with_embeddings(Arc::new(OpenAiEmbeddings::new(api_key)));

// Custom cache size
let workspace = Workspace::new("user_123", pool)
    .with_embeddings_cached(
        Arc::new(OpenAiEmbeddings::new(api_key)),
        EmbeddingCacheConfig { max_entries: 50_000 }
    );

// Tests: skip cache
let workspace = Workspace::new("user_123", pool)
    .with_embeddings_uncached(Arc::new(MockEmbeddings::new(1536)));
```

**Cache key**: `SHA-256(model_name + "\0" + text)` — same text with different models produces different cache entries.

**Thundering herd**: Multiple concurrent callers with the same uncached key will each call the inner provider. Last writer wins in the LRU cache.

---

## Document Chunking

**Source**: `src/workspace/chunker.rs`

Documents are split into overlapping chunks for search indexing:

```rust
pub struct ChunkConfig {
    pub chunk_size: usize,        // Default: 800 words (~800 tokens)
    pub overlap_percent: f32,     // Default: 0.15 (15% overlap)
    pub min_chunk_size: usize,    // Default: 50 words
}
```

**Chunking strategy**:
1. Split content by whitespace into words
2. Create chunks of `chunk_size` words with `overlap_percent` overlap
3. Merge trailing chunks smaller than `min_chunk_size` with previous chunk
4. Content smaller than `chunk_size` returns as single chunk

**Example**:
```rust
let config = ChunkConfig {
    chunk_size: 10,
    overlap_percent: 0.2,  // 2 word overlap
    min_chunk_size: 3,
};
let chunks = chunk_document("one two three ... twenty", config);
// Creates overlapping chunks: [one..ten], [nine..twenty]
```

---

## Hybrid Search

**Source**: `src/workspace/search.rs`

### Fusion Strategies

Two strategies for combining full-text search (FTS) and vector similarity:

**1. Reciprocal Rank Fusion (RRF)** — Default:
```
score(d) = Σ 1/(k + rank(d)) for each method where d appears
```
- Default k=60
- Documents appearing in both results get boosted scores
- Rank-based, ignores absolute score values

**2. Weighted Score Fusion**:
```
score = fts_weight * (1/fts_rank) + vector_weight * (1/vector_rank)
```
- Configurable weights (default 0.5 each)
- Converts ranks to scores, then combines
- Normalized to [0,1] range

### Search Configuration

```rust
let config = SearchConfig::default()
    .with_limit(10)           // Max results
    .with_rrf_k(60)           // RRF constant
    .with_min_score(0.1)      // Filter low scores
    .with_fusion_strategy(FusionStrategy::Rrf)
    .with_fts_weight(0.7)     // For WeightedScore
    .with_vector_weight(0.3); // For WeightedScore

// Builder shortcuts
let fts_only = SearchConfig::default().fts_only();
let vector_only = SearchConfig::default().vector_only();
```

### Backend Differences

| Feature | PostgreSQL | libSQL |
|---------|------------|--------|
| FTS | `ts_rank_cd` | FTS5 |
| Vector | pgvector cosine distance | libsql_vector_idx |
| Dimension | Fixed by schema | Dynamic (set by `ensure_vector_index()`) |

---

## Privacy Layer

**Source**: `src/workspace/privacy.rs`

### Privacy Classifier

Guards writes to shared memory layers by detecting sensitive content:

```rust
pub trait PrivacyClassifier: Send + Sync {
    fn classify(&self, content: &str) -> SensitivityResult;
}
```

**Default patterns** (`PatternPrivacyClassifier`):
- SSN: `\b\d{3}-\d{2}-\d{4}\b`
- Credit cards: `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`
- Credentials: `password`, `api_key`, `auth_token`, `secret_key`

**Intentionally excluded** (false positives in household contexts):
- Health vocabulary (doctor, medication, therapy)
- Contact info (email addresses, phone numbers)
- General personal mentions

### Configurable Classifier

Operators can customize patterns:

```rust
use ironclaw::workspace::privacy::ConfigurablePrivacyClassifier;

let classifier = ConfigurablePrivacyClassifier::new(vec![
    r"\b\d{3}-\d{2}-\d{4}\b".into(),  // SSN only
]).unwrap();

let workspace = Workspace::new("user_123", pool)
    .with_privacy_classifier(Arc::new(classifier));
```

### Redirect Behavior

When a privacy classifier is configured AND writing to a shared layer:
1. Content is classified
2. If sensitive → redirect to private layer (same path, different scope)
3. `WriteResult::redirected` flag indicates redirect occurred
4. Subsequent multi-scope reads return the private copy (primary scope wins)

**Important**: No classifier is set by default. Writes go exactly where requested. The LLM chooses the correct layer via system prompt guidance.

---

## Memory Layers

**Source**: `src/workspace/layer.rs`

Layers map to synthetic `user_id` values in the database:

```rust
pub struct MemoryLayer {
    pub name: String,              // e.g., "private", "shared"
    pub scope: String,             // user_id for DB queries
    pub writable: bool,            // Default: true
    pub sensitivity: LayerSensitivity,  // Private or Shared
}
```

**Default configuration** (single private layer):
```rust
let layers = MemoryLayer::default_for_user("alice");
// [MemoryLayer { name: "private", scope: "alice", writable: true, sensitivity: Private }]
```

**Multi-layer setup**:
```rust
let layers = vec![
    MemoryLayer {
        name: "private".into(),
        scope: "alice".into(),
        writable: true,
        sensitivity: LayerSensitivity::Private,
    },
    MemoryLayer {
        name: "shared".into(),
        scope: "shared".into(),
        writable: true,
        sensitivity: LayerSensitivity::Shared,
    },
    MemoryLayer {
        name: "reports".into(),
        scope: "reports".into(),
        writable: false,  // Read-only
        sensitivity: LayerSensitivity::Shared,
    },
];

let workspace = Workspace::new("alice", pool)
    .with_memory_layers(layers);
```

**Layer operations**:
- `MemoryLayer::find(layers, "shared")` - Find layer by name
- `MemoryLayer::private_layer(layers)` - Get first private layer
- `MemoryLayer::read_scopes(layers)` - Get all scope values
- `MemoryLayer::writable_scopes(layers)` - Get writable scopes only

---

## Multi-Scope Reads & Identity Isolation

**Critical Security Feature**: Identity files are exempt from multi-scope reads.

| File | Read Method | Rationale |
|------|-------------|-----------|
| AGENTS.md | `read_primary()` | Agent instructions are per-user |
| SOUL.md | `read_primary()` | Core values are per-user |
| USER.md | `read_primary()` | User context is per-user |
| IDENTITY.md | `read_primary()` | Identity is per-user |
| TOOLS.md | `read_primary()` | Tool config is per-user |
| BOOTSTRAP.md | `read_primary()` | Onboarding is per-user |
| MEMORY.md | `read()` | Shared memory is a feature |
| daily/*.md | `read()` | Shared daily logs are a feature |

**Why**: Without this, a user with read access to another scope could silently inherit that scope's identity if their own copy is missing. The agent would present itself as the wrong user — a correctness and security issue.

**Design rule**: If you want shared identity across users, seed the same content into each user's scope at setup time. Don't rely on multi-scope fallback for identity files.

---

## Code Patterns

### Basic Usage

```rust
use std::sync::Arc;
use ironclaw::workspace::{Workspace, OpenAiEmbeddings, paths};

// Create workspace (embeddings wrapped in LRU cache by default)
let workspace = Workspace::new("user_123", pool)
    .with_embeddings(Arc::new(OpenAiEmbeddings::new(api_key)));

// Read/write any path
let doc = workspace.read("projects/alpha/notes.md").await?;
workspace.write("context/priorities.md", "# Priorities\n\n1. Feature X").await?;
workspace.append("daily/2024-01-15.md", "Completed task X").await?;

// Convenience methods
workspace.append_memory("User prefers dark mode").await?;
workspace.append_daily_log("Session note").await?;

// List directory
let entries = workspace.list("projects/").await?;
for entry in entries {
    if entry.is_directory {
        println!("📁 {}/", entry.name());
    } else {
        println!("📄 {}", entry.name());
    }
}

// Search (hybrid FTS + vector)
let results = workspace.search("dark mode preference", 5).await?;
for result in results {
    println!("{} (score: {:.2})", result.document_path, result.score);
}

// Get system prompt from identity files
let prompt = workspace.system_prompt().await?;
```

### Layer-Aware Writes

```rust
use ironclaw::workspace::privacy::PatternPrivacyClassifier;

// Configure privacy classifier for shared layer protection
let classifier = PatternPrivacyClassifier::new()?;
let workspace = Workspace::new("user_123", pool)
    .with_memory_layers(vec![
        MemoryLayer { name: "private".into(), scope: "user_123".into(), writable: true, sensitivity: Private },
        MemoryLayer { name: "shared".into(), scope: "shared".into(), writable: true, sensitivity: Shared },
    ])
    .with_privacy_classifier(Arc::new(classifier));

// Write to shared layer (will redirect if sensitive content detected)
let result = workspace.write_to_layer("shared", "notes.md", "Meeting notes").await?;
if result.redirected {
    println!("Content redirected to {} layer", result.actual_layer);
}

// Force write to shared layer (bypass privacy check)
workspace.write_to_layer("shared", "notes.md", "Sensitive info", true).await?;
```

### Multi-Scope Reads

```rust
// User can read from both their own workspace and a shared workspace
let workspace = Workspace::new("alice", pool)
    .with_additional_read_scopes(vec!["shared".to_string()]);

// Read operations span both scopes
let doc = workspace.read("MEMORY.md").await?;  // Merged from alice + shared

// Identity files always from primary scope only
let identity = workspace.read("IDENTITY.md").await?;  // Always alice's identity
```

### System Prompt Building

```rust
// Build system prompt from identity files (excludes MEMORY.md in group chats)
let prompt = workspace.system_prompt_for_context(is_group_chat).await?;

// With timezone-aware daily log dates
use chrono_tz::America/New_York;
let prompt = workspace.system_prompt_for_context_tz(false, America::New_York).await?;
```

---

## Heartbeat System

**Source**: `src/agent/` (heartbeat runner)

Proactive periodic execution:

1. Reads `HEARTBEAT.md` checklist
2. Runs agent turn with checklist prompt
3. If findings, notifies via channel
4. If nothing, agent replies "HEARTBEAT_OK" (no notification)

```rust
use ironclaw::agent::{HeartbeatConfig, spawn_heartbeat};

let config = HeartbeatConfig::default()
    .with_interval(Duration::from_secs(60 * 30))  // 30 minutes
    .with_notify("user_123", "telegram");

spawn_heartbeat(config, workspace, llm, response_tx);
```

**Seed template**: `src/workspace/seeds/HEARTBEAT.md` — used as fallback if file doesn't exist (never written to DB automatically).

---

## Bootstrap Ritual

**First-run onboarding**:

1. Fresh workspace: `BOOTSTRAP.md` is seeded with ritual instructions
2. Agent reads `BOOTSTRAP.md` at session start (injected first in system prompt)
3. Agent completes ritual tasks
4. Agent deletes `BOOTSTRAP.md`
5. `profile_onboarding_completed` setting marks bootstrap as done

**Safety net**: If `profile_onboarding_completed` is set but `BOOTSTRAP.md` still exists, injection is suppressed to avoid repeating the ritual.

**Seed**: `src/workspace/seeds/BOOTSTRAP.md`

---

## Error Handling

**WorkspaceError types**:
- `DocumentNotFound` - File doesn't exist
- `ChunkingFailed` - Failed to create/update chunks
- `SearchFailed` - Database query error
- `InjectionRejected` - Prompt injection detected in system-prompt file
- `LayerNotFound` - Named memory layer doesn't exist
- `LayerReadOnly` - Attempted write to read-only layer
- `PrivacyRedirectFailed` - Couldn't redirect sensitive content to private layer

**Prompt injection protection**:
- Writes to system-prompt files (`AGENTS.md`, `SOUL.md`, `USER.md`, `IDENTITY.md`, `MEMORY.md`, `TOOLS.md`, `HEARTBEAT.md`, `BOOTSTRAP.md`, `PROFILE.md`, `ASSISTANT_DIRECTIVES.md`) are scanned
- High/critical severity patterns → write rejected
- Warnings logged for lower severity matches

---

## Reference

- **Source**: `src/workspace/`
- **Module docs**: `src/workspace/mod.rs`
- **README**: `src/workspace/README.md`
- **Seeds**: `src/workspace/seeds/`

---

**Related**:
- architecture/concepts/config-precedence.md
- context-system/standards/structure.md
- project-intelligence/technical-domain.md
