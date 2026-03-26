# Database Abstraction Layer

**Purpose**: Backend-agnostic persistence layer supporting PostgreSQL and libSQL (Turso's SQLite fork)

**Last Updated**: 2026-03-26

---

## Quick Reference

```bash
# Default build (PostgreSQL)
cargo build

# libSQL/Turso build
cargo build --no-default-features --features libsql

# Both backends
cargo build --features "postgres,libsql"

# Test each backend in isolation
cargo check                                           # postgres (default)
cargo check --no-default-features --features libsql   # libsql only
cargo check --all-features                            # both
```

**Rule**: All new persistence features must support both backends.

---

## Dual-Backend Architecture

IronClaw uses a **trait-based abstraction** allowing runtime selection between two database backends:

| Backend | Feature Flag | Use Case | Connection Model |
|---------|-------------|----------|------------------|
| **PostgreSQL** | `postgres` (default) | Production deployments, full-featured | `deadpool-postgres` pool |
| **libSQL** | `libsql` | Embedded/edge, local dev, Turso cloud | New connection per operation |

### Backend Selection

Backend is selected at runtime via `DatabaseConfig.backend`:

```rust
// From environment: DATABASE_BACKEND=postgres|libsql
match config.backend {
    DatabaseBackend::Postgres => PgBackend::new(config).await?,
    DatabaseBackend::LibSql => LibSqlBackend::new_local(db_path).await?,
}
```

### Deployment Modes

**PostgreSQL**:
- Standard server deployment
- Requires PostgreSQL 15+ (for pgvector)
- Managed via `migrations/V1__*.sql` through `V9__*.sql`

**libSQL** (three modes):
1. **Local embedded**: `LIBSQL_PATH=~/.ironclaw/data.db`
2. **Turso cloud sync**: `LIBSQL_URL=libsql://xxx.turso.io` + `LIBSQL_AUTH_TOKEN`
3. **In-memory** (tests): `LibSqlBackend::new_memory()`

---

## DB Trait Structure

The `Database` supertrait is composed of **seven sub-traits** (~78 async methods total):

| Sub-trait | Methods | Covers |
|-----------|---------|--------|
| `ConversationStore` | 12 | Conversations, messages, metadata |
| `JobStore` | 13 | Agent jobs, actions, LLM calls, estimation |
| `SandboxStore` | 13 | Sandbox jobs, job events |
| `RoutineStore` | 15 | Routines, routine runs, scheduling |
| `ToolFailureStore` | 4 | Self-repair tracking |
| `SettingsStore` | 8 | Per-user key-value settings |
| `WorkspaceStore` | 13+ | Memory documents, chunks, hybrid search |

### Trait Definition (simplified)

```rust
#[async_trait]
pub trait Database:
    ConversationStore
    + JobStore
    + SandboxStore
    + RoutineStore
    + ToolFailureStore
    + SettingsStore
    + WorkspaceStore
    + Send
    + Sync
{
    async fn run_migrations(&self) -> Result<(), DatabaseError>;
}
```

### Usage Pattern

```rust
// Leaf consumers can depend on narrowest sub-trait needed
async fn process_job(db: &impl JobStore) { ... }

// Full system access uses Database supertrait
async fn full_system(db: Arc<dyn Database>) { ... }
```

---

## Migrations

### PostgreSQL Migrations

Managed by `refinery` crate. Located in `migrations/` directory:

```
migrations/
├── V1__initial.sql                    # Base schema
├── V2__leak_detection.sql
├── V3__routines.sql
├── ...
└── V9__flexible_embedding_dimension.sql
```

Applied automatically on startup via `PgBackend::run_migrations()`.

### libSQL Migrations

**Consolidated schema** in `src/db/libsql_migrations.rs`:
- Single `SCHEMA` constant with all `CREATE TABLE IF NOT EXISTS` statements
- No ALTER TABLE support (SQLite limitation)
- Idempotent — safe to run multiple times

**Incremental migrations** (V9+) tracked in `_migrations` table:

```rust
pub const INCREMENTAL_MIGRATIONS: &[(i64, &str, &str)] = &[
    (9, "flexible_embedding_dimension", "..."),
    (12, "job_token_budget", "..."),
    (13, "routine_notify_user_nullable", "..."),
];
```

Each migration:
1. Checks `_migrations` table for prior application
2. Runs in transaction for atomicity
3. Records version on success

### Migration Best Practices

**Adding new persistence**:
1. Add method signature to sub-trait in `src/db/mod.rs`
2. Implement in `postgres.rs` (delegate to `Store`/`Repository`)
3. Implement in `libsql/<module>.rs` (SQLite-dialect SQL)
4. Add PostgreSQL migration: new `migrations/VN__*.sql`
5. Add libSQL migration: update `libsql_migrations.rs`

**Pattern**: Fix the pattern, not the instance. A fix to `postgres.rs` that doesn't also fix the libSQL module is half a fix.

---

## Connection Pooling

### PostgreSQL: Connection Pool

Uses `deadpool-postgres` for connection pooling:

```rust
pub struct PgBackend {
    store: Store,      // Wraps deadpool-postgres Pool
    repo: Repository,  // Borrows pool reference
}

impl PgBackend {
    pub fn pool(&self) -> Pool {
        self.store.pool()
    }
}
```

**Characteristics**:
- Pool size configured via `DATABASE_POOL_SIZE`
- Fully concurrent reads/writes
- Connections reused across operations
- Automatic connection recycling

### libSQL: Connection Per Operation

libSQL creates fresh connections per operation:

```rust
impl LibSqlBackend {
    pub async fn connect(&self) -> Result<Connection, DatabaseError> {
        let conn = self.db.connect()?;
        conn.query("PRAGMA busy_timeout = 5000", ()).await?;
        Ok(conn)
    }
}
```

**Characteristics**:
- WAL mode enabled for concurrent reads
- Write serialization (one writer at a time)
- 5-second busy timeout for write contention
- Retry logic (3 attempts with exponential backoff)
- Connection closed when dropped

### Sharing Database Handles

Satellite stores (SecretsStore, WasmToolStore) need backend-specific handles:

```rust
// libSQL: Share Arc<LibSqlDatabase>, each store creates own connections
let backend = LibSqlBackend::new_local(path).await?;
let shared_db = backend.shared_db(); // Arc<LibSqlDatabase>
let secrets_store = LibSqlSecretsStore::new(shared_db, crypto);

// PostgreSQL: Share Pool clone
let pg = PgBackend::new(config).await?;
let pool = pg.pool(); // Cloneable Pool
let secrets_store = PostgresSecretsStore::new(pool, crypto);
```

---

## SQL Dialect Differences

| Feature | PostgreSQL | libSQL |
|---------|-----------|--------|
| UUIDs | `UUID` type | `TEXT` (hex string) |
| Timestamps | `TIMESTAMPTZ` | `TEXT` (ISO-8601 RFC 3339) |
| JSON | `JSONB` | `TEXT` (JSON encoded) |
| Numeric/Decimal | `NUMERIC` | `TEXT` (preserves precision) |
| Arrays | `TEXT[]` | `TEXT` (JSON-encoded array) |
| Booleans | `BOOLEAN` | `INTEGER` (0/1) |
| Vector embeddings | `VECTOR` (unbounded dim) | `F32_BLOB(N)` (dynamic dimension) |
| Full-text search | `tsvector` + `ts_rank_cd` | FTS5 virtual table + triggers |
| JSON path update | `jsonb_set(col, '{key}', val)` | `json_patch(col, '{"key": val}')` |
| Stored procedures | PL/pgSQL functions | Triggers only |

### Critical Gotchas

**Boolean storage** (libSQL):
```rust
// Writing: convert bool to i64
conn.execute("INSERT INTO ... (enabled) VALUES (?1)", [1i64])?;

// Reading: convert i64 to bool
let enabled = row.get::<i64>(idx)? != 0;
```

**Timestamp format** (libSQL):
```rust
// Write: RFC 3339 with milliseconds
fn fmt_ts(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

// Read: Multi-format parser with fallbacks
fn parse_timestamp(s: &str) -> Result<DateTime<Utc>, String> {
    // Tries: RFC3339 → naive with ms → naive without ms
}
```

**JSON merge patch** (libSQL):
- Uses RFC 7396 JSON Merge Patch (`json_patch`)
- Replaces top-level keys entirely
- **Cannot** do partial nested updates
- PostgreSQL uses `jsonb_set` for path-targeted updates

**Vector dimension** (libSQL):
- Dynamically created as `F32_BLOB(N)` during migrations
- Dimension inferred from `EMBEDDING_DIMENSION` env var
- V9 migration changed to flexible `BLOB` (any dimension)
- Index rebuilt on dimension change

---

## Code Patterns

### Pattern 1: Sub-trait Implementation

```rust
// src/db/mod.rs - Define in sub-trait
#[async_trait]
pub trait JobStore: Send + Sync {
    async fn save_job(&self, ctx: &JobContext) -> Result<(), DatabaseError>;
}

// src/db/postgres.rs - PostgreSQL impl
#[async_trait]
impl JobStore for PgBackend {
    async fn save_job(&self, ctx: &JobContext) -> Result<(), DatabaseError> {
        self.store.save_job(ctx).await  // Delegate to Store
    }
}

// src/db/libsql/jobs.rs - libSQL impl
#[async_trait]
impl JobStore for LibSqlBackend {
    async fn save_job(&self, ctx: &JobContext) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        conn.execute(
            "INSERT INTO agent_jobs (...) VALUES (...)",
            params![...],
        ).await.map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }
}
```

### Pattern 2: Helper Functions (libSQL)

```rust
// src/db/libsql/mod.rs - Shared utilities

/// Extract text with NULL → empty string default
pub(crate) fn get_text(row: &libsql::Row, idx: i32) -> String {
    row.get::<String>(idx).unwrap_or_default()
}

/// Extract optional text (NULL → None, "" → Some(""))
pub(crate) fn get_opt_text(row: &libsql::Row, idx: i32) -> Option<String> {
    row.get::<String>(idx).ok()
}

/// Convert Option<&str> to libsql::Value (preserves NULL)
pub(crate) fn opt_text(s: Option<&str>) -> libsql::Value {
    match s {
        Some(s) => libsql::Value::Text(s.to_string()),
        None => libsql::Value::Null,
    }
}

/// Boolean: i64 ↔ bool conversion
pub(crate) fn get_opt_bool(row: &libsql::Row, idx: i32) -> Option<bool> {
    row.get::<i64>(idx).ok().map(|v| v != 0)
}
```

### Pattern 3: Testing libSQL

```rust
#[tokio::test]
async fn test_my_feature() {
    // In-memory database (no files, no cleanup)
    let backend = LibSqlBackend::new_memory().await.unwrap();
    backend.run_migrations().await.unwrap();
    
    // backend implements Database — call any trait method
    backend.save_job(&ctx).await.unwrap();
    let loaded = backend.get_job(ctx.job_id).await.unwrap();
}

#[tokio::test]
async fn test_concurrent_writes() {
    // Temp file for shared state (in-memory is connection-local)
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let backend = LibSqlBackend::new_local(&db_path).await.unwrap();
    backend.run_migrations().await.unwrap();
    
    // Spawn concurrent operations
    let mut handles = Vec::new();
    for i in 0..20 {
        let conn = backend.connect().await.unwrap();
        handles.push(tokio::spawn(async move {
            conn.execute("INSERT INTO ...", params![i]).await
        }));
    }
}
```

### Pattern 4: Multi-Scope Queries

WorkspaceStore supports multi-user scope queries with optimized SQL:

```rust
// Default implementation (loops over user_ids)
async fn hybrid_search_multi(
    &self,
    user_ids: &[String],
    ...
) -> Result<Vec<SearchResult>, WorkspaceError> {
    let mut all_results = Vec::new();
    for uid in user_ids {
        let results = self.hybrid_search(uid, ...).await?;
        all_results.extend(results);
    }
    // Merge and re-sort by score
    Ok(all_results)
}

// PostgreSQL override (single SQL query with ANY($1::text[]))
async fn hybrid_search_multi(
    &self,
    user_ids: &[String],
    ...
) -> Result<Vec<SearchResult>, WorkspaceError> {
    self.repo.hybrid_search_multi(user_ids, ...).await
    // Uses: WHERE user_id = ANY($1::text[])
}
```

---

## Current Limitations

### libSQL

- **Secrets store**: `LibSqlSecretsStore` exists but not plumbed through main startup
- **Settings reload**: `Config::from_db` skipped (requires `Store`)
- **No incremental schema changes**: All columns must exist in base schema
- **No encryption at rest**: Only secrets encrypted; other data plaintext
- **Write serialization**: WAL mode allows concurrent readers, one writer
- **Busy timeout**: 5s may cause timeouts under high write concurrency

### PostgreSQL

- Requires external server (Docker or installed)
- pgvector extension mandatory (PostgreSQL 15+)
- More complex deployment for edge/embedded scenarios

---

## Key Tables

**Core**: `conversations`, `conversation_messages`, `agent_jobs`, `job_actions`, `job_events`, `llm_calls`, `estimation_snapshots`

**Workspace**: `memory_documents`, `memory_chunks`, `memory_chunks_fts` (virtual), `heartbeat_state`

**Security**: `secrets`, `wasm_tools`, `tool_capabilities`, `leak_detection_patterns`, `leak_detection_events`, `secret_usage_log`

**Automation**: `routines`, `routine_runs`, `settings`, `tool_failures`

**Tracking**: `_migrations` (libSQL only)

---

## Related

- `src/db/CLAUDE.md` — Developer guide with detailed examples
- `src/db/mod.rs` — Trait definitions
- `src/db/postgres.rs` — PostgreSQL backend
- `src/db/libsql/` — libSQL backend modules
- `src/db/libsql_migrations.rs` — libSQL schema
- `migrations/` — PostgreSQL migrations

---

**Quality Note**: This document reflects the actual implementation as of 2026-03-26. Always verify against source code for critical changes.
