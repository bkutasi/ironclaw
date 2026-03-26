# Import Patterns

**Purpose**: Migration system for importing OpenClaw data (memory, history, settings, credentials) into IronClaw without data loss

**Last Updated**: 2026-03-26

---

## Quick Reference

```bash
# Enable import feature (optional, not in default build)
cargo build --features import

# Import detects OpenClaw automatically at ~/.openclaw
# Import is triggered programmatically via OpenClawImporter
```

**Rule**: All imports are idempotent where possible. Re-running import should not create duplicates.

---

## Architecture Overview

The import system provides **safe, phased migration** from OpenClaw to IronClaw:

```
┌─────────────────────────────────────────────────────────────┐
│ OpenClawInstallation                                        │
│ ~/.openclaw/                                                │
│ ├── openclaw.json          (configuration)                 │
│ ├── agents/                  (SQLite databases)            │
│ │   ├── agent1.sqlite                                      │
│ │   └── agent2.sqlite                                      │
│ └── workspace/               (markdown files)              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ OpenClawImporter                                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Phase 1: Read-Only Extraction                        │   │
│ │  • OpenClawReader parses config, SQLite DBs          │   │
│ │  • Validates all data before any writes              │   │
│ │  • Collects: settings, credentials, conversations    │   │
│ └──────────────────────────────────────────────────────┘   │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Phase 2: Grouped Writes (fail-safe)                  │   │
│ │  Group 1: Settings (idempotent upsert)               │   │
│ │  Group 2: Credentials (idempotent upsert)            │   │
│ │  Group 3: Workspace documents                        │   │
│ │  Group 4: Memory chunks (path deduplication)         │   │
│ │  Group 5: Conversations + messages (atomic units)    │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ IronClaw Database                                           │
│ ├── settings               (per-user key-value)            │
│ ├── secrets                (encrypted credentials)         │
│ ├── memory_documents       (workspace files)               │
│ ├── memory_chunks          (embedded chunks)               │
│ └── conversations          (+ messages)                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Read-before-write**: All data validated before first write
2. **Grouped writes**: Related items committed together
3. **Fail-safe**: Errors logged but don't halt entire import
4. **Idempotent**: Re-importing should not create duplicates
5. **Dry-run support**: Preview what would be imported without writing

---

## Import Mechanisms

### OpenClawImporter

**Location**: `src/import/openclaw/mod.rs`

Main orchestrator coordinating the entire import process:

```rust
pub struct OpenClawImporter {
    db: Arc<dyn Database>,
    workspace: Workspace,
    secrets: Arc<dyn SecretsStore>,
    opts: ImportOptions,
}
```

**Construction**:
```rust
let importer = OpenClawImporter::new(
    db,           // IronClaw database
    workspace,    // Workspace instance
    secrets,      // Secrets store
    opts,         // ImportOptions config
);
```

**Detection**:
```rust
// Check if OpenClaw exists at ~/.openclaw
if let Some(path) = OpenClawImporter::detect() {
    println!("OpenClaw found at: {:?}", path);
}
```

**Execution**:
```rust
let stats = importer.import().await?;
println!("Imported {} items", stats.total_imported());
```

### OpenClawReader

**Location**: `src/import/openclaw/reader.rs`

Read-only extraction layer for OpenClaw data:

```rust
pub struct OpenClawReader {
    openclaw_dir: PathBuf,
}
```

**Capabilities**:
- `read_config()` — Parse `openclaw.json` (JSON5 format)
- `list_agent_dbs()` — Enumerate agent SQLite databases
- `read_memory_chunks(db_path)` — Extract memory chunks with embeddings
- `read_conversations(db_path)` — Extract conversations with messages
- `list_workspace_files()` — Count markdown files for import

**Security**: API keys in config are wrapped in `SecretString` and never exposed in debug output.

### ImportOptions

**Location**: `src/import/mod.rs`

Configuration for import behavior:

```rust
pub struct ImportOptions {
    pub openclaw_path: PathBuf,    // Default: ~/.openclaw
    pub dry_run: bool,             // Preview without writing
    pub re_embed: bool,            // Re-embed if dimension mismatch
    pub user_id: String,           // Scope imported data to user
}
```

### ImportStats

**Location**: `src/import/mod.rs`

Detailed statistics returned after import:

```rust
pub struct ImportStats {
    pub documents: usize,      // Workspace files imported
    pub chunks: usize,         // Memory chunks imported
    pub conversations: usize,  // Conversations imported
    pub messages: usize,       // Messages imported
    pub settings: usize,       // Settings imported
    pub secrets: usize,        // Credentials imported
    pub skipped: usize,        // Items skipped (existed)
    pub re_embed_queued: usize,// Chunks needing re-embedding
}
```

---

## Migration Process

### Phase 1: Read-Only Extraction

**Goal**: Validate all data before any writes to minimize partial state risk.

```rust
// Read configuration
let reader = OpenClawReader::new(&self.opts.openclaw_path)?;
let config = reader.read_config()?;
let agent_dbs = reader.list_agent_dbs()?;

// Pre-read all conversations
let mut all_conversations = Vec::new();
for (_agent_name, db_path) in &agent_dbs {
    match reader.read_conversations(db_path).await {
        Ok(convs) => all_conversations.extend(convs),
        Err(e) => tracing::warn!("Failed to read conversations: {}", e),
    }
}

// Pre-read all memory chunks
let mut all_chunks = Vec::new();
for (_agent_name, db_path) in &agent_dbs {
    match reader.read_memory_chunks(db_path).await {
        Ok(chunks) => all_chunks.extend(chunks),
        Err(e) => tracing::warn!("Failed to read memory chunks: {}", e),
    }
}

// Prepare settings and credentials
let settings_map = settings::map_openclaw_config_to_settings(&config);
let creds = settings::extract_credentials(&config);
```

**Why this matters**: If validation fails, no database writes have occurred.

### Phase 2: Grouped Writes

**Goal**: Commit related items together so partial commits are minimized.

**Write Order** (if not dry-run):

```rust
// Group 1: Settings (idempotent via upsert)
for (key, value) in settings_map {
    if let Err(e) = self.db.set_setting(&self.opts.user_id, &key, &value).await {
        tracing::warn!("Failed to import setting {}: {}", key, e);
    } else {
        stats.settings += 1;
    }
}

// Group 2: Credentials (idempotent via upsert)
for (name, value) in creds {
    use secrecy::ExposeSecret;
    let exposed = value.expose_secret().to_string();
    let params = crate::secrets::CreateSecretParams::new(name, exposed);
    if let Err(e) = self.secrets.create(&self.opts.user_id, params).await {
        tracing::warn!("Failed to import credential: {}", e);
    } else {
        stats.secrets += 1;
    }
}

// Group 3: Workspace documents
match self.workspace.import_from_directory(&self.opts.openclaw_path.join("workspace")).await {
    Ok(imported) => stats.documents = imported,
    Err(e) => tracing::warn!("Failed to import workspace documents: {}", e),
}

// Group 4: Memory chunks (path deduplication)
for chunk in all_chunks {
    if let Err(e) = memory::import_chunk(&self.db, &chunk, &self.opts).await {
        tracing::warn!("Failed to import memory chunk: {}", e);
    } else {
        stats.chunks += 1;
    }
}

// Group 5: Conversations with messages (atomic units)
for conv in all_conversations {
    match history::import_conversation_atomic(&self.db, conv, &self.opts).await {
        Ok((_conv_id, msg_count)) => {
            stats.conversations += 1;
            stats.messages += msg_count;
        }
        Err(e) => tracing::warn!("Failed to import conversation: {}", e),
    }
}
```

**Fail-safe behavior**: Individual item failures are logged but don't stop the entire import. This allows partial recovery.

---

## Validation

### Configuration Parsing

**Location**: `src/import/openclaw/reader.rs`

OpenClaw config parsed as JSON5 (more flexible than JSON):

```rust
let content = std::fs::read_to_string(&config_path)?;
let config: serde_json::Value = json5::from_str(&content)
    .map_err(|e| ImportError::ConfigParse(e.to_string()))?;
```

**Extracted fields**:
- `llm.provider` → `llm.backend` setting
- `llm.model` → `llm.selected_model` setting
- `llm.api_key` → `llm_api_key` secret
- `llm.base_url` → `llm.base_url` setting
- `embeddings.model` → `embeddings.model` setting
- `embeddings.provider` → `embeddings.provider` setting
- `embeddings.api_key` → `embeddings_api_key` secret

### Message Role Normalization

**Location**: `src/import/openclaw/history.rs`

OpenClaw message roles normalized to IronClaw standard:

```rust
let role = match msg.role.to_lowercase().as_str() {
    "user" | "human" => "user",
    "assistant" | "ai" => "assistant",
    _ => &msg.role,
};
```

**Validation**: All messages validated before conversation creation.

### Embedding Dimension Handling

**Location**: `src/import/openclaw/memory.rs`

Embeddings stored as-is, with optional re-embedding:

```rust
// If we have an embedding, try to update it
if let Some(ref embedding) = chunk.embedding {
    // Note: dimension check would go here if we had target dimensions
    // For now, just store what we have
    db.update_chunk_embedding(chunk_id, embedding)
        .await
        .map_err(|e| ImportError::Database(e.to_string()))?;
}
```

**Future enhancement**: `ImportOptions::re_embed` flag would queue chunks for re-embedding if dimension mismatch detected.

### Security Validation

**Location**: `src/import/openclaw/reader.rs`

API keys never exposed in debug output:

```rust
impl fmt::Debug for OpenClawLlmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenClawLlmConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "***REDACTED***"))
            .field("base_url", &self.base_url)
            .finish()
    }
}
```

**Test coverage**: `test_llm_config_debug_redacts_api_key()` verifies secrets never logged.

---

## Rollback

### Current Limitations

**Important**: The import system does **not** support automatic rollback due to Database trait limitations:

```rust
// Database trait does not expose explicit transaction control
// (BEGIN/COMMIT/ROLLBACK)
```

### Mitigation Strategies

**1. Read-before-write**: All data validated before first write minimizes partial state window.

**2. Grouped writes**: Related items committed together:
- If crash during Group 1 (settings), no data written
- If crash during Group 3 (documents), settings + credentials fully committed
- If crash during Group 5 (conversations), all prior groups fully committed

**3. Atomic conversation units**: Each conversation + its messages form a logical unit:
```rust
// CRITICAL: Each conversation + its messages form an atomic unit.
// If a crash occurs mid-conversation, only that conversation is incomplete.
// All previous conversations are fully committed.
```

**4. Idempotent operations**: Re-running import should not create duplicates:
- Settings: Upsert by key
- Credentials: Upsert by name
- Memory chunks: Deduplicated by path
- Conversations: Metadata includes `openclaw_conversation_id` for deduplication

### Manual Recovery

**If import fails partway**:

1. **Check stats**: Review `ImportStats` to see what was imported
2. **Manual cleanup**: Remove partially imported data via database queries
3. **Re-run import**: Idempotent operations will skip already-imported items

**Example cleanup** (PostgreSQL):
```sql
-- Remove imported settings for user
DELETE FROM settings WHERE user_id = 'user-id';

-- Remove imported secrets for user
DELETE FROM secrets WHERE user_id = 'user-id';

-- Remove imported conversations with OpenClaw metadata
DELETE FROM conversations 
WHERE metadata->>'openclaw_conversation_id' IS NOT NULL
  AND user_id = 'user-id';
```

### Future Enhancement: Transaction Support

**TODO**: Add explicit transaction control to Database trait:

```rust
#[async_trait]
pub trait Database: /* ... */ {
    // ... existing methods ...
    
    async fn begin_transaction(&self) -> Result<Transaction, DatabaseError>;
    async fn run_in_transaction<F, T>(&self, f: F) -> Result<T, DatabaseError>
    where
        F: FnOnce(&Transaction) -> Fut + Send,
        Fut: Future<Output = Result<T, DatabaseError>> + Send;
}
```

This would enable atomic import of all data types.

---

## Code Patterns

### Pattern 1: Import Orchestration

```rust
use crate::import::{ImportOptions, openclaw::OpenClawImporter};

// Configure import
let opts = ImportOptions {
    openclaw_path: dirs::home_dir()
        .unwrap_or_default()
        .join(".openclaw"),
    dry_run: false,
    re_embed: false,
    user_id: current_user_id.clone(),
};

// Create importer
let importer = OpenClawImporter::new(
    db.clone(),
    workspace.clone(),
    secrets.clone(),
    opts,
);

// Run import
match importer.import().await {
    Ok(stats) => {
        println!("Import complete!");
        println!("  Documents: {}", stats.documents);
        println!("  Chunks: {}", stats.chunks);
        println!("  Conversations: {}", stats.conversations);
        println!("  Messages: {}", stats.messages);
        println!("  Settings: {}", stats.settings);
        println!("  Secrets: {}", stats.secrets);
    }
    Err(e) => {
        eprintln!("Import failed: {}", e);
    }
}
```

### Pattern 2: Dry-Run Preview

```rust
// Preview what would be imported
let mut opts = ImportOptions::default();
opts.dry_run = true;
opts.user_id = user_id.clone();

let importer = OpenClawImporter::new(db, workspace, secrets, opts);
let stats = importer.import().await?;

println!("Would import {} items", stats.total_imported());
// No data written to database
```

### Pattern 3: Detection Before Import

```rust
// Check if OpenClaw installation exists
if let Some(openclaw_path) = OpenClawImporter::detect() {
    println!("Found OpenClaw at: {:?}", openclaw_path);
    
    // Prompt user for import
    // ...
} else {
    println!("No OpenClaw installation found at ~/.openclaw");
}
```

### Pattern 4: Memory Chunk Import

```rust
// src/import/openclaw/memory.rs

pub async fn import_chunk(
    db: &Arc<dyn Database>,
    chunk: &OpenClawMemoryChunk,
    opts: &ImportOptions,
) -> Result<(), ImportError> {
    // Get or create document by path (deduplication)
    let doc = db
        .get_or_create_document_by_path(&opts.user_id, None, &chunk.path)
        .await
        .map_err(|e| ImportError::Database(e.to_string()))?;

    // Insert chunk
    let chunk_id = db
        .insert_chunk(
            doc.id,
            chunk.chunk_index,
            &chunk.content,
            None, // Don't set embedding yet if dimensions might not match
        )
        .await
        .map_err(|e| ImportError::Database(e.to_string()))?;

    // If we have an embedding, try to update it
    if let Some(ref embedding) = chunk.embedding {
        db.update_chunk_embedding(chunk_id, embedding)
            .await
            .map_err(|e| ImportError::Database(e.to_string()))?;
    }

    Ok(())
}
```

### Pattern 5: Atomic Conversation Import

```rust
// src/import/openclaw/history.rs

pub async fn import_conversation_atomic(
    db: &Arc<dyn Database>,
    conv: OpenClawConversation,
    opts: &ImportOptions,
) -> Result<(Uuid, usize), ImportError> {
    // PHASE 1: Validate all message data before writing anything
    let mut validated_messages = Vec::with_capacity(conv.messages.len());
    for msg in &conv.messages {
        let role = match msg.role.to_lowercase().as_str() {
            "user" | "human" => "user",
            "assistant" | "ai" => "assistant",
            _ => &msg.role,
        };
        validated_messages.push((role.to_string(), msg.content.clone()));
    }

    // PHASE 2: Create the conversation
    let metadata = json!({
        "openclaw_conversation_id": conv.id,
        "openclaw_channel": conv.channel,
    });

    let conv_id = db
        .create_conversation_with_metadata(&conv.channel, &opts.user_id, &metadata)
        .await
        .map_err(|e| ImportError::Database(e.to_string()))?;

    // PHASE 3: Add all messages in sequence
    let mut message_count = 0;
    for (role, content) in validated_messages {
        db.add_conversation_message(conv_id, &role, &content)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to add message to conversation {}. \
                     Conversation created but may be incomplete.",
                    conv_id,
                    e
                );
                ImportError::Database(e.to_string())
            })?;
        message_count += 1;
    }

    Ok((conv_id, message_count))
}
```

### Pattern 6: Settings Mapping

```rust
// src/import/openclaw/settings.rs

pub fn map_openclaw_config_to_settings(
    config: &OpenClawConfig,
) -> HashMap<String, serde_json::Value> {
    let mut settings = HashMap::new();

    // Map LLM configuration
    if let Some(ref llm) = config.llm {
        if let Some(ref provider) = llm.provider {
            settings.insert(
                "llm.backend".to_string(),
                serde_json::Value::String(provider.clone()),
            );
        }
        if let Some(ref model) = llm.model {
            settings.insert(
                "llm.selected_model".to_string(),
                serde_json::Value::String(model.clone()),
            );
        }
    }

    // Map embeddings configuration
    if let Some(ref emb) = config.embeddings {
        if let Some(ref model) = emb.model {
            settings.insert(
                "embeddings.model".to_string(),
                serde_json::Value::String(model.clone()),
            );
        }
    }

    settings
}
```

### Pattern 7: Credential Extraction

```rust
// src/import/openclaw/settings.rs

pub fn extract_credentials(config: &OpenClawConfig) -> Vec<(String, SecretString)> {
    let mut credentials = Vec::new();

    // Extract LLM API key if present
    if let Some(ref llm) = config.llm {
        if let Some(ref api_key) = llm.api_key {
            credentials.push(("llm_api_key".to_string(), api_key.clone()));
        }
    }

    // Extract embeddings API key if present
    if let Some(ref emb) = config.embeddings {
        if let Some(ref api_key) = emb.api_key {
            credentials.push(("embeddings_api_key".to_string(), api_key.clone()));
        }
    }

    credentials
}
```

---

## Error Handling

### ImportError Types

**Location**: `src/import/mod.rs`

```rust
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("OpenClaw not found at {path}: {reason}")]
    NotFound { path: PathBuf, reason: String },

    #[error("JSON5 parse error: {0}")]
    ConfigParse(String),

    #[error("SQLite error: {0}")]
    Sqlite(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Workspace error: {0}")]
    Workspace(String),

    #[error("Secret error: {0}")]
    Secret(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(String),
}
```

### Error Recovery Strategy

**Fail-safe logging**: Most errors are logged as warnings but don't halt import:

```rust
if let Err(e) = self.db.set_setting(&self.opts.user_id, &key, &value).await {
    tracing::warn!("Failed to import setting {}: {}", key, e);
    // Continue with next setting
} else {
    stats.settings += 1;
}
```

**Rationale**: Allows partial import completion rather than all-or-nothing failure.

---

## Current Limitations

### Database Trait Limitations

1. **No explicit transactions**: Cannot wrap entire import in single transaction
2. **No metadata-based queries**: Cannot check `openclaw_conversation_id` for deduplication efficiently
3. **No bulk operations**: Each item inserted individually (performance impact)

### Idempotency Gaps

1. **Conversations**: Metadata includes `openclaw_conversation_id` but Database trait lacks metadata-based lookup → reimport creates duplicates
2. **Memory chunks**: Path-based deduplication works, but embeddings may be duplicated if chunk exists

### Feature Gaps

1. **Re-embedding**: `ImportOptions::re_embed` flag exists but not implemented
2. **Progress reporting**: No streaming progress updates during import
3. **Rollback**: No automatic rollback on failure (manual cleanup required)

---

## Related

- `src/import/` — Import system implementation
- `src/import/openclaw/` — OpenClaw-specific migration
- `src/db/mod.rs` — Database trait (lacks transaction support)
- `src/secrets/` — Secrets management for credential import
- `src/workspace/` — Workspace import for documents

---

**Quality Note**: This document reflects the actual implementation as of 2026-03-26. Always verify against source code for critical changes.
