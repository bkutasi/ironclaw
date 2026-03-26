# Secrets and Configuration Management

**Purpose**: How IronClaw manages encrypted secrets, bootstrap configuration, and DB-backed settings with proper precedence

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Core Concept

IronClaw uses a **three-layer configuration architecture** with strict precedence: **Environment Variables** (highest) → **TOML/Bootstrap Files** (`~/.ironclaw/.env`, `config.toml`) → **Database Settings** → **Defaults** (lowest). Secrets are encrypted with AES-256-GCM and stored in PostgreSQL/libSQL, while bootstrap config (like `DATABASE_URL`) lives in `.env` files.

---

## Key Points

- **Three-layer precedence**: Env vars > TOML/bootstrap files > DB settings > defaults
- **Encrypted secrets**: AES-256-GCM with per-secret HKDF key derivation
- **Master key storage**: OS keychain (macOS/Linux) or `SECRETS_MASTER_KEY` env var
- **Bootstrap config**: `~/.ironclaw/.env` for chicken-and-egg settings (DB URL, backend type)
- **DB-backed settings**: User preferences stored in database, re-loadable at runtime
- **Thread-safe injection**: Secrets injected into env overlay without unsafe `set_var` calls
- **LLM re-resolution**: Config re-resolves LLM providers after secrets are loaded

---

## Quick Example

```bash
# 1. Master key → OS keychain (auto) or env var (CI/Docker)
export SECRETS_MASTER_KEY="0123456789abcdef..."  # 64 hex chars (32 bytes)

# 2. Bootstrap config → ~/.ironclaw/.env (loaded before DB connection)
DATABASE_URL="postgres://user:pass@localhost:5432/ironclaw"
DATABASE_BACKEND="postgres"
ONBOARD_COMPLETED="true"

# 3. TOML config → ~/.ironclaw/config.toml (overrides DB, overridden by env)
[agent]
name = "my-agent"
max_parallel_jobs = 10

[llm]
backend = "anthropic"

# 4. Secrets stored encrypted in DB (via setup wizard or API)
# Example: openai_api_key, anthropic_oauth_token, stripe_key
```

---

## 3-Layer Architecture

### Layer 1: Environment Variables (Highest Priority)

**Purpose**: Runtime overrides, secrets, CI/CD configuration

**Sources**:
- Process environment (`std::env::var`)
- Runtime overlay (`INJECTED_VARS` - thread-safe map for secrets)
- `RUNTIME_ENV_OVERRIDES` - dynamic overrides via `set_runtime_env()`

**Priority Order**:
```
Real env vars > RUNTIME_ENV_OVERRIDES > INJECTED_VARS (secrets from DB)
```

**Key Variables**:
```bash
# Secrets
SECRETS_MASTER_KEY          # Master encryption key (32+ bytes)

# Database
DATABASE_URL                # Connection string (postgres://...)
DATABASE_BACKEND            # "postgres" or "libsql"
DATABASE_POOL_SIZE          # Connection pool size

# LLM Providers
LLM_BACKEND                 # "nearai", "anthropic", "openai", "ollama", etc.
NEARAI_API_KEY              # NEAR AI API key
ANTHROPIC_OAUTH_TOKEN       # Anthropic OAuth token
OPENAI_API_KEY              # OpenAI API key

# Features
BUILDER_ENABLED             # Enable/disable builder mode
SANDBOX_POLICY              # "readonly", "workspace_write", "full_access"
```

**Code Pattern** (`src/config/helpers.rs`):
```rust
// Check real env first, then runtime overlays, then injected secrets
pub fn optional_env(key: &str) -> Result<Option<String>, ConfigError> {
    // 1. Real env vars (always win)
    if let Ok(val) = std::env::var(key) {
        if !val.is_empty() {
            return Ok(Some(val));
        }
    }
    
    // 2. Runtime overrides (set via set_runtime_env)
    if let Some(val) = RUNTIME_ENV_OVERRIDES.get(key) {
        return Ok(Some(val));
    }
    
    // 3. Injected secrets from DB (INJECTED_VARS)
    if let Some(val) = INJECTED_VARS.get(key) {
        return Ok(Some(val));
    }
    
    Ok(None)
}
```

### Layer 2: Bootstrap & TOML Files

**Purpose**: Persistent configuration that survives restarts, loaded before DB connection

**Files**:
- `~/.ironclaw/.env` - Bootstrap vars (dotenvy format)
- `~/.ironclaw/config.toml` - Full config overlay (TOML format)

**Bootstrap Loading** (`src/bootstrap.rs`):
```rust
/// Load env vars from ~/.ironclaw/.env
/// Priority: explicit env vars > ./.env > ~/.ironclaw/.env > auto-detect
pub fn load_ironclaw_env() {
    let path = ironclaw_env_path();
    
    // Load .env file (dotenvy never overwrites existing vars)
    if path.exists() {
        let _ = dotenvy::from_path(&path);
    }
    
    // Auto-detect libsql if DATABASE_BACKEND unset and ironclaw.db exists
    if std::env::var("DATABASE_BACKEND").is_err() {
        let default_db = dirs::home_dir()
            .unwrap_or_default()
            .join(".ironclaw")
            .join("ironclaw.db");
        if default_db.exists() {
            crate::config::set_runtime_env("DATABASE_BACKEND", "libsql");
        }
    }
}
```

**Writing Bootstrap Vars**:
```rust
// Write DATABASE_URL to ~/.ironclaw/.env (creates parent dirs, sets 0o600 perms)
pub fn save_database_url(url: &str) -> std::io::Result<()> {
    save_bootstrap_env(&[("DATABASE_URL", url)])
}

// Update or add single var, preserving existing content
pub fn upsert_bootstrap_var(key: &str, value: &str) -> std::io::Result<()> {
    // Reads existing .env, replaces line for key if exists, appends otherwise
}

// Batch update multiple vars
pub fn upsert_bootstrap_vars(vars: &[(&str, &str)]) -> std::io::Result<()> {
    // Preserves user-added vars not in the update list
}
```

**TOML Config Overlay**:
```rust
// Load from DB, then overlay TOML (TOML values win over DB)
pub async fn from_db_with_toml(
    store: &(dyn SettingsStore + Sync),
    user_id: &str,
    toml_path: Option<&Path>,
) -> Result<Self, ConfigError> {
    let mut db_settings = store.get_all_settings(user_id).await?;
    
    // Overlay TOML (values win over DB settings)
    Self::apply_toml_overlay(&mut db_settings, toml_path)?;
    
    Self::build(&db_settings).await
}
```

### Layer 3: Database-Backed Settings

**Purpose**: User preferences, feature flags, runtime-modifiable configuration

**Storage**: Flat key-value map in `settings` table (JSONB values)

**Structure** (`src/settings.rs`):
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    // Bootstrap config (persisted but requires env for chicken-and-egg)
    pub owner_id: Option<String>,
    pub database_backend: Option<String>,
    pub database_url: Option<String>,
    
    // Secrets config
    pub secrets_master_key_source: KeySource,  // "keychain", "env", "none"
    
    // LLM config
    pub llm_backend: Option<String>,
    pub selected_model: Option<String>,
    pub ollama_base_url: Option<String>,
    
    // Nested settings
    pub agent: AgentSettings,
    pub embeddings: EmbeddingsSettings,
    pub channels: ChannelSettings,
    pub heartbeat: HeartbeatSettings,
    pub wasm: WasmSettings,
    pub sandbox: SandboxSettings,
    pub safety: SafetySettings,
    pub builder: BuilderSettings,
}
```

**DB Map Conversion**:
```rust
// Flatten Settings to dotted paths for DB storage
// Example: "agent.name" => "ironclaw", "agent.max_parallel_jobs" => "5"
pub fn to_db_map(&self) -> HashMap<String, serde_json::Value> {
    let json = serde_json::to_value(self).unwrap();
    let mut map = HashMap::new();
    collect_settings_json(&json, String::new(), &mut map);
    map.remove("owner_id");  // Don't persist owner_id in DB
    map
}

// Reconstruct Settings from DB map
pub fn from_db_map(map: &HashMap<String, serde_json::Value>) -> Self {
    let mut settings = Self::default();
    for (key, value) in map {
        settings.set(key, &value.to_string()).ok();
    }
    settings
}
```

**Merge Priority** (TOML > DB):
```rust
/// Merge values from `other` into `self`, preferring `other` for
/// fields that differ from default (used for TOML overlay)
pub fn merge_from(&mut self, other: &Self) {
    let default_json = serde_json::to_value(Self::default()).unwrap();
    let other_json = serde_json::to_value(other).unwrap();
    let mut self_json = serde_json::to_value(&*self).unwrap();
    
    // Only merge non-default values from other
    merge_non_default(&mut self_json, &other_json, &default_json);
    
    *self = serde_json::from_value(self_json).unwrap();
}
```

---

## Secrets Encryption

### Security Model

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Secret Lifecycle                                │
│                                                                              │
│   User stores secret ──► Encrypt with AES-256-GCM ──► Store in PostgreSQL  │
│                          (per-secret key via HKDF)                          │
│                                                                              │
│   WASM requests HTTP ──► Host checks allowlist ──► Decrypt secret ──►       │
│                          & allowed_secrets        (in memory only)           │
│                                                         │                    │
│                                                         ▼                    │
│                          Inject into request ──► Execute HTTP call          │
│                          (WASM never sees value)                            │
│                                                         │                    │
│                                                         ▼                    │
│                          Leak detector scans ──► Return response to WASM   │
│                          response for secrets                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cryptographic Details (`src/secrets/crypto.rs`)

**Algorithm**: AES-256-GCM with per-secret key derivation via HKDF-SHA256

**Key Derivation**:
```text
master_key (from env) ─┬─► HKDF-SHA256 ─► derived_key (per secret)
                       │
per-secret salt ───────┘
```

**Constants**:
```rust
const KEY_SIZE: usize = 32;      // AES-256 key (32 bytes)
const NONCE_SIZE: usize = 12;    // GCM nonce (12 bytes)
const SALT_SIZE: usize = 32;     // Per-secret salt (32 bytes)
const TAG_SIZE: usize = 16;      // GCM auth tag (16 bytes)
```

**Encryption Flow**:
```rust
pub fn encrypt(&self, plaintext: &[u8]) -> Result<(Vec<u8>, Vec<u8>), SecretError> {
    // 1. Generate random salt for this secret
    let salt = Self::generate_salt();
    
    // 2. Derive per-secret key via HKDF
    let derived_key = self.derive_key(&salt)?;
    
    // 3. Create AES-256-GCM cipher
    let cipher = Aes256Gcm::new_from_slice(&derived_key)?;
    
    // 4. Generate random nonce
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    
    // 5. Encrypt: ciphertext = plaintext + auth_tag
    let ciphertext = cipher.encrypt(&nonce, plaintext)?;
    
    // 6. Combine: encrypted_value = nonce || ciphertext || tag
    let mut encrypted = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    encrypted.extend_from_slice(&nonce);
    encrypted.extend_from_slice(&ciphertext);
    
    Ok((encrypted, salt))
}
```

**Decryption Flow**:
```rust
pub fn decrypt(
    &self,
    encrypted_value: &[u8],
    salt: &[u8],
) -> Result<DecryptedSecret, SecretError> {
    // 1. Validate encrypted_value length (must have nonce + tag)
    if encrypted_value.len() < NONCE_SIZE + TAG_SIZE {
        return Err(SecretError::DecryptionFailed("too short".into()));
    }
    
    // 2. Derive same key using stored salt
    let derived_key = self.derive_key(salt)?;
    
    // 3. Split: nonce || ciphertext
    let (nonce_bytes, ciphertext) = encrypted_value.split_at(NONCE_SIZE);
    let nonce = Nonce::from_slice(nonce_bytes);
    
    // 4. Decrypt (GCM verifies auth tag automatically)
    let plaintext = cipher.decrypt(nonce, ciphertext)?;
    
    DecryptedSecret::from_bytes(plaintext)
}
```

### Master Key Storage (`src/secrets/keychain.rs`)

**Resolution Order**:
1. `SECRETS_MASTER_KEY` environment variable (hex-encoded)
2. OS keychain (macOS Keychain / Linux secret-service)
3. None (secrets disabled)

**Platform Support**:
- **macOS**: `security-framework` (Keychain Services)
- **Linux**: `secret-service` (GNOME Keyring, KWallet)
- **Other**: Env var only

**OS Keychain Integration**:
```rust
// macOS: Store in Keychain
pub async fn store_master_key(key: &[u8]) -> Result<(), SecretError> {
    let key_hex = key.iter().map(|b| format!("{:02x}", b)).collect();
    set_generic_password("ironclaw", "master_key", key_hex.as_bytes())
}

// macOS: Retrieve from Keychain
pub async fn get_master_key() -> Result<Vec<u8>, SecretError> {
    let password = get_generic_password("ironclaw", "master_key")?;
    let hex_str = String::from_utf8(password)?;
    hex_to_bytes(&hex_str)
}
```

---

## Bootstrap Loading

### Startup Sequence

```text
1. Early Bootstrap (before main())
   └─► Load ./.env (project-local, higher priority)
   └─► Load ~/.ironclaw/.env (user-global, lower priority)
   └─► Auto-detect DATABASE_BACKEND=libsql if ironclaw.db exists

2. Config Loading (Config::from_env or Config::from_db)
   └─► Load bootstrap settings from env/.env
   └─► Load DB settings (if DB connected)
   └─► Apply TOML overlay (~/.ironclaw/config.toml)
   └─► Build Config struct with resolved values

3. Secrets Injection (AppBuilder::init_secrets)
   └─► Resolve master key (env > OS keychain)
   └─► Create SecretsCrypto instance
   └─► Load encrypted secrets from DB
   └─► Decrypt and inject into INJECTED_VARS overlay
   └─► Re-resolve LLM config (picks up injected keys)

4. Runtime Ready
   └─► All config layers merged
   └─► Secrets available via optional_env()
   └─► LLM providers configured with API keys
```

### Bootstrap File Management

**Writing `.env`** (with escaping for special chars):
```rust
pub fn save_bootstrap_env_to(
    path: &Path,
    vars: &[(&str, &str)],
) -> std::io::Result<()> {
    // Create parent dirs
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Escape backslashes and quotes (prevent injection)
    let mut content = String::new();
    for (key, value) in vars {
        let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
        content.push_str(&format!("{}=\"{}\"\n", key, escaped));
    }
    
    std::fs::write(path, &content)?;
    restrict_file_permissions(path)?;  // chmod 0o600 on Unix
    Ok(())
}
```

**Upserting Vars** (preserves user additions):
```rust
pub fn upsert_bootstrap_vars_to(
    path: &Path,
    vars: &[(&str, &str)],
) -> std::io::Result<()> {
    let keys_being_written: HashSet<&str> = vars.iter().map(|(k, _)| *k).collect();
    
    // Read existing content
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    
    // Keep lines not being updated
    let mut result = String::new();
    for line in existing.lines() {
        let is_overwritten = line
            .split_once('=')
            .map(|(k, _)| keys_being_written.contains(k.trim()))
            .unwrap_or(false);
        
        if !is_overwritten {
            result.push_str(line);
            result.push('\n');
        }
    }
    
    // Append new/updated vars
    for (key, value) in vars {
        let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
        result.push_str(&format!("{}=\"{}\"\n", key, escaped));
    }
    
    std::fs::write(path, &result)?;
    Ok(())
}
```

---

## DB-Backed Settings

### Settings Table Schema

```sql
CREATE TABLE settings (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, key)
);
```

### Key-Value Storage

**Example Entries**:
```
user_id | key                      | value
--------|--------------------------|---------------------------
"default"| agent.name              | "ironclaw"
"default"| agent.max_parallel_jobs | 5
"default"| llm_backend             | "anthropic"
"default"| selected_model          | "claude-3-5-sonnet-20241022"
"default"| heartbeat.enabled       | false
"default"| sandbox.policy          | "readonly"
```

### Settings Store Trait (`src/db/*.rs`)

```rust
#[async_trait]
pub trait SettingsStore: Send + Sync {
    /// Get all settings as flat key-value map
    async fn get_all_settings(
        &self,
        user_id: &str,
    ) -> Result<HashMap<String, serde_json::Value>, DbError>;
    
    /// Set a single setting
    async fn set_setting(
        &self,
        user_id: &str,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<(), DbError>;
    
    /// Set all settings (bulk upsert)
    async fn set_all_settings(
        &self,
        user_id: &str,
        settings: &HashMap<String, serde_json::Value>,
    ) -> Result<(), DbError>;
    
    /// Check if user has any settings
    async fn has_settings(&self, user_id: &str) -> Result<bool, DbError>;
}
```

### Dotted Path Access

```rust
// Get nested setting by dotted path
settings.get("agent.name")           // Some("ironclaw")
settings.get("agent.max_parallel_jobs") // Some("5")
settings.get("heartbeat.enabled")    // Some("false")

// Set nested setting
settings.set("agent.name", "mybot")?;
settings.set("heartbeat.enabled", "true")?;
settings.set("channels.wasm_channel_owner_ids.telegram", "123456")?;

// List all settings
let list = settings.list();  // Vec<(String, String)>
```

---

## Code Patterns

### Loading Config at Startup

```rust
use ironclaw::config::Config;
use ironclaw::secrets::{SecretsCrypto, SecretsStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load bootstrap env (~/.ironclaw/.env)
    crate::bootstrap::load_ironclaw_env();
    
    // 2. Load config from env (before DB connection)
    let mut config = Config::from_env().await?;
    
    // 3. Connect to database
    let db = crate::db::Database::connect(&config.database).await?;
    
    // 4. Re-load config from DB with TOML overlay
    config = Config::from_db_with_toml(
        &db.settings_store(),
        &config.owner_id,
        Some(&Settings::default_toml_path()),
    ).await?;
    
    // 5. Initialize secrets (inject API keys from encrypted store)
    if let Some(master_key) = config.secrets.master_key() {
        let crypto = Arc::new(SecretsCrypto::new(master_key.clone())?);
        let secrets_store = create_secrets_store(crypto, &db.handles());
        
        if let Some(store) = &secrets_store {
            // Inject LLM keys from DB into env overlay
            inject_llm_keys_from_secrets(store.as_ref(), &config.owner_id).await;
            
            // Re-resolve LLM config (picks up injected keys)
            config.re_resolve_llm(
                Some(db.settings_store()),
                &config.owner_id,
                Some(&Settings::default_toml_path()),
            ).await?;
        }
    }
    
    // 6. Config ready - all layers merged
    println!("Config loaded: owner={}, llm={}", 
             config.owner_id, 
             config.llm.backend);
    
    Ok(())
}
```

### Injecting Secrets into Env Overlay

```rust
use ironclaw::config::inject_llm_keys_from_secrets;
use ironclaw::secrets::SecretsStore;

/// Load API keys from encrypted secrets into thread-safe overlay
pub async fn inject_llm_keys_from_secrets(
    secrets: &dyn SecretsStore,
    user_id: &str,
) {
    // Well-known provider mappings
    let mappings: Vec<(&str, &str)> = vec![
        ("llm_nearai_api_key", "NEARAI_API_KEY"),
        ("llm_anthropic_oauth_token", "ANTHROPIC_OAUTH_TOKEN"),
    ];
    
    // Dynamic mappings from provider registry
    let registry = ProviderRegistry::load();
    let dynamic: Vec<(String, String)> = registry
        .selectable()
        .iter()
        .filter_map(|def| {
            def.api_key_env.as_ref().and_then(|env_var| {
                def.setup.as_ref()
                    .and_then(|s| s.secret_name())
                    .map(|secret_name| (secret_name.to_string(), env_var.clone()))
            })
        })
        .collect();
    
    // Load each secret (skip if explicit env var set)
    let mut injected = HashMap::new();
    for (secret_name, env_var) in mappings.iter().chain(dynamic.iter()) {
        // Skip if already set in real env
        if let Ok(val) = std::env::var(env_var) {
            if !val.is_empty() {
                continue;
            }
        }
        
        // Decrypt from DB
        if let Ok(decrypted) = secrets.get_decrypted(user_id, secret_name).await {
            injected.insert(env_var.to_string(), decrypted.expose().to_string());
            tracing::debug!("Loaded secret '{}' for '{}'", secret_name, env_var);
        }
    }
    
    // Also load from OS credential store (no DB required)
    inject_os_credential_store_tokens(&mut injected);
    
    // Merge into global overlay (visible to optional_env())
    merge_injected_vars(injected);
}
```

### Thread-Safe Env Overrides

```rust
use ironclaw::config::{set_runtime_env, env_or_override};

// Set runtime override (thread-safe alternative to unsafe set_var)
set_runtime_env("NEARAI_API_KEY", "nra_...");

// Read with priority: real env > runtime override > injected secrets
if let Some(api_key) = env_or_override("NEARAI_API_KEY") {
    println!("API key available: {}...", &api_key[..8]);
}

// Used by config resolution
pub fn optional_env(key: &str) -> Result<Option<String>, ConfigError> {
    // Checks real env, then RUNTIME_ENV_OVERRIDES, then INJECTED_VARS
}
```

---

## Verification

```bash
# Check bootstrap config loaded
./target/release/ironclaw 2>&1 | grep -E "(Loaded configuration|DATABASE_BACKEND)"

# Check secrets master key source
./target/release/ironclaw config get secrets_master_key_source

# List all DB-backed settings
./target/release/ironclaw config list

# Check if secrets are loaded
./target/release/ironclaw secrets list

# Verify env var precedence
export TEST_VAR="from_shell"
echo "TEST_VAR=from_dotenv" >> ~/.ironclaw/.env
./target/release/ironclaw  # Should see "from_shell" (env wins)
```

---

## Reference

- Secrets module: `src/secrets/`
  - Crypto: `src/secrets/crypto.rs`
  - Store: `src/secrets/store.rs`
  - Keychain: `src/secrets/keychain.rs`
  - Types: `src/secrets/types.rs`
- Config module: `src/config/`
  - Main: `src/config/mod.rs`
  - Helpers: `src/config/helpers.rs`
  - Bootstrap: `src/bootstrap.rs`
  - Secrets: `src/config/secrets.rs`
- Settings: `src/settings.rs`
- DB stores: `src/db/*/settings.rs`

---

**Related**:
- architecture/concepts/config-precedence.md
- architecture/lookup/env-variables.md
- architecture/guides/env-config.md
