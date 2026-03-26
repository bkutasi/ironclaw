<!-- Context: architecture/concepts | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->
# Extensions System

**Purpose**: Runtime lifecycle management for channels, tools, and MCP server integrations

**Last Updated**: 2026-03-26

---

## Quick Reference

**Four Extension Types**:
- **WasmChannel** — Sandboxed WASM messaging channels (Telegram, HTTP webhooks)
- **WasmTool** — Sandboxed WASM capability modules (file ops, API calls)
- **McpServer** — External Model Context Protocol servers (HTTP, OAuth 2.1)
- **ChannelRelay** — External channel via channel-relay service (Slack)

**Lifecycle Commands**:
```
tool_search("telegram")      → Find in registry or discover online
tool_install("telegram")     → Download/copy WASM or configure MCP
tool_auth("telegram")        → Check/start authentication
tool_activate("telegram")    → Load and register with runtime
tool_remove("telegram")      → Uninstall and clean up
```

**Directories**:
- `~/.ironclaw/channels/` — WASM channel binaries (*.wasm, *.capabilities.json)
- `~/.ironclaw/tools/` — WASM tool binaries
- `src/extensions/` — Manager, registry, discovery logic

---

## Architecture Overview

The extensions system unifies four runtime kinds under a single lifecycle API:

```
┌─────────────────────────────────────────────────────────────┐
│                    ExtensionManager                          │
│  (orchestrates search, install, auth, activate, remove)     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ WasmChannel  │  │  WasmTool    │  │    McpServer     │  │
│  │   (WASM)     │  │   (WASM)     │  │  (HTTP + OAuth)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐                                           │
│  │ ChannelRelay │                                           │
│  │  (External)  │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
  ┌─────────────┐    ┌─────────────┐      ┌─────────────┐
  │   Registry  │    │  Discovery  │      │   Secrets   │
  │  (builtin + │    │  (online +  │      │   Store     │
  │   cached)   │    │  GitHub)    │      │ (OAuth +    │
  └─────────────┘    └─────────────┘      │  tokens)    │
                                          └─────────────┘
```

**Key Components** (`src/extensions/`):
- **`manager.rs`** — Central lifecycle dispatcher (8000+ lines)
- **`registry.rs`** — Curated catalog with fuzzy search
- **`discovery.rs`** — Online MCP server discovery (well-known URLs, GitHub)
- **`mod.rs`** — Shared types (`ExtensionKind`, `RegistryEntry`, `AuthResult`, etc.)

---

## Extension Lifecycle

### 1. Search

```rust
// Search built-in registry first, then discover online if no results
let results = manager.search("telegram", discover: true).await?;
```

**Search Strategy**:
1. Query built-in registry (fuzzy match on name, keywords, description)
2. If empty results, probe common MCP URL patterns (`mcp.{service}.com`)
3. Search GitHub for `topic:mcp-server` repositories
4. Validate discovered URLs via `.well-known/oauth-protected-resource`
5. Cache validated discoveries for future lookups

**Scoring** (`registry.rs:187-224`):
- Exact name match: +100
- Partial name match: +50
- Keyword match: +40
- Display name match: +30
- Description match: +10

### 2. Install

```rust
// Install from registry (preferred)
manager.install("telegram", url: None, kind_hint: None, user_id).await?;

// Or install from explicit URL
manager.install("custom", url: Some("https://..."), kind_hint: Some(McpServer), user_id).await?;
```

**Install Paths by Kind**:

| Kind | Primary Source | Fallback | Destination |
|------|---------------|-----------|-------------|
| `WasmChannel` | WASM download URL | Build from source | `~/.ironclaw/channels/` |
| `WasmTool` | WASM download URL | Build from source | `~/.ironclaw/tools/` |
| `McpServer` | MCP server URL | — | Config + secrets store |
| `ChannelRelay` | Registry only | — | In-memory tracking |

**Install Flow** (`manager.rs:1204-1208`):
1. Validate extension name (alphanumeric + hyphens only)
2. Look up registry entry (prefer `kind_hint` to resolve name collisions)
3. Dispatch to kind-specific installer:
   - WASM: Download binary + capabilities.json
   - MCP: Validate URL, store config
   - Relay: Add to installed set
4. Return `InstallResult` with status message

### 3. Authenticate

```rust
// Check auth status (may initiate OAuth for MCP)
let auth_result = manager.auth("notion", user_id).await?;

match auth_result.status {
    AuthStatus::Authenticated => { /* ready */ }
    AuthStatus::AwaitingAuthorization { auth_url, .. } => { /* open browser */ }
    AuthStatus::AwaitingToken { instructions, .. } => { /* show UI prompt */ }
    AuthStatus::NeedsSetup { setup_url, .. } => { /* configure OAuth creds */ }
    AuthStatus::NoAuthRequired => { /* no auth needed */ }
}
```

**Auth Methods by Kind**:

| Kind | Method | Flow |
|------|--------|------|
| `McpServer` | OAuth 2.1 + DCR | Dynamic Client Registration or pre-configured OAuth |
| `WasmTool` | Capabilities file | Token/key from capabilities.json auth section |
| `WasmChannel` | Bot token / OAuth | Manual token entry or OAuth (e.g., Telegram bot token) |
| `ChannelRelay` | Relay OAuth | OAuth via channel-relay service |

**Gateway Mode** (`manager.rs:596-618`):
- When `should_use_gateway_mode()` returns `true`, OAuth flows return auth URLs to frontend
- Uses `IRONCLAW_OAUTH_CALLBACK_URL` or `tunnel_url` for redirect URIs
- Stores pending flows in `pending_oauth_flows` registry for callback completion

### 4. Activate

```rust
// Activate an installed + authenticated extension
let result = manager.activate("telegram", user_id).await?;
// → ActivateResult { tools_loaded: [...], message: "..." }
```

**Activation by Kind**:

**WasmChannel** (`manager.rs:2000+`):
1. Load WASM module from `~/.ironclaw/channels/{name}.wasm`
2. Parse capabilities.json for endpoints, secrets schema
3. Resolve secrets from store
4. Instantiate WASM with host functions (HTTP, logging, storage)
5. Register endpoints with `WasmChannelRouter`
6. Add to `active_channel_names` set + persist to DB
7. Broadcast status via SSE

**WasmTool** (`manager.rs:1900+`):
1. Load WASM module from `~/.ironclaw/tools/{name}.wasm`
2. Parse capabilities.json for tool definitions
3. Register tools with `ToolRegistry` (prefixed: `{name}_{tool}`)
4. Register hooks if defined (`plugin.tool:{name}::...`)

**McpServer** (`manager.rs:1800+`):
1. Create `McpClient` with server URL
2. Establish SSE/streaming transport
3. List + register tools from MCP server
4. Store client in `mcp_clients` map

**ChannelRelay** (`manager.rs:2200+`):
1. Fetch signing secret from relay `/relay/signing-secret`
2. Configure webhook endpoint at relay
3. Create channel via `ChannelManager`
4. Store `event_tx` for webhook event delivery

### 5. Remove

```rust
// Remove extension + clean up
let message = manager.remove("telegram", user_id).await?;
```

**Cleanup by Kind**:

| Kind | Cleanup Actions |
|------|----------------|
| `WasmChannel` | Remove from active set, delete .wasm + .capabilities.json, revoke credential mappings, persist changes |
| `WasmTool` | Unregister tools + hooks, delete files, evict from WASM cache, revoke credentials |
| `McpServer` | Unregister tools (prefix match), remove MCP client, delete config |
| `ChannelRelay` | Remove from installed set, delete team_id setting, clear webhook state, shutdown channel |

**OAuth Cleanup** (`manager.rs:1489-1501`):
- Abort pending auth listener tasks (TCP mode)
- Remove stale entries from `pending_oauth_flows`

### 6. Upgrade (WASM only)

```rust
// Upgrade all outdated WASM extensions to match host WIT version
let result = manager.upgrade(name: None, user_id).await?;

// Or upgrade a specific extension
let result = manager.upgrade(name: Some("telegram"), user_id).await?;
```

**Upgrade Flow** (`manager.rs:1665-1850`):
1. Read current WIT version from capabilities.json
2. Compare against host WIT version (`WIT_TOOL_VERSION` or `WIT_CHANNEL_VERSION`)
3. If outdated, look up in registry for newer version
4. Delete old .wasm + .capabilities.json (preserve secrets)
5. Reinstall from registry entry
6. Return `UpgradeResult` with per-extension outcomes

---

## MCP Integration

### Discovery (`discovery.rs`)

**Well-Known URL Probing**:
```rust
// Tries patterns like:
https://mcp.{service}.com
https://mcp.{service}.app
https://mcp.{service}.dev
https://{service}.com/mcp
```

**GitHub Search**:
- Query: `api.github.com/search/repositories?q={query}+topic:mcp-server`
- Filters: Repos with `mcp` or `model-context-protocol` topics
- Returns: Repo homepage (potential MCP endpoint) or repo URL

**Validation** (`discovery.rs:204-237`):
1. GET `{origin}/.well-known/oauth-protected-resource` → 200 + JSON = confirmed
2. Fallback: HEAD request to URL itself (accept 200-299, 401, 403, 405)

### OAuth Flow (`manager.rs:1040-1110`)

**Gateway Mode** (web UI):
1. `start_gateway_oauth_flow()` builds CSRF state param
2. Rewrites OAuth URL with platform state
3. Stores flow in `pending_oauth_flows` registry
4. Returns `AuthStatus::AwaitingAuthorization { auth_url }` to frontend
5. Gateway `/oauth/callback` handler completes token exchange

**CLI Mode** (terminal):
1. Opens browser via `open::that()`
2. Spawns TCP listener on port 9876 for callback
3. Exchanges code for token
4. Stores token in secrets store

**DCR (Dynamic Client Registration)**:
- If server supports DCR, auto-registers client at runtime
- No manual OAuth app setup required

---

## Extension Registry

### Structure (`registry.rs`)

```rust
pub struct RegistryEntry {
    pub name: String,              // "telegram", "notion"
    pub display_name: String,      // "Telegram Channel", "Notion"
    pub kind: ExtensionKind,       // WasmChannel, McpServer, etc.
    pub description: String,
    pub keywords: Vec<String>,     // ["messaging", "bot"]
    pub source: ExtensionSource,   // Where to get it
    pub fallback_source: Option<ExtensionSource>,
    pub auth_hint: AuthHint,       // How auth works
    pub version: Option<String>,   // Semver
}
```

### Sources

```rust
pub enum ExtensionSource {
    McpUrl { url: String },
    WasmDownload { wasm_url: String, capabilities_url: Option<String> },
    WasmBuildable { source_dir: String, build_dir: Option<String>, crate_name: Option<String> },
    Discovered { url: String },           // From online discovery
    ChannelRelay { relay_url: String },
}
```

### Auth Hints

```rust
pub enum AuthHint {
    Dcr,                                  // Dynamic Client Registration (zero-config OAuth)
    OAuthPreConfigured { setup_url: String },  // Manual OAuth app setup
    CapabilitiesAuth,                     // Auth defined in capabilities.json
    None,                                 // No auth needed
    ChannelRelayOAuth,                    // OAuth via relay service
}
```

### Search Algorithm (`registry.rs:49-90`)

1. Split query into lowercase tokens
2. Score each entry:
   - Exact name: +100 per token
   - Partial name: +50
   - Keyword: +40
   - Display name: +30
   - Description: +10
3. Sort by score descending
4. Return merged results (builtin + cached discoveries)

**Collision Handling** (`registry.rs:122-148`):
- Same name, different kind → both coexist (e.g., "telegram" as WasmChannel + WasmTool)
- `get_with_kind(name, kind_hint)` resolves collisions
- Deduplication by `(name, kind)` pair in `new_with_catalog()` and `cache_discovered()`

---

## Code Patterns

### Search + Install + Activate

```rust
// User: "add telegram"
let results = manager.search("telegram", discover: true).await?;
let entry = results.first().ok_or("not found")?;

// Install
manager.install(&entry.name, None, None, user_id).await?;

// Auth (may return auth URL for gateway mode)
let auth = manager.auth(&entry.name, user_id).await?;
if let AuthStatus::AwaitingAuthorization { auth_url, .. } = auth.status {
    return Ok(auth_url); // Frontend opens browser
}

// Activate
let result = manager.activate(&entry.name, user_id).await?;
Ok(format!("Activated with tools: {:?}", result.tools_loaded))
```

### Check Auth Status (UI Polling)

```rust
// Web UI polls this to show auth progress
let auth = manager.auth("notion", user_id).await?;
match auth.status {
    AuthStatus::Authenticated => json!({ status: "ready" }),
    AuthStatus::AwaitingAuthorization { auth_url } => {
        json!({ status: "authorizing", auth_url })
    }
    AuthStatus::AwaitingToken { instructions, setup_url } => {
        json!({ status: "needs_token", instructions, setup_url })
    }
    AuthStatus::NeedsSetup { instructions, setup_url } => {
        json!({ status: "needs_setup", instructions, setup_url })
    }
    AuthStatus::NoAuthRequired => json!({ status: "no_auth" }),
}
```

### List Extensions (UI Dashboard)

```rust
// Show installed + available extensions
let extensions = manager.list(kind_filter: None, include_available: true, user_id).await?;

for ext in extensions {
    println!("{} ({}) - installed: {}, active: {}, authenticated: {}",
        ext.name, ext.kind, ext.installed, ext.active, ext.authenticated);
}
```

### Upgrade Check

```rust
// Check all WASM extensions for WIT version mismatches
let result = manager.upgrade(None, user_id).await?;
for outcome in result.results {
    match outcome.status.as_str() {
        "upgraded" => println!("✓ {} upgraded", outcome.name),
        "already_up_to_date" => println!("○ {} up to date", outcome.name),
        "not_in_registry" => println!("⚠ {} not in registry", outcome.name),
        "failed" => println!("✗ {} failed: {}", outcome.name, outcome.detail),
        _ => {}
    }
}
```

---

## 📂 Codebase References

**Core Modules**:
- `src/extensions/mod.rs` — Shared types, enums, serialization
- `src/extensions/manager.rs` — Lifecycle orchestration (8000+ lines)
- `src/extensions/registry.rs` — Catalog + fuzzy search
- `src/extensions/discovery.rs` — Online MCP discovery

**Supporting Infrastructure**:
- `src/tools/mcp/` — MCP client, session, OAuth auth
- `src/tools/wasm/` — WASM tool loader, runtime, capabilities
- `src/channels/wasm/` — WASM channel loader, router, runtime
- `src/channels/relay/` — Channel-relay client, webhook handling
- `src/secrets/` — Encrypted secrets store (OAuth tokens, API keys)
- `src/registry/catalog/` — MCP server catalog (JSON files)

**Type Definitions** (`mod.rs`):
- `ExtensionKind` — WasmChannel, WasmTool, McpServer, ChannelRelay
- `RegistryEntry` — Extension metadata
- `ExtensionSource` — Where to get extension
- `AuthHint` — Authentication method
- `AuthResult` / `AuthStatus` — Auth state machine
- `InstalledExtension` — Runtime status
- `InstallResult` / `ActivateResult` / `ConfigureResult` — Operation outcomes

---

## Deep Dive

**OAuth Implementation**: See `src/tools/mcp/auth.rs` for DCR, token exchange, PKCE
**WASM Runtime**: See `src/tools/wasm/runtime.rs` and `src/channels/wasm/runtime.rs`
**Channel-Relay Protocol**: See `src/channels/relay/client.rs` for webhook signing, event streaming
**Registry Catalog**: See `src/registry/catalog/mod.rs` for MCP server JSON schema

---

## Related

- concepts/wasm-channels.md
- architecture/concepts/config-precedence.md
- standards/security-patterns.md
- tools/mcp-authentication.md (if exists)
