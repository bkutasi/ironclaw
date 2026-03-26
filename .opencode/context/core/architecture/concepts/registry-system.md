<!-- Context: core/architecture/concepts | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->

# Registry System

**Purpose**: Central catalog and installation system for WASM tools, channels, and MCP servers

**Last Updated**: 2026-03-26

> ⚠️ **Quality Warning**: This knowledge was harvested from AI-generated summaries and may contain inaccuracies. Requires constant revision and verification against actual source code and behavior. Do not treat as authoritative reference without validation.

---

## Quick Reference

**Registry Structure**:
```
registry/
├── tools/*.json          # One manifest per WASM tool
├── channels/*.json       # One manifest per WASM channel
├── mcp-servers/*.json    # One manifest per MCP server
└── _bundles.json         # Bundle definitions (google, messaging, default)
```

**Installation Commands**:
```
ironclaw tool install github          # Install from registry
ironclaw tool install --build github  # Build from source (ignore artifacts)
ironclaw tool install google          # Install bundle (all Google tools)
```

**Key Types**:
- `RegistryCatalog` — Loads manifests from disk or embedded
- `RegistryInstaller` — Handles build-from-source and artifact download
- `ExtensionManifest` — Single extension metadata (name, source, artifacts, auth)
- `BundleDefinition` — Grouping of related extensions with shared auth

**Catalog Loading**: Disk (`registry/` dir) → Embedded (compiled JSON) fallback

---

## Architecture Overview

The registry system provides a **unified extension catalog** for discovering, installing, and managing WASM tools, WASM channels, and MCP servers. It supports two distribution modes:

1. **Disk-based** — `registry/` directory with JSON manifests (development, source builds)
2. **Embedded** — Manifests compiled into binary via `build.rs` (release binaries)

```
┌─────────────────────────────────────────────────────────────┐
│                  RegistryCatalog                             │
│  (loads manifests from disk or embedded)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    Tools     │  │   Channels   │  │    MCP Servers   │  │
│  │  (WASM)      │  │   (WASM)     │  │   (HTTP/OAuth)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              _bundles.json                            │  │
│  │  (google, messaging, default bundles + shared auth)   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
   ┌─────────────┐              ┌─────────────┐
   │  Installer  │              │  Discovery  │
   │  (build or  │              │  (search,   │
   │  download)  │              │  resolve)   │
   └─────────────┘              └─────────────┘
```

**Key Components** (`src/registry/`):
- **`catalog.rs`** — Loads manifests, provides list/search/resolve operations (765 lines)
- **`manifest.rs`** — Serde structs for extension metadata (590 lines)
- **`installer.rs`** — Build-from-source and artifact download with fallback (1339 lines)
- **`artifacts.rs`** — WASM artifact resolution and building (377 lines)
- **`embedded.rs`** — Embedded catalog compiled into binary (103 lines)

**Design Principles**:
1. **Dual distribution** — Disk or embedded catalog (transparent fallback)
2. **Artifact-first with source fallback** — Prefer pre-built WASM, build if unavailable
3. **Checksum verification** — SHA256 required for all artifact downloads
4. **Bundle support** — Install related extensions together with shared auth
5. **Collision handling** — Same name across kinds requires qualified lookup (`tools/github` vs `channels/github`)

---

## Registry Catalog

**Location**: `src/registry/catalog.rs`

**Purpose**: Central index of all available extensions with list, search, and resolve operations.

### Loading Strategy

**Priority Order**:
1. **Disk** — Search for `registry/` directory relative to:
   - Current working directory
   - Executable location (up to 3 levels up)
   - `CARGO_MANIFEST_DIR` (compile-time, dev builds)
2. **Embedded** — Fallback to JSON blob compiled by `build.rs`

```rust
// Load from disk or fall back to embedded
let catalog = RegistryCatalog::load_or_embedded()?;

// Or load from specific directory
let catalog = RegistryCatalog::load(&PathBuf::from("registry/"))?;
```

### Directory Structure

```
registry/
├── tools/
│   ├── github.json
│   ├── gmail.json
│   └── slack.json
├── channels/
│   ├── telegram.json
│   ├── discord.json
│   └── slack.json
├── mcp-servers/
│   ├── notion.json
│   └── stripe.json
└── _bundles.json
```

### Manifest Keying

Manifests are keyed by `"{kind}/{name}"`:
- `"tools/github"` — GitHub WASM tool
- `"channels/telegram"` — Telegram WASM channel
- `"mcp-servers/notion"` — Notion MCP server

**Collision Handling**:
- Same name across kinds → both coexist (e.g., `tools/slack` and `channels/slack`)
- Bare name lookup (`catalog.get("slack")`) → returns `None` if ambiguous
- Use qualified key (`catalog.get("tools/slack")`) to disambiguate

### Search & Lookup

**Methods**:
```rust
// Get by exact key or bare name (returns None if ambiguous)
catalog.get("github")           // Ok if unique
catalog.get("tools/github")     // Always works

// Strict lookup with explicit error for ambiguous names
catalog.get_strict("slack")     // Err(AmbiguousName) if both tool and channel exist

// Search by query (matches name, display_name, description, keywords)
let results = catalog.search("messaging");  // Returns scored results

// List with filters
let tools = catalog.list(Some(ManifestKind::Tool), None);
let defaults = catalog.list(None, Some("default"));  // By tag

// Bundle resolution
let (manifests, missing) = catalog.resolve_bundle("google")?;
let (manifests, bundle_def) = catalog.resolve("google")?;  // Works for bundles or singles
```

**Search Scoring** (`catalog.rs:397-437`):
- Exact name match: +10 points
- Name contains: +5 points
- Exact display_name match: +8 points
- Display_name contains: +4 points
- Description contains: +2 points
- Keyword match: +6 points (exact), +3 points (contains)
- Tag match: +4 points

### Bundle Support

**Structure** (`_bundles.json`):
```json
{
  "bundles": {
    "google": {
      "display_name": "Google Suite",
      "description": "Gmail, Calendar, Drive, Docs, Sheets, Slides",
      "extensions": [
        "tools/gmail",
        "tools/google-calendar",
        "tools/google-docs",
        "tools/google-drive",
        "tools/google-sheets",
        "tools/google-slides"
      ],
      "shared_auth": "google_oauth_token"
    },
    "default": {
      "display_name": "Recommended Set",
      "description": "Core tools and channels for a productive setup",
      "extensions": [
        "tools/github",
        "tools/gmail",
        "tools/google-calendar",
        "tools/google-drive",
        "tools/slack-tool",
        "channels/telegram",
        "channels/slack"
      ],
      "shared_auth": null
    }
  }
}
```

**Bundle Resolution**:
```rust
// Check if name is a bundle
if catalog.is_bundle("google") {
    let (manifests, bundle) = catalog.resolve("google")?;
    // manifests: Vec<&ExtensionManifest>
    // bundle: Some(&BundleDefinition)
}
```

---

## Extension Manifest

**Location**: `src/registry/manifest.rs`

**Purpose**: Metadata schema for individual extensions.

### Manifest Structure

```json
{
  "name": "github",
  "display_name": "GitHub",
  "kind": "tool",
  "version": "0.2.2",
  "wit_version": "0.3.0",
  "description": "GitHub integration for issues, PRs, repos, and code search",
  "keywords": ["git", "code", "issues", "pull-requests"],
  "source": {
    "dir": "tools-src/github",
    "capabilities": "github-tool.capabilities.json",
    "crate_name": "github-tool"
  },
  "artifacts": {
    "wasm32-wasip2": {
      "url": "https://github.com/nearai/ironclaw/releases/download/ironclaw-v0.22.0/tool-github-0.2.2-wasm32-wasip2.tar.gz",
      "sha256": "70b55af593193d8fa495c0f702ea23284d83a624124f8a5f7564916ec5032c3f",
      "capabilities_url": null
    }
  },
  "auth_summary": {
    "method": "manual",
    "provider": "GitHub",
    "secrets": ["github_token"],
    "shared_auth": null,
    "setup_url": "https://github.com/settings/tokens"
  },
  "tags": ["default", "development"]
}
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✓ | Unique identifier (matches crate name stem) |
| `display_name` | string | ✓ | Human-readable name |
| `kind` | enum | ✓ | `tool`, `channel`, or `mcp_server` |
| `version` | string | ✗ | Semver from Cargo.toml (optional for MCP) |
| `wit_version` | string | ✗ | WIT interface version for WASM extensions |
| `description` | string | ✓ | One-line description |
| `keywords` | string[] | ✗ | Search keywords beyond name |
| `source` | SourceSpec | ✗ | Source code location (WASM only) |
| `artifacts` | Map<string, ArtifactSpec> | ✗ | Pre-built binaries by target triple |
| `auth_summary` | AuthSummary | ✗ | Auth requirements (WASM only) |
| `tags` | string[] | ✗ | Filtering tags (e.g., "default", "messaging") |
| `url` | string | ✗ | MCP server URL (MCP only) |
| `auth` | string | ✗ | MCP auth method: "dcr", "oauth_pre_configured:<url>", "none" |

### Source Spec (WASM)

```json
{
  "dir": "tools-src/github",
  "capabilities": "github-tool.capabilities.json",
  "crate_name": "github-tool"
}
```

- `dir` — Path relative to repo root
- `capabilities` — Capabilities filename relative to source dir
- `crate_name` — Rust crate name for `cargo component build`

### Artifact Spec

```json
{
  "wasm32-wasip2": {
    "url": "https://.../tool-github-0.2.2-wasm32-wasip2.tar.gz",
    "sha256": "70b55af...",
    "capabilities_url": null
  }
}
```

- `url` — Download URL (`.wasm` or `.tar.gz` bundle)
- `sha256` — Hex SHA256 checksum (required for artifact installs)
- `capabilities_url` — Separate capabilities file download (if not in bundle)

**Supported Target Triples**:
- `wasm32-wasip1`
- `wasm32-wasip2` (preferred)
- `wasm32-wasi`
- `wasm32-unknown-unknown`

### Auth Summary (WASM)

```json
{
  "method": "oauth",
  "provider": "Google",
  "secrets": ["google_oauth_token"],
  "shared_auth": "google_oauth_token",
  "setup_url": "https://console.cloud.google.com/apis/credentials"
}
```

- `method` — `"oauth"`, `"manual"`, or `"none"`
- `provider` — Display name for auth provider
- `secrets` — Secret names required
- `shared_auth` — Shared secret name across bundle members
- `setup_url` — URL for credential setup

### MCP Server Manifest

```json
{
  "name": "notion",
  "display_name": "Notion",
  "kind": "mcp_server",
  "description": "Connect to Notion for reading and writing pages",
  "keywords": ["notes", "wiki", "docs"],
  "url": "https://mcp.notion.com/mcp",
  "auth": "dcr"
}
```

**Auth Methods**:
- `"dcr"` — Dynamic Client Registration (zero-config OAuth)
- `"oauth_pre_configured:<setup_url>"` — Manual OAuth app setup
- `"none"` — No auth needed

---

## Registry Installer

**Location**: `src/registry/installer.rs`

**Purpose**: Install extensions via build-from-source or artifact download with automatic fallback.

### Installation Modes

**Mode Selection**:
```rust
// Auto-select: prefer artifact if available, else build from source
installer.install(manifest, force: false, prefer_build: false).await?;

// Force build from source (ignore artifacts)
installer.install_from_source(manifest, force: false).await?;

// Artifact with source fallback
installer.install_with_source_fallback(manifest, force: false).await?;
```

### Artifact Download Flow

**Security Checks**:
1. **HTTPS required** — Reject `http://` URLs
2. **Host allowlist** — Only GitHub/gcp.githubusercontent.com allowed
3. **SHA256 verification** — Checksum required before download
4. **Path traversal rejection** — Validate `source.dir` is safe relative path

```rust
// Allowed artifact hosts
const ALLOWED_ARTIFACT_HOSTS: &[&str] = &[
    "github.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
    "raw.githubusercontent.com",
];
```

**Download Process**:
1. Validate manifest inputs (`validate_manifest_install_inputs()`)
2. Check if already installed (unless `force=true`)
3. Download artifact from URL
4. Verify SHA256 checksum
5. Extract tar.gz or copy bare `.wasm`
6. Fetch capabilities (from bundle, separate URL, or source tree)
7. Install to `~/.ironclaw/tools/` or `~/.ironclaw/channels/`

**Format Detection**:
- **tar.gz bundle** — Extract `{name}.wasm` + `{name}.capabilities.json`
- **bare .wasm** — Copy WASM, fetch capabilities separately if available

### Source Build Flow

**Build Process**:
1. Validate `source.dir` starts with correct prefix:
   - Tools: `tools-src/`
   - Channels: `channels-src/`
2. Check source directory exists
3. Run `cargo component build --release`
4. Find built artifact in `target/<triple>/release/`
5. Copy WASM + capabilities to install directory

```rust
// Build WASM component
let wasm_path = crate::registry::artifacts::build_wasm_component(
    &source_dir,
    &source.crate_name,
    true  // release build
).await?;
```

### Source Fallback Policy

**When Fallback Occurs**:
- Artifact download fails (404, timeout, network error)
- Checksum mismatch on `releases/latest` URL (moving target)

**When Fallback Blocked**:
- `AlreadyInstalled` error
- `InvalidManifest` error
- Checksum mismatch on version-pinned URL (security concern)
- Source directory doesn't exist

```rust
fn should_attempt_source_fallback(err: &RegistryError) -> bool {
    match err {
        // Moving-target URL — OK to fallback
        RegistryError::ChecksumMismatch { url, .. } 
            => url.contains("github.com/nearai/ironclaw/releases/latest/"),
        
        // Structural problems — no fallback
        RegistryError::AlreadyInstalled { .. } 
        | RegistryError::InvalidManifest { .. } => false,
        
        // Transient errors — fallback OK
        _ => true,
    }
}
```

### Bundle Installation

```rust
let (outcomes, auth_hints) = installer.install_bundle(
    &manifests,      // &[&ExtensionManifest]
    &bundle,         // &BundleDefinition
    force: false,
    prefer_build: false,
).await;
```

**Auth Hint Collection**:
- Shared auth reminder if bundle has `shared_auth`
- Setup URLs for each unique auth provider
- Error summary for failed installations

---

## WASM Artifacts

**Location**: `src/registry/artifacts.rs`

**Purpose**: Unified WASM artifact resolution — find, build, and install WASM components.

### Artifact Resolution

**Target Directory Resolution**:
```rust
// Check CARGO_TARGET_DIR env var first, else <crate_dir>/target
let target_dir = resolve_target_dir(&crate_dir);
```

**Artifact Search** (priority order):
1. `wasm32-wasip1`
2. `wasm32-wasip2` (preferred)
3. `wasm32-wasi`
4. `wasm32-unknown-unknown`

```rust
// Find specific artifact by crate name
let wasm_path = find_wasm_artifact(&crate_dir, "github-tool", "release");

// Fallback: find any .wasm file
let wasm_path = find_any_wasm_artifact(&crate_dir, "release");
```

**Name Normalization**: Hyphens converted to underscores (`github-tool` → `github_tool.wasm`)

### Building WASM Components

**Async Build**:
```rust
let wasm_path = build_wasm_component(
    &source_dir,
    &crate_name,
    release: true,
).await?;
```

**Sync Build** (CLI use):
```rust
let wasm_path = build_wasm_component_sync(&source_dir, release: true)?;
```

**Build Process**:
1. Check `cargo-component` availability
2. Run `cargo component build [--release]`
3. Stream output to terminal
4. Search target directories for `.wasm` file

### Installation

```rust
let wasm_dst = install_wasm_files(
    &wasm_src,       // Path to built/downloaded WASM
    &source_dir,     // Source dir for capabilities lookup
    "github-tool",   // Extension name
    &target_dir,     // ~/.ironclaw/tools/ or channels/
    force: false,
).await?;
```

**Capabilities Lookup** (in order):
1. `{name}.capabilities.json`
2. `{name}-tool.capabilities.json`
3. `capabilities.json`

---

## Embedded Catalog

**Location**: `src/registry/embedded.rs`

**Purpose**: Compile registry manifests into binary for distribution without `registry/` directory.

### Generation (build.rs)

```rust
// build.rs generates OUT_DIR/embedded_catalog.json
let mut raw = EmbeddedCatalogRaw {
    tools: vec![],
    channels: vec![],
    mcp_servers: vec![],
    bundles: BundlesFile::default(),
};

// Load all manifests from registry/ directory
for entry in read_dir("registry/tools")? {
    let manifest: ExtensionManifest = parse_json(entry)?;
    raw.tools.push(manifest);
}
// ... same for channels and mcp-servers

// Write to OUT_DIR
fs::write(
    out_dir.join("embedded_catalog.json"),
    serde_json::to_string(&raw)?,
)?;
```

### Runtime Loading

```rust
// Cached singleton (OnceLock)
fn parsed_catalog() -> &'static ParsedCatalog {
    static CACHE: OnceLock<ParsedCatalog> = OnceLock::new();
    CACHE.get_or_init(|| {
        let raw: EmbeddedCatalogRaw = serde_json::from_str(EMBEDDED_CATALOG)?;
        // Build HashMap keyed by "tools/<name>", "channels/<name>", etc.
    })
}

// Public API
let manifests = load_embedded();        // HashMap<String, ExtensionManifest>
let bundles = load_embedded_bundles();  // HashMap<String, BundleDefinition>
```

---

## Error Handling

### RegistryError Variants

```rust
pub enum RegistryError {
    DirectoryNotFound(PathBuf),
    ManifestRead { path: PathBuf, reason: String },
    ManifestParse { path: PathBuf, reason: String },
    ExtensionNotFound(String),
    AlreadyInstalled { name: String, path: PathBuf },
    DownloadFailed { url: String, reason: String },
    InvalidManifest { name: String, field: &'static str, reason: String },
    ChecksumMismatch { url: String, expected_sha256: String, actual_sha256: String },
    MissingChecksum { name: String },
    SourceFallbackUnavailable { name: String, source_dir: PathBuf, artifact_error: Box<RegistryError> },
    InstallFallbackFailed { name: String, artifact_error: Box<RegistryError>, source_error: Box<RegistryError> },
    AmbiguousName { name: String, kind_a: &'static str, prefix_a: &'static str, kind_b: &'static str, prefix_b: &'static str },
    BundleNotFound(String),
    BundlesRead(String),
    Io(#[from] std::io::Error),
}
```

### Error Messages

**User-Facing** (intentionally omit internal details):
```rust
// ChecksumMismatch — omits URL to avoid leaking internal artifact URLs
#[error("Checksum verification failed: expected {expected_sha256}, got {actual_sha256}")]
ChecksumMismatch { url: String, expected_sha256: String, actual_sha256: String }

// AmbiguousName — provides disambiguation hints
#[error("Ambiguous name '{name}': exists as both {kind_a} and {kind_b}. Use '{prefix_a}/{name}' or '{prefix_b}/{name}'.")]
AmbiguousName { ... }
```

---

## Code Patterns

### Load Catalog (Disk or Embedded)

```rust
use crate::registry::RegistryCatalog;

// Try disk first, fall back to embedded
let catalog = RegistryCatalog::load_or_embedded()?;

// List all extensions
for manifest in catalog.all() {
    println!("{}/{} - {}", manifest.kind, manifest.name, manifest.description);
}

// Search
let results = catalog.search("messaging");
for m in results {
    println!("{}: {}", m.name, m.display_name);
}
```

### Resolve Extension or Bundle

```rust
// Check if name is bundle or single extension
let (manifests, bundle_def) = catalog.resolve("google")?;

if let Some(bundle) = bundle_def {
    println!("Installing bundle: {}", bundle.display_name);
    if let Some(shared) = &bundle.shared_auth {
        println!("Shared auth: {}", shared);
    }
} else {
    println!("Installing single extension: {}", manifests[0].name);
}
```

### Install Extension

```rust
use crate::registry::RegistryInstaller;

let installer = RegistryInstaller::with_defaults(repo_root);

// Auto-select install mode (artifact or source)
let outcome = installer.install(&manifest, force: false, prefer_build: false).await?;

println!("Installed {} to {}", outcome.name, outcome.wasm_path.display());
if !outcome.warnings.is_empty() {
    for warning in outcome.warnings {
        eprintln!("Warning: {}", warning);
    }
}
```

### Install Bundle

```rust
let (outcomes, auth_hints) = installer.install_bundle(
    &manifests,
    &bundle,
    force: false,
    prefer_build: false,
).await;

for outcome in outcomes {
    println!("✓ Installed {}", outcome.name);
}

for hint in auth_hints {
    println!("{}", hint);
}
```

### Validate Manifest

```rust
// Validate before install (rejects path traversal, invalid hosts, etc.)
validate_manifest_install_inputs(&manifest)?;

// Check for artifact availability
let has_artifact = manifest
    .artifacts
    .get("wasm32-wasip2")
    .and_then(|a| a.url.as_ref())
    .is_some();

if has_artifact {
    println!("Pre-built artifact available");
} else {
    println!("Must build from source");
}
```

---

## Security Considerations

### 1. Artifact Host Allowlist

**Threat**: Malicious manifest points to attacker-controlled URL for supply-chain attack.

**Defense**: Hardcoded allowlist — only GitHub/gcp.githubusercontent.com permitted.

```rust
fn is_allowed_artifact_host(host: &str) -> bool {
    ALLOWED_ARTIFACT_HOSTS
        .iter()
        .any(|allowed| host.eq_ignore_ascii_case(allowed))
        || host.ends_with(".githubusercontent.com")
}
```

Unknown hosts → `InvalidManifest` error (not source fallback).

### 2. Checksum Verification

**Threat**: Tampered artifact during download.

**Defense**: SHA256 required before download, verified after.

```rust
let expected_sha = artifact.sha256.as_ref()
    .ok_or_else(|| RegistryError::MissingChecksum { name: manifest.name.clone() })?;

let bytes = download_artifact(url).await?;
verify_sha256(&bytes, expected_sha, url)?;
```

**Missing Checksum Policy**:
- Returns `MissingChecksum` error (allows source fallback via `should_attempt_source_fallback()`)
- Bootstrapping: New extensions can be installed from source before checksums populated

### 3. Path Traversal Prevention

**Threat**: Manifest `source.dir` contains `../` to escape repo root.

**Defense**: Validate path components before install.

```rust
let has_unsafe_component = source_path.components().any(|component| {
    matches!(component, Component::ParentDir | Component::RootDir | Component::Prefix(_) | Component::CurDir)
});

if source_path.is_absolute() || has_unsafe_component {
    return Err(RegistryError::InvalidManifest {
        name: manifest.name.clone(),
        field: "source.dir",
        reason: "must be a safe relative path without traversal segments".to_string(),
    });
}
```

### 4. Name Validation

**Threat**: Malicious extension name breaks filesystem or CLI parsing.

**Defense**: Alphanumeric + hyphens/underscores only.

```rust
let is_valid_name = !manifest.name.is_empty()
    && manifest.name.chars().all(|c| {
        c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_'
    });
```

### 5. Decompression Bomb Protection

**Threat**: Malicious tar.gz with massive entry exhausts memory.

**Defense**: 100 MB cap per entry.

```rust
const MAX_ENTRY_SIZE: u64 = 100 * 1024 * 1024;

if entry.size() > MAX_ENTRY_SIZE {
    return Err(RegistryError::DownloadFailed {
        reason: format!("archive entry too large ({} bytes)", entry.size()),
    });
}
```

### 6. URL Validation

**Threat**: HTTP URL leaks credentials in transit.

**Defense**: HTTPS required.

```rust
if parsed.scheme() != "https" {
    return Err(RegistryError::InvalidManifest {
        field: "artifacts.wasm32-wasip2.url",
        reason: "URL must use https".to_string(),
    });
}
```

---

## Related Files

**Core Implementation**:
- `src/registry/mod.rs` — Module root, re-exports (25 lines)
- `src/registry/catalog.rs` — Catalog loading, search, resolve (765 lines)
- `src/registry/manifest.rs` — Manifest structs, serde, conversion (590 lines)
- `src/registry/installer.rs` — Install logic, artifact download, source fallback (1339 lines)
- `src/registry/artifacts.rs` — WASM artifact resolution, building (377 lines)
- `src/registry/embedded.rs` — Embedded catalog compilation (103 lines)

**Build Integration**:
- `build.rs` — Generates `embedded_catalog.json` from `registry/` directory

**Registry Directory**:
- `registry/tools/*.json` — WASM tool manifests
- `registry/channels/*.json` — WASM channel manifests
- `registry/mcp-servers/*.json` — MCP server manifests
- `registry/_bundles.json` — Bundle definitions

**Integration Points**:
- `src/extensions/manager.rs` — Uses registry for extension discovery/install
- `src/cli/tool.rs` — CLI commands for registry operations
- `src/tools/wasm/loader.rs` — Loads installed WASM tools
- `src/channels/wasm/loader.rs` — Loads installed WASM channels

---

## Common Pitfalls

### ❌ Using bare name for ambiguous extensions

```rust
// WRONG: Returns None if both tools/slack and channels/slack exist
let manifest = catalog.get("slack");

// RIGHT: Use qualified key
let manifest = catalog.get("tools/slack");
```

### ❌ Forgetting to check for bundle before single extension

```rust
// WRONG: Assumes name is always single extension
let manifest = catalog.get_strict("google")?;  // Fails - "google" is a bundle

// RIGHT: Check bundle first or use resolve()
let (manifests, bundle) = catalog.resolve("google")?;
```

### ❌ Installing without checksum verification

```rust
// WRONG: Missing sha256 in manifest
"artifacts": {
    "wasm32-wasip2": { "url": "https://...", "sha256": null }
}
// → RegistryError::MissingChecksum

// RIGHT: Populate sha256 before release
"artifacts": {
    "wasm32-wasip2": {
        "url": "https://...",
        "sha256": "70b55af..."
    }
}
```

### ❌ Hardcoding target triple in artifact lookup

```rust
// WRONG: Only checks wasm32-wasip2
let artifact = manifest.artifacts.get("wasm32-wasip2");

// RIGHT: Use installer which handles all triples
installer.install(manifest, force, prefer_build).await?;
```

### ❌ Ignoring source fallback policy

```rust
// WRONG: Assume all installs use artifacts
match installer.install_from_artifact(manifest, force).await {
    Ok(outcome) => { /* success */ }
    Err(e) => { /* fail immediately */ }
}

// RIGHT: Use install_with_source_fallback for automatic fallback
match installer.install_with_source_fallback(manifest, force).await {
    Ok(outcome) => { /* success (artifact or source) */ }
    Err(e) => { /* both paths failed */ }
}
```

---

## Testing Patterns

### Unit Test: Catalog Loading

```rust
#[test]
fn test_load_catalog() {
    let tmp = tempfile::tempdir().unwrap();
    create_test_registry(tmp.path());
    
    let catalog = RegistryCatalog::load(tmp.path()).unwrap();
    assert_eq!(catalog.all().len(), 4);  // 2 tools, 1 channel, 1 MCP
}

#[test]
fn test_load_or_embedded_succeeds() {
    // Should always succeed: disk or embedded
    let catalog = RegistryCatalog::load_or_embedded().unwrap();
    assert!(!catalog.all().is_empty() || !catalog.bundle_names().is_empty());
}
```

### Unit Test: Search Scoring

```rust
#[test]
fn test_search_exact_match() {
    let catalog = load_test_catalog();
    let results = catalog.search("github");
    
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "github");
}

#[test]
fn test_search_by_keyword() {
    let catalog = load_test_catalog();
    let results = catalog.search("messaging");
    
    assert!(!results.is_empty());  // Should find telegram, slack, etc.
}
```

### Unit Test: Bundle Resolution

```rust
#[test]
fn test_resolve_bundle() {
    let catalog = load_test_catalog();
    
    let (manifests, missing) = catalog.resolve_bundle("default").unwrap();
    assert_eq!(manifests.len(), 3);
    assert!(missing.is_empty());
    
    assert!(catalog.resolve_bundle("nonexistent").is_err());
}

#[test]
fn test_bundle_entries_resolve_against_real_registry() {
    // Catch stale bundle refs after renames
    let catalog = RegistryCatalog::load_or_embedded().unwrap();
    
    for bundle_name in catalog.bundle_names() {
        let (manifests, missing) = catalog.resolve_bundle(bundle_name).unwrap();
        assert!(
            missing.is_empty(),
            "Bundle '{}' has unresolved entries: {:?}",
            bundle_name,
            missing
        );
    }
}
```

### Unit Test: Checksum Verification

```rust
#[test]
fn test_verify_sha256_valid() {
    use sha2::{Digest, Sha256};
    let data = b"hello world";
    let mut hasher = Sha256::new();
    hasher.update(data);
    let hash = format!("{:x}", hasher.finalize());
    
    assert!(verify_sha256(data, &hash, "test://url").is_ok());
}

#[test]
fn test_verify_sha256_invalid() {
    let err = verify_sha256(b"data", "0000", "test://url")
        .expect_err("checksum mismatch");
    
    assert!(matches!(err, RegistryError::ChecksumMismatch { .. }));
}
```

### Integration Test: Source Fallback Policy

```rust
#[test]
fn test_source_fallback_on_latest_url_mismatch() {
    let latest_mismatch = RegistryError::ChecksumMismatch {
        url: "https://github.com/nearai/ironclaw/releases/latest/download/github-wasm32-wasip2.tar.gz".to_string(),
        expected_sha256: "aaa".to_string(),
        actual_sha256: "bbb".to_string(),
    };
    
    assert!(
        should_attempt_source_fallback(&latest_mismatch),
        "ChecksumMismatch on releases/latest URL should allow source fallback"
    );
    
    let pinned_mismatch = RegistryError::ChecksumMismatch {
        url: "https://github.com/nearai/ironclaw/releases/download/v0.7.0/github-0.2.0-wasm32-wasip2.tar.gz".to_string(),
        expected_sha256: "aaa".to_string(),
        actual_sha256: "bbb".to_string(),
    };
    
    assert!(
        !should_attempt_source_fallback(&pinned_mismatch),
        "ChecksumMismatch on version-pinned URL must remain a hard block"
    );
}
```

---

## Related Context

- `extensions-system.md` — Extension lifecycle management (search, install, auth, activate)
- `wasm-channels.md` — WASM channel runtime and sandboxing
- `config-precedence.md` — Configuration layers and bootstrap order
- `security-patterns.md` — Overall security architecture
