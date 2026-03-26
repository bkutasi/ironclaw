<!-- Context: core/architecture/pairing | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->

# Pairing System

**Purpose**: DM access control for channels — gates direct messages from unknown senders. Only approved senders can message the agent.

**Last Updated**: 2026-03-26

## Quick Reference

**Update Triggers**: DM policy changes | New channel additions | Pairing flow modifications | Security updates

**Audience**: Developers, AI agents, integration engineers, security reviewers

**Key Files**:
- `src/pairing/mod.rs` - Module exports and overview
- `src/pairing/store.rs` - `PairingStore`, `PairingRequest`, allowFrom management
- `src/cli/pairing.rs` - CLI commands (`ironclaw pairing list/approve`)
- `src/channels/web/server.rs` - Web API handlers (`/api/pairing/{channel}`)
- `src/channels/web/types.rs` - Pairing DTOs (`PairingListResponse`, `PairingApproveRequest`)

**Commands**:
```bash
ironclaw pairing list telegram           # List pending requests
ironclaw pairing list telegram --json    # JSON output
ironclaw pairing approve telegram ABC12345  # Approve by code
```

**API Endpoints**:
- `GET /api/pairing/{channel}` - List pending requests
- `POST /api/pairing/{channel}/approve` - Approve a pairing code

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pairing System                              │
│                                                                     │
│  ┌─────────────────┐     ┌─────────────────┐                       │
│  │  PairingStore   │     │  AllowFrom List │                       │
│  │  (pending)      │     │  (approved)     │                       │
│  │                 │     │                 │                       │
│  │ - pairing.json  │     │ - allowFrom.json│                       │
│  │ - pending codes │     │ - approved IDs  │                       │
│  └────────┬────────┘     └────────┬────────┘                       │
│           │                      │                                  │
│           └──────────┬───────────┘                                  │
│                      │                                              │
│                      ▼                                              │
│            ┌─────────────────┐                                     │
│            │  Channel Filter │                                     │
│            │  (DM Policy)    │                                     │
│            └────────┬────────┘                                     │
│                     │                                               │
│         ┌───────────┼───────────┐                                  │
│         ▼           ▼           ▼                                  │
│    "open"     "allowlist"  "pairing"                               │
│    (all OK)   (check list) (check + reply)                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Core Pattern**: File-based pairing store (`~/.ironclaw/{channel}-pairing.json`) + allowFrom list (`~/.ironclaw/{channel}-allowFrom.json`) with rate limiting and TTL expiration.

---

## DM Policy Modes

Channels support three DM policies (configured in `src/config/channels.rs`):

| Policy | Behavior | Use Case |
|--------|----------|----------|
| `"open"` | All messages accepted | Public bots, open access |
| `"allowlist"` | Only pre-approved senders | Private assistants |
| `"pairing"` (default) | Allowlist + send pairing code to unknown | Controlled onboarding |

**Policy Check Flow**:
```rust
// Pseudocode from channel message handling
if dm_policy == "open" {
    accept_message();
} else if dm_policy == "allowlist" || dm_policy == "pairing" {
    if pairing_store.is_sender_allowed(channel, sender_id, username)? {
        accept_message();
    } else if dm_policy == "pairing" {
        // Upsert pairing request and reply with code
        let result = pairing_store.upsert_request(channel, sender_id, metadata)?;
        reply_with_pairing_code(result.code);
    } else {
        reject_message();
    }
}
```

---

## Pairing Mechanism

### PairingRequest Structure

```rust
pub struct PairingRequest {
    pub id: String,                        // Sender identifier (e.g., Telegram user_id)
    pub code: String,                      // 8-char pairing code (e.g., "ABC12345")
    pub created_at: String,                // RFC3339 timestamp
    pub last_seen_at: String,              // RFC3339 timestamp (updated on reuse)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,   // Channel-specific metadata (e.g., chat_id)
}
```

### Code Generation

- **Length**: 8 characters
- **Alphabet**: `ABCDEFGHJKLMNPQRSTUVWXYZ23456789` (base32-like, excludes ambiguous chars)
- **Uniqueness**: Checked against existing pending requests (max 500 retries + fallback suffix)
- **Case Insensitive**: Approval normalizes to uppercase

### TTL and Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `PAIRING_CODE_LENGTH` | 8 | Code length |
| `PAIRING_PENDING_TTL_SECS` | 900 (15 min) | Request expiration |
| `PAIRING_PENDING_MAX` | 3 | Max pending per channel |
| `PAIRING_APPROVE_RATE_LIMIT` | 10 | Max failed attempts |
| `PAIRING_APPROVE_RATE_WINDOW_SECS` | 300 (5 min) | Rate limit window |

**Expiration**: Requests older than 15 minutes are automatically purged on read/write operations.

---

## Device Linking Flow

### Step-by-Step Flow

```
1. Unknown sender messages channel
   ↓
2. Channel checks pairing policy
   - If "pairing" mode and sender not in allowFrom
   ↓
3. Channel upserts pairing request
   - store.upsert_request(channel, sender_id, metadata)
   - Returns (code, created: bool)
   ↓
4. Channel replies with pairing instructions
   "To connect, run: ironclaw pairing approve telegram ABC12345"
   ↓
5. User runs CLI command on host
   ironclaw pairing approve telegram ABC12345
   ↓
6. PairingStore validates code
   - Checks rate limit
   - Finds and removes matching request
   - Adds sender_id to allowFrom list
   ↓
7. Future messages from sender accepted
```

### CLI Commands

**List pending requests**:
```bash
ironclaw pairing list telegram
# Output:
# Pairing requests (2):
#   ABC12345  user123  chat_id=456  2026-03-26T18:00:00+00:00
#   XYZ98765  user456  chat_id=789  2026-03-26T18:05:00+00:00

ironclaw pairing list telegram --json
# Output:
# [
#   {"id":"user123","code":"ABC12345","created_at":"...","meta":{"chat_id":456}},
#   ...
# ]
```

**Approve a code**:
```bash
ironclaw pairing approve telegram ABC12345
# Success: "Approved telegram sender user123."
# Failure: "No pending pairing request found for code: ABC12345"
# Rate limited: "Too many failed approve attempts. Wait a few minutes before trying again."
```

---

## Authentication Flow

### Approval Process

```rust
pub fn approve(
    &self,
    channel: &str,
    code: &str,
) -> Result<Option<PairingRequest>, PairingStoreError> {
    let code = code.trim().to_uppercase();
    
    // 1. Check rate limit
    if self.is_approve_rate_limited(channel)? {
        return Err(PairingStoreError::ApproveRateLimited);
    }
    
    // 2. Find and remove matching request
    let idx = store.requests.iter().position(|r| r.code.to_uppercase() == code);
    let entry = match idx {
        Some(i) => store.requests.remove(i),
        None => {
            self.record_failed_approve(channel)?;  // Track for rate limiting
            return Ok(None);
        }
    };
    
    // 3. Add to allowFrom list
    self.add_allow_from(channel, &entry.id)?;
    
    Ok(Some(entry))
}
```

### Rate Limiting

**Failed Attempt Tracking**:
- Stored in `~/.ironclaw/{channel}-approve-attempts.json`
- Records timestamp of each failed approve attempt
- Purges attempts older than 5 minutes
- Triggers rate limit after 10 failures within window

**Rate Limit Error**:
```rust
PairingStoreError::ApproveRateLimited
// Message: "Too many failed approve attempts. Wait a few minutes before trying again."
```

### AllowFrom List

**Structure**:
```json
{
  "version": 1,
  "allowFrom": ["user123", "user456", "@username"]
}
```

**Normalization**:
- Entries stored as provided (case preserved)
- Lookup is case-insensitive
- Username lookup strips `@` prefix for matching

**Check Logic**:
```rust
pub fn is_sender_allowed(
    &self,
    channel: &str,
    id: &str,
    username: Option<&str>,
) -> Result<bool, PairingStoreError> {
    let allow = self.read_allow_from(channel)?;
    
    // Check by ID
    if allow.iter().any(|e| e.trim() == id.trim()) {
        return Ok(true);
    }
    
    // Check by username (with/without @)
    if let Some(u) = username {
        let u = u.trim().to_lowercase();
        let u_norm = u.strip_prefix('@').unwrap_or(&u);
        if allow.iter().any(|e| {
            e.trim().to_lowercase() == u || 
            e.trim().to_lowercase() == format!("@{}", u_norm)
        }) {
            return Ok(true);
        }
    }
    
    Ok(false)
}
```

---

## Session Sharing

### Per-Channel Isolation

Each channel maintains independent pairing state:
- `~/.ironclaw/telegram-pairing.json`
- `~/.ironclaw/slack-pairing.json`
- `~/.ironclaw/signal-pairing.json`

**Channel Key Sanitization**:
```rust
fn safe_channel_key(channel: &str) -> Result<String, PairingStoreError> {
    let raw = channel.trim().to_lowercase();
    let safe = raw
        .chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect::<String>()
        .replace("..", "_");
    Ok(safe)
}
```

**Invalid Channels**: Empty strings or-only underscores are rejected.

### Metadata Preservation

Pairing requests can carry channel-specific metadata:
```json
{
  "id": "user123",
  "code": "ABC12345",
  "created_at": "2026-03-26T18:00:00+00:00",
  "meta": {
    "chat_id": -123456,
    "username": "@john"
  }
}
```

**Usage**: Metadata aids in routing and identification but is not used for access control decisions (only `id` is added to allowFrom).

---

## Code Patterns

### Creating Pairing Store

```rust
use crate::pairing::PairingStore;

// Default (~/.ironclaw)
let store = PairingStore::new();

// Custom base directory (testing)
let store = PairingStore::with_base_dir(temp_dir.path().to_path_buf());
```

### Upserting Pairing Request

```rust
let result = store.upsert_request(
    "telegram",
    "user123",
    Some(serde_json::json!({"chat_id": -123456})),
)?;

if result.created {
    println!("New pairing request: {}", result.code);
} else {
    println!("Existing code: {}", result.code);
}
```

### Checking Access

```rust
if store.is_sender_allowed("telegram", "user123", Some("@john"))? {
    // Accept message
} else {
    // Reject or send pairing code
}
```

### Listing Pending Requests

```rust
let requests = store.list_pending("telegram")?;
for req in &requests {
    println!("{} - {} - {}", req.code, req.id, req.created_at);
}
```

### Approving Code

```rust
match store.approve("telegram", "ABC12345") {
    Ok(Some(entry)) => println!("Approved sender: {}", entry.id),
    Ok(None) => println!("Invalid or expired code"),
    Err(PairingStoreError::ApproveRateLimited) => {
        println!("Rate limited - wait before retrying")
    }
    Err(e) => println!("Error: {}", e),
}
```

---

## Web API

### List Pending Requests

**Endpoint**: `GET /api/pairing/{channel}`

**Response** (`PairingListResponse`):
```json
{
  "channel": "telegram",
  "requests": [
    {
      "code": "ABC12345",
      "sender_id": "user123",
      "meta": {"chat_id": -123456},
      "created_at": "2026-03-26T18:00:00+00:00"
    }
  ]
}
```

### Approve Code

**Endpoint**: `POST /api/pairing/{channel}/approve`

**Request** (`PairingApproveRequest`):
```json
{
  "code": "ABC12345"
}
```

**Success Response**:
```json
{
  "success": true,
  "message": "Pairing approved for sender 'user123'"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid or expired code
- `429 Too Many Requests`: Rate limited

---

## Error Handling

### PairingStoreError Variants

```rust
pub enum PairingStoreError {
    #[error("Invalid channel: {0}")]
    InvalidChannel(String),
    
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Rate limit: too many failed approve attempts; try again later")]
    ApproveRateLimited,
}
```

### Common Failure Modes

| Scenario | Error | Resolution |
|----------|-------|------------|
| Invalid channel name | `InvalidChannel` | Use valid channel (non-empty, not special chars) |
| Code not found | `Ok(None)` from `approve()` | Verify code, check expiration |
| Too many failures | `ApproveRateLimited` | Wait 5+ minutes |
| File locked | `Io` (lock timeout) | Retry operation |
| Corrupted JSON | `Json` | Delete corrupted file, recreate |

---

## Security Considerations

### Brute-Force Mitigation

1. **Short TTL**: 15-minute expiration reduces attack window
2. **Rate Limiting**: 10 failed attempts per 5 minutes
3. **Large Code Space**: 8 chars from 32-symbol alphabet = ~32^8 combinations
4. **Case Insensitivity**: Reduces user error, doesn't weaken security

### File-Based Storage

**Pros**:
- Simple, no database dependency
- Human-readable state
- Easy backup/restore

**Cons**:
- No encryption at rest
- File locking required for concurrency
- Limited to single-instance scenarios

**Recommendation**: Suitable for personal/single-user deployments. Multi-instance deployments should migrate to database-backed storage.

### Channel Isolation

Each channel's pairing state is completely isolated. Compromise of one channel's pairing file does not affect others.

---

## Related Files

**Core Implementation**:
- `src/pairing/mod.rs` - Module overview
- `src/pairing/store.rs` - Full implementation

**CLI**:
- `src/cli/pairing.rs` - CLI commands
- `src/cli/mod.rs` - Command registration

**Web API**:
- `src/channels/web/server.rs` - Handlers (lines 2495-2540)
- `src/channels/web/types.rs` - DTOs (lines 497-518)

**Integration Points**:
- `src/extensions/manager.rs` - Extension manager uses pairing store
- `src/channels/wasm/wrapper.rs` - WASM channel host functions for pairing
- `src/config/channels.rs` - DM policy configuration
- `src/settings.rs` - Settings for DM policy

**Related Context**:
- `channels-system.md` - Channel architecture, message flow
- `agent-system.md` - Session/thread model
- `websocket-security.md` - Auth patterns

---

## Testing

### Unit Tests (from `store.rs`)

```rust
#[test]
fn test_upsert_request_creates_new() {
    let (store, _) = test_store();
    let result = store
        .upsert_request("telegram", "user123", Some(serde_json::json!({"chat_id": 456})))
        .unwrap();
    assert!(result.created);
    assert_eq!(result.code.len(), PAIRING_CODE_LENGTH);
}

#[test]
fn test_approve_adds_to_allow_from() {
    let (store, _) = test_store();
    let r = store.upsert_request("telegram", "user456", None).unwrap();
    let approved = store.approve("telegram", &r.code).unwrap();
    assert!(approved.is_some());
    let allow = store.read_allow_from("telegram").unwrap();
    assert_eq!(allow, vec!["user456"]);
}

#[test]
fn test_approve_rate_limited_after_many_failures() {
    let (store, _) = test_store();
    for _ in 0..PAIRING_APPROVE_RATE_LIMIT {
        let _ = store.approve("telegram", "WRONG01");
    }
    let err = store.approve("telegram", "WRONG02").unwrap_err();
    assert!(matches!(err, PairingStoreError::ApproveRateLimited));
}
```

### CLI Tests (from `pairing.rs`)

```rust
#[test]
fn test_approve_valid_code_returns_ok() {
    let (store, _) = test_store();
    let r = store.upsert_request("telegram", "user1", None).unwrap();
    let result = run_pairing_command_with_store(
        &store,
        PairingCommand::Approve {
            channel: "telegram".to_string(),
            code: r.code,
        },
    );
    assert!(result.is_ok());
}
```
