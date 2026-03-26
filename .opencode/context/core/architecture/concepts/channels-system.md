<!-- Context: core/architecture/channels | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->

# Channels System

**Purpose**: Multi-channel input normalization and message routing for Ironclaw - a secure personal AI assistant.

**Last Updated**: 2026-03-26

## Quick Reference

**Update Triggers**: Channel additions | Message flow changes | Web gateway updates | WASM channel changes

**Audience**: Developers, AI agents, integration engineers

**Key Files**:
- `src/channels/mod.rs` - Module exports and architecture overview
- `src/channels/channel.rs` - `Channel` trait, `IncomingMessage`, `OutgoingResponse`, `StatusUpdate`
- `src/channels/manager.rs` - `ChannelManager` for multi-channel coordination
- `src/channels/web/mod.rs` - Web gateway (HTTP/SSE/WebSocket)
- `src/channels/wasm/` - Dynamic WASM channel runtime

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ChannelManager                              │
│                                                                     │
│   ┌──────────────┐   ┌─────────────┐   ┌─────────────┐             │
│   │ ReplChannel  │   │HttpChannel  │   │WasmChannel  │   ...       │
│   └──────┬───────┘   └──────┬──────┘   └──────┬──────┘             │
│          │                 │                 │                      │
│          └─────────────────┴─────────────────┘                      │
│                            │                                        │
│                   select_all (futures)                              │
│                            │                                        │
│                            ▼                                        │
│                     MessageStream                                   │
│                            │                                        │
│                            ▼                                        │
│                    Agent Loop                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Core Pattern**: All channels normalize external input into `IncomingMessage`; `ChannelManager` merges all active channel streams into a single `MessageStream` for the agent loop.

---

## Message Normalization

### IncomingMessage Structure

All channels convert external messages into a unified format:

```rust
pub struct IncomingMessage {
    pub id: Uuid,                        // Unique message ID
    pub channel: String,                 // Source channel name
    pub user_id: String,                 // Storage/persistence scope
    pub owner_id: String,                // Stable instance owner
    pub sender_id: String,               // Channel-specific sender
    pub user_name: Option<String>,       // Display name
    pub content: String,                 // Message content
    pub thread_id: Option<String>,       // Thread ID for threading
    pub conversation_scope_id: Option<String>, // Stable conversation scope
    pub received_at: DateTime<Utc>,      // Receipt timestamp
    pub metadata: serde_json::Value,     // Channel-specific metadata
    pub timezone: Option<String>,        // Client timezone (IANA)
    pub attachments: Vec<IncomingAttachment>, // File/media attachments
    pub(crate) is_internal: bool,        // Internal-only flag
}
```

**Key Fields**:
- `user_id`: Persistence scope (owner-capable channels use stable owner ID)
- `sender_id`: Channel-specific routing target (e.g., Telegram `chat_id`)
- `metadata`: Carries channel-specific routing info (e.g., `chat_id`, `signal_target`)
- `is_internal`: Marks messages from background tasks (bypasses user-input pipeline)

### IncomingAttachment

```rust
pub struct IncomingAttachment {
    pub id: String,                      // Channel-specific ID (e.g., Telegram file_id)
    pub kind: AttachmentKind,            // Audio | Image | Document
    pub mime_type: String,               // MIME type
    pub filename: Option<String>,        // Original filename
    pub size_bytes: Option<u64>,         // File size
    pub source_url: Option<String>,      // Download URL
    pub storage_key: Option<String>,     // Host-side storage key
    pub extracted_text: Option<String>,  // OCR/transcript
    pub data: Vec<u8>,                   // Raw bytes (small files)
    pub duration_secs: Option<u32>,      // Audio/video duration
}
```

### Routing Target Extraction

Proactive replies extract routing targets from metadata:

```rust
pub fn routing_target_from_metadata(metadata: &serde_json::Value) -> Option<String> {
    metadata
        .get("signal_target")
        .or_else(|| metadata.get("chat_id"))
        .or_else(|| metadata.get("target"))
        .and_then(|v| match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        })
}
```

**Fallback**: If no metadata key found, uses `sender_id`.

---

## ChannelManager

### Responsibilities

1. **Channel Registration**: `add()`, `hot_add()` for dynamic channel addition
2. **Stream Merging**: `start_all()` merges all channel streams + injection channel
3. **Response Routing**: `respond()` routes to specific channel
4. **Status Broadcasting**: `send_status()` sends status updates
5. **Proactive Messages**: `broadcast()`, `broadcast_all()` for alerts

### Injection Channel

Background tasks (job monitors, routines) inject messages without full `Channel` impl:

```rust
let inject_tx = manager.inject_sender();
inject_tx.send(IncomingMessage::new("injected", "system", "alert")).await?;
```

**Implementation**: `mpsc::Sender<IncomingMessage>` merged into `select_all()` stream.

### Hot-Add Pattern

```rust
pub async fn hot_add(&self, channel: Box<dyn Channel>) -> Result<(), ChannelError> {
    // 1. Shut down existing channel with same name
    // 2. Start new channel stream
    // 3. Register in channels map
    // 4. Spawn task to forward stream messages through inject_tx
}
```

**Use Case**: WASM channel activation without agent restart.

---

## Web Gateway Architecture

### Transport Layers

```
Browser ─── POST /api/chat/send ──► Agent Loop
        ◄── GET  /api/chat/events ── SSE stream
        ─── GET  /api/chat/ws ─────► WebSocket (bidirectional)
```

**SSE**: Server-sent events for one-way streaming (default)
**WebSocket**: Bidirectional (messages + approvals over same connection)

### GatewayState

Shared state held by `GatewayChannel`:

```rust
pub struct GatewayState {
    pub msg_tx: RwLock<Option<mpsc::Sender<IncomingMessage>>>,
    pub sse: Arc<SseManager>,
    pub workspace: Option<Arc<Workspace>>,
    pub session_manager: Option<Arc<SessionManager>>,
    pub log_broadcaster: Option<Arc<LogBroadcaster>>,
    pub extension_manager: Option<Arc<ExtensionManager>>,
    pub tool_registry: Option<Arc<ToolRegistry>>,
    pub job_manager: Option<Arc<ContainerJobManager>>,
    pub ws_tracker: Option<Arc<WsConnectionTracker>>,
    pub chat_rate_limiter: PerUserRateLimiter,
    // ... optional subsystems
}
```

**Pattern**: All fields are `Option<Arc<T>>` - gateway starts even with optional subsystems disabled.

### SSE Event Types

Events use `#[serde(tag = "type")]` serialization:

| Type | Payload | When |
|------|---------|------|
| `response` | `{content, thread_id}` | Final agent response |
| `thinking` | `{message, thread_id}` | Agent reasoning status |
| `tool_started` | `{name, thread_id}` | Tool execution begins |
| `tool_completed` | `{name, success, error?, parameters?}` | Tool finishes |
| `approval_needed` | `{request_id, tool_name, description, parameters, allow_always}` | Tool requires approval |
| `auth_required` | `{extension_name, instructions?, auth_url?, setup_url?}` | Extension needs auth |
| `heartbeat` | (empty) | SSE keepalive (30s) |

**Full list**: See `src/channels/web/types.rs` - `AppEvent` enum (re-exported from `ironclaw_common`).

### WebSocket Envelope

SSE events wrapped for WebSocket:

```json
{"type":"event","event_type":"response","data":{"content":"hello","thread_id":"t1"}}
{"type":"pong"}
```

**Client→Server**:
```json
{"type":"message","content":"hello","thread_id":"t1"}
{"type":"approval","request_id":"uuid","action":"approve"}
{"type":"ping"}
```

### Connection Limits

- **Max connections**: 100 (SSE + WebSocket combined)
- **Broadcast buffer**: 256 events (slow clients miss events - reconnect expected)
- **SSE keepalive**: Empty event every 30 seconds

### Auth

**Bearer Token**: `Authorization: Bearer <GATEWAY_AUTH_TOKEN>` required for all protected routes.

**Query Token**: SSE/WebSocket endpoints accept `?token=xxx` (browser `EventSource` limitation).

**Rate Limiting**: 30 messages per 60 seconds per user (sliding window).

---

## Message Flow

### Inbound Flow (Channel → Agent)

```
1. External API (Telegram, HTTP, etc.)
   ↓
2. Channel builds IncomingMessage
   - Extracts sender_id, chat_id from metadata
   - Attaches files, timezone
   ↓
3. Channel.start() returns MessageStream
   ↓
4. ChannelManager.start_all() merges streams
   ↓
5. Agent loop receives IncomingMessage
   ↓
6. Session/Thread resolution
   ↓
7. LLM processing
```

### Outbound Flow (Agent → Channel)

```
1. Agent produces response
   ↓
2. Channel.respond(msg, OutgoingResponse)
   ↓
3. Extract routing target from msg.metadata
   - routing_target_from_metadata() → chat_id
   ↓
4. Channel-specific delivery
   - Web: SSE broadcast_for_user(user_id, event)
   - Telegram: POST to sendMessage API with chat_id
   ↓
5. Client receives response
```

### Status Update Flow

```
1. Agent emits StatusUpdate (Thinking, ToolStarted, etc.)
   ↓
2. Channel.send_status(status, metadata)
   ↓
3. Extract user_id from metadata
   ↓
4. Web: SseManager.broadcast_for_user(user_id, AppEvent)
   ↓
5. All connected browser tabs for that user receive event
```

**Metadata Round-Trip**: `chat_id` flows from incoming message metadata → agent loop → `send_status` metadata → SSE broadcast.

---

## WASM Channels

### Dynamic Loading

WASM channels allow runtime loading of channel implementations:

```
┌─────────────────────────────────────────────────────────────┐
│ Host (Rust)                                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ WasmChannel                                          │  │
│  │  - Loads WASM module from disk/URL                   │  │
│  │  - Provides host functions: emit_message, respond    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↑↓                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ WASM Module (Guest)                                  │  │
│  │  - on_message(msg) → IncomingMessage                 │  │
│  │  - on_respond(metadata_json) → delivery              │  │
│  │  - on_status(status_json) → status update            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Metadata Boundary

```rust
// Inbound: JSON → IncomingMessage.metadata
let metadata_json = serde_json::to_string(&telegram_metadata)?;
host.emit_message(content, &metadata_json)?;

// Outbound: IncomingMessage.metadata → JSON → channel-specific struct
let metadata: TelegramMetadata = serde_json::from_str(&msg.metadata)?;
send_to_telegram(metadata.chat_id, response).await?;
```

**Failure Mode**: Internal messages (`is_internal = true`) have `metadata: Null` → deserialization fails → silent drop (correct behavior).

---

## Code Patterns

### Creating IncomingMessage

```rust
let msg = IncomingMessage::new("telegram", user_id, "hello")
    .with_thread(thread_id)
    .with_timezone("America/New_York")
    .with_metadata(serde_json::json!({"chat_id": -123456}))
    .with_sender_id(sender_id)
    .with_owner_id(owner_id);
```

### Responding to Message

```rust
async fn respond(
    &self,
    msg: &IncomingMessage,
    response: OutgoingResponse,
) -> Result<(), ChannelError> {
    let thread_id = msg.thread_id.clone().ok_or(NoThreadError)?;
    
    // Web gateway: broadcast via SSE
    self.state.sse.broadcast_for_user(
        &msg.user_id,
        AppEvent::Response {
            content: response.content,
            thread_id,
        },
    );
    
    Ok(())
}
```

### Sending Status Updates

```rust
async fn send_status(
    &self,
    status: StatusUpdate,
    metadata: &serde_json::Value,
) -> Result<(), ChannelError> {
    let thread_id = metadata
        .get("thread_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    
    let event = match status {
        StatusUpdate::Thinking(msg) => AppEvent::Thinking {
            message: msg,
            thread_id,
        },
        StatusUpdate::ToolCompleted { name, success, error, parameters } => {
            AppEvent::ToolCompleted {
                name,
                success,
                error,
                parameters,
                thread_id,
            }
        },
        // ... other variants
    };
    
    // Scope to user if available
    if let Some(uid) = metadata.get("user_id").and_then(|v| v.as_str()) {
        self.state.sse.broadcast_for_user(uid, event);
    } else {
        self.state.sse.broadcast(event); // Global (heartbeat, etc.)
    }
    
    Ok(())
}
```

### Proactive Broadcast

```rust
async fn broadcast(
    &self,
    user_id: &str,
    response: OutgoingResponse,
) -> Result<(), ChannelError> {
    let thread_id = response.thread_id.clone().ok_or(NoThreadError)?;
    
    self.state.sse.broadcast_for_user(
        user_id,
        AppEvent::Response {
            content: response.content,
            thread_id,
        },
    );
    
    Ok(())
}
```

**Note**: Requires stored routing target (e.g., `last_broadcast_metadata` for Telegram).

---

## Related Files

**Core**:
- `src/channels/mod.rs` - Module exports
- `src/channels/channel.rs` - Channel trait, message types
- `src/channels/manager.rs` - ChannelManager

**Web Gateway**:
- `src/channels/web/mod.rs` - GatewayChannel implementation
- `src/channels/web/server.rs` - Axum server, routes, GatewayState
- `src/channels/web/types.rs` - Request/response DTOs, AppEvent, WebSocket messages
- `src/channels/web/sse.rs` - SseManager, broadcast channel
- `src/channels/web/ws.rs` - WebSocket handler, WsConnectionTracker

**WASM**:
- `src/channels/wasm/mod.rs` - WASM channel runtime
- `src/channels/wasm/host.rs` - Host function bindings
- `src/channels/wasm/runtime.rs` - Wasmtime runtime setup

**Related Context**:
- `technical-domain.md` - Overall architecture, LoopDelegate pattern
- `agent-system.md` - Agent loop, session management
- `websocket-security.md` - Auth, rate limiting, CORS
