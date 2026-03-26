<!-- Context: development/integration/webhooks | Priority: high | Version: 1.0 | Updated: 2026-03-26 -->

# Webhook Patterns

**Purpose**: Generic webhook ingress for tool-driven event processing - external webhook providers POST payloads that are normalized by tools into `system_event`s.

**Last Updated**: 2026-03-26

## Quick Reference

**Update Triggers**: Webhook handling changes | New signature verification methods | Routing updates | Security patches

**Audience**: Developers integrating external webhook providers, AI agents implementing webhook handlers

**Endpoints**:
- `POST /webhook/tools/{tool}` - Standard webhook ingestion
- `POST /webhook/tools/{tool}/{*rest}` - Webhook with REST path segments
- `GET /webhook/tools/{tool}` - Health check for webhook-capable tools

**Body Limit**: 64 KB (`MAX_WEBHOOK_BODY_BYTES`)

---

## Architecture Overview

```
External Provider                    Ironclaw Webhook Ingress                    Tool Implementation
┌──────────────────┐                ┌─────────────────────────┐                 ┌──────────────────┐
│ GitHub/GitLab/   │  POST          │ /webhook/tools/{tool}   │  Tool execute() │                  │
│ Discord/Slack/   │ ──────────────►│                         │────────────────►│  normalize_event │
│ Stripe/etc.      │  + headers     │  1. Validate auth       │  params:       │  emit_events: [] │
│                  │  + body        │  2. Parse payload       │  {             │                  │
│                  │                │  3. Call tool.execute() │   action:      │                  │
│                  │                │  4. Emit system events  │   "handle_     │                  │
│                  │                │                         │   webhook",     │                  │
│                  │                │                         │   webhook: {...}│                  │
│                  │                │                         │ })             │                  │
│                  │                │                         │                 │                  │
│                  │  202 ACCEPTED  │                         │  ToolOutput     │                  │
│                  │ ◄──────────────│                         │◄────────────────│                  │
│                  │  {status,      │                         │  {emit_events:  │                  │
│                  │   tool,        │                         │   [...]}        │                  │
│                  │   events,      │                         │                 │                  │
│                  │   routines}    │                         │                 │                  │
└──────────────────┘                └─────────────────────────┘                 └──────────────────┘
```

**Key Components**:
1. **Generic Ingress**: Single endpoint handles all webhook providers
2. **Tool-Driven Processing**: Each tool defines its own webhook handling logic
3. **Signature Verification**: Multiple auth mechanisms supported (secrets, HMAC, ed25519)
4. **Event Emission**: Tools emit `system_event`s that trigger routines via `RoutineEngine`

---

## Authentication & Signature Verification

### WebhookCapability Configuration

Tools declare webhook support via `webhook_capability()` method:

```rust
fn webhook_capability(&self) -> Option<crate::tools::wasm::WebhookCapability> {
    Some(WebhookCapability {
        // Simple secret-based auth
        secret_name: Some("github_webhook_secret".to_string()),
        secret_header: Some("x-webhook-secret".to_string()),
        
        // OR HMAC signature (GitHub/GitLab style)
        hmac_secret_name: Some("hmac_secret".to_string()),
        hmac_signature_header: Some("x-hub-signature-256".to_string()),
        hmac_prefix: Some("sha256=".to_string()),
        hmac_timestamp_header: None, // GitHub doesn't use timestamp
        
        // OR ed25519 signature (Discord style)
        signature_key_secret_name: Some("discord_public_key".to_string()),
        
        ..Default::default()
    })
}
```

### Authentication Mechanisms

**At least one mechanism MUST be configured**. The system validates this at request time.

#### 1. Secret Header (Simple)

```rust
// Configuration
secret_name: Some("my_webhook_secret"),
secret_header: Some("x-webhook-secret"), // defaults to "x-webhook-secret"

// Verification flow:
// 1. Fetch decrypted secret from SecretsStore
// 2. Compare header value using constant-time equality (ct_eq)
// 3. Reject if mismatch or missing
```

**Security**: Uses `subtle::ConstantTimeEq` to prevent timing attacks.

#### 2. HMAC SHA256 (GitHub/GitLab/Slack)

**Without Timestamp** (GitHub style):
```rust
hmac_secret_name: Some("github_secret"),
hmac_signature_header: Some("x-hub-signature-256"),
hmac_prefix: Some("sha256="),
hmac_timestamp_header: None, // Skip timestamp validation
```

**With Timestamp** (Slack style):
```rust
hmac_secret_name: Some("slack_signing_secret"),
hmac_signature_header: Some("x-slack-signature"),
hmac_timestamp_header: Some("x-slack-request-timestamp"),
```

**Verification Flow**:
1. Fetch HMAC secret from SecretsStore
2. Extract signature from configured header
3. Extract timestamp (if required)
4. Validate timestamp freshness (within 5 minutes)
5. Compute HMAC-SHA256 of body
6. Compare using constant-time equality

#### 3. Ed25519 Signature (Discord)

```rust
signature_key_secret_name: Some("discord_public_key"),
```

**Verification Flow**:
1. Fetch public key from SecretsStore
2. Extract signature from `x-signature-ed25519` header
3. Extract timestamp from `x-signature-timestamp` header
4. Validate timestamp freshness
5. Verify ed25519 signature over `timestamp + body`

### Signature Verification Functions

All signature verification delegates to `crate::channels::wasm::signature`:

- `verify_discord_signature()` - Ed25519 for Discord
- `verify_slack_signature()` - HMAC with timestamp for Slack
- `verify_hmac_sha256_prefixed()` - Generic HMAC with prefix (GitHub)

---

## Routing

### Endpoint Structure

```
/webhook/tools/{tool}           - Standard webhook
/webhook/tools/{tool}/{*rest}   - Webhook with additional path segments
```

**Path Parameters**:
- `{tool}` - Target tool name (required, must exist in ToolRegistry)
- `{*rest}` - Optional additional path segments (captured but not processed by default)

### Route Registration

```rust
use crate::webhooks::{routes, ToolWebhookState};

let state = ToolWebhookState {
    tools: Arc::new(tool_registry),
    routine_engine: routine_engine_slot,
    user_id: user_id.clone(),
    secrets_store: Some(secrets_store),
};

let app = Router::new()
    .merge(routes(state))
    .layer(DefaultBodyLimit::max(64 * 1024)); // 64 KB limit
```

### Tool Lookup & Validation

**Health Check** (`GET /webhook/tools/{tool}`):
- Returns `200 OK` if tool exists AND declares `webhook_capability()`
- Returns `404 NOT_FOUND` if tool doesn't exist OR doesn't support webhooks

**Request Processing**:
1. Lookup tool in `ToolRegistry`
2. Verify tool declares `webhook_capability()`
3. Validate authentication (see Authentication section)
4. Execute tool with webhook params

---

## Payload Processing

### Request Normalization

All webhook requests are normalized into a standard structure before passing to tool:

```rust
let params = serde_json::json!({
    "action": "handle_webhook",
    "webhook": {
        "method": "POST",                      // HTTP method
        "path": "/webhook/tools/github",       // Full path
        "query": {"ref": "main"},              // Query params (HashMap<String, String>)
        "headers": {"x-github-event": "push"}, // All headers (filtered to valid UTF-8)
        "body_json": {"action": "opened"},     // Parsed JSON (if valid)
        "body_raw": "{\"action\":\"opened\"}"  // Raw body as string
    }
});
```

### Tool Execution Context

```rust
let ctx = JobContext::with_user(
    user_id.clone(),
    format!("webhook:{tool}"),
    "Process external webhook",
);

let output = tool_impl.execute(params, &ctx).await?;
```

### Expected Tool Output

Tools MUST return a JSON object with optional `emit_events` array:

```rust
#[derive(Debug, Deserialize)]
struct ToolWebhookOutput {
    #[serde(default)]
    emit_events: Vec<SystemEventIntent>,
}

#[derive(Debug, Deserialize)]
struct SystemEventIntent {
    source: String,        // Event source identifier
    event_type: String,    // Event type (e.g., "push", "issue.opened")
    #[serde(default)]
    payload: serde_json::Value,  // Event payload
}
```

**Example Tool Response**:
```json
{
  "emit_events": [
    {
      "source": "github",
      "event_type": "push",
      "payload": {
        "ref": "refs/heads/main",
        "repository": "owner/repo",
        "commits": 3
      }
    }
  ]
}
```

### Event Emission & Routine Firing

If tool emits events:

```rust
for event in parsed.emit_events {
    fired_routines += engine
        .emit_system_event(
            &event.source,
            &event.event_type,
            &event.payload,
            Some(&user_id),
        )
        .await;
}
```

**Response**:
```json
{
  "status": "accepted",
  "tool": "github",
  "emitted_events": 1,
  "fired_routines": 2
}
```

---

## Code Patterns

### Implementing a Webhook Handler Tool

```rust
use async_trait::async_trait;
use crate::tools::{Tool, ToolOutput, ToolError};
use crate::context::JobContext;
use crate::tools::wasm::WebhookCapability;

#[async_trait]
impl Tool for GitHubWebhookTool {
    fn name(&self) -> &str {
        "github_webhook"
    }
    
    fn description(&self) -> &str {
        "Handle GitHub webhook events"
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    
    fn webhook_capability(&self) -> Option<WebhookCapability> {
        Some(WebhookCapability {
            hmac_secret_name: Some("github_webhook_secret".to_string()),
            hmac_signature_header: Some("x-hub-signature-256".to_string()),
            hmac_prefix: Some("sha256=".to_string()),
            ..Default::default()
        })
    }
    
    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        // Extract webhook data
        let webhook = params["webhook"].as_object()
            .ok_or_else(|| ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: "Missing webhook data".to_string(),
            })?;
        
        let event_type = webhook["headers"]["x-github-event"]
            .as_str()
            .unwrap_or("unknown");
        
        let body = &webhook["body_json"];
        
        // Normalize into system events
        let mut emit_events = Vec::new();
        
        match event_type {
            "push" => {
                emit_events.push(serde_json::json!({
                    "source": "github",
                    "event_type": "push",
                    "payload": {
                        "ref": body["ref"],
                        "repository": body["repository"]["full_name"],
                        "commits": body["commits"].as_array().map(|c| c.len()).unwrap_or(0)
                    }
                }));
            }
            "issues" => {
                let action = body["action"].as_str().unwrap_or("unknown");
                emit_events.push(serde_json::json!({
                    "source": "github",
                    "event_type": format!("issue.{}", action),
                    "payload": {
                        "issue": body["issue"]["number"],
                        "repository": body["repository"]["full_name"]
                    }
                }));
            }
            _ => {
                tracing::info!(event_type = %event_type, "Unhandled GitHub event");
            }
        }
        
        Ok(ToolOutput::success(
            serde_json::json!({ "emit_events": emit_events }),
            std::time::Duration::from_millis(10),
        ))
    }
}
```

### Error Handling Pattern

```rust
// In webhook handler
let output = match tool_impl.execute(params, &ctx).await {
    Ok(out) => out,
    Err(e) => {
        tracing::warn!(tool = %tool, error = %e, "Webhook tool execution failed");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Tool execution failed" })),
        );
    }
};

// Validate output structure
let parsed: ToolWebhookOutput = match serde_json::from_value(output.result) {
    Ok(v) => v,
    Err(_) => {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Tool webhook response must be a JSON object (optionally with 'emit_events' array)"
            })),
        );
    }
};
```

### Testing Webhook Handlers

```rust
#[tokio::test]
async fn accepts_with_valid_hmac_signature() {
    use hmac::Mac;
    
    let tools = Arc::new(ToolRegistry::new());
    tools.register(Arc::new(HmacWebhookTool)).await;
    
    let secrets = Arc::new(InMemorySecretsStore::new(Arc::new(
        SecretsCrypto::new(secrecy::SecretString::from(
            "test-key-at-least-32-chars-long!!".to_string(),
        ))
        .expect("crypto"),
    )));
    secrets
        .create(
            "test",
            CreateSecretParams::new("hmac_secret", "github-secret"),
        )
        .await
        .expect("secret create");
    
    let app = routes(ToolWebhookState {
        tools,
        routine_engine: Arc::new(tokio::sync::RwLock::new(None)),
        user_id: "test".to_string(),
        secrets_store: Some(secrets),
    });
    
    let payload = br#"{"action":"opened"}"#;
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(b"github-secret")
        .expect("hmac key");
    mac.update(payload);
    let sig = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));
    
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/webhook/tools/hmac_webhook")
        .header("content-type", "application/json")
        .header("x-hub-signature-256", sig)
        .body(Body::from(payload.to_vec()))
        .expect("request");
    
    let resp = ServiceExt::<axum::http::Request<Body>>::oneshot(app, req)
        .await
        .expect("response");
    
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}
```

---

## Security Considerations

### 1. Constant-Time Comparison

All secret/signature comparisons use `subtle::ConstantTimeEq` to prevent timing attacks:

```rust
if !bool::from(expected.as_bytes().ct_eq(provided.as_bytes())) {
    return Err("Invalid webhook secret".to_string());
}
```

### 2. Body Size Limits

Webhook bodies are limited to 64 KB via `DefaultBodyLimit` layer:

```rust
if body.len() > MAX_WEBHOOK_BODY_BYTES {
    return (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(serde_json::json!({
            "error": format!("Webhook body exceeds {} bytes", MAX_WEBHOOK_BODY_BYTES)
        })),
    );
}
```

### 3. Secrets Store Requirement

Webhook authentication requires `SecretsStore` to be available. Requests fail with `503 SERVICE UNAVAILABLE` if secrets store is missing.

### 4. Misconfiguration Detection

Tools that declare `webhook_capability()` but configure NO authentication mechanism are rejected:

```rust
if cfg.secret_name.is_none()
    && cfg.signature_key_secret_name.is_none()
    && cfg.hmac_secret_name.is_none()
{
    return Err(
        "Webhook capability misconfigured: at least one auth mechanism must be configured"
            .to_string(),
    );
}
```

### 5. Timestamp Validation

HMAC signatures with timestamps are validated for freshness (within 5 minutes of current time) to prevent replay attacks.

---

## Related Files

- `src/webhooks/mod.rs` - Webhook ingress implementation (712 lines)
- `src/tools/wasm/mod.rs` - `WebhookCapability` struct definition
- `src/channels/wasm/signature.rs` - Signature verification functions
- `src/agent/routine_engine.rs` - `RoutineEngine` for event-driven automation
- `src/secrets/` - Secrets storage and encryption

## Related Context

- **Technical Domain** → `../../project-intelligence/technical-domain.md`
- **Code Quality** → `../../core/standards/code-quality.md`
- **Security Patterns** → `../../core/standards/security-patterns.md` (if exists)
