<!-- Context: core/architecture/concepts | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->

# Safety Layer: Prompt Injection Defense

**Core Idea**: The safety layer protects against prompt injection attacks, secret leakage, and policy violations by sanitizing external data before it reaches the LLM and scanning outputs for credential exposure.

## Quick Reference

**Protection Layers**:
1. **Input Validation** — Reject malformed/dangerous inputs early
2. **Prompt Injection Detection** — Detect and neutralize injection attempts
3. **Policy Enforcement** — Apply configurable safety rules
4. **Leak Detection** — Block secret exfiltration (API keys, tokens, credentials)

**Key Locations**:
- `crates/ironclaw_safety/src/` — Core safety crate implementation
- `src/safety/mod.rs` — Re-export for `crate::safety::*` imports

**When It Activates**:
- ✅ Tool outputs before reaching LLM
- ✅ External content (emails, webhooks, API responses)
- ✅ Inbound user messages (secret detection)
- ✅ Outbound HTTP requests from WASM tools

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Safety Layer Pipeline                               │
│                                                                              │
│   External Data ──► Validator ──► Sanitizer ──► Policy ──► LeakDetector    │
│       │                │              │           │            │            │
│       │                │              │           │            │            │
│       ▼                ▼              ▼           ▼            ▼            │
│   Reject if       Check length   Detect      Check rules   Scan for       │
│   invalid         & encoding     injection  (block/warn)  secrets         │
│                                                                              │
│   Final Output: Wrapped in <tool_output> delimiters for LLM                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    WASM Tool Request Flow                                   │
│                                                                              │
│   WASM ──► Allowlist ──► Leak Scan ──► Credential ──► Execute ──► Response │
│            Validator     (request)     Injector       Request      │        │
│                                                                    ▼        │
│                                      WASM ◀── Leak Scan ◀── Response       │
│                                               (response)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Components** (`crates/ironclaw_safety/src/`):
- `lib.rs` — `SafetyLayer` unified interface
- `sanitizer.rs` — Prompt injection detection
- `leak_detector.rs` — Secret leak prevention
- `policy.rs` — Configurable safety rules
- `validator.rs` — Input validation
- `credential_detect.rs` — HTTP credential detection

---

## Input Validation

**Purpose**: Catch malformed or suspicious inputs before processing.

**Validation Checks**:
- Length limits (default: 1-100,000 bytes)
- Null byte detection (`\x00`)
- Forbidden patterns (configurable)
- Excessive whitespace ratio (>90% warns)
- Character repetition (>20 repeats warns)

**Usage**:
```rust
use ironclaw_safety::{SafetyLayer, SafetyConfig};

let config = SafetyConfig {
    max_output_length: 100_000,
    injection_check_enabled: true,
};
let safety = SafetyLayer::new(&config);

// Validate user input
let result = safety.validate_input(user_message);
if !result.is_valid {
    // Handle validation errors
    for error in result.errors {
        eprintln!("{}: {}", error.code, error.message);
    }
}
```

**Error Codes**:
- `Empty` — Input is empty
- `TooLong` / `TooShort` — Length violation
- `InvalidEncoding` — Contains null bytes
- `ForbiddenContent` — Matches forbidden pattern
- `SuspiciousPattern` — Unusual structure detected

---

## Prompt Injection Defense

**Purpose**: Detect and neutralize attempts to override system instructions or manipulate the LLM.

### Detection Patterns

**Aho-Corasick Multi-Pattern Matching** (fast, case-insensitive):

| Pattern | Severity | Description |
|---------|----------|-------------|
| `ignore previous` | High | Override instructions |
| `ignore all previous` | Critical | Full context reset |
| `disregard` | Medium | Potential override |
| `forget everything` | High | Context reset attempt |
| `you are now` | High | Role manipulation |
| `act as` / `pretend to be` | Medium | Role manipulation |
| `system:` / `assistant:` / `user:` | Critical-High | Message injection |
| `<\|` / `\|>` | Critical | Special token injection |
| `[INST]` / `[/INST]` | Critical | Instruction token injection |
| `new instructions` | High | Instruction injection |
| ````system` | High | Code block injection |

**Regex Patterns** (complex detection):
- `base64[:\s]+[A-Za-z0-9+/=]{50,}` — Encoded payloads
- `eval\s*\(` / `exec\s*\(` — Code execution attempts
- `\x00` — Null byte injection

### Sanitization Actions

**Critical Severity** → Content is escaped:
- `<\|` → `\<\|` (zero-width space insertion)
- `\|>` → `\|\\>` 
- `[INST]` → `\[INST]`
- `\x00` → Removed entirely
- Line-starting `system:`, `user:`, `assistant:` → Prefixed with `[ESCAPED]`

**Lower Severity** → Warnings logged, content passes through

**Usage**:
```rust
let sanitizer = Sanitizer::new();
let result = sanitizer.sanitize(external_content);

// Check for warnings
for warning in &result.warnings {
    tracing::warn!(
        "Injection detected: {} (severity: {:?})",
        warning.description,
        warning.severity
    );
}

// Content is escaped if critical issues found
if result.was_modified {
    // Critical patterns were neutralized
}
```

### Wrapping for LLM

**Structural Boundary Protection**:
```rust
// Wrap tool output with security delimiters
let wrapped = safety.wrap_for_llm("search_tool", tool_output);
// Produces: <tool_output name="search_tool">\n...content...\n</tool_output>

// Closing tags are escaped to prevent boundary injection:
// </tool_output> → <​/tool_output> (zero-width space after <)

// Unwrap when processing response
let original = SafetyLayer::unwrap_tool_output(&wrapped);
```

**External Content Wrapper**:
```rust
use ironclaw_safety::wrap_external_content;

let wrapped = wrap_external_content(
    "email from alice@example.com",
    email_content
);
// Adds SECURITY NOTICE warning about untrusted content
// Escapes closing delimiter to prevent boundary escape
```

---

## Policy Enforcement

**Purpose**: Apply configurable rules to block or warn on dangerous content patterns.

### Default Policy Rules

| Rule ID | Pattern | Action | Severity |
|---------|---------|--------|----------|
| `system_file_access` | `/etc/passwd`, `/etc/shadow`, `.ssh/`, `.aws/credentials` | Block | Critical |
| `crypto_private_key` | `private key` + 64 hex chars | Block | Critical |
| `shell_injection` | `; rm -rf`, `; curl ... \| sh` | Block | Critical |
| `encoded_exploit` | `base64_decode`, `eval(base64`, `atob(` | Sanitize | High |
| `sql_pattern` | `DROP TABLE`, `DELETE FROM`, etc. | Warn | Medium |
| `obfuscated_string` | 500+ chars without spaces | Warn | Medium |
| `excessive_urls` | 10+ URLs in sequence | Warn | Low |

**Actions**:
- `Block` — Content rejected entirely
- `Sanitize` — Dangerous patterns neutralized
- `Warn` — Logged but allowed
- `Review` — Requires human approval (future)

**Usage**:
```rust
let policy = Policy::default();

// Check content against all rules
let violations = policy.check(content);
if violations.iter().any(|r| r.action == PolicyAction::Block) {
    // Reject content
    return Err("Content blocked by safety policy");
}

// Custom rules
let mut policy = Policy::new();
policy.add_rule(PolicyRule::new(
    "custom_rule",
    "Description of what this blocks",
    r"dangerous_pattern",
    Severity::High,
    PolicyAction::Block,
)?);
```

---

## Leak Detection

**Purpose**: Prevent accidental exposure of secrets (API keys, tokens, credentials) in tool outputs, logs, or outbound requests.

### Detected Secret Types

| Pattern | Severity | Action | Example |
|---------|----------|--------|---------|
| `openai_api_key` | Critical | Block | `sk-proj-...` |
| `anthropic_api_key` | Critical | Block | `sk-ant-api...` |
| `aws_access_key` | Critical | Block | `AKIA...` |
| `github_token` | Critical | Block | `ghp_...`, `gho_...` |
| `github_fine_grained_pat` | Critical | Block | `github_pat_...` |
| `stripe_api_key` | Critical | Block | `sk_live_...`, `sk_test_...` |
| `nearai_session` | Critical | Block | `sess_...` |
| `pem_private_key` | Critical | Block | `-----BEGIN PRIVATE KEY-----` |
| `ssh_private_key` | Critical | Block | `-----BEGIN OPENSSH PRIVATE KEY-----` |
| `google_api_key` | High | Block | `AIza...` |
| `slack_token` | High | Block | `xoxb-...`, `xoxp-...` |
| `twilio_api_key` | High | Block | `SK...` |
| `sendgrid_api_key` | High | Block | `SG....` |
| `bearer_token` | High | Redact | `Bearer eyJ...` |
| `auth_header` | High | Redact | `Authorization: ...` |
| `high_entropy_hex` | Medium | Warn | 64-char hex strings |

### Scan Actions

- **Block** — Content rejected, error returned
- **Redact** — Secret replaced with `[REDACTED]`
- **Warn** — Logged but content passes through

**Usage**:
```rust
let detector = LeakDetector::new();

// Scan and clean (returns Err if should block)
match detector.scan_and_clean(output) {
    Ok(cleaned) => {
        // Content is clean or redacted
        send_to_llm(&cleaned);
    }
    Err(LeakDetectionError::SecretLeakBlocked { pattern, preview }) => {
        // Blocked secret leak
        tracing::error!("Secret leak blocked: {} - {}", pattern, preview);
    }
}

// Scan outbound HTTP request
detector.scan_http_request(url, headers, body)?;
```

### HTTP Request Scanning

**Critical for WASM tools** — Scans at three points:
1. **URL** — Query params, path
2. **Headers** — All header values
3. **Body** — Request body (lossy UTF-8 conversion)

```rust
// Blocks if secret detected in any part
detector.scan_http_request(
    "https://api.example.com/data?key=AKIAIOSFODNN7EXAMPLE", // ← Blocks: AWS key
    &[("Content-Type", "application/json")],
    Some(b"{\"query\": \"test\"}"),
)?;
```

**Known Limitation**: Percent-encoded secrets bypass detection (e.g., `AKIA%49...` instead of `AKIAI...`). Scanner operates on raw strings, not decoded forms.

---

## Credential Detection (HTTP Tools)

**Purpose**: Detect manually-provided credentials in HTTP tool parameters to trigger approval workflows.

**Detection Targets**:
- **Header names**: `Authorization`, `X-API-Key`, `Cookie`, etc.
- **Header values**: `Bearer ...`, `Basic ...`, `Token ...`
- **Query params**: `api_key=`, `access_token=`, `secret=`
- **URL userinfo**: `https://user:pass@host/`

**Usage**:
```rust
use ironclaw_safety::params_contain_manual_credentials;

let params = serde_json::json!({
    "method": "GET",
    "url": "https://api.example.com/data?api_key=secret123",
    "headers": {"Authorization": "Bearer token"}
});

if params_contain_manual_credentials(&params) {
    // Require human approval before executing
    require_approval();
}
```

**False Positive Avoidance**:
- `X-Idempotency-Key` — NOT detected (not a credential)
- `Content-Type`, `Accept` — NOT detected

---

## Code Patterns

### Basic Safety Layer Setup

```rust
use ironclaw_safety::{SafetyLayer, SafetyConfig};

// Configure safety layer
let config = SafetyConfig {
    max_output_length: 100_000,  // Truncate larger outputs
    injection_check_enabled: true,
};

let safety = SafetyLayer::new(&config);

// Sanitize tool output before LLM
let output = safety.sanitize_tool_output("web_search", raw_output);

// Check for warnings
for warning in &output.warnings {
    tracing::warn!("Sanitization warning: {}", warning.description);
}

// Wrap for LLM with structural boundaries
let wrapped = safety.wrap_for_llm("web_search", &output.content);

// Send to LLM
send_to_llm(ChatMessage::tool_result(wrapped));
```

### Inbound Message Screening

```rust
// Screen user messages for accidental secret exposure
if let Some(warning) = safety.scan_inbound_for_secrets(user_message) {
    // Reject message before sending to LLM
    return Err(warning);
}
```

### External Content Handling

```rust
use ironclaw_safety::wrap_external_content;

// Wrap untrusted external data (emails, webhooks, API responses)
let safe_content = wrap_external_content(
    "webhook from GitHub",
    &webhook_payload
);

// Inject into conversation with security notice
messages.push(ChatMessage::user(safe_content));
```

### Custom Policy Rules

```rust
use ironclaw_safety::{Policy, PolicyRule, PolicyAction, Severity};

let mut policy = Policy::new();

// Add custom rule for your use case
policy.add_rule(PolicyRule::new(
    "no_internal_ips",
    "Block requests to internal IP addresses",
    r"(127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)",
    Severity::High,
    PolicyAction::Block,
)?);

// Check content
if policy.is_blocked(content) {
    return Err("Content violates safety policy");
}
```

### WASM Tool Protection

```rust
// In WASM tool HTTP transport
let leak_detector = LeakDetector::new();

// Before executing request
leak_detector.scan_http_request(&url, &headers, body)?;

// After receiving response
let clean_response = leak_detector.scan_and_clean(&response_body)?;
```

---

## Known Limitations

### Unicode Bypass Vectors

The following can bypass pattern detection (documented in adversarial tests):

- **Zero-width characters** (`\u{200B}`, `\u{200C}`, `\u{200D}`) inserted into patterns break literal matching
- **Combining diacriticals** change string matching (e.g., `s\u{0301}ystem:` ≠ `system:`)
- **Percent-encoding** in URLs bypasses raw string scanning

### Performance Characteristics

- **Aho-Corasick**: O(n + m) where n = text length, m = pattern count
- **Regex patterns**: Compiled once, reused across scans
- **100KB inputs**: All scans complete in <100ms (verified by adversarial tests)

### Whitespace Ratio Calculation

Uses byte length (`input.len()`) instead of char count for denominator. This makes the ratio artificially low for multibyte UTF-8 content (e.g., CJK characters).

---

## 📂 Codebase References

**Core Implementation**:
- `crates/ironclaw_safety/src/lib.rs` — `SafetyLayer` unified interface
- `crates/ironclaw_safety/src/sanitizer.rs` — Injection detection (lines 1-725)
- `crates/ironclaw_safety/src/leak_detector.rs` — Secret scanning (lines 1-1336)
- `crates/ironclaw_safety/src/policy.rs` — Policy rules (lines 1-535)
- `crates/ironclaw_safety/src/validator.rs` — Input validation (lines 1-776)
- `crates/ironclaw_safety/src/credential_detect.rs` — HTTP credential detection

**Integration Points**:
- `src/safety/mod.rs` — Re-export for `crate::safety::*`
- `src/worker/job.rs` — Tool output sanitization (lines 674, 781-803)
- `src/tools/execute.rs` — Tool result processing (lines 118-133)
- `src/tools/wasm/wrapper.rs` — WASM leak detection (lines 294-308)
- `src/workspace/mod.rs` — Write sanitization (lines 112-130)
- `src/llm/provider.rs` — Message sanitization (line 457)

**Configuration**:
- `SafetyConfig` struct — Max length, injection check toggle
- Default policy rules — `Policy::default()` (lines 127-223 in policy.rs)
- Default leak patterns — `default_patterns()` (lines 414-533 in leak_detector.rs)

---

## Related

- `sandbox/proxy/policy.rs` — Network policy decisions (allowlist, credential injection)
- `tools/wasm/wrapper.rs` — WASM sandbox boundary protection
- `project-intelligence/concepts/tool-protection.md` — Protected tool handling
- `NETWORK_SECURITY.md` — Broader network security model
