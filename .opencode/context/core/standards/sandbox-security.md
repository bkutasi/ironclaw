<!-- Context: standards/security/sandbox | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->
# Sandbox Security Standards

## Quick Reference

**Security Model**: Defense in depth with container isolation, network proxy, and credential injection

**Critical Invariants**:
- ✅ No credentials ever enter containers (injected by proxy only)
- ✅ All network traffic routes through validating proxy (allowlist enforced)
- ✅ Containers run as non-root (UID 1000)
- ✅ Read-only root filesystem (except workspace mount)
- ✅ All Linux capabilities dropped except CHOWN
- ✅ Auto-cleanup after execution (--rm + explicit removal)
- ✅ Timeout enforcement (commands killed after limit)

**Policies**:
- `ReadOnly` — Read workspace, proxied network (explore code, fetch docs)
- `WorkspaceWrite` — Read/write workspace, proxied network (build, test)
- `FullAccess` — Full host access (DANGER: requires double opt-in)

---

## Security Model

**Defense in Depth**: Multiple layers ensure compromise of one layer doesn't breach the system

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Sandbox System                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        SandboxManager                                │    │
│  │                                                                      │    │
│  │  • Coordinates container creation and execution                     │    │
│  │  • Manages proxy lifecycle                                          │    │
│  │  • Enforces resource limits                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│           │                              │                                   │
│           ▼                              ▼                                   │
│  ┌──────────────────┐          ┌───────────────────┐                        │
│  │   Container      │          │   Network Proxy   │                        │
│  │   Runner         │          │                   │                        │
│  │                  │          │  • Allowlist      │                        │
│  │  • Create        │◀────────▶│  • Credentials    │                        │
│  │  • Execute       │          │  • Logging        │                        │
│  │  • Cleanup       │          │                   │                        │
│  └──────────────────┘          └───────────────────┘                        │
│           │                              │                                   │
│           ▼                              ▼                                   │
│  ┌──────────────────┐          ┌───────────────────┐                        │
│  │     Docker       │          │     Internet      │                        │
│  │                  │          │   (allowed hosts) │                        │
│  └──────────────────┘          └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Layer 1: Container Isolation** — Commands run in ephemeral Docker containers with strict limits
**Layer 2: Network Proxy** — All traffic validated against allowlist, credentials injected externally
**Layer 3: Resource Limits** — Memory, CPU, timeout, and output size enforcement
**Layer 4: Credential Injection** — Secrets never enter containers, injected by proxy on egress

---

## Container Isolation

### Security Configuration

```rust
// From src/sandbox/container.rs
let host_config = HostConfig {
    // Drop all capabilities, add back only CHOWN
    cap_drop: Some(vec!["ALL".to_string()]),
    cap_add: Some(vec!["CHOWN".to_string()]),
    
    // Prevent privilege escalation
    security_opt: Some(vec!["no-new-privileges:true".to_string()]),
    
    // Read-only root filesystem (workspace mount exception)
    readonly_rootfs: Some(policy != SandboxPolicy::FullAccess),
    
    // Resource limits
    memory: Some((limits.memory_bytes) as i64),  // Default: 2GB
    cpu_shares: Some(limits.cpu_shares as i64),  // Default: 1024
    
    // Tmpfs for temporary storage (cleared on exit)
    tmpfs: Some([
        ("/tmp".to_string(), "size=512M".to_string()),
        ("/home/sandbox/.cargo/registry".to_string(), "size=1G".to_string()),
    ].into_iter().collect()),
    
    // Non-root user
    user: Some("1000:1000".to_string()),
    
    ..Default::default()
};
```

### Isolation Properties

| Property | Setting | Purpose |
|----------|---------|---------|
| User | UID 1000:1000 | Non-root execution |
| Root filesystem | Read-only | Prevent persistence |
| Capabilities | ALL dropped, CHOWN added | Minimal privileges |
| Privilege escalation | `no-new-privileges:true` | Block setuid attacks |
| Network | Bridge (proxied) | Controlled egress |
| Temp storage | Tmpfs (512M-1G) | Auto-cleanup |
| Auto-remove | `--rm` + explicit cleanup | No container persistence |

### Workspace Mounts by Policy

```rust
let binds = match policy {
    SandboxPolicy::ReadOnly => {
        vec![format!("{}:/workspace:ro", working_dir_str)]
    }
    SandboxPolicy::WorkspaceWrite => {
        vec![format!("{}:/workspace:rw", working_dir_str)]
    }
    SandboxPolicy::FullAccess => {
        vec![
            format!("{}:/workspace:rw", working_dir_str),
            "/tmp:/tmp:rw".to_string(),
        ]
    }
};
```

---

## Network Policies

### Proxy Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Network Proxy                               │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ HTTP Proxy  │───▶│   Policy    │───▶│ Credential Resolver │  │
│  │   Server    │    │   Decider   │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                  │                                     │
│         │                  ▼                                     │
│         │           ┌─────────────┐                             │
│         │           │  Allowlist  │                             │
│         │           │  Validator  │                             │
│         │           └─────────────┘                             │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Internet                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow

1. **Container** sets `http_proxy=http://host.docker.internal:PORT`
2. **Proxy** receives request, extracts host from URL
3. **Policy Decider** checks allowlist
4. **Credential Resolver** injects secrets if domain matches mapping
5. **Proxy** forwards validated request to internet
6. **Response** returned to container

### Policy Decisions

```rust
pub enum NetworkDecision {
    /// Allow the request as-is
    Allow,
    
    /// Allow with credential injection
    AllowWithCredentials {
        secret_name: String,      // e.g., "OPENAI_API_KEY"
        location: CredentialLocation,  // Header, QueryParam, Bearer
    },
    
    /// Deny the request
    Deny {
        reason: String,  // e.g., "host 'evil.com' not in allowlist"
    },
}
```

### CONNECT Tunnel Handling

**Important**: HTTPS tunneling (CONNECT method) bypasses credential injection since traffic is encrypted. Containers needing authenticated HTTPS should:
1. Fetch credentials via orchestrator's `GET /worker/{id}/credentials` endpoint
2. Set credentials as environment variables directly

```rust
// From src/sandbox/proxy/http.rs
// NOTE: Credential injection is not possible through CONNECT tunnels
// since the proxy cannot inspect or modify TLS-encrypted traffic without MITM.
```

---

## Allowlists

### Domain Validation

```rust
// From src/sandbox/proxy/allowlist.rs
pub struct DomainAllowlist {
    patterns: Vec<DomainPattern>,  // Supports exact and wildcard patterns
}

impl DomainAllowlist {
    /// Check if a domain is allowed
    pub fn is_allowed(&self, host: &str) -> DomainValidationResult {
        // Wildcard: *.example.com matches foo.example.com, example.com
        // Exact: api.example.com matches only api.example.com
    }
}
```

### Default Allowlist

```rust
// From src/sandbox/config.rs
pub fn default_allowlist() -> Vec<String> {
    vec![
        // Package registries
        "crates.io", "static.crates.io", "index.crates.io",
        "registry.npmjs.org", "proxy.golang.org",
        "pypi.org", "files.pythonhosted.org",
        
        // Documentation
        "docs.rs", "doc.rust-lang.org", "nodejs.org",
        "go.dev", "docs.python.org",
        
        // Version control (read-only)
        "github.com", "raw.githubusercontent.com",
        "api.github.com", "codeload.github.com",
        
        // Common APIs (credentials injected by proxy)
        "api.openai.com", "api.anthropic.com", "api.near.ai",
    ]
}
```

### Wildcard Pattern Matching

```rust
// From src/sandbox/proxy/policy.rs
fn host_matches_pattern(host: &str, pattern: &str) -> bool {
    let pattern_lower = pattern.to_lowercase();
    if pattern_lower == host {
        return true;
    }
    
    // Support wildcard: *.example.com matches sub.example.com
    if let Some(suffix) = pattern_lower.strip_prefix("*.")
        && host.ends_with(suffix)
        && host.len() > suffix.len()
    {
        let prefix = &host[..host.len() - suffix.len()];
        if prefix.ends_with('.') || prefix.is_empty() {
            return true;
        }
    }
    
    false
}
```

### Security Tests (Adversarial)

```rust
// From src/sandbox/proxy/allowlist.rs tests

#[test]
fn test_subdomain_bypass_attempt() {
    let allowlist = DomainAllowlist::new(&["api.example.com".to_string()]);
    
    // Exact match should work
    assert!(allowlist.is_allowed("api.example.com").is_allowed());
    
    // Subdomain of exact match should NOT be allowed
    assert!(!allowlist.is_allowed("evil.api.example.com").is_allowed());
    
    // Similar-looking domains should NOT be allowed
    assert!(!allowlist.is_allowed("api.example.com.evil.com").is_allowed());
    assert!(!allowlist.is_allowed("api-example.com").is_allowed());
}

#[test]
fn test_ip_address_not_matched_by_domain() {
    let allowlist = DomainAllowlist::new(&["example.com".to_string()]);
    
    // IP addresses should NOT match domain names
    assert!(!allowlist.is_allowed("93.184.216.34").is_allowed());
    assert!(!allowlist.is_allowed("127.0.0.1").is_allowed());
}
```

---

## Security Boundaries

### Policy Comparison

| Policy | Filesystem | Network | Credentials | Use Case |
|--------|------------|---------|-------------|----------|
| `ReadOnly` | /workspace (ro) | Proxied (allowlist) | Injected by proxy | Explore code, fetch docs |
| `WorkspaceWrite` | /workspace (rw) | Proxied (allowlist) | Injected by proxy | Build, test, generate |
| `FullAccess` | Full host | Full (no proxy) | Environment vars | Legacy/trusted ops |

### FullAccess Blast Radius

**WARNING**: `FullAccess` policy bypasses all sandbox protections:

```rust
// From src/sandbox/config.rs
/// **BLAST RADIUS**: This bypasses Docker entirely and executes commands
/// via `sh -c` directly on the host with the agent process's full
/// privileges. If prompt injection bypasses tool approval, arbitrary
/// host shell commands can run. File system, network, and environment
/// are completely unrestricted.
///
/// Requires `SANDBOX_ALLOW_FULL_ACCESS=true` as a second opt-in.
FullAccess,
```

### Double Opt-In for FullAccess

```rust
// From src/sandbox/manager.rs
if policy == SandboxPolicy::FullAccess {
    if !self.config.allow_full_access {
        return Err(SandboxError::Config {
            reason: "FullAccess policy requires SANDBOX_ALLOW_FULL_ACCESS=true".to_string(),
        });
    }
    // Log only the binary name to avoid leaking secrets in command args
    let binary = command.split_whitespace().next().unwrap_or("<empty>");
    tracing::warn!(
        binary = %binary,
        cwd = %cwd.display(),
        "[FullAccess] Executing command directly on host (no sandbox isolation)"
    );
    return self.execute_direct(command, cwd, env).await;
}
```

---

## Enforcement Patterns

### Retry Logic for Transient Failures

```rust
// From src/sandbox/manager.rs
const MAX_SANDBOX_RETRIES: u32 = 2;
let mut last_err: Option<SandboxError> = None;

for attempt in 0..=MAX_SANDBOX_RETRIES {
    if attempt > 0 {
        let delay = std::time::Duration::from_secs(1 << attempt); // 2s, 4s
        tokio::time::sleep(delay).await;
    }
    
    match self.try_execute_in_container(command, cwd, policy, env.clone()).await {
        Ok(output) => return Ok(output),
        Err(e) if is_transient_sandbox_error(&e) => {
            last_err = Some(e);
            // Retry
        }
        Err(e) => return Err(e),  // Non-transient, fail immediately
    }
}
```

### Transient vs Non-Transient Errors

```rust
fn is_transient_sandbox_error(err: &SandboxError) -> bool {
    matches!(
        err,
        SandboxError::DockerNotAvailable { .. }
            | SandboxError::ContainerCreationFailed { .. }
            | SandboxError::ContainerStartFailed { .. }
    )
}
```

**Retry**: Docker daemon glitches, container creation races, image pull failures
**Don't Retry**: Timeouts, execution failures, network blocks, config errors

### Output Truncation

```rust
// From src/sandbox/container.rs
fn append_with_limit(buffer: &mut String, text: &str, limit: usize) -> bool {
    if buffer.len() >= limit {
        return true;  // Already at limit
    }
    
    let remaining = limit - buffer.len();
    if text.len() <= remaining {
        buffer.push_str(text);
        return false;
    }
    
    // Truncate at UTF-8 boundary
    let end = crate::util::floor_char_boundary(text, remaining);
    buffer.push_str(&text[..end]);
    true  // Truncated
}
```

### Resource Limits

```rust
// From src/sandbox/config.rs
impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            memory_bytes: 2 * 1024 * 1024 * 1024,  // 2 GB
            cpu_shares: 1024,
            timeout: Duration::from_secs(120),
            max_output_bytes: 64 * 1024,  // 64 KB
        }
    }
}
```

---

## Best Practices

✅ **Always use sandboxed policies** (`ReadOnly` or `WorkspaceWrite`) for untrusted code
✅ **Keep allowlist minimal** — only domains required for the task
✅ **Never expose credentials** in command arguments or container environment
✅ **Use wildcard patterns carefully** — `*.example.com` allows all subdomains
✅ **Monitor proxy logs** for blocked requests (potential attacks)
✅ **Test adversarial scenarios** — subdomain bypasses, IP addresses, case sensitivity

❌ **Never enable `FullAccess`** without explicit user consent and understanding
❌ **Never add broad allowlist entries** like `*` or overly permissive wildcards
❌ **Never modify container security opts** without security review
❌ **Never skip credential injection** for domains that require auth

**Golden Rule**: If a command could compromise the host if malicious, it must run in the sandbox with proxied network access.
