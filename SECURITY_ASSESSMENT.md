# Ironclaw Security Assessment

**Date**: March 21, 2026 | **Posture**: Good (7.5/10) | **Critical Issues**: 0

---

## Executive Summary

Ironclaw has **strong security foundations**: AES-256-GCM encryption, Docker sandboxing, zero-exposure credential model, and defense-in-depth architecture. Priority improvements: secure `.env` permissions and migrate tokens to secrets manager.

---

## Current Security Posture

### ✅ Strengths

| Area | Implementation |
|------|----------------|
| **Secrets Management** | AES-256-GCM encryption, OS keychain integration, per-secret key derivation (HKDF-SHA256), leak detection (15+ patterns) |
| **Docker Sandbox** | Non-root (UID 1000), `cap_drop: ALL`, read-only root FS, tmpfs mounts, network allowlist, resource limits |
| **API Security** | Bearer token auth (constant-time), rate limiting (30/60s chat, 60/min webhook), HMAC-SHA256 signatures, CORS restrictions |
| **Safety Layer** | Input sanitization, injection detection, policy enforcement, environment scrubbing |

### ⚠️ Priority Improvements

| Priority | Issue | Mitigation |
|----------|-------|------------|
| **HIGH** | `.env` file plaintext storage | Migrate to secrets manager, `chmod 600` |
| **MEDIUM** | `SANDBOX_POLICY=workspace_write` | Appropriate for personal use; never use `full_access` |
| **MEDIUM** | No security audit logging | Add `RUST_LOG=ironclaw::secrets=debug` |
| **LOW** | Database SSL `prefer` mode | Use `require` for remote PostgreSQL |

---

## Key Findings

### Secrets Flow
```
~/.ironclaw/.env → Ironclaw reads → AES-256-GCM encrypt → PostgreSQL (encrypted) → HTTP proxy injects → External APIs
```

### Risk Assessment

| Threat | Likelihood | Impact | Status |
|--------|------------|--------|--------|
| `.env` file theft | Medium | High | 🟡 Mitigate with chmod 600 + secrets manager |
| Prompt injection | High | Medium | ✅ Safety layer + sandbox |
| Sandbox escape | Low | Critical | ✅ Docker hardening |
| Credential leak | Low | High | ✅ Zero-exposure model |
| Network MITM | Low | High | ✅ HTTPS, localhost DB |

**Overall Risk**: 🟢 **LOW** (appropriate for personal use)

---

## Immediate Actions (< 30 min)

```bash
# 1. Secure .env permissions
chmod 600 ~/.ironclaw/.env
chmod 700 ~/.ironclaw

# 2. Verify sandbox policy
grep SANDBOX_POLICY ~/.ironclaw/.env  # Should be: workspace_write

# 3. Check master key status
./target/release/ironclaw doctor | grep -A2 "Master Key"

# 4. Migrate tokens to secrets manager
./target/release/ironclaw secrets set telegram_bot_token "YOUR_TOKEN"
./target/release/ironclaw secrets list
```

---

## Verification Commands

```bash
# File permissions
ls -la ~/.ironclaw/.env  # Should show: -rw-------

# Sandbox status
./target/release/ironclaw sandbox status

# Secret access test
./target/release/ironclaw sandbox test --network api.telegram.org

# Recent security events
tail -100 /tmp/ironclaw.log | grep -iE "error|warn|secret"
```

---

## Architecture Overview

```
User Input → Auth (Bearer Token) → Rate Limiting → Safety Layer (Injection/Leak Detection)
    ↓
Docker Sandbox (Non-root, Cap Drop, Network Proxy)
    ↓
Secrets (Encrypted at Rest, Injected at HTTP Proxy)
    ↓
External APIs (HTTPS Only)
```

---

**Next Review**: June 21, 2026 (quarterly) | **Full docs**: See original assessment in git history
