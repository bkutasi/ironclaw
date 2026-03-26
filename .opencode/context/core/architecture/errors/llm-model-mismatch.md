<!-- Context: architecture/errors | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Error: LLM Model Mismatch

**Purpose**: Fix 404 errors from invalid LLM model configuration

**Last Updated**: 2026-03-19

---

## Symptom

```json
{
  "error": "404 Not Found",
  "message": "Model stepfun-ai/step-3.5-flash not available"
}
```

IronClaw fails to start or returns errors when calling LLM APIs.

---

## Cause

`~/.ironclaw/.env` file contains invalid or unavailable model name. This file **overrides** script defaults, so even if startup script specifies correct model, the `.env` value takes precedence.

**Common scenario**:
- Was: `LLM_MODEL="stepfun-ai/step-3.5-flash"` (returns 404)
- Fixed: `LLM_MODEL="z-ai/glm5"` (NVIDIA NIM, available)

---

## Solution

### 1. Check Current Model
```bash
cat ~/.ironclaw/.env | grep LLM_MODEL
```

### 2. Update to Valid Model
```bash
# Edit ~/.ironclaw/.env
nano ~/.ironclaw/.env

# Change:
LLM_MODEL="z-ai/glm5"  # NVIDIA NIM (recommended)
```

### 3. Verify Model Availability
```bash
# Test with curl
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NGC_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-ai/glm5",
    "messages": [{"role": "user", "content": "test"}]
  }'
```

### 4. Restart IronClaw
```bash
pkill ironclaw
./target/release/ironclaw
```

---

## Prevention

- **Check NVIDIA NIM catalog** before setting model: https://build.nvidia.com/explore/discover
- **Don't hardcode models** in scripts - use environment variables
- **Test model availability** before adding to `.env`

---

## 📂 Codebase References

**Configuration**:
- `~/.ironclaw/.env` - User environment overrides
- `src/config/env.rs` - Environment variable parsing

**LLM Integration**:
- `src/llm/nvidia_nim.rs` - NVIDIA NIM client
- `src/llm/mod.rs` - LLM provider selection

---

## Related

- guides/env-config.md
- lookup/env-variables.md
- errors/common-build-issues.md
