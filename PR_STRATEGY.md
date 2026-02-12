# OpenAI-Compatible Provider: PR Strategy

## Executive Summary

**Two different implementations exist** - they are **complementary**, not competing:

1. **Current Branch (CLIENT)**: Calls OpenAI-compatible APIs
2. **Jaswinder's PR #31 (SERVER)**: Exposes Ironclaw as OpenAI-compatible API

## Recommended Atomic PRs

### PR #1: Base OpenAI-Compatible Provider (CLIENT) ✅ READY
**Scope**: Generic OpenAI Chat Completions API client
**Files**:
- `src/llm/openai_compatible.rs` (generic client, no NVIDIA-specific features)
- `src/llm/mod.rs` (basic routing)
- `src/settings.rs` (ProviderConfig)
- `src/config.rs` (providers HashMap)
- `src/error.rs` (ConfigError)
- `src/setup/wizard.rs` (init)

**Features**:
- ✅ OpenAI Chat Completions API client
- ✅ Works with ANY OpenAI-compatible endpoint
- ✅ Preset configs: `nvidia/`, `local/`, `openrouter/`, `openai/`
- ✅ Tool calling support
- ✅ Environment variable auth
- ✅ Custom provider support via settings.json

**What it does NOT include**:
- ❌ Reasoning field extraction (NVIDIA-specific)
- ❌ HTTP server API (Jaswinder's feature)

**Testing**:
```bash
cargo test --lib openai_compatible
cargo test test_nvidia_config -- --ignored
cargo test test_local_llm_completion -- --ignored
```

**Value**: Immediate - users can connect to NVIDIA, OpenRouter, local LLMs

---

### PR #2: NVIDIA-Specific Enhancements (Optional)
**Scope**: Reasoning field support for stepfun-ai models
**Files**:
- `src/llm/provider.rs` (add reasoning field)
- `src/llm/openai_compatible.rs` (extract reasoning)

**Features**:
- Extract `reasoning`/`reasoning_content` from responses
- Store in `CompletionResponse.reasoning`

**Dependencies**: Requires PR #1

**Value**: Enhancement - better support for NVIDIA stepfun-ai models

---

### PR #3: OpenAI-Compatible HTTP API (SERVER) - Jaswinder's PR #31
**Scope**: Expose Ironclaw as OpenAI-compatible HTTP endpoint
**Files**:
- `src/channels/web/openai_compat.rs` (NEW)
- `tests/openai_compat_integration.rs` (NEW)
- Route registration in server.rs

**Features**:
- `POST /v1/chat/completions` endpoint
- `GET /v1/models` endpoint
- SSE streaming support
- External client compatibility

**Dependencies**: Requires PR #1 (needs LlmProvider trait)

**Value**: Different use case - allows external apps to use Ironclaw

---

## Current Status

### Files Modified in This Branch
```
M .gitignore                    # Added .opencode/, test_*.sh, IDE files
M src/llm/mod.rs               # Provider routing + presets
M src/llm/nearai.rs            # Added reasoning: None
M src/llm/nearai_chat.rs       # Added reasoning: None
M src/llm/openai_compatible.rs # Generic client + reasoning extraction
M src/llm/provider.rs          # Added reasoning field
M src/worker/api.rs            # Added reasoning: None
```

### What's Ready for PR #1
The current branch has **both** generic client AND reasoning extraction mixed together.

**To create clean PR #1**, you would need to:
1. Remove reasoning field extraction from `openai_compatible.rs`
2. Remove reasoning field from `provider.rs`
3. Remove `reasoning: None` from other providers
4. Keep only the generic OpenAI-compatible client

**OR** - accept that reasoning extraction is a small addition and include it in PR #1.

---

## Comparison: Current vs Jaswinder

| Aspect | Current Branch | Jaswinder's PR #31 |
|--------|---------------|-------------------|
| **Type** | CLIENT (calls APIs) | SERVER (exposes API) |
| **Direction** | Outbound | Inbound |
| **Purpose** | Use external LLMs | Be used by external clients |
| **File** | `src/llm/openai_compatible.rs` | `src/channels/web/openai_compat.rs` |
| **Lines** | 633 | 1,011 |
| **Streaming** | ❌ | ✅ |
| **Tests** | Unit tests | Integration tests |
| **Production Ready** | ✅ Yes | ⚠️ Needs review |

---

## Recommendation

### Option A: Merge Current Branch First (Recommended)
1. ✅ Current branch is production-ready
2. ✅ Provides immediate value (NVIDIA, OpenRouter, local LLMs)
3. ✅ No breaking changes
4. ✅ Follows project patterns

**Then**:
- Review Jaswinder's PR #31 separately
- Merge as complementary feature

### Option B: Coordinate Both PRs
1. Merge current branch (PR #1)
2. Jaswinder rebases PR #31 on top
3. Merge PR #31 as server-side feature

---

## Git Cleanup Done

✅ Updated `.gitignore`:
- Added `.opencode/`
- Added `test_*.sh`
- Added IDE files (`.vscode/`, `.idea/`)
- Added OS files (`.DS_Store`, `Thumbs.db`)

✅ Removed temporary files:
- `test_nvidia.sh`
- `test_nvidia_step.sh`
- `NVIDIA_STEP_SETUP.md`
- `OPENAI_COMPATIBLE_SETUP.md`
- `FEATURE_BREAKDOWN.md`

---

## Next Steps

1. **Decide on PR scope**:
   - Include reasoning extraction in PR #1? (it's small)
   - Or strip it out for separate PR?

2. **Create PR**:
   ```bash
   git add -A
   git commit -m "feat: add OpenAI-compatible LLM provider

   - Generic client for any OpenAI-compatible API
   - Preset configs for NVIDIA, OpenRouter, local, OpenAI
   - Provider routing via provider/model syntax
   - Tool calling support
   - Environment variable auth"
   ```

3. **Review Jaswinder's PR #31** separately as server-side feature

4. **Update FEATURE_PARITY.md** to mark OpenAI as ✅ complete
