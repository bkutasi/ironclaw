# LLM Model Mismatch

This document covers issues when the configured LLM model is invalid, unavailable, or misconfigured.

## Overview

IronClaw supports multiple LLM providers via the `LLM_BACKEND` and `LLM_MODEL` environment variables. Using an invalid model name causes all LLM requests to fail, resulting in bot non-responsiveness.

## Common Invalid Models

### ❌ stepfun-ai/step-3.5-flash

**Status**: Returns 404 Not Found

**Why**: Model was deprecated or never available on NVIDIA NIM

**Detection**:
```bash
curl -X POST https://api.nvcf.nvidia.com/v2/nvcf/pfx/functions/invoke \
  -H "Authorization: Bearer $NGC_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"stepfun-ai/step-3.5-flash","messages":[{"role":"user","content":"test"}]}'
# Returns: 404 Not Found
```

**Fix**:
```bash
# Change to valid model
LLM_MODEL="z-ai/glm5"
```

## Valid Models by Provider

### NVIDIA NIM (Recommended)

```bash
LLM_BACKEND=nearai
LLM_MODEL="z-ai/glm5"                    # ✅ GLM-5 FP8 (best)
LLM_MODEL="meta/llama3-70b-instruct"     # ✅ Llama 3 70B
LLM_MODEL="mistralai/mixtral-8x7b-instruct"  # ✅ Mixtral 8x7B
```

### Ollama (Local)

```bash
LLM_BACKEND=ollama
LLM_MODEL="llama3.2"                     # ✅ Llama 3.2
LLM_MODEL="mistral"                      # ✅ Mistral 7B
LLM_MODEL="gemma2"                       # ✅ Gemma 2
```

### OpenAI Compatible

```bash
LLM_BACKEND=openai_compatible
LLM_BASE_URL="https://openrouter.ai/api/v1"
LLM_MODEL="anthropic/claude-sonnet-4"    # ✅ Claude Sonnet
LLM_MODEL="meta-llama/llama-3-70b-instruct"  # ✅ Llama 3
```

### Together AI

```bash
LLM_BACKEND=openai_compatible
LLM_BASE_URL="https://api.together.xyz/v1"
LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"  # ✅ Llama 3.3
```

### Fireworks AI

```bash
LLM_BACKEND=openai_compatible
LLM_BASE_URL="https://api.fireworks.ai/inference/v1"
LLM_MODEL="accounts/fireworks/models/llama4-maverick-instruct-basic"  # ✅ Llama 4
```

## Detection

### Method 1: Test API Directly

```bash
# NVIDIA NIM test
curl -X POST https://api.nvcf.nvidia.com/v2/nvcf/pfx/functions/invoke \
  -H "Authorization: Bearer $NGC_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"'$LLM_MODEL'",
    "messages":[{"role":"user","content":"test"}]
  }'

# Check response:
# 200 = model valid
# 404 = model not found
# 401 = invalid API key
```

### Method 2: Check IronClaw Logs

```bash
RUST_LOG=debug cargo run 2>&1 | grep -i "model\|404\|invalid"

# Look for:
# "Model not found: stepfun-ai/step-3.5-flash"
# "LLM API returned 404"
# "Invalid model configuration"
```

### Method 3: Use Diagnostic Tool

```bash
telegram_check_config
# Output will show:
# ❌ LLM Model: stepfun-ai/step-3.5-flash (INVALID - returns 404)
#    → Fix: Change to "z-ai/glm5" in ~/.ironclaw/.env
```

## Symptoms

When LLM model is invalid:

1. **Bot receives messages** ✅
2. **Bot doesn't respond** ❌
3. **No errors in webhook logs** ⚠️
4. **LLM API returns 404** ❌
5. **Pending updates increase** ⚠️

## Fix Steps

### Step 1: Identify Current Model

```bash
grep LLM_MODEL ~/.ironclaw/.env
# Output: LLM_MODEL="stepfun-ai/step-3.5-flash"
```

### Step 2: Choose Valid Model

For NVIDIA NIM (recommended):
```bash
LLM_MODEL="z-ai/glm5"
```

### Step 3: Update Configuration

```bash
# Edit ~/.ironclaw/.env
nano ~/.ironclaw/.env

# Change line:
LLM_MODEL="z-ai/glm5"

# Save and exit
```

### Step 4: Restart IronClaw

```bash
pkill ironclaw
cargo run
```

### Step 5: Verify Fix

```bash
# Send test message to bot
"Hello, are you working?"

# Should receive response within 30 seconds
```

## Auto-Fix

Use the telegram-debug skill:

```bash
telegram_fix_common_issues
# Output:
# Issue: LLM model invalid (stepfun-ai/step-3.5-flash)
#   ✅ Fixed: Changed to z-ai/glm5 in ~/.ironclaw/.env
```

## Provider-Specific Configuration

### NVIDIA NIM Setup

```bash
# ~/.ironclaw/.env
LLM_BACKEND=nearai
LLM_MODEL="z-ai/glm5"
NGC_KEY="nvapi-your-api-key-here"
NEARAI_BASE_URL="https://private.near.ai"
```

Get NGC key: https://org.ngc.nvidia.com/setup/personal-keys

### Ollama Setup

```bash
# ~/.ironclaw/.env
LLM_BACKEND=ollama
LLM_MODEL="llama3.2"
OLLAMA_BASE_URL="http://localhost:11434"

# Pull model first
ollama pull llama3.2
```

### OpenRouter Setup

```bash
# ~/.ironclaw/.env
LLM_BACKEND=openai_compatible
LLM_BASE_URL="https://openrouter.ai/api/v1"
LLM_MODEL="anthropic/claude-sonnet-4"
LLM_API_KEY="sk-or-your-api-key"
```

## Model Comparison

| Model | Provider | Speed | Quality | Cost |
|-------|----------|-------|---------|------|
| z-ai/glm5 | NVIDIA NIM | Fast | High | Free tier |
| meta/llama3-70b-instruct | NVIDIA NIM | Medium | High | Free tier |
| llama3.2 | Ollama | Fast | Medium | Free (local) |
| claude-sonnet-4 | OpenRouter | Fast | Very High | Paid |

## Troubleshooting

### Still Getting 404 After Fix

1. **Check file saved**: `cat ~/.ironclaw/.env | grep LLM_MODEL`
2. **Check IronClaw restarted**: `ps aux | grep ironclaw`
3. **Check NGC_KEY valid**: Test API directly
4. **Check backend**: `LLM_BACKEND=nearai` for NVIDIA NIM

### Model Returns 401 Unauthorized

```bash
# Check NGC_KEY format
echo $NGC_KEY
# Should start with "nvapi-"

# Regenerate key if needed
# https://org.ngc.nvidia.com/setup/personal-keys
```

### Model Too Slow

Try smaller model:
```bash
# Instead of 70B model
LLM_MODEL="meta/llama3-8b-instruct"  # Faster, less VRAM
```

## Related

- `integrations/errors/bot-no-response.md` - Bot response troubleshooting
- `docs/LLM_PROVIDERS.md` - Full LLM provider setup guide
- `skills/telegram-debug/SKILL.md` - Automated diagnostics
