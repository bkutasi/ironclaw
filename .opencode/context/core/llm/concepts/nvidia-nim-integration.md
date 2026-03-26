# NVIDIA NIM API Integration

**Core Idea**: NVIDIA NIM provides OpenAI-compatible LLM inference at `integrate.api.nvidia.com` with Bearer token auth. The Step-3.5-Flash model (196B params, ~11B active) offers 256K context and 100-300 tok/s throughput for coding and agentic workflows.

**Key Points**:
- **Base URL**: `https://integrate.api.nvidia.com/v1` — do NOT use `api.nvcf.nvidia.com`
- **Auth**: `Authorization: Bearer nvapi-xxx` header; keys obtained from build.nvidia.com
- **Endpoint**: `POST /v1/chat/completions` — standard OpenAI chat format with `messages` array
- **Model ID format**: `stepfun-ai/step-3-5-flash` (hyphens, not dots)
- **Step-3.5-Flash specs**: 256K context, sparse MoE (196B total / 11B active), Apache 2.0 license

**Quick Example**:
```bash
curl https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVAPI_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"stepfun-ai/step-3-5-flash",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":1024,"temperature":0.7}'
```

**Reference**: https://docs.api.nvidia.com/nim/docs/api-quickstart

**Related**: [rig-openai-provider.md](./rig-openai-provider.md)
