# Rig-core OpenAI Provider: Endpoint Construction

**Core Idea**: rig-core's OpenAI provider defaults to the newer Responses API (`/responses`), but NVIDIA NIM only supports the Chat Completions API (`/chat/completions`). You must explicitly call `.completions_api()` to avoid 404 errors.

**Key Points**:
- **Default endpoint is `/responses`** — `client.completion_model("model")` hits `/v1/responses`, which NVIDIA doesn't support
- **`.completions_api()` switches to `/chat/completions`** — call it on the model or client to use the traditional endpoint
- **`base_url` must include `/v1`** — e.g., `https://integrate.api.nvidia.com/v1`; rig appends the path with a `/` separator
- **Two client types exist**: `Client` (Responses, default) and `CompletionsClient` (Chat Completions)
- **URL construction**: `build_uri()` concatenates `base_url + "/" + path.trim_start_matches('/')`

**Quick Example**:
```rust
use rig::providers::openai;

// Option 1: Switch at model level
let client = openai::Client::builder()
    .api_key("nvapi-xxx")
    .base_url("https://integrate.api.nvidia.com/v1")
    .build()?;
let model = client.completion_model("stepfun-ai/step-3-5-flash").completions_api();

// Option 2: Use CompletionsClient directly
let client = openai::CompletionsClient::builder()
    .api_key("nvapi-xxx")
    .base_url("https://integrate.api.nvidia.com/v1")
    .build()?;
let model = client.completion_model("stepfun-ai/step-3-5-flash");
```

**Reference**: rig-core v0.30.0 source — `rig/providers/openai/client.rs`, `rig/client/mod.rs`

**Related**: [nvidia-nim-integration.md](./nvidia-nim-integration.md)
