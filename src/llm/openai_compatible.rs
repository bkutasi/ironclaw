//! Generic OpenAI-compatible API provider implementation.
//!
//! Works with any OpenAI-compatible endpoint including:
//! - OpenRouter (https://openrouter.ai)
//! - NVIDIA API (https://integrate.api.nvidia.com)
//! - Local LLMs (llama.cpp, Jan, LM Studio, etc.)
//! - OpenAI API
//! - Any other OpenAI-compatible service
//!
//! # Quick Setup
//!
//! ## NVIDIA API
//! ```bash
//! export NVIDIA_API_KEY="your-ngc-key"
//! ```
//! Model: `nvidia/moonshotai/kimi-k2.5`
//!
//! ## Local LLM (localhost:3000)
//! ```bash
//! # No API key needed for most local providers
//! ```
//! Model: `local/your-model-name`

use async_trait::async_trait;
use reqwest::Client;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::llm::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, Role, ToolCall,
    ToolCompletionRequest, ToolCompletionResponse,
};

/// Configuration for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleConfig {
    /// Provider name for error messages.
    pub provider_name: String,
    /// Base URL for the API (e.g., "https://openrouter.ai/api/v1").
    pub base_url: String,
    /// API key (optional for local providers like llama.cpp).
    pub api_key: Option<secrecy::SecretString>,
    /// Model identifier to use.
    pub model: String,
}

impl OpenAiCompatibleConfig {
    /// Create configuration for NVIDIA API.
    ///
    /// # Arguments
    /// * `api_key` - Your NVIDIA NGC API key
    /// * `model` - Model name (e.g., "moonshotai/kimi-k2.5")
    pub fn nvidia(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_name: "nvidia".to_string(),
            base_url: "https://integrate.api.nvidia.com/v1".to_string(),
            api_key: Some(secrecy::SecretString::from(api_key.into())),
            model: model.into(),
        }
    }

    /// Create configuration for local LLM (e.g., llama.cpp, Jan, LM Studio).
    ///
    /// # Arguments
    /// * `base_url` - Base URL (e.g., "http://localhost:3000/v1")
    /// * `model` - Model name (optional for many local providers)
    pub fn local(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_name: "local".to_string(),
            base_url: base_url.into(),
            api_key: None,
            model: model.into(),
        }
    }

    /// Create configuration for OpenRouter.
    ///
    /// # Arguments
    /// * `api_key` - Your OpenRouter API key
    /// * `model` - Model name (e.g., "anthropic/claude-3.5-sonnet")
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_name: "openrouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: Some(secrecy::SecretString::from(api_key.into())),
            model: model.into(),
        }
    }

    /// Create configuration for OpenAI.
    ///
    /// # Arguments
    /// * `api_key` - Your OpenAI API key
    /// * `model` - Model name (e.g., "gpt-4o")
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_name: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: Some(secrecy::SecretString::from(api_key.into())),
            model: model.into(),
        }
    }
}

/// Generic OpenAI-compatible API provider.
pub struct OpenAiCompatibleProvider {
    client: Client,
    config: OpenAiCompatibleConfig,
}

impl OpenAiCompatibleProvider {
    /// Create a new OpenAI-compatible provider.
    pub fn new(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| Client::new());

        Ok(Self { client, config })
    }

    fn api_url(&self, path: &str) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{}/{}", base, path.trim_start_matches('/'))
    }

    fn api_key(&self) -> Option<String> {
        self.config
            .api_key
            .as_ref()
            .map(|k| k.expose_secret().to_string())
    }

    /// Send a request to the chat completions API.
    async fn send_request<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        body: &T,
    ) -> Result<R, LlmError> {
        let url = self.api_url("chat/completions");
        tracing::debug!("Sending request to {}: {}", self.config.provider_name, url);

        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

        if let Some(api_key) = self.api_key() {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.json(body).send().await.map_err(|e| {
            tracing::error!("{} request failed: {}", self.config.provider_name, e);
            LlmError::RequestFailed {
                provider: self.config.provider_name.clone(),
                reason: e.to_string(),
            }
        })?;

        let status = response.status();
        let response_text = response.text().await.unwrap_or_default();

        tracing::debug!("{} response status: {}", self.config.provider_name, status);
        tracing::trace!("{} response body: {}", self.config.provider_name, response_text);

        if !status.is_success() {
            return Err(self.parse_error(status, &response_text));
        }

        serde_json::from_str(&response_text).map_err(|e| LlmError::InvalidResponse {
            provider: self.config.provider_name.clone(),
            reason: format!("JSON parse error: {}. Raw: {}", e, response_text),
        })
    }

    /// Parse HTTP error response into appropriate LlmError.
    fn parse_error(&self, status: reqwest::StatusCode, response_text: &str) -> LlmError {
        match status.as_u16() {
            401 => LlmError::AuthFailed {
                provider: self.config.provider_name.clone(),
            },
            429 => LlmError::RateLimited {
                provider: self.config.provider_name.clone(),
                retry_after: None,
            },
            _ => LlmError::RequestFailed {
                provider: self.config.provider_name.clone(),
                reason: format!("HTTP {}: {}", status, response_text),
            },
        }
    }

    /// Parse finish reason from string.
    fn parse_finish_reason(&self, reason: Option<&str>, has_tool_calls: bool) -> FinishReason {
        match reason {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolUse,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => {
                if has_tool_calls {
                    FinishReason::ToolUse
                } else {
                    FinishReason::Unknown
                }
            }
        }
    }

    /// Parse tool calls from response.
    fn parse_tool_calls(&self, tool_calls: Option<Vec<ChatCompletionToolCall>>) -> Vec<ToolCall> {
        tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| {
                let arguments = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments,
                }
            })
            .collect()
    }

    /// Fetch available models.
    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let url = self.api_url("models");
        let mut request = self.client.get(&url);

        if let Some(api_key) = self.api_key() {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await.map_err(|e| LlmError::RequestFailed {
            provider: self.config.provider_name.clone(),
            reason: format!("Failed to fetch models: {}", e),
        })?;

        let status = response.status();
        let response_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            return Err(self.parse_error(status, &response_text));
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelEntry>,
        }

        #[derive(Deserialize)]
        struct ModelEntry {
            id: String,
        }

        let resp: ModelsResponse = serde_json::from_str(&response_text).map_err(|e| {
            LlmError::InvalidResponse {
                provider: self.config.provider_name.clone(),
                reason: format!("JSON parse error: {}", e),
            }
        })?;

        Ok(resp.data.into_iter().map(|m| m.id).collect())
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let messages: Vec<ChatCompletionMessage> =
            req.messages.into_iter().map(|m| m.into()).collect();

        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools: None,
            tool_choice: None,
        };

        let response: ChatCompletionResponse = self.send_request(&request).await?;

        let choice = response.choices.into_iter().next().ok_or_else(|| {
            LlmError::InvalidResponse {
                provider: self.config.provider_name.clone(),
                reason: "No choices in response".to_string(),
            }
        })?;

        let content = choice.message.content.unwrap_or_default();
        let finish_reason = self.parse_finish_reason(choice.finish_reason.as_deref(), false);

        // Extract reasoning from either `reasoning` or `reasoning_content` field
        let reasoning = choice
            .message
            .reasoning
            .or(choice.message.reasoning_content);

        Ok(CompletionResponse {
            content,
            finish_reason,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            response_id: response.id,
            reasoning,
        })
    }

    async fn complete_with_tools(
        &self,
        req: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let messages: Vec<ChatCompletionMessage> =
            req.messages.into_iter().map(|m| m.into()).collect();

        let tools: Vec<ChatCompletionTool> = req
            .tools
            .into_iter()
            .map(|t| ChatCompletionTool {
                tool_type: "function".to_string(),
                function: ChatCompletionFunction {
                    name: t.name,
                    description: Some(t.description),
                    parameters: Some(t.parameters),
                },
            })
            .collect();

        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice: req.tool_choice,
        };

        let response: ChatCompletionResponse = self.send_request(&request).await?;

        let choice = response.choices.into_iter().next().ok_or_else(|| {
            LlmError::InvalidResponse {
                provider: self.config.provider_name.clone(),
                reason: "No choices in response".to_string(),
            }
        })?;

        let content = choice.message.content;
        let tool_calls = self.parse_tool_calls(choice.message.tool_calls);
        let finish_reason =
            self.parse_finish_reason(choice.finish_reason.as_deref(), !tool_calls.is_empty());

        // Extract reasoning from either `reasoning` or `reasoning_content` field
        let reasoning = choice
            .message
            .reasoning
            .or(choice.message.reasoning_content);

        Ok(ToolCompletionResponse {
            content,
            tool_calls,
            finish_reason,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            response_id: response.id,
            reasoning,
        })
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
        // Default costs - override for specific providers
        (dec!(0.000003), dec!(0.000015))
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        OpenAiCompatibleProvider::list_models(self).await
    }
}

// OpenAI-compatible Chat Completions API types

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatCompletionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatCompletionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatCompletionToolCall>>,
}

impl From<ChatMessage> for ChatCompletionMessage {
    fn from(msg: ChatMessage) -> Self {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let tool_calls = msg.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| ChatCompletionToolCall {
                    id: tc.id,
                    call_type: "function".to_string(),
                    function: ChatCompletionToolCallFunction {
                        name: tc.name,
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect()
        });
        Self {
            role: role.to_string(),
            content: if msg.content.is_empty() {
                None
            } else {
                Some(msg.content)
            },
            tool_call_id: msg.tool_call_id,
            name: msg.name,
            tool_calls,
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ChatCompletionFunction,
}

#[derive(Debug, Serialize)]
struct ChatCompletionFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<String>,
    choices: Vec<ChatCompletionChoice>,
    #[serde(default)]
    usage: ChatCompletionUsage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponseMessage {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    /// Reasoning content from models that return thinking process separately (e.g., stepfun-ai models)
    #[serde(default)]
    reasoning: Option<String>,
    /// Alternative field name for reasoning content used by some providers
    #[serde(default, rename = "reasoning_content")]
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionToolCall {
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    call_type: String,
    function: ChatCompletionToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize, Default)]
struct ChatCompletionUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[allow(dead_code)]
    #[serde(default)]
    total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_conversion() {
        let msg = ChatMessage::user("Hello");
        let chat_msg: ChatCompletionMessage = msg.into();
        assert_eq!(chat_msg.role, "user");
        assert_eq!(chat_msg.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_tool_message_conversion() {
        let msg = ChatMessage::tool_result("call_123", "my_tool", "result");
        let chat_msg: ChatCompletionMessage = msg.into();
        assert_eq!(chat_msg.role, "tool");
        assert_eq!(chat_msg.tool_call_id, Some("call_123".to_string()));
        assert_eq!(chat_msg.name, Some("my_tool".to_string()));
    }

    #[test]
    fn test_empty_content_becomes_none() {
        let msg = ChatMessage::assistant("");
        let chat_msg: ChatCompletionMessage = msg.into();
        assert_eq!(chat_msg.content, None);
    }

    #[test]
    fn test_nvidia_config() {
        let config = OpenAiCompatibleConfig::nvidia("test-key", "moonshotai/kimi-k2.5");
        assert_eq!(config.provider_name, "nvidia");
        assert_eq!(config.base_url, "https://integrate.api.nvidia.com/v1");
        assert_eq!(config.model, "moonshotai/kimi-k2.5");
        assert!(config.api_key.is_some());
    }

    #[test]
    fn test_local_config() {
        let config = OpenAiCompatibleConfig::local("http://localhost:3000/v1", "llama-3.2");
        assert_eq!(config.provider_name, "local");
        assert_eq!(config.base_url, "http://localhost:3000/v1");
        assert_eq!(config.model, "llama-3.2");
        assert!(config.api_key.is_none());
    }

    /// Integration test for local LLM providers.
    ///
    /// Configure via environment variables:
    /// - `LOCAL_LLM_URL`: Base URL (e.g., "http://localhost:3000/v1")
    /// - `LOCAL_LLM_MODEL`: Model name (e.g., "Jan-V3")
    /// - `LOCAL_LLM_API_KEY`: Optional API key
    ///
    /// Run with: cargo test test_local_llm_completion --ignored -- --nocapture
    #[tokio::test]
    #[ignore = "Requires local LLM server. Set LOCAL_LLM_URL and LOCAL_LLM_MODEL env vars."]
    async fn test_local_llm_completion() {
        let base_url = std::env::var("LOCAL_LLM_URL")
            .expect("Set LOCAL_LLM_URL (e.g., http://localhost:3000/v1)");
        let model = std::env::var("LOCAL_LLM_MODEL").expect("Set LOCAL_LLM_MODEL (e.g., Jan-V3)");
        let api_key = std::env::var("LOCAL_LLM_API_KEY").ok();

        let config = OpenAiCompatibleConfig::local(base_url, model);
        let config = OpenAiCompatibleConfig {
            api_key: api_key.map(secrecy::SecretString::from),
            ..config
        };

        let provider = OpenAiCompatibleProvider::new(config).expect("Failed to create provider");

        let request = CompletionRequest {
            messages: vec![ChatMessage::user("Say 'hello' in one word.")],
            temperature: Some(0.1),
            max_tokens: Some(10),
            stop_sequences: None,
            metadata: std::collections::HashMap::new(),
        };

        let response = provider.complete(request).await.expect("Completion failed");

        println!("Response: {:?}", response);
        assert!(!response.content.is_empty(), "Response should not be empty");
    }

    /// Integration test for NVIDIA API.
    ///
    /// Run with: cargo test test_nvidia_api --ignored -- --nocapture
    #[tokio::test]
    #[ignore = "Requires NVIDIA_API_KEY env var."]
    async fn test_nvidia_api() {
        let api_key = std::env::var("NVIDIA_API_KEY").expect("Set NVIDIA_API_KEY");

        let config = OpenAiCompatibleConfig::nvidia(api_key, "moonshotai/kimi-k2.5");
        let provider = OpenAiCompatibleProvider::new(config).expect("Failed to create provider");

        let request = CompletionRequest {
            messages: vec![ChatMessage::user("Hello, how are you?")],
            temperature: Some(0.7),
            max_tokens: Some(50),
            stop_sequences: None,
            metadata: std::collections::HashMap::new(),
        };

        let response = provider.complete(request).await.expect("Completion failed");

        println!("Response: {:?}", response);
        assert!(!response.content.is_empty(), "Response should not be empty");
    }
}
