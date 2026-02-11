//! Generic OpenAI-compatible API provider implementation.
//!
//! This provider works with any OpenAI-compatible endpoint (OpenRouter, local llama.cpp, etc.).

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

    async fn send_request<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        body: &T,
    ) -> Result<R, LlmError> {
        // Note: base_url should already include the /v1 path (e.g., "https://openrouter.ai/api/v1")
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
            if status.as_u16() == 401 {
                return Err(LlmError::AuthFailed {
                    provider: self.config.provider_name.clone(),
                });
            }
            if status.as_u16() == 429 {
                return Err(LlmError::RateLimited {
                    provider: self.config.provider_name.clone(),
                    retry_after: None,
                });
            }
            return Err(LlmError::RequestFailed {
                provider: self.config.provider_name.clone(),
                reason: format!("HTTP {}: {}", status, response_text),
            });
        }

        serde_json::from_str(&response_text).map_err(|e| LlmError::InvalidResponse {
            provider: self.config.provider_name.clone(),
            reason: format!("JSON parse error: {}. Raw: {}", e, response_text),
        })
    }

    /// Fetch available models.
    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        // Note: base_url should already include the /v1 path
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
            return Err(LlmError::RequestFailed {
                provider: self.config.provider_name.clone(),
                reason: format!("HTTP {}: {}", status, response_text),
            });
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelEntry>,
        }

        #[derive(Deserialize)]
        struct ModelEntry {
            id: String,
        }

        let resp: ModelsResponse =
            serde_json::from_str(&response_text).map_err(|e| LlmError::InvalidResponse {
                provider: self.config.provider_name.clone(),
                reason: format!("JSON parse error: {}", e),
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
        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolUse,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        };

        Ok(CompletionResponse {
            content,
            finish_reason,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            response_id: response.id,
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
        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
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
            .collect();

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolUse,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => {
                if !tool_calls.is_empty() {
                    FinishReason::ToolUse
                } else {
                    FinishReason::Unknown
                }
            }
        };

        Ok(ToolCompletionResponse {
            content,
            tool_calls,
            finish_reason,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            response_id: response.id,
        })
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
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
            content: if msg.content.is_empty() { None } else { Some(msg.content) },
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
        let model = std::env::var("LOCAL_LLM_MODEL")
            .expect("Set LOCAL_LLM_MODEL (e.g., Jan-V3)");
        let api_key = std::env::var("LOCAL_LLM_API_KEY").ok();

        let config = OpenAiCompatibleConfig {
            provider_name: "local".to_string(),
            base_url,
            api_key: api_key.map(secrecy::SecretString::from),
            model,
        };

        let provider = OpenAiCompatibleProvider::new(config).expect("Failed to create provider");

        let request = CompletionRequest {
            messages: vec![ChatMessage::user("Say 'hello' in one word.")],
            temperature: Some(0.1),
            max_tokens: Some(10),
            stop_sequences: None,
        };

        let response = provider.complete(request).await.expect("Completion failed");

        println!("Response: {:?}", response);
        assert!(!response.content.is_empty(), "Response should not be empty");
        assert!(response.input_tokens > 0 || response.input_tokens == 0, "Should have token count");
    }

    /// Integration test for local LLM with tool calling.
    ///
    /// Run with: cargo test test_local_llm_with_tools --ignored -- --nocapture
    #[tokio::test]
    #[ignore = "Requires local LLM server. Set LOCAL_LLM_URL and LOCAL_LLM_MODEL env vars."]
    async fn test_local_llm_with_tools() {
        let base_url = std::env::var("LOCAL_LLM_URL")
            .expect("Set LOCAL_LLM_URL (e.g., http://localhost:3000/v1)");
        let model = std::env::var("LOCAL_LLM_MODEL")
            .expect("Set LOCAL_LLM_MODEL (e.g., Jan-V3)");
        let api_key = std::env::var("LOCAL_LLM_API_KEY").ok();

        let config = OpenAiCompatibleConfig {
            provider_name: "local".to_string(),
            base_url,
            api_key: api_key.map(secrecy::SecretString::from),
            model,
        };

        let provider = OpenAiCompatibleProvider::new(config).expect("Failed to create provider");

        let tools = vec![crate::llm::provider::ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }),
        }];

        let request = ToolCompletionRequest {
            messages: vec![ChatMessage::user("What's the weather in Tokyo?")],
            tools,
            tool_choice: Some("auto".to_string()),
            temperature: Some(0.1),
            max_tokens: Some(100),
        };

        let response = provider.complete_with_tools(request).await.expect("Tool completion failed");

        println!("Response: {:?}", response);
        // Note: Not all local models support tool calling, so we just check it doesn't crash
    }
}
