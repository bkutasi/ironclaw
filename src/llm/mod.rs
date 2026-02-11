//! LLM integration for the agent.
//!
//! Supports multiple providers:
//! - **NEAR AI** (Responses API or Chat Completions API)
//! - **OpenRouter** and other OpenAI-compatible APIs
//! - **Local** providers like llama.cpp
//!
//! Provider routing is done via prefix in `selected_model`:
//! - `openrouter/pony-alpha` → OpenRouter API
//! - `local/llama-3.2-3b` → localhost:3000 (llama.cpp)
//! - `fireworks::accounts/...` → NEAR AI proxy (existing default)
//! - No prefix → NEAR AI (existing default)

mod nearai;
mod nearai_chat;
mod openai_compatible;
mod provider;
mod reasoning;
pub mod session;

pub use nearai::{ModelInfo, NearAiProvider};
pub use nearai_chat::NearAiChatProvider;
pub use openai_compatible::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
pub use provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, ModelMetadata,
    Role, ToolCall, ToolCompletionRequest, ToolCompletionResponse, ToolDefinition, ToolResult,
};
pub use reasoning::{ActionPlan, Reasoning, ReasoningContext, RespondResult, ToolSelection};
pub use session::{SessionConfig, SessionManager, create_session_manager};

use std::collections::HashMap;
use std::sync::Arc;

use secrecy::SecretString;

use crate::config::{LlmConfig, NearAiApiMode};
use crate::error::LlmError;
use crate::settings::ProviderConfig;

/// Parsed model specification with optional provider prefix.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Provider name (e.g., "openrouter", "local", or empty for default).
    pub provider: Option<String>,
    /// Model identifier (e.g., "pony-alpha", "llama-3.2-3b").
    pub model: String,
}

impl ModelSpec {
    /// Parse a model string with optional provider prefix.
    ///
    /// Formats:
    /// - `provider/model` → provider and model
    /// - `provider::path` → NEAR AI style (provider = None, model as-is)
    /// - `model` → no provider (uses default)
    pub fn parse(s: &str) -> Self {
        if let Some(slash_pos) = s.find('/') {
            let provider = s[..slash_pos].to_string();
            let model = s[slash_pos + 1..].to_string();
            Self {
                provider: Some(provider),
                model,
            }
        } else {
            Self {
                provider: None,
                model: s.to_string(),
            }
        }
    }
}

/// Create an LLM provider based on configuration.
///
/// - For `Responses` mode: Requires a session manager for authentication
/// - For `ChatCompletions` mode: Uses API key from config (session not needed)
/// - For custom providers: Uses provider config from settings
pub fn create_llm_provider(
    config: &LlmConfig,
    session: Arc<SessionManager>,
) -> Result<Arc<dyn LlmProvider>, LlmError> {
    create_llm_provider_with_providers(config, session, HashMap::new())
}

/// Create an LLM provider with custom provider configurations.
pub fn create_llm_provider_with_providers(
    config: &LlmConfig,
    session: Arc<SessionManager>,
    providers: HashMap<String, ProviderConfig>,
) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let model_spec = ModelSpec::parse(&config.nearai.model);

    if let Some(provider_name) = &model_spec.provider {
        if let Some(provider_config) = providers.get(provider_name) {
            return create_custom_provider(provider_name, provider_config, &model_spec.model);
        }
        return Err(LlmError::ConfigError {
            reason: format!("Unknown provider: '{}'. Add it to ~/.ironclaw/settings.json under 'providers'.", provider_name),
        });
    }

    match config.nearai.api_mode {
        NearAiApiMode::Responses => {
            tracing::info!("Using Responses API (chat-api) with session auth");
            Ok(Arc::new(NearAiProvider::new(
                config.nearai.clone(),
                session,
            )))
        }
        NearAiApiMode::ChatCompletions => {
            tracing::info!("Using Chat Completions API (cloud-api) with API key auth");
            Ok(Arc::new(NearAiChatProvider::new(config.nearai.clone())?))
        }
    }
}

fn create_custom_provider(
    provider_name: &str,
    config: &ProviderConfig,
    model: &str,
) -> Result<Arc<dyn LlmProvider>, LlmError> {
    if config.base_url.trim().is_empty() {
        return Err(LlmError::ConfigError {
            reason: format!("Provider '{}' has empty base_url", provider_name),
        });
    }

    let api_key = config
        .api_key_env
        .as_ref()
        .and_then(|env_var| {
            match std::env::var(env_var) {
                Ok(val) => Some(val),
                Err(_) => {
                    tracing::warn!(
                        "Provider '{}' configured with api_key_env='{}' but environment variable not set",
                        provider_name, env_var
                    );
                    None
                }
            }
        })
        .map(SecretString::from);

    tracing::info!(
        "Using custom provider '{}' at {} with model '{}'",
        provider_name,
        config.base_url,
        model
    );

    let openai_config = OpenAiCompatibleConfig {
        provider_name: provider_name.to_string(),
        base_url: config.base_url.clone(),
        api_key,
        model: model.to_string(),
    };

    Ok(Arc::new(OpenAiCompatibleProvider::new(openai_config)?))
}
