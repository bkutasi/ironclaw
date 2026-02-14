//! Embedding providers for semantic search.
//!
//! Embeddings convert text into dense vectors that capture semantic meaning.
//! Similar concepts have similar vectors, enabling semantic search.

use async_trait::async_trait;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};

/// Error type for embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Rate limited, retry after {retry_after:?}")]
    RateLimited {
        retry_after: Option<std::time::Duration>,
    },

    #[error("Authentication failed")]
    AuthFailed,

    #[error("Text too long: {length} > {max}")]
    TextTooLong { length: usize, max: usize },
}

impl From<reqwest::Error> for EmbeddingError {
    fn from(e: reqwest::Error) -> Self {
        EmbeddingError::HttpError(e.to_string())
    }
}

/// Trait for embedding providers.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Get the embedding dimension.
    fn dimension(&self) -> usize;

    /// Get the model name.
    fn model_name(&self) -> &str;

    /// Maximum input length in characters.
    fn max_input_length(&self) -> usize;

    /// Generate an embedding for a single text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embeddings for multiple texts (batched).
    ///
    /// Default implementation calls embed() for each text.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }
}

/// OpenAI embedding provider using text-embedding-ada-002 or text-embedding-3-small.
pub struct OpenAiEmbeddings {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimension: usize,
}

impl OpenAiEmbeddings {
    /// Create a new OpenAI embedding provider with the default model.
    ///
    /// Uses text-embedding-3-small which has 1536 dimensions.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
        }
    }

    /// Use text-embedding-ada-002 model.
    pub fn ada_002(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: "text-embedding-ada-002".to_string(),
            dimension: 1536,
        }
    }

    /// Use text-embedding-3-large model.
    pub fn large(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: "text-embedding-3-large".to_string(),
            dimension: 3072,
        }
    }

    /// Use a custom model with specified dimension.
    pub fn with_model(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            dimension,
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddings {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn max_input_length(&self) -> usize {
        // text-embedding-3-small/large: 8191 tokens (~32k chars)
        // text-embedding-ada-002: 8191 tokens
        32_000
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.len() > self.max_input_length() {
            return Err(EmbeddingError::TextTooLong {
                length: text.len(),
                max: self.max_input_length(),
            });
        }

        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("No embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = OpenAiEmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        let status = response.status();

        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(EmbeddingError::AuthFailed);
        }

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(std::time::Duration::from_secs);
            return Err(EmbeddingError::RateLimited { retry_after });
        }

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::HttpError(format!(
                "Status {}: {}",
                status, error_text
            )));
        }

        let result: OpenAiEmbeddingResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }
}

/// NEAR AI embedding provider using the NEAR AI API.
///
/// Uses session-based auth or custom API key authentication.
pub struct NearAiEmbeddings {
    client: reqwest::Client,
    base_url: String,
    /// Optional session manager for session-based auth.
    session: Option<std::sync::Arc<crate::llm::SessionManager>>,
    /// Optional API key for custom authentication (e.g., NVIDIA, Azure).
    api_key: Option<String>,
    model: String,
    dimension: usize,
}

impl NearAiEmbeddings {
    /// Create a new NEAR AI embedding provider with session-based auth.
    ///
    /// Uses the same session manager as the LLM provider for auth.
    pub fn new(
        base_url: impl Into<String>,
        session: std::sync::Arc<crate::llm::SessionManager>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            session: Some(session),
            api_key: None,
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
        }
    }

    /// Create a new NEAR AI embedding provider with custom API key auth.
    ///
    /// This is used for providers like NVIDIA, Azure OpenAI, etc. that use
    /// Bearer token authentication instead of session-based auth.
    pub fn with_api_key(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            session: None,
            api_key: Some(api_key.into()),
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
        }
    }

    /// Use a specific model.
    pub fn with_model(mut self, model: impl Into<String>, dimension: usize) -> Self {
        self.model = model.into();
        self.dimension = dimension;
        self
    }
}

#[derive(Debug, Serialize)]
struct NearAiEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Debug, Deserialize)]
struct NearAiEmbeddingResponse {
    data: Vec<NearAiEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct NearAiEmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for NearAiEmbeddings {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn max_input_length(&self) -> usize {
        32_000
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.len() > self.max_input_length() {
            return Err(EmbeddingError::TextTooLong {
                length: text.len(),
                max: self.max_input_length(),
            });
        }

        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("No embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        use secrecy::ExposeSecret;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = NearAiEmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        // Get authorization header - prefer API key if provided, otherwise use session
        let auth_value = if let Some(ref api_key) = self.api_key {
            format!("Bearer {}", api_key)
        } else if let Some(ref session) = self.session {
            let token = session
                .get_token()
                .await
                .map_err(|_| EmbeddingError::AuthFailed)?;
            format!("Bearer {}", token.expose_secret())
        } else {
            return Err(EmbeddingError::AuthFailed);
        };

        let url = format!("{}/v1/embeddings", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", auth_value)
            .json(&request)
            .send()
            .await?;

        let status = response.status();

        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(EmbeddingError::AuthFailed);
        }

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(std::time::Duration::from_secs);
            return Err(EmbeddingError::RateLimited { retry_after });
        }

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::HttpError(format!(
                "Status {}: {}",
                status, error_text
            )));
        }

        let result: NearAiEmbeddingResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }
}

/// OpenAI-compatible embedding provider (e.g., Ollama, llama.cpp, vLLM).
///
/// Supports any endpoint that implements the OpenAI /v1/embeddings API.
/// Defaults to Ollama at http://localhost:11434 if no base_url is provided.
pub struct OpenAiCompatibleEmbeddings {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    dimension: usize,
}

impl OpenAiCompatibleEmbeddings {
    /// Create a new OpenAI-compatible embedding provider from config.
    pub fn new(config: crate::config::OpenAiCompatibleEmbeddingsConfig) -> Self {
        let dimension = config.dimensions.unwrap_or(768);
        Self {
            client: reqwest::Client::new(),
            base_url: config.base_url,
            api_key: config.api_key.map(|k| k.expose_secret().to_string()),
            model: config.model,
            dimension,
        }
    }

    /// Create a new OpenAI-compatible embedding provider with explicit values.
    ///
    /// Defaults to Ollama at http://localhost:11434 if base_url is empty.
    pub fn with_params(
        base_url: impl Into<String>,
        api_key: Option<impl Into<String>>,
        model: impl Into<String>,
        dimension: usize,
    ) -> Self {
        let base_url = base_url.into();
        // Default to Ollama if no base_url provided
        let base_url = if base_url.is_empty() {
            "http://localhost:11434".to_string()
        } else {
            base_url
        };
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key: api_key.map(|k| k.into()),
            model: model.into(),
            dimension,
        }
    }

    /// Create with default Ollama settings (nomic-embed-text-v1, 768 dimensions).
    pub fn ollama_defaults() -> Self {
        Self::with_params(
            "http://localhost:11434",
            None::<String>,
            "nomic-embed-text-v1",
            768,
        )
    }
}

/// Request/Response structs for OpenAI-compatible embeddings API.
/// Reuses the same format as OpenAiEmbeddings.
#[derive(Debug, Serialize)]
struct OpenAiCompatibleEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Debug, Deserialize)]
struct OpenAiCompatibleEmbeddingResponse {
    data: Vec<OpenAiCompatibleEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct OpenAiCompatibleEmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for OpenAiCompatibleEmbeddings {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn max_input_length(&self) -> usize {
        32_000
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.len() > self.max_input_length() {
            return Err(EmbeddingError::TextTooLong {
                length: text.len(),
                max: self.max_input_length(),
            });
        }

        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("No embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = OpenAiCompatibleEmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));

        let mut request_builder = self.client.post(&url).json(&request);

        // Add Authorization header if API key is present
        if let Some(ref api_key) = self.api_key {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request_builder.send().await?;

        let status = response.status();

        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(EmbeddingError::AuthFailed);
        }

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(std::time::Duration::from_secs);
            return Err(EmbeddingError::RateLimited { retry_after });
        }

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::HttpError(format!(
                "Status {}: {}",
                status, error_text
            )));
        }

        let result: OpenAiCompatibleEmbeddingResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }
}

/// A mock embedding provider for testing.
///
/// Generates deterministic embeddings based on text hash.
/// Useful for unit and integration tests.
pub struct MockEmbeddings {
    dimension: usize,
}

impl MockEmbeddings {
    /// Create a new mock embeddings provider with the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddings {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "mock-embedding"
    }

    fn max_input_length(&self) -> usize {
        10_000
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Generate a deterministic embedding based on text hash
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut embedding = Vec::with_capacity(self.dimension);
        let mut seed = hash;
        for _ in 0..self.dimension {
            // Simple LCG for deterministic random values
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
            embedding.push(value);
        }

        // Normalize to unit length
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for x in &mut embedding {
                *x /= magnitude;
            }
        }

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embeddings() {
        let provider = MockEmbeddings::new(128);

        let embedding = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding.len(), 128);

        // Check normalization (should be unit vector)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mock_embeddings_deterministic() {
        let provider = MockEmbeddings::new(64);

        let emb1 = provider.embed("test").await.unwrap();
        let emb2 = provider.embed("test").await.unwrap();

        // Same input should produce same embedding
        assert_eq!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_mock_embeddings_batch() {
        let provider = MockEmbeddings::new(64);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 64);
        assert_eq!(embeddings[1].len(), 64);

        // Different texts should produce different embeddings
        assert_ne!(embeddings[0], embeddings[1]);
    }

    #[test]
    fn test_openai_embeddings_config() {
        let provider = OpenAiEmbeddings::new("test-key");
        assert_eq!(provider.dimension(), 1536);
        assert_eq!(provider.model_name(), "text-embedding-3-small");

        let provider = OpenAiEmbeddings::large("test-key");
        assert_eq!(provider.dimension(), 3072);
        assert_eq!(provider.model_name(), "text-embedding-3-large");
    }
}
