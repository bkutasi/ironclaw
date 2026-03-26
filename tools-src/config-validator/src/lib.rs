//! IronClaw Configuration Validator WASM Tool
//!
//! Validates IronClaw configuration data and catches common issues.
//! This tool validates configuration data passed to it - it cannot read
//! files directly (WASM sandbox). Use via host tool invocation.

wit_bindgen::generate!({
    world: "sandboxed-tool",
    path: "../../wit/tool.wit",
});

use serde::{Deserialize, Serialize};

/// Valid LLM models that are known to work with NVIDIA NIM
const VALID_LLM_MODELS: &[&str] = &[
    "z-ai/glm5",
    "z-ai/glm4",
    "meta/llama3-70b-instruct",
    "meta/llama3-8b-instruct",
    "mistralai/mistral-large",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "nvidia/nemotron-4-340b-instruct",
];

/// Invalid models that commonly cause issues
const INVALID_LLM_MODELS: &[&str] = &[
    "stepfun-ai/step-3.5-flash", // Returns 404
];

/// Default database password for Docker PostgreSQL
const DOCKER_POSTGRES_PASSWORD: &str = "yourpass";

/// Common wrong password
const COMMON_WRONG_PASSWORD: &str = "ironclaw_pass";

/// Configuration check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Status: pass, fail, or warning
    pub status: CheckStatus,
    /// Check name
    pub name: String,
    /// Detailed message
    pub message: String,
    /// Suggested fix if applicable
    pub fix_suggestion: Option<String>,
}

/// Check status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Warning,
    Fail,
}

struct ConfigValidatorTool;

impl exports::near::agent::tool::Guest for ConfigValidatorTool {
    fn execute(req: exports::near::agent::tool::Request) -> exports::near::agent::tool::Response {
        match execute_inner(&req.params) {
            Ok(result) => exports::near::agent::tool::Response {
                output: Some(result),
                error: None,
            },
            Err(e) => exports::near::agent::tool::Response {
                output: None,
                error: Some(e),
            },
        }
    }

    fn schema() -> String {
        SCHEMA.to_string()
    }

    fn description() -> String {
        "Validate IronClaw configuration including environment variables, \
         LLM model settings, and database URL format. Accepts configuration \
         data as input and returns validation results. Helps catch common \
         issues like invalid LLM models or wrong database passwords."
            .to_string()
    }
}

#[derive(Debug, Deserialize)]
struct ToolParams {
    /// Environment variables to validate (JSON object)
    env_vars: serde_json::Value,
    /// Which check to run: "env", "full_report"
    check: Option<String>,
}

fn execute_inner(params: &str) -> Result<String, String> {
    let params: ToolParams =
        serde_json::from_str(params).map_err(|e| format!("Invalid parameters: {e}"))?;

    let check_type = params.check.as_deref().unwrap_or("env");

    match check_type {
        "env" => check_env(&params.env_vars),
        "full_report" => generate_full_report(&params.env_vars),
        _ => Err(format!("Unknown check type: {}", check_type)),
    }
}

fn check_env(env_vars: &serde_json::Value) -> Result<String, String> {
    let env_map = env_vars.as_object().ok_or("env_vars must be an object")?;

    let mut results: Vec<CheckResult> = Vec::new();

    // Check NGC_KEY
    if let Some(ngc_key) = env_map.get("NGC_KEY").and_then(|v| v.as_str()) {
        results.push(check_ngc_key(ngc_key));
    } else {
        results.push(CheckResult {
            status: CheckStatus::Fail,
            name: "NGC Key".to_string(),
            message: "NGC_KEY not set".to_string(),
            fix_suggestion: Some(
                "Set NGC_KEY=nvapi-your-key in environment or .env file".to_string(),
            ),
        });
    }

    // Check LLM_MODEL
    if let Some(model) = env_map.get("LLM_MODEL").and_then(|v| v.as_str()) {
        results.push(check_llm_model(model));
    } else {
        results.push(CheckResult {
            status: CheckStatus::Warning,
            name: "LLM Model".to_string(),
            message: "LLM_MODEL not set (will use default)".to_string(),
            fix_suggestion: Some("Set LLM_MODEL=z-ai/glm5".to_string()),
        });
    }

    // Check DATABASE_URL
    if let Some(db_url) = env_map.get("DATABASE_URL").and_then(|v| v.as_str()) {
        results.push(check_database_url(db_url));
    } else {
        results.push(CheckResult {
            status: CheckStatus::Warning,
            name: "Database URL".to_string(),
            message: "DATABASE_URL not set".to_string(),
            fix_suggestion: Some(
                "Set DATABASE_URL=postgres://postgres:yourpass@localhost:5433/ironclaw".to_string(),
            ),
        });
    }

    // Check TELEGRAM_BOT_TOKEN (optional)
    if let Some(token) = env_map.get("TELEGRAM_BOT_TOKEN").and_then(|v| v.as_str()) {
        results.push(check_telegram_token(token));
    }

    // Check TUNNEL_URL (optional)
    if let Some(url) = env_map.get("TUNNEL_URL").and_then(|v| v.as_str()) {
        results.push(check_tunnel_url(url));
    }

    serde_json::to_string_pretty(&results).map_err(|e| format!("Failed to serialize results: {e}"))
}

fn generate_full_report(env_vars: &serde_json::Value) -> Result<String, String> {
    let mut report =
        String::from("IronClaw Configuration Report\n==============================\n\n");

    let results = parse_check_env_result(&check_env(env_vars)?);
    for result in &results {
        report.push_str(&format!(
            "{} {}: {}\n",
            status_icon(&result.status),
            result.name,
            result.message
        ));
        if let Some(fix) = &result.fix_suggestion {
            report.push_str(&format!("   → Fix: {}\n", fix));
        }
    }

    report.push('\n');

    // Count issues
    let issues: Vec<_> = results
        .iter()
        .filter(|r| r.status != CheckStatus::Pass)
        .collect();

    if issues.is_empty() {
        report.push_str("Issues Found: 0\nConfiguration looks good!\n");
    } else {
        report.push_str(&format!("Issues Found: {}\n", issues.len()));
        for (i, issue) in issues.iter().enumerate() {
            report.push_str(&format!("{}. {} - {}\n", i + 1, issue.name, issue.message));
        }
        report.push_str("\nNote: Fix issues by updating ~/.ironclaw/.env file\n");
    }

    Ok(report)
}

fn status_icon(status: &CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "✅",
        CheckStatus::Warning => "⚠️",
        CheckStatus::Fail => "❌",
    }
}

fn parse_check_env_result(json: &str) -> Vec<CheckResult> {
    serde_json::from_str(json).unwrap_or_default()
}

fn check_ngc_key(key: &str) -> CheckResult {
    // Check if starts with "nvapi-" and has valid characters
    if key.starts_with("nvapi-")
        && key.len() > 6
        && key[6..]
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        let truncated = if key.len() > 9 {
            format!("nvapi-{}***", &key[6..9])
        } else {
            "nvapi-***".to_string()
        };
        CheckResult {
            status: CheckStatus::Pass,
            name: "NGC Key".to_string(),
            message: format!("Configured ({})", truncated),
            fix_suggestion: None,
        }
    } else {
        CheckResult {
            status: CheckStatus::Fail,
            name: "NGC Key".to_string(),
            message: "Invalid format (must start with 'nvapi-')".to_string(),
            fix_suggestion: Some(
                "Get key from https://org.ngc.nvidia.com/setup/personal-keys".to_string(),
            ),
        }
    }
}

fn check_llm_model(model: &str) -> CheckResult {
    if INVALID_LLM_MODELS.contains(&model) {
        CheckResult {
            status: CheckStatus::Warning,
            name: "LLM Model".to_string(),
            message: format!("{} (INVALID - returns 404)", model),
            fix_suggestion: Some("Change to \"z-ai/glm5\"".to_string()),
        }
    } else if VALID_LLM_MODELS.contains(&model) {
        CheckResult {
            status: CheckStatus::Pass,
            name: "LLM Model".to_string(),
            message: format!("{} (valid)", model),
            fix_suggestion: None,
        }
    } else {
        CheckResult {
            status: CheckStatus::Warning,
            name: "LLM Model".to_string(),
            message: format!("{} (unknown model)", model),
            fix_suggestion: Some(
                "Verify model at https://build.nvidia.com/explore/discover".to_string(),
            ),
        }
    }
}

fn check_database_url(url: &str) -> CheckResult {
    // Simple postgres URL parsing: postgres://user:pass@host:port/db
    if !url.starts_with("postgres://") {
        return CheckResult {
            status: CheckStatus::Fail,
            name: "Database URL".to_string(),
            message: "Invalid URL format (must start with postgres://)".to_string(),
            fix_suggestion: Some("Use format: postgres://user:pass@host:port/database".to_string()),
        };
    }

    // Extract password if present
    let after_scheme = &url[11..]; // Skip "postgres://"
    if let Some(at_pos) = after_scheme.find('@') {
        let user_pass = &after_scheme[..at_pos];
        if let Some(colon_pos) = user_pass.find(':') {
            let password = &user_pass[colon_pos + 1..];
            if password == COMMON_WRONG_PASSWORD {
                return CheckResult {
                    status: CheckStatus::Warning,
                    name: "Database URL".to_string(),
                    message: format!("Password '{}' may not match Docker setup", password),
                    fix_suggestion: Some(format!(
                        "Docker PostgreSQL uses \"{}\"",
                        DOCKER_POSTGRES_PASSWORD
                    )),
                };
            }
        }
    }

    CheckResult {
        status: CheckStatus::Pass,
        name: "Database URL".to_string(),
        message: "Format valid".to_string(),
        fix_suggestion: None,
    }
}

fn check_telegram_token(token: &str) -> CheckResult {
    // Format: digits:alphanumeric_with_underscores_dashes
    if let Some(colon_pos) = token.find(':') {
        let numeric_part = &token[..colon_pos];
        let alpha_part = &token[colon_pos + 1..];

        if !numeric_part.is_empty()
            && numeric_part.chars().all(|c| c.is_ascii_digit())
            && !alpha_part.is_empty()
            && alpha_part
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return CheckResult {
                status: CheckStatus::Pass,
                name: "Telegram Bot".to_string(),
                message: "Configured".to_string(),
                fix_suggestion: None,
            };
        }
    }

    CheckResult {
        status: CheckStatus::Fail,
        name: "Telegram Bot".to_string(),
        message: "Invalid token format".to_string(),
        fix_suggestion: Some("Re-get token from @BotFather on Telegram".to_string()),
    }
}

fn check_tunnel_url(url: &str) -> CheckResult {
    if url.starts_with("https://") && url.len() > 8 {
        CheckResult {
            status: CheckStatus::Pass,
            name: "Tunnel".to_string(),
            message: format!("{} (HTTPS)", url),
            fix_suggestion: None,
        }
    } else {
        CheckResult {
            status: CheckStatus::Warning,
            name: "Tunnel".to_string(),
            message: format!("{} (not HTTPS)", url),
            fix_suggestion: Some("Tunnel URL should use HTTPS for security".to_string()),
        }
    }
}

const SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "env_vars": {
            "type": "object",
            "description": "Environment variables to validate (key-value pairs)",
            "properties": {
                "NGC_KEY": { "type": "string" },
                "LLM_MODEL": { "type": "string" },
                "DATABASE_URL": { "type": "string" },
                "TELEGRAM_BOT_TOKEN": { "type": "string" },
                "TUNNEL_URL": { "type": "string" }
            }
        },
        "check": {
            "type": "string",
            "description": "Which check to run: 'env' for detailed results, 'full_report' for formatted report",
            "enum": ["env", "full_report"],
            "default": "env"
        }
    },
    "required": ["env_vars"],
    "additionalProperties": false
}"#;

export!(ConfigValidatorTool);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngc_key_valid() {
        assert!(check_ngc_key("nvapi-test123").status == CheckStatus::Pass);
        assert!(check_ngc_key("nvapi-abc_def").status == CheckStatus::Pass);
    }

    #[test]
    fn test_ngc_key_invalid() {
        assert!(check_ngc_key("test123").status == CheckStatus::Fail);
        assert!(check_ngc_key("nvapi_").status == CheckStatus::Fail);
    }

    #[test]
    fn test_telegram_token_valid() {
        assert!(check_telegram_token("123456:ABC-DEF1234").status == CheckStatus::Pass);
        assert!(check_telegram_token("789:xyz_123").status == CheckStatus::Pass);
    }

    #[test]
    fn test_telegram_token_invalid() {
        assert!(check_telegram_token("invalid").status == CheckStatus::Fail);
        assert!(check_telegram_token("123456").status == CheckStatus::Fail);
    }

    #[test]
    fn test_https_url_valid() {
        assert!(check_tunnel_url("https://example.com").status == CheckStatus::Pass);
        assert!(check_tunnel_url("https://tunnel.ngrok.io").status == CheckStatus::Pass);
    }

    #[test]
    fn test_https_url_invalid() {
        assert!(check_tunnel_url("http://example.com").status == CheckStatus::Warning);
        assert!(check_tunnel_url("ftp://example.com").status == CheckStatus::Warning);
    }

    #[test]
    fn test_llm_model_validation() {
        assert!(VALID_LLM_MODELS.contains(&"z-ai/glm5"));
        assert!(INVALID_LLM_MODELS.contains(&"stepfun-ai/step-3.5-flash"));
    }

    #[test]
    fn test_database_password_check() {
        assert_eq!(COMMON_WRONG_PASSWORD, "ironclaw_pass");
        assert_eq!(DOCKER_POSTGRES_PASSWORD, "yourpass");
    }

    #[test]
    fn test_database_url_valid() {
        assert!(
            check_database_url("postgres://postgres:yourpass@localhost:5433/ironclaw").status
                == CheckStatus::Pass
        );
    }

    #[test]
    fn test_database_url_wrong_password() {
        let result =
            check_database_url("postgres://postgres:ironclaw_pass@localhost:5433/ironclaw");
        assert!(result.status == CheckStatus::Warning);
        assert!(result.message.contains("ironclaw_pass"));
    }
}
