//! IronClaw Tunnel Monitor WASM Tool
//!
//! Monitors Cloudflare tunnel health and auto-recovers when it dies.
//! Provides proactive monitoring, auto-recovery, and Telegram webhook sync.
//!
//! # Tools
//!
//! - `tunnel_check_status`: Check if tunnel is alive and accessible
//! - `tunnel_test_endpoint`: Test webhook endpoint accessibility
//! - `tunnel_restart_if_dead`: Auto-restart dead tunnel
//! - `tunnel_update_webhook`: Update Telegram webhook with new tunnel URL

wit_bindgen::generate!({
    world: "sandboxed-tool",
    path: "../../wit/tool.wit",
});

use serde::{Deserialize, Serialize};
use std::process::Command;

// ============================================================================
// Constants
// ============================================================================

/// Default tunnel URL file path
const TUNNEL_URL_FILE: &str = "/tmp/cloudflared-ephemeral-url.txt";

/// Default port for local webhook service
const DEFAULT_LOCAL_PORT: u16 = 8081;

/// Cloudflare error codes that indicate tunnel issues
const CLOUDFLARE_TUNNEL_ERRORS: &[u16] = &[530, 522, 521, 502, 524, 1033];

// ============================================================================
// Data Structures
// ============================================================================

/// Tunnel health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TunnelStatus {
    Healthy,
    ProcessNotRunning,
    UrlNotFound,
    DnsFailed,
    EndpointUnreachable,
    TelegramWebhookMismatch,
    Unknown,
}

/// Individual health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Check name
    pub name: String,
    /// Status: pass, fail, or warning
    pub status: CheckStatus,
    /// Detailed message
    pub message: String,
    /// Additional details if available
    pub details: Option<String>,
}

/// Check status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Warning,
    Fail,
}

/// Complete tunnel health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelHealthReport {
    /// Overall tunnel status
    pub status: TunnelStatus,
    /// Individual health checks
    pub checks: Vec<HealthCheck>,
    /// Tunnel URL if available
    pub tunnel_url: Option<String>,
    /// Process PID if running
    pub process_pid: Option<u32>,
    /// Tunnel age in minutes if available
    pub tunnel_age_minutes: Option<u32>,
    /// Recovery actions taken if any
    pub recovery_actions: Vec<String>,
}

/// Webhook sync result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookSyncResult {
    /// Whether webhook was updated
    pub updated: bool,
    /// Current webhook URL
    pub current_url: String,
    /// Expected webhook URL
    pub expected_url: String,
    /// Telegram webhook info
    pub telegram_info: Option<TelegramWebhookInfo>,
    /// Error message if any
    pub error: Option<String>,
}

/// Telegram webhook information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramWebhookInfo {
    pub url: String,
    pub has_custom_certificate: bool,
    pub pending_update_count: u32,
    pub last_error_message: Option<String>,
    pub max_connections: Option<u32>,
}

/// Recovery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,
    /// New tunnel URL if recovered
    pub new_url: Option<String>,
    /// New process PID
    pub new_pid: Option<u32>,
    /// Webhook updated
    pub webhook_updated: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Actions taken
    pub actions: Vec<String>,
}

// ============================================================================
// WASM Tool Implementation
// ============================================================================

struct TunnelMonitorTool;

impl exports::near::agent::tool::Guest for TunnelMonitorTool {
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
        "Monitor Cloudflare tunnel health and auto-recover when it dies. \
         Provides four tools: tunnel_check_status (check if tunnel is alive), \
         tunnel_test_endpoint (test webhook accessibility), \
         tunnel_restart_if_dead (auto-restart dead tunnel), \
         tunnel_update_webhook (sync Telegram webhook with tunnel URL). \
         Use for proactive monitoring and automatic recovery of ephemeral tunnels."
            .to_string()
    }
}

// ============================================================================
// Request Handling
// ============================================================================

#[derive(Debug, Deserialize)]
struct ToolParams {
    /// Tool to execute
    tool: String,
    /// Tool-specific parameters
    #[serde(default)]
    params: serde_json::Value,
}

fn execute_inner(params: &str) -> Result<String, String> {
    let params: ToolParams =
        serde_json::from_str(params).map_err(|e| format!("Invalid parameters: {e}"))?;

    match params.tool.as_str() {
        "tunnel_check_status" => check_status(&params.params),
        "tunnel_test_endpoint" => test_endpoint(&params.params),
        "tunnel_restart_if_dead" => restart_if_dead(&params.params),
        "tunnel_update_webhook" => update_webhook(&params.params),
        _ => Err(format!("Unknown tool: {}", params.tool)),
    }
}

// ============================================================================
// Tool: tunnel_check_status
// ============================================================================

fn check_status(params: &serde_json::Value) -> Result<String, String> {
    let config = parse_status_params(params)?;
    let mut report = TunnelHealthReport {
        status: TunnelStatus::Unknown,
        checks: Vec::new(),
        tunnel_url: None,
        process_pid: None,
        tunnel_age_minutes: None,
        recovery_actions: Vec::new(),
    };

    // Check 1: Is cloudflared process running?
    let process_check = check_cloudflared_process();
    report.process_pid = process_check.details.as_ref().and_then(|d| {
        d.split_whitespace()
            .find(|s| s.chars().all(char::is_numeric))
            .and_then(|s| s.parse().ok())
    });
    report.checks.push(process_check.clone());

    if process_check.status == CheckStatus::Fail {
        report.status = TunnelStatus::ProcessNotRunning;
        return serialize_report(&report);
    }

    // Check 2: Read tunnel URL from file
    let url_check = read_tunnel_url();
    report.tunnel_url = url_check.details.clone();
    report.checks.push(url_check.clone());

    if url_check.status == CheckStatus::Fail {
        report.status = TunnelStatus::UrlNotFound;
        return serialize_report(&report);
    }

    // Check 3: DNS resolution
    if let Some(url) = &report.tunnel_url {
        let dns_check = check_dns_resolution(url);
        report.checks.push(dns_check.clone());

        if dns_check.status == CheckStatus::Fail {
            report.status = TunnelStatus::DnsFailed;
            return serialize_report(&report);
        }

        // Check 4: Endpoint accessibility
        let endpoint_check = test_endpoint_accessibility(url, &config);
        report.checks.push(endpoint_check.clone());

        if endpoint_check.status == CheckStatus::Fail {
            report.status = TunnelStatus::EndpointUnreachable;
            return serialize_report(&report);
        }

        // Check 5: Telegram webhook sync (optional)
        if config.check_telegram {
            let webhook_check = check_telegram_webhook_sync(
                url,
                &config.telegram_bot_token,
                &config.telegram_webhook_path,
            );
            report.checks.push(webhook_check.clone());

            if webhook_check.status == CheckStatus::Fail {
                report.status = TunnelStatus::TelegramWebhookMismatch;
            } else {
                report.status = TunnelStatus::Healthy;
            }
        } else {
            report.status = TunnelStatus::Healthy;
        }
    }

    serialize_report(&report)
}

#[derive(Debug, Clone)]
struct StatusConfig {
    check_telegram: bool,
    telegram_bot_token: String,
    telegram_webhook_path: String,
    timeout_seconds: u64,
}

fn parse_status_params(params: &serde_json::Value) -> Result<StatusConfig, String> {
    let obj = params.as_object().ok_or("params must be an object")?;

    let check_telegram = obj
        .get("check_telegram")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let telegram_bot_token = obj
        .get("telegram_bot_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let telegram_webhook_path = obj
        .get("telegram_webhook_path")
        .and_then(|v| v.as_str())
        .unwrap_or("/webhook/telegram")
        .to_string();

    let timeout_seconds = obj
        .get("timeout_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(5);

    Ok(StatusConfig {
        check_telegram,
        telegram_bot_token,
        telegram_webhook_path,
        timeout_seconds,
    })
}

fn check_cloudflared_process() -> HealthCheck {
    match Command::new("pgrep").arg("-f").arg("cloudflared").output() {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let pid = stdout.trim().split('\n').next().unwrap_or("").to_string();
                HealthCheck {
                    name: "Process".to_string(),
                    status: CheckStatus::Pass,
                    message: format!("cloudflared running (PID: {})", pid),
                    details: Some(pid),
                }
            } else {
                HealthCheck {
                    name: "Process".to_string(),
                    status: CheckStatus::Fail,
                    message: "cloudflared not running".to_string(),
                    details: None,
                }
            }
        }
        Err(e) => HealthCheck {
            name: "Process".to_string(),
            status: CheckStatus::Fail,
            message: format!("Failed to check process: {}", e),
            details: None,
        },
    }
}

fn read_tunnel_url() -> HealthCheck {
    match std::fs::read_to_string(TUNNEL_URL_FILE) {
        Ok(content) => {
            let url = content.trim().to_string();
            if url.starts_with("https://") && url.contains("trycloudflare.com") {
                HealthCheck {
                    name: "URL".to_string(),
                    status: CheckStatus::Pass,
                    message: url.clone(),
                    details: Some(url),
                }
            } else {
                HealthCheck {
                    name: "URL".to_string(),
                    status: CheckStatus::Warning,
                    message: format!("URL found but may be invalid: {}", url),
                    details: Some(url),
                }
            }
        }
        Err(_) => HealthCheck {
            name: "URL".to_string(),
            status: CheckStatus::Fail,
            message: "Tunnel URL file not found".to_string(),
            details: None,
        },
    }
}

fn check_dns_resolution(url: &str) -> HealthCheck {
    // Extract hostname from URL
    let hostname = url
        .strip_prefix("https://")
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or("");

    match Command::new("dig").arg("+short").arg(hostname).output() {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let ip = stdout.lines().next().unwrap_or("").to_string();
                if !ip.is_empty() {
                    HealthCheck {
                        name: "DNS".to_string(),
                        status: CheckStatus::Pass,
                        message: format!("Resolved ({})", ip),
                        details: Some(ip),
                    }
                } else {
                    HealthCheck {
                        name: "DNS".to_string(),
                        status: CheckStatus::Fail,
                        message: "DNS returned empty response".to_string(),
                        details: None,
                    }
                }
            } else {
                HealthCheck {
                    name: "DNS".to_string(),
                    status: CheckStatus::Fail,
                    message: "DNS lookup failed".to_string(),
                    details: None,
                }
            }
        }
        Err(e) => HealthCheck {
            name: "DNS".to_string(),
            status: CheckStatus::Warning,
            message: format!("dig command not available: {}", e),
            details: None,
        },
    }
}

fn test_endpoint_accessibility(url: &str, config: &StatusConfig) -> HealthCheck {
    let test_url = format!("{}{}", url, config.telegram_webhook_path);

    match Command::new("curl")
        .arg("-I")
        .arg("-s")
        .arg("-o")
        .arg("/dev/null")
        .arg("-w")
        .arg("%{http_code}")
        .arg("--connect-timeout")
        .arg(config.timeout_seconds.to_string())
        .arg(&test_url)
        .output()
    {
        Ok(output) => {
            let status_code = String::from_utf8_lossy(&output.stdout);
            match status_code.trim().parse::<u16>() {
                Ok(code) => {
                    if code == 200 {
                        HealthCheck {
                            name: "Endpoint".to_string(),
                            status: CheckStatus::Pass,
                            message: format!("Accessible (HTTP {})", code),
                            details: Some(code.to_string()),
                        }
                    } else if CLOUDFLARE_TUNNEL_ERRORS.contains(&code) {
                        HealthCheck {
                            name: "Endpoint".to_string(),
                            status: CheckStatus::Fail,
                            message: format!("Cloudflare error (HTTP {})", code),
                            details: Some(format!("Cloudflare tunnel error code {}", code)),
                        }
                    } else {
                        HealthCheck {
                            name: "Endpoint".to_string(),
                            status: CheckStatus::Warning,
                            message: format!("Returned HTTP {}", code),
                            details: Some(code.to_string()),
                        }
                    }
                }
                Err(_) => HealthCheck {
                    name: "Endpoint".to_string(),
                    status: CheckStatus::Fail,
                    message: "Invalid response from endpoint".to_string(),
                    details: None,
                },
            }
        }
        Err(e) => HealthCheck {
            name: "Endpoint".to_string(),
            status: CheckStatus::Fail,
            message: format!("Endpoint unreachable: {}", e),
            details: None,
        },
    }
}

fn check_telegram_webhook_sync(
    tunnel_url: &str,
    bot_token: &str,
    webhook_path: &str,
) -> HealthCheck {
    if bot_token.is_empty() {
        return HealthCheck {
            name: "Telegram Webhook".to_string(),
            status: CheckStatus::Warning,
            message: "Bot token not provided".to_string(),
            details: None,
        };
    }

    let expected_url = format!("{}{}", tunnel_url, webhook_path);

    // Get current webhook info
    match get_telegram_webhook_info(bot_token) {
        Ok(info) => {
            if info.url == expected_url {
                HealthCheck {
                    name: "Telegram Webhook".to_string(),
                    status: CheckStatus::Pass,
                    message: "Synced".to_string(),
                    details: Some(format!(
                        "URL: {}, Pending: {}",
                        info.url, info.pending_update_count
                    )),
                }
            } else {
                HealthCheck {
                    name: "Telegram Webhook".to_string(),
                    status: CheckStatus::Fail,
                    message: "Webhook URL mismatch".to_string(),
                    details: Some(format!("Expected: {}, Current: {}", expected_url, info.url)),
                }
            }
        }
        Err(e) => HealthCheck {
            name: "Telegram Webhook".to_string(),
            status: CheckStatus::Warning,
            message: format!("Failed to check webhook: {}", e),
            details: None,
        },
    }
}

fn get_telegram_webhook_info(bot_token: &str) -> Result<TelegramWebhookInfo, String> {
    let url = format!("https://api.telegram.org/bot{}/getWebhookInfo", bot_token);

    match Command::new("curl").arg("-s").arg(&url).output() {
        Ok(output) => {
            if output.status.success() {
                let response = String::from_utf8_lossy(&output.stdout);
                parse_webhook_info(&response)
            } else {
                Err("Failed to get webhook info".to_string())
            }
        }
        Err(e) => Err(format!("curl failed: {}", e)),
    }
}

fn parse_webhook_info(response: &str) -> Result<TelegramWebhookInfo, String> {
    #[derive(Deserialize)]
    struct TelegramResponse {
        ok: bool,
        result: Option<WebhookResult>,
    }

    #[derive(Deserialize)]
    struct WebhookResult {
        url: String,
        has_custom_certificate: bool,
        pending_update_count: u32,
        last_error_message: Option<String>,
        max_connections: Option<u32>,
    }

    let parsed: TelegramResponse =
        serde_json::from_str(response).map_err(|e| format!("JSON parse error: {}", e))?;

    if !parsed.ok {
        return Err("Telegram API returned error".to_string());
    }

    let result = parsed.result.ok_or("No webhook info in response")?;

    Ok(TelegramWebhookInfo {
        url: result.url,
        has_custom_certificate: result.has_custom_certificate,
        pending_update_count: result.pending_update_count,
        last_error_message: result.last_error_message,
        max_connections: result.max_connections,
    })
}

fn serialize_report(report: &TunnelHealthReport) -> Result<String, String> {
    serde_json::to_string_pretty(report).map_err(|e| format!("Serialization error: {}", e))
}

// ============================================================================
// Tool: tunnel_test_endpoint
// ============================================================================

fn test_endpoint(params: &serde_json::Value) -> Result<String, String> {
    let config = parse_endpoint_params(params)?;

    let url = if let Some(tunnel_url) = &config.tunnel_url {
        format!("{}{}", tunnel_url, config.path)
    } else {
        // Try to read from file
        match std::fs::read_to_string(TUNNEL_URL_FILE) {
            Ok(content) => format!("{}{}", content.trim(), config.path),
            Err(_) => return Err("Tunnel URL not provided and file not found".to_string()),
        }
    };

    let _start = std::time::Instant::now();

    match Command::new("curl")
        .arg("-s")
        .arg("-o")
        .arg("/dev/null")
        .arg("-w")
        .arg("%{http_code}|%{time_total}|%{size_download}")
        .arg("--connect-timeout")
        .arg(config.timeout_seconds.to_string())
        .arg(&url)
        .output()
    {
        Ok(output) => {
            let metrics = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = metrics.trim().split('|').collect();

            if parts.len() >= 3 {
                let status_code = parts[0].parse::<u16>().unwrap_or(0);
                let response_time = parts[1].parse::<f64>().unwrap_or(0.0);
                let response_size = parts[2].parse::<u64>().unwrap_or(0);

                let result = serde_json::json!({
                    "url": url,
                    "status_code": status_code,
                    "response_time_ms": (response_time * 1000.0) as u64,
                    "response_size_bytes": response_size,
                    "success": status_code >= 200 && status_code < 300,
                    "error": if status_code == 0 {
                        Some("Connection failed".to_string())
                    } else if CLOUDFLARE_TUNNEL_ERRORS.contains(&status_code) {
                        Some(format!("Cloudflare error {}", status_code))
                    } else {
                        None
                    }
                });

                Ok(result.to_string())
            } else {
                Err("Invalid curl output format".to_string())
            }
        }
        Err(e) => Err(format!("curl failed: {}", e)),
    }
}

#[derive(Debug, Clone)]
struct EndpointConfig {
    tunnel_url: Option<String>,
    path: String,
    method: String,
    timeout_seconds: u64,
}

fn parse_endpoint_params(params: &serde_json::Value) -> Result<EndpointConfig, String> {
    let obj = params.as_object().ok_or("params must be an object")?;

    let tunnel_url = obj
        .get("tunnel_url")
        .and_then(|v| v.as_str())
        .map(String::from);
    let path = obj
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("/webhook/telegram")
        .to_string();
    let method = obj
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("GET")
        .to_string();
    let timeout_seconds = obj
        .get("timeout_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(5);

    Ok(EndpointConfig {
        tunnel_url,
        path,
        method,
        timeout_seconds,
    })
}

// ============================================================================
// Tool: tunnel_restart_if_dead
// ============================================================================

fn restart_if_dead(params: &serde_json::Value) -> Result<String, String> {
    let config = parse_restart_params(params)?;
    let mut result = RecoveryResult {
        success: false,
        new_url: None,
        new_pid: None,
        webhook_updated: false,
        error: None,
        actions: Vec::new(),
    };

    // Check if tunnel is already healthy
    let status_check = check_status(&serde_json::Value::Object(serde_json::Map::new()));

    if let Ok(report_str) = status_check {
        let report: TunnelHealthReport =
            serde_json::from_str(&report_str).unwrap_or_else(|_| TunnelHealthReport {
                status: TunnelStatus::Unknown,
                checks: Vec::new(),
                tunnel_url: None,
                process_pid: None,
                tunnel_age_minutes: None,
                recovery_actions: Vec::new(),
            });

        if report.status == TunnelStatus::Healthy {
            result.success = true;
            result.new_url = report.tunnel_url;
            result.new_pid = report.process_pid;
            result
                .actions
                .push("Tunnel already healthy, no restart needed".to_string());
            return serialize_recovery(&result);
        }
    }

    // Kill old cloudflared process
    result
        .actions
        .push("Killing old cloudflared process".to_string());
    if let Err(e) = kill_cloudflared() {
        result.actions.push(format!("Warning: {}", e));
    }

    // Wait for process to stop
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Start new tunnel
    result
        .actions
        .push("Starting new cloudflared tunnel".to_string());
    match start_cloudflared(config.local_port) {
        Ok((pid, url)) => {
            result.new_pid = Some(pid);
            result.new_url = Some(url.clone());
            result
                .actions
                .push(format!("New tunnel started (PID: {})", pid));
            result.actions.push(format!("New URL: {}", url));

            // Save URL to file
            if std::fs::write(TUNNEL_URL_FILE, &url).is_ok() {
                result.actions.push("URL saved to file".to_string());
            }

            // Update Telegram webhook if configured
            if !config.telegram_bot_token.is_empty() {
                let webhook_url = format!("{}{}", url, config.telegram_webhook_path);
                result.actions.push("Updating Telegram webhook".to_string());
                match set_telegram_webhook(&config.telegram_bot_token, &webhook_url) {
                    Ok(_) => {
                        result.webhook_updated = true;
                        result
                            .actions
                            .push("Webhook updated successfully".to_string());
                    }
                    Err(e) => {
                        result.actions.push(format!("Webhook update failed: {}", e));
                    }
                }
            }

            result.success = true;
        }
        Err(e) => {
            result.error = Some(e.clone());
            result
                .actions
                .push(format!("Failed to start tunnel: {}", e));
        }
    }

    serialize_recovery(&result)
}

#[derive(Debug, Clone)]
struct RestartConfig {
    local_port: u16,
    telegram_bot_token: String,
    telegram_webhook_path: String,
}

fn parse_restart_params(params: &serde_json::Value) -> Result<RestartConfig, String> {
    let obj = params.as_object().ok_or("params must be an object")?;

    let local_port = obj
        .get("local_port")
        .and_then(|v| v.as_u64())
        .map(|v| v as u16)
        .unwrap_or(DEFAULT_LOCAL_PORT);

    let telegram_bot_token = obj
        .get("telegram_bot_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let telegram_webhook_path = obj
        .get("telegram_webhook_path")
        .and_then(|v| v.as_str())
        .unwrap_or("/webhook/telegram")
        .to_string();

    Ok(RestartConfig {
        local_port,
        telegram_bot_token,
        telegram_webhook_path,
    })
}

fn kill_cloudflared() -> Result<(), String> {
    Command::new("pkill")
        .arg("-f")
        .arg("cloudflared")
        .output()
        .map_err(|e| format!("Failed to kill cloudflared: {}", e))?;
    Ok(())
}

fn start_cloudflared(port: u16) -> Result<(u32, String), String> {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let mut child = Command::new("cloudflared")
        .arg("tunnel")
        .arg("--url")
        .arg(format!("http://localhost:{}", port))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn cloudflared: {}", e))?;

    let pid = child.id();
    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let reader = BufReader::new(stdout);

    // Wait for URL to appear in logs (timeout 10 seconds)
    let url_regex = regex::Regex::new(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        .map_err(|e| format!("Regex error: {}", e))?;

    let mut extracted_url = None;
    let start = std::time::Instant::now();

    for line in reader.lines() {
        if start.elapsed().as_secs() > 10 {
            break;
        }

        if let Ok(log_line) = line {
            if let Some(mat) = url_regex.find(&log_line) {
                extracted_url = Some(mat.as_str().to_string());
                break;
            }
        }
    }

    // Detach process (don't wait for it)
    // In a real implementation, you'd want to properly daemonize
    std::mem::forget(child);

    let url = extracted_url.ok_or("Failed to extract tunnel URL from logs")?;
    Ok((pid, url))
}

fn set_telegram_webhook(bot_token: &str, url: &str) -> Result<(), String> {
    let api_url = format!("https://api.telegram.org/bot{}/setWebhook", bot_token);

    let response = Command::new("curl")
        .arg("-s")
        .arg("-X")
        .arg("POST")
        .arg(&api_url)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(format!(r#"{{"url":"{}"}}"#, url))
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    let response_str = String::from_utf8_lossy(&response.stdout);

    #[derive(Deserialize)]
    struct SetWebhookResponse {
        ok: bool,
        description: Option<String>,
    }

    let parsed: SetWebhookResponse =
        serde_json::from_str(&response_str).map_err(|e| format!("JSON parse error: {}", e))?;

    if parsed.ok {
        Ok(())
    } else {
        Err(parsed
            .description
            .unwrap_or_else(|| "Unknown Telegram API error".to_string()))
    }
}

fn serialize_recovery(result: &RecoveryResult) -> Result<String, String> {
    serde_json::to_string_pretty(result).map_err(|e| format!("Serialization error: {}", e))
}

// ============================================================================
// Tool: tunnel_update_webhook
// ============================================================================

fn update_webhook(params: &serde_json::Value) -> Result<String, String> {
    let config = parse_webhook_params(params)?;

    // Get current tunnel URL
    let tunnel_url = if let Some(url) = &config.tunnel_url {
        url.clone()
    } else {
        match std::fs::read_to_string(TUNNEL_URL_FILE) {
            Ok(content) => content.trim().to_string(),
            Err(_) => return Err("Tunnel URL not provided and file not found".to_string()),
        }
    };

    let expected_webhook_url = format!("{}{}", tunnel_url, config.webhook_path);

    // Get current Telegram webhook info
    let telegram_info = get_telegram_webhook_info(&config.bot_token)?;
    let current_url = telegram_info.url.clone();

    let mut result = WebhookSyncResult {
        updated: false,
        current_url: current_url.clone(),
        expected_url: expected_webhook_url.clone(),
        telegram_info: Some(telegram_info.clone()),
        error: None,
    };

    // Check if update needed
    if current_url == expected_webhook_url {
        result.updated = false;
        return serialize_webhook_result(&result);
    }

    // Update webhook
    let api_url = format!(
        "https://api.telegram.org/bot{}/setWebhook",
        config.bot_token
    );

    let response = Command::new("curl")
        .arg("-s")
        .arg("-X")
        .arg("POST")
        .arg(&api_url)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(format!(
            r#"{{"url":"{}","drop_pending_updates":{}}}"#,
            expected_webhook_url, config.drop_pending_updates
        ))
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    let response_str = String::from_utf8_lossy(&response.stdout);

    #[derive(Deserialize)]
    struct SetWebhookResponse {
        ok: bool,
        description: Option<String>,
    }

    let parsed: SetWebhookResponse =
        serde_json::from_str(&response_str).map_err(|e| format!("JSON parse error: {}", e))?;

    if parsed.ok {
        result.updated = true;
        result.current_url = expected_webhook_url;

        // Verify update
        if let Ok(new_info) = get_telegram_webhook_info(&config.bot_token) {
            result.telegram_info = Some(new_info);
        }
    } else {
        result.error = parsed
            .description
            .or_else(|| Some("Telegram API error".to_string()));
    }

    serialize_webhook_result(&result)
}

#[derive(Debug, Clone)]
struct WebhookConfig {
    bot_token: String,
    tunnel_url: Option<String>,
    webhook_path: String,
    drop_pending_updates: bool,
}

fn parse_webhook_params(params: &serde_json::Value) -> Result<WebhookConfig, String> {
    let obj = params.as_object().ok_or("params must be an object")?;

    let bot_token = obj
        .get("bot_token")
        .and_then(|v| v.as_str())
        .ok_or("bot_token is required")?
        .to_string();

    let tunnel_url = obj
        .get("tunnel_url")
        .and_then(|v| v.as_str())
        .map(String::from);
    let webhook_path = obj
        .get("webhook_path")
        .and_then(|v| v.as_str())
        .unwrap_or("/webhook/telegram")
        .to_string();

    let drop_pending_updates = obj
        .get("drop_pending_updates")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(WebhookConfig {
        bot_token,
        tunnel_url,
        webhook_path,
        drop_pending_updates,
    })
}

fn serialize_webhook_result(result: &WebhookSyncResult) -> Result<String, String> {
    serde_json::to_string_pretty(result).map_err(|e| format!("Serialization error: {}", e))
}

// ============================================================================
// Schema
// ============================================================================

const SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "description": "Tool to execute",
            "enum": [
                "tunnel_check_status",
                "tunnel_test_endpoint",
                "tunnel_restart_if_dead",
                "tunnel_update_webhook"
            ]
        },
        "params": {
            "type": "object",
            "description": "Tool-specific parameters",
            "properties": {
                "tunnel_url": {
                    "type": "string",
                    "description": "Cloudflare tunnel URL (optional, reads from file if not provided)"
                },
                "local_port": {
                    "type": "integer",
                    "description": "Local port for tunnel (default: 8081)"
                },
                "path": {
                    "type": "string",
                    "description": "Webhook path to test (default: /webhook/telegram)"
                },
                "telegram_bot_token": {
                    "type": "string",
                    "description": "Telegram bot token for webhook operations"
                },
                "bot_token": {
                    "type": "string",
                    "description": "Telegram bot token (alias for telegram_bot_token)"
                },
                "telegram_webhook_path": {
                    "type": "string",
                    "description": "Telegram webhook path (default: /webhook/telegram)"
                },
                "webhook_path": {
                    "type": "string",
                    "description": "Webhook path (alias for telegram_webhook_path)"
                },
                "check_telegram": {
                    "type": "boolean",
                    "description": "Whether to check Telegram webhook sync (default: false)"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default: 5)"
                },
                "drop_pending_updates": {
                    "type": "boolean",
                    "description": "Drop pending updates when updating webhook (default: false)"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method for endpoint test (default: GET)"
                }
            }
        }
    },
    "required": ["tool"],
    "additionalProperties": false
}"#;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunnel_status_serialization() {
        let status = TunnelStatus::Healthy;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"healthy\"");
    }

    #[test]
    fn test_check_status_serialization() {
        let status = CheckStatus::Pass;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"pass\"");
    }

    #[test]
    fn test_health_check_creation() {
        let check = HealthCheck {
            name: "Test".to_string(),
            status: CheckStatus::Pass,
            message: "OK".to_string(),
            details: Some("details".to_string()),
        };
        assert_eq!(check.name, "Test");
        assert_eq!(check.status, CheckStatus::Pass);
    }

    #[test]
    fn test_cloudflare_error_codes() {
        assert!(CLOUDFLARE_TUNNEL_ERRORS.contains(&530));
        assert!(CLOUDFLARE_TUNNEL_ERRORS.contains(&522));
        assert!(CLOUDFLARE_TUNNEL_ERRORS.contains(&1033));
    }

    #[test]
    fn test_default_port() {
        assert_eq!(DEFAULT_LOCAL_PORT, 8081);
    }

    #[test]
    fn test_tunnel_url_file_path() {
        assert_eq!(TUNNEL_URL_FILE, "/tmp/cloudflared-ephemeral-url.txt");
    }
}
