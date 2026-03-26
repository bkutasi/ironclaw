<!-- Context: development/integration/cli | Priority: high | Version: 1.0 | Updated: 2026-03-26 -->

# CLI Patterns

**Purpose**: Command-line interface architecture, command parsing, REPL structure, and interactive mode handling

**Last Updated**: 2026-03-26

## Quick Reference

**Update Triggers**: CLI command changes | New subcommands | Parsing logic updates | Interactive mode improvements

**Audience**: Developers extending CLI functionality, AI agents implementing command handlers

**Entry Point**: `src/cli/mod.rs` - Main CLI structure and command routing

**Key Commands**:
- `ironclaw run` - Start the agent (default)
- `ironclaw config` - Manage settings
- `ironclaw tool` - Manage WASM tools
- `ironclaw mcp` - Manage MCP servers
- `ironclaw memory` - Query workspace memory
- `ironclaw routines` - Manage scheduled routines
- `ironclaw service` - OS service management
- `ironclaw skills` - Manage SKILL.md-based skills
- `ironclaw doctor` - Run diagnostics

**Dependencies**: `clap` (parsing), `clap_complete` (shell completions), `crossterm` (interactive input)

---

## Architecture Overview

```
User Input                    CLI Parser                          Command Handler
┌──────────────────┐         ┌─────────────────────┐             ┌──────────────────┐
│                  │         │  src/cli/mod.rs     │             │  src/cli/*.rs    │
│ ironclaw <cmd>   │  ──────►│  Cli struct         │  ──────────►│  run_*_command() │
│ [args...]        │         │  Command enum       │             │                  │
│                  │         │                     │             │                  │
│                  │         │  - Parses args      │             │  - DB access     │
│                  │         │  - Routes to cmd    │             │  - Business logic│
│                  │         │  - Handles errors   │             │  - Output formatting│
└──────────────────┘         └─────────────────────┘             └──────────────────┘
```

**Key Components**:
1. **Cli struct** - Top-level parser with global flags (`--cli-only`, `--no-db`, `--message`, `--config`, `--no-onboard`)
2. **Command enum** - All subcommands with their arguments
3. **Command handlers** - Individual `run_*_command()` functions per subcommand
4. **Shared helpers** - `init_secrets_store()`, DB connection utilities

**Design Principles**:
- Each subcommand has its own module in `src/cli/`
- Command handlers are async and return `anyhow::Result<()>`
- DB connection is optional where possible (graceful degradation)
- Secrets store initialized on-demand for auth-related commands

---

## Commands

### Command Structure

All commands follow this pattern:

```rust
#[derive(Subcommand, Debug, Clone)]
pub enum XxxCommand {
    /// Subcommand description
    Action {
        /// Argument description
        arg_name: Type,
        
        #[arg(short, long)]
        flag: Option<Type>,
    },
}

pub async fn run_xxx_command(cmd: XxxCommand) -> anyhow::Result<()> {
    match cmd {
        XxxCommand::Action { arg_name, flag } => {
            // Implementation
        }
    }
}
```

### Global Flags (Cli struct)

| Flag | Description | Scope |
|------|-------------|-------|
| `--cli-only` | Run in interactive CLI mode only (disable other channels) | Global |
| `--no-db` | Skip database connection (for testing) | Global |
| `-m, --message <MSG>` | Single message mode - send one message and exit | Global |
| `-c, --config <PATH>` | Configuration file path | Global |
| `--no-onboard` | Skip first-run onboarding check | Global |

### Core Commands

#### `run` - Start Agent
```rust
/// Run the agent (default if no subcommand given)
Run
```
- Starts the IronClaw agent in default mode
- Default behavior when no subcommand is provided

#### `onboard` - Interactive Setup Wizard
```rust
Onboard {
    #[arg(long)]
    skip_auth: bool,
    
    #[arg(long)]
    quick: bool,
    
    #[arg(long, value_delimiter = ',')]
    step: Vec<String>,  // provider, channels, model, database, security
}
```
- Guides through initial configuration
- Supports partial reconfiguration via `--step`
- Quick mode auto-defaults everything except LLM provider/model

#### `config` - Settings Management
```rust
Config(ConfigCommand)

enum ConfigCommand {
    Init { output: Option<PathBuf>, force: bool },
    List { filter: Option<String> },
    Get { path: String },
    Set { path: String, value: String },
    Reset { path: String },
    Path,
}
```
- Settings stored in database with env > DB > default precedence
- `config init` generates TOML config file
- `config list --filter <prefix>` filters by key prefix

#### `tool` - WASM Tool Management
```rust
Tool(ToolCommand)

enum ToolCommand {
    Install { path: PathBuf, name: Option<String>, capabilities: Option<PathBuf>, ... },
    List { dir: Option<PathBuf>, verbose: bool },
    Remove { name: String, dir: Option<PathBuf> },
    Info { name_or_path: String, dir: Option<PathBuf>, user: String },
    Auth { name: String, dir: Option<PathBuf>, user: String },
    Setup { name: String, dir: Option<PathBuf>, user: String },
}
```
- Install from source directory or `.wasm` file
- Auto-detects `capabilities.json`
- OAuth and manual auth flows supported
- Setup configures `required_secrets` from capabilities

#### `mcp` - MCP Server Management
```rust
Mcp(McpCommand)

enum McpCommand {
    Add(Box<McpAddArgs>),  // Supports http, stdio, unix transports
    Remove { name: String },
    List { verbose: bool },
    Auth { name: String, user: String },
    Test { name: String, user: String },
    Toggle { name: String, enable: bool, disable: bool },
}
```
- Transports: HTTP (default), stdio, unix
- OAuth authentication with DCR (Dynamic Client Registration)
- Server config stored in DB or disk fallback

#### `memory` - Workspace Memory
```rust
Memory(MemoryCommand)

enum MemoryCommand {
    Search { query: String, limit: usize },
    Read { path: String },
    Write { path: String, content: Option<String>, append: bool },
    Tree { path: String, depth: usize },
    Status,
}
```
- Hybrid full-text + semantic search
- Workspace file operations (read/write/append)
- Directory tree visualization
- Status shows document count and identity files

#### `routines` - Scheduled Automation
```rust
Routines(RoutinesCommand)

enum RoutinesCommand {
    List { trigger: Option<String>, disabled: bool, json: bool },
    Create { name: String, schedule: String, prompt: String, ... },
    Edit { name: String, schedule: Option<String>, prompt: Option<String>, ... },
    Enable { name: String },
    Disable { name: String },
    Delete { name: String, yes: bool },
    History { name: String, limit: i64, json: bool },
}
```
- Cron schedules (6-field: sec min hour day month weekday)
- Timezone support (IANA format)
- Cooldown between fires
- Run history with status tracking

#### `service` - OS Service Management
```rust
Service(ServiceCommand)

enum ServiceCommand {
    Install, Start, Stop, Status, Uninstall,
}
```
- launchd on macOS
- systemd on Linux

#### `skills` - SKILL.md Management
```rust
Skills(SkillsCommand)

enum SkillsCommand {
    List { verbose: bool, json: bool },
    Search { query: String, json: bool },
    Info { name: String, json: bool },
}
```
- Discovers skills from configured directories
- Search queries ClawHub registry
- Shows trust level, source, activation patterns

#### `doctor` - Diagnostics
```rust
Doctor
```
- Probes external dependencies
- Validates configuration
- Checks system health

#### `logs` - Gateway Logs
```rust
Logs(LogsCommand)
```
- Tail gateway logs
- Stream live output via SSE
- Adjust log level

#### `status` - System Health
```rust
Status
```
- Displays health and diagnostics info

#### `completion` - Shell Completions
```rust
Completion(Completion)

struct Completion {
    #[arg(long)]
    shell: Shell,  // bash, zsh, fish, powershell, elvish
}
```
- Zsh: Guards `compdef` call for safe sourcing
- Other shells: Standard clap_complete output

---

## REPL Structure

IronClaw does **not** have a traditional REPL. Instead, it uses:

### Interactive Mode (via `--cli-only`)

When `--cli-only` flag is set:
1. Disables other channels (web, Telegram, etc.)
2. Runs in terminal-only mode
3. Processes user input line-by-line

### Single Message Mode (via `--message`)

```bash
ironclaw --message "What is the weather?"
```
- Sends one message and exits
- Useful for scripting
- No interactive session

### Onboarding Wizard

The `onboard` command provides an interactive wizard:
```rust
// Example: Re-authenticate prompt
print!("  Re-authenticate? [y/N]: ");
std::io::stdout().flush()?;

let mut input = String::new();
std::io::stdin().read_line(&mut input)?;

if !input.trim().eq_ignore_ascii_case("y") {
    return Ok(());
}
```

**Interactive Patterns**:
- Yes/No prompts with `[y/N]` convention
- Input validation with retry
- Hidden input for secrets (via `crossterm`)
- Progress indicators with Unicode symbols (✓, ✗, ●, ○)

---

## Command Parsing

### Clap Configuration

```rust
#[derive(Parser, Debug)]
#[command(name = "ironclaw")]
#[command(about = "Secure personal AI assistant...")]
#[command(long_about = "...")]
#[command(version)]
#[command(color = ColorChoice::Auto)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
    
    #[arg(long, global = true)]
    pub cli_only: bool,
    
    // ... other global flags
}
```

### Argument Types

**Positional Arguments**:
```rust
/// Server name (e.g., "notion", "github")
pub name: String,

/// Server URL (e.g., "https://mcp.notion.com")
pub url: Option<String>,
```

**Named Flags**:
```rust
#[arg(short, long)]
verbose: bool,

#[arg(long, default_value = "5")]
limit: usize,
```

**Repeated Arguments**:
```rust
/// Command arguments (stdio transport, can be repeated)
#[arg(long = "arg", num_args = 1..)]
pub cmd_args: Vec<String>,

/// Environment variables (KEY=VALUE format, can be repeated)
#[arg(long = "env", value_parser = parse_env_var)]
pub env: Vec<(String, String)>,
```

**Custom Parsers**:
```rust
fn parse_header(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find(':')
        .ok_or_else(|| format!("invalid header format '{}', expected KEY:VALUE", s))?;
    Ok((s[..pos].trim().to_string(), s[pos + 1..].trim().to_string()))
}

fn parse_env_var(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid env var format '{}', expected KEY=VALUE", s))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}
```

### Conflicts & Validation

```rust
#[arg(long, conflicts_with_all = ["channels_only", "quick", "step"])]
provider_only: bool,

#[arg(long, value_delimiter = ',', conflicts_with_all = ["channels_only", "provider_only", "quick"])]
step: Vec<String>,
```

**Runtime Validation**:
```rust
let transport_lower = transport.to_lowercase();

let mut config = match transport_lower.as_str() {
    "stdio" => {
        let cmd = command
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--command is required for stdio transport"))?;
        // ...
    }
    "http" => {
        let url_val = url
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("URL is required for http transport"))?;
        // ...
    }
    other => {
        anyhow::bail!(
            "Unknown transport type '{}'. Supported: http, stdio, unix",
            other
        );
    }
};
```

---

## Interactive Mode

### Hidden Input (for Secrets)

Uses `crossterm` for raw terminal mode:

```rust
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal,
};

fn read_hidden_input() -> anyhow::Result<String> {
    let mut input = String::new();
    terminal::enable_raw_mode()?;

    loop {
        if let Event::Key(key_event) = event::read()? {
            match key_event.code {
                KeyCode::Enter => break,
                KeyCode::Backspace => {
                    if !input.is_empty() {
                        input.pop();
                        print!("\x08 \x08");
                        std::io::stdout().flush()?;
                    }
                }
                KeyCode::Char('c') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                    terminal::disable_raw_mode()?;
                    return Err(anyhow::anyhow!("Interrupted"));
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    print!("*");
                    std::io::stdout().flush()?;
                }
                _ => {}
            }
        }
    }

    terminal::disable_raw_mode()?;
    Ok(input)
}
```

### Confirmation Prompts

```rust
// Yes/No with default
print!("  Delete this routine? [y/N] ");
std::io::Write::flush(&mut std::io::stdout())?;

let mut input = String::new();
std::io::stdin().read_line(&mut input)?;
if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
    println!("Cancelled.");
    return Ok(());
}

// Replace existing
print!("  Replace existing credentials? [y/N]: ");
std::io::stdout().flush()?;

let mut input = String::new();
std::io::stdin().read_line(&mut input)?;
if !input.trim().eq_ignore_ascii_case("y") {
    println!();
    println!("  Keeping existing credentials.");
    return Ok(());
}
```

### Progress & Status Display

```rust
// Success/Failure indicators
println!("  ✓ Added MCP server '{}'", name);
println!("  ✗ Authentication failed: {}", e);

// Status bullets
let status = if server.enabled { "●" } else { "○" };
println!("  {} {} - {}", status, server.name, display);

// Section headers with box drawing
println!("╔════════════════════════════════════════════════════════════════╗");
println!("║  {:^62}║", format!("{} Authentication", display_name));
println!("╚════════════════════════════════════════════════════════════════╝");
```

### Table Formatting

```rust
// Dynamic column width
let max_key_len = all.iter().map(|(k, _)| k.len()).max().unwrap_or(0);

for (key, value) in all {
    let display_value = if value.len() > 60 {
        format!("{}...", &value[..57])
    } else {
        value
    };
    println!("  {:width$}  {}", key, display_value, width = max_key_len);
}

// Fixed-width table
println!(
    "{:<36}  {:<20}  {:<8}  {:<8}  {:<22}  {:<22}  {:>5}",
    "ID", "NAME", "TRIGGER", "STATUS", "NEXT FIRE", "LAST RUN", "RUNS"
);
println!("{}", "-".repeat(130));
```

---

## Code Patterns

### Command Handler Template

```rust
//! Module doc: CLI subcommand definitions for `ironclaw <cmd>`.

use clap::Subcommand;

#[derive(Subcommand, Debug, Clone)]
pub enum XxxCommand {
    /// Subcommand description
    Action {
        #[arg(short, long)]
        flag: Option<Type>,
    },
}

/// Run the xxx command.
pub async fn run_xxx_command(cmd: XxxCommand) -> anyhow::Result<()> {
    match cmd {
        XxxCommand::Action { flag } => {
            // Implementation
        }
    }
}
```

### DB Connection Pattern

```rust
/// Bootstrap a DB connection for config commands (backend-agnostic).
async fn connect_db() -> anyhow::Result<Arc<dyn crate::db::Database>> {
    let config = crate::config::Config::from_env()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    crate::db::connect_from_config(&config.database)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))
}

/// Load settings: DB if available, else disk.
async fn load_settings(store: Option<&dyn crate::db::Database>) -> Settings {
    if let Some(store) = store {
        match store.get_all_settings(DEFAULT_USER_ID).await {
            Ok(map) if !map.is_empty() => return Settings::from_db_map(&map),
            _ => {}
        }
    }
    Settings::default()
}
```

### Secrets Store Initialization

```rust
/// Initialize a secrets store from environment config.
///
/// Shared helper for CLI subcommands (`mcp auth`, `tool auth`, etc.) that need
/// access to encrypted secrets without spinning up the full AppBuilder.
pub async fn init_secrets_store()
-> anyhow::Result<Arc<dyn crate::secrets::SecretsStore + Send + Sync>> {
    let config = crate::config::Config::from_env().await?;
    let master_key = config.secrets.master_key().ok_or_else(|| {
        anyhow::anyhow!(
            "SECRETS_MASTER_KEY not set. Run 'ironclaw onboard' first or set it in .env"
        )
    })?;

    let crypto = Arc::new(crate::secrets::SecretsCrypto::new(master_key.clone())?);

    Ok(crate::db::create_secrets_store(&config.database, crypto).await?)
}
```

### Graceful Degradation (DB Optional)

```rust
pub async fn run_config_command(cmd: ConfigCommand) -> anyhow::Result<()> {
    // Try to connect to the DB for settings access
    let db: Option<Arc<dyn crate::db::Database>> = match connect_db().await {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!(
                "Warning: Could not connect to database ({}), using disk fallback",
                e
            );
            None
        }
    };

    let db_ref = db.as_deref();
    match cmd {
        ConfigCommand::Init { output, force } => init_toml(db_ref, output, force).await,
        ConfigCommand::List { filter } => list_settings(db_ref, filter).await,
        // ...
    }
}
```

### Output Formatting Utilities

```rust
/// Truncate a string to a maximum character length.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(2)).collect();
        format!("{}..", truncated)
    }
}

/// Format bytes as human-readable size.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;

    if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format a datetime relative to now (e.g. "in 2h", "3m ago").
fn format_relative(dt: DateTime<Utc>) -> String {
    let now = Utc::now();
    let diff = dt.signed_duration_since(now);
    let secs = diff.num_seconds();

    if secs.abs() < 60 {
        if secs >= 0 { "in <1m" } else { "<1m ago" }.to_string()
    } else if secs.abs() < 3600 {
        let mins = secs.abs() / 60;
        if secs >= 0 {
            format!("in {}m", mins)
        } else {
            format!("{}m ago", mins)
        }
    } else if secs.abs() < 86400 {
        let hours = secs.abs() / 3600;
        if secs >= 0 {
            format!("in {}h", hours)
        } else {
            format!("{}h ago", hours)
        }
    } else {
        let days = secs.abs() / 86400;
        if secs >= 0 {
            format!("in {}d", days)
        } else {
            format!("{}d ago", days)
        }
    }
}
```

### Error Handling Pattern

```rust
// Validate and provide actionable error messages
let tool_name = name.unwrap_or_else(|| {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
});

// Check for name conflicts
if db.get_routine_by_name(user_id, name).await?.is_some() {
    anyhow::bail!("Routine '{}' already exists", name);
}

// Validate cron expression
let next_fire = next_cron_fire(schedule, timezone)
    .map_err(|e| anyhow::anyhow!("Invalid cron schedule: {e}"))?;
```

### Testing CLI Commands

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_command_parsing() {
        // Verify command structure is valid
        #[derive(clap::Parser)]
        struct TestCli {
            #[command(subcommand)]
            cmd: XxxCommand,
        }

        TestCli::command().debug_assert();
    }

    #[test]
    fn test_helper_function() {
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
    }

    #[tokio::test]
    async fn test_command_execution() {
        // Use test harness for DB-backed commands
        let harness = crate::testing::TestHarnessBuilder::new().build().await;
        let db = harness.db.clone();
        
        // Execute command
        run_xxx_command(cmd, db, "test_user").await.unwrap();
        
        // Verify side effects
        // ...
    }
}
```

---

## Security Considerations

### 1. Secret Handling

- Never log secret values (only names)
- Use constant-time comparison for secret validation
- Hide input during manual entry (crossterm raw mode)
- Validate secret format before saving

### 2. Path Traversal Prevention

```rust
fn validate_tool_name(name: &str) -> anyhow::Result<()> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.contains('\0')
    {
        anyhow::bail!(
            "Invalid tool name '{}': must not contain path separators or '..'",
            name
        );
    }
    Ok(())
}
```

### 3. Input Validation

- Validate transport types against whitelist
- Check required arguments before proceeding
- Validate cron expressions, timezones, etc.
- Sanitize user input in error messages

### 4. Database Fallback

- Gracefully degrade when DB unavailable
- Never crash on DB connection failure
- Warn user when using disk fallback
- Require DB for write operations (explicit error)

---

## Related Files

- `src/cli/mod.rs` - Main CLI structure (427 lines)
- `src/cli/config.rs` - Configuration management (322 lines)
- `src/cli/tool.rs` - WASM tool management (1400 lines)
- `src/cli/mcp.rs` - MCP server management (704 lines)
- `src/cli/memory.rs` - Workspace memory commands (295 lines)
- `src/cli/routines.rs` - Routine management (802 lines)
- `src/cli/skills.rs` - SKILL.md management (375 lines)
- `src/cli/service.rs` - OS service management (37 lines)
- `src/cli/completion.rs` - Shell completions (78 lines)
- `src/cli/oauth_defaults.rs` - OAuth helper functions

## Related Context

- **Technical Domain** → `../../project-intelligence/technical-domain.md`
- **Code Quality** → `../../core/standards/code-quality.md`
- **Documentation Standards** → `../../core/standards/documentation.md`
- **Security Patterns** → `../../core/standards/security-patterns.md` (if exists)
