# Secure Setup Guide

> **Purpose**: Comprehensive security-focused configuration guide for production deployments of IronClaw. Covers environment variables, database hardening, secrets management, and security best practices.

---

## Table of Contents

- [Part 1: Introduction & Overview](#part-1-introduction--overview)
- [Part 2: Environment Variables (Complete Reference)](#part-2-environment-variables-complete-reference)
- [Part 3: Database Configuration](#part-3-database-configuration)
- [Part 4: Channel Security Configuration](#part-4-channel-security-configuration)
- [Part 5: Secrets Management](#part-5-secrets-management)
- [Part 6: Production Hardening](#part-6-production-hardening)
- [Part 7: Verification & Troubleshooting](#part-7-verification--troubleshooting)
- [Appendix A: Quick Reference](#appendix-a-quick-reference)
- [Appendix B: Security Checklist](#appendix-b-security-checklist)

---

## Part 1: Introduction & Overview

### Why This Guide Exists

IronClaw is designed with **security-first principles**. Unlike cloud AI assistants that store your data on remote servers, IronClaw keeps everything local and encrypted. However, proper configuration is essential to maintain this security posture.

This guide covers:

- **Environment variables** — Complete reference for all configuration options
- **Database hardening** — PostgreSQL and libSQL security configurations
- **Secrets management** — Encryption, keychain integration, and rotation
- **Network security** — Tunnel configuration, webhook validation, and allowlisting
- **Production checklists** — Step-by-step verification for deployments

### Security Model

IronClaw implements **defense in depth** across multiple layers:

```
┌────────────────────────────────────────────────────────────────┐
│                      External Requests                          │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  Tunnel / HTTP    │  TLS termination       │
│                    └─────────┬─────────┘                        │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  Webhook Secret   │  HMAC validation       │
│                    │    Validation     │                        │
│                    └─────────┬─────────┘                        │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  Channel Sandbox  │  WASM isolation        │
│                    │   (Capability-    │                        │
│                    │    based perms)   │                        │
│                    └─────────┬─────────┘                        │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  Secret Injector  │  Credential injection  │
│                    │  (Leak detection) │  at host boundary      │
│                    └─────────┬─────────┘                        │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  Database Layer   │  Encrypted storage     │
│                    │  (AES-256-GCM)    │                        │
│                    └───────────────────┘                        │
└────────────────────────────────────────────────────────────────┘
```

### Key Security Features

| Layer | Protection | Implementation |
|-------|------------|----------------|
| **Network** | TLS encryption | Tunnel providers (ngrok, Cloudflare) |
| **Authentication** | Webhook secrets | HMAC token validation |
| **Isolation** | WASM sandbox | Capability-based permissions |
| **Secrets** | Encryption at rest | AES-256-GCM with master key |
| **Credentials** | Leak detection | Pattern scanning on I/O |
| **Database** | Access control | Role-based permissions |

### Prerequisites

Before proceeding with secure setup:

- [ ] IronClaw installed (`ironclaw onboard` completed)
- [ ] PostgreSQL 15+ with `pgvector` extension **OR** libSQL configured
- [ ] OS keychain available (macOS Keychain, Linux Secret Service, Windows Credential Manager)
- [ ] Tunnel provider account (ngrok, Cloudflare, or similar) for webhook channels
- [ ] Text editor for configuration files

### Quick Start: Secure Defaults

For most users, the setup wizard configures secure defaults:

```bash
# Run the interactive setup wizard
ironclaw onboard
```

The wizard will:

1. Configure database connection with proper permissions
2. Generate and store master key in OS keychain
3. Set up encrypted secrets storage
4. Configure webhook secrets for channels
5. Enable prompt injection defenses

**For production deployments**, continue reading to harden each layer.

---

## Part 2: Environment Variables (Complete Reference)

### Overview

IronClaw reads configuration from multiple sources in this order (lowest to highest priority):

1. `~/.ironclaw/.env` — Bootstrap variables (database backend, paths)
2. System environment variables — Shell or process environment
3. Database settings table — Runtime configuration
4. CLI flags — Override everything for single commands

### Bootstrap Variables (`~/.ironclaw/.env`)

These variables are **required before database connection** and must be in the `.env` file:

| Variable | Values | Required | Description |
|----------|--------|----------|-------------|
| `DATABASE_BACKEND` | `postgres`, `libsql` | Yes | Database backend selection |
| `DATABASE_URL` | PostgreSQL connection string | If `postgres` | Full connection URL with credentials |
| `LIBSQL_PATH` | Filesystem path | If `libsql` | Local database file path (default: `~/.ironclaw/ironclaw.db`) |
| `LIBSQL_URL` | Turso sync URL | Optional | Remote sync endpoint for libSQL |

**Example `.env` file (PostgreSQL):**
```env
DATABASE_BACKEND="postgres"
DATABASE_URL="postgres://ironclaw:secure_password@localhost:5432/ironclaw"
```

**Example `.env` file (libSQL):**
```env
DATABASE_BACKEND="libsql"
LIBSQL_PATH="/home/user/.ironclaw/ironclaw.db"
```

### Database Connection Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `DATABASE_URL` | `postgres://user:pass@host:port/db` | — | PostgreSQL connection string |
| `DATABASE_POOL_SIZE` | Integer | `10` | Maximum database connections |
| `DATABASE_TIMEOUT_SECS` | Integer | `30` | Connection timeout |
| `LIBSQL_PATH` | Filesystem path | `~/.ironclaw/ironclaw.db` | Local libSQL file |
| `LIBSQL_URL` | `https://...` | — | Turso remote sync URL |
| `LIBSQL_AUTH_TOKEN` | JWT token | — | Turso authentication |

**PostgreSQL URL format:**
```
postgres://[user[:password]@][host][:port][/dbname][?params]
```

**Example with SSL:**
```env
DATABASE_URL="postgres://ironclaw:pass@localhost:5432/ironclaw?sslmode=require"
```

### Security & Secrets Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `SECRETS_MASTER_KEY` | 32-byte hex string | — | Master encryption key (alternative to keychain) |
| `SECRETS_BACKEND` | `auto`, `postgres`, `libsql` | `auto` | Secrets storage backend |
| `ALLOW_KEYCHAIN` | `true`, `false` | `true` | Enable OS keychain integration |

**Generating a master key:**
```bash
# Generate a random 32-byte key
openssl rand -hex 32
```

**Using master key (not recommended for production):**
```env
SECRETS_MASTER_KEY="a1b2c3d4e5f6..."  # 64 hex characters
```

> **Security Note**: Prefer OS keychain over `SECRETS_MASTER_KEY` env var. The keychain provides hardware-backed encryption and access control. Only use env var for headless deployments without keychain support.

### LLM Provider Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `LLM_BACKEND` | `nearai`, `anthropic`, `openai`, `ollama`, `openai_compatible` | — | Primary LLM provider |
| `NEARAI_SESSION_TOKEN` | JWT token | — | NEAR AI authentication (from `ironclaw onboard`) |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | — | Anthropic API key |
| `OPENAI_API_KEY` | `sk-...` | — | OpenAI API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `http://localhost:11434` | Ollama endpoint |
| `LLM_COMPATIBLE_BASE_URL` | Custom URL | — | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | API key | — | For OpenAI-compatible providers |

**Example configurations:**

**Anthropic:**
```env
LLM_BACKEND="anthropic"
ANTHROPIC_API_KEY="sk-ant-api03-..."
```

**Ollama (local, no auth):**
```env
LLM_BACKEND="ollama"
OLLAMA_BASE_URL="http://localhost:11434"
```

**OpenAI-compatible:**
```env
LLM_BACKEND="openai_compatible"
LLM_COMPATIBLE_BASE_URL="https://api.example.com/v1"
LLM_API_KEY="your-api-key"
```

### Embeddings Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `EMBEDDING_ENABLED` | `true`, `false` | `true` | Enable semantic search |
| `EMBEDDING_PROVIDER` | `nearai`, `openai` | — | Embeddings provider |
| `EMBEDDING_MODEL` | Model ID | `text-embedding-3-small` | Embedding model |

**Example:**
```env
EMBEDDING_ENABLED="true"
EMBEDDING_PROVIDER="openai"
EMBEDDING_MODEL="text-embedding-3-small"
```

### Channel & Network Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `TUNNEL_URL` | HTTPS URL | — | Public webhook endpoint |
| `TUNNEL_PROVIDER` | `ngrok`, `cloudflare`, `localtunnel` | — | Tunnel service |
| `HTTP_WEBHOOK_ENABLED` | `true`, `false` | `false` | Enable HTTP webhook channel |
| `HTTP_WEBHOOK_SECRET` | Secret token | Auto-generated | Webhook HMAC secret |
| `HTTP_WEBHOOK_PATH` | URL path | `/webhook` | Webhook endpoint path |
| `REPL_ENABLED` | `true`, `false` | `true` | Enable REPL channel |
| `WEB_GATEWAY_ENABLED` | `true`, `false` | `false` | Enable web UI gateway |
| `WEB_GATEWAY_PORT` | Port number | `8080` | Web gateway listen port |

**Example (Cloudflare Tunnel):**
```env
TUNNEL_PROVIDER="cloudflare"
TUNNEL_URL="https://abc123.ngrok-free.app"
HTTP_WEBHOOK_ENABLED="true"
HTTP_WEBHOOK_SECRET="your-webhook-secret"
```

### Channel-Specific Variables

**Telegram:**
```env
TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
TELEGRAM_WEBHOOK_SECRET="optional-hmac-secret"
```

**Slack:**
```env
SLACK_BOT_TOKEN="xoxb-..."
SLACK_SIGNING_SECRET="..."
```

### WASM Sandbox Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `WASM_ENABLED` | `true`, `false` | `true` | Enable WASM tool sandbox |
| `WASM_CHANNELS_DIR` | Directory path | `~/.ironclaw/channels` | WASM channel location |
| `WASM_TOOLS_DIR` | Directory path | `~/.ironclaw/tools` | WASM tools location |
| `WASM_MEMORY_LIMIT_MB` | Integer | `128` | Per-WASM memory limit |
| `WASM_TIMEOUT_MS` | Integer | `30000` | Per-WASM execution timeout |

### Security Hardening Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `PROMPT_INJECTION_DEFENSE` | `enabled`, `disabled` | `enabled` | Enable injection detection |
| `CONTENT_SANITIZATION` | `enabled`, `disabled` | `enabled` | Sanitize external content |
| `ENDPOINT_ALLOWLIST_STRICT` | `true`, `false` | `true` | Strict HTTP allowlisting |
| `SECRET_LEAK_DETECTION` | `enabled`, `disabled` | `enabled` | Scan for credential leaks |
| `AUDIT_LOG_ENABLED` | `true`, `false` | `true` | Log all tool executions |

**Production hardening:**
```env
PROMPT_INJECTION_DEFENSE="enabled"
CONTENT_SANITIZATION="enabled"
ENDPOINT_ALLOWLIST_STRICT="true"
SECRET_LEAK_DETECTION="enabled"
AUDIT_LOG_ENABLED="true"
```

### Logging & Debug Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `RUST_LOG` | Log level | `info` | Global log level |
| `IRONCLAW_LOG` | Log level | `info` | IronClaw-specific log level |
| `LOG_FORMAT` | `json`, `pretty` | `pretty` | Log output format |
| `LOG_FILE` | File path | — | Write logs to file |

**Debug mode:**
```env
RUST_LOG="ironclaw=debug,wasm=debug"
LOG_FORMAT="json"
LOG_FILE="/var/log/ironclaw/ironclaw.log"
```

### Heartbeat & Background Tasks

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `HEARTBEAT_ENABLED` | `true`, `false` | `false` | Enable periodic execution |
| `HEARTBEAT_INTERVAL_MINS` | Integer | `30` | Interval between heartbeats |
| `HEARTBEAT_NOTIFY_CHANNEL` | Channel name | — | Send heartbeat reports to channel |

**Example:**
```env
HEARTBEAT_ENABLED="true"
HEARTBEAT_INTERVAL_MINS="60"
HEARTBEAT_NOTIFY_CHANNEL="telegram"
```

### Environment Variable Priority

When the same variable is set in multiple places:

```
CLI flags > Process env > ~/.ironclaw/.env > Database settings > Defaults
```

**Example:**
```bash
# This overrides DATABASE_URL in .env file
DATABASE_URL="postgres://override@host/db" ironclaw run
```

---

## Part 3: Database Configuration

### Overview

IronClaw supports two database backends:

| Backend | Use Case | Security Features |
|---------|----------|-------------------|
| **PostgreSQL** | Production, multi-user | Row-level security, SSL, role-based access |
| **libSQL** | Single-user, local-first | File permissions, optional Turso sync |

### PostgreSQL Configuration

#### Installation

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS (Homebrew)
brew install postgresql@15

# Enable pgvector extension
sudo apt install postgresql-15-pgvector  # Debian/Ubuntu
# OR build from source: https://github.com/pgvector/pgvector
```

#### Database Setup

```bash
# Connect as postgres superuser
sudo -u postgres psql

# Create database and user
CREATE DATABASE ironclaw;
CREATE USER ironclaw WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE ironclaw TO ironclaw;

# Enable pgvector extension
\c ironclaw
CREATE EXTENSION IF NOT EXISTS vector;

# Set up schema (optional, for multi-tenant isolation)
CREATE SCHEMA IF NOT EXISTS ironclaw_schema;
GRANT ALL ON SCHEMA ironclaw_schema TO ironclaw;
```

#### Connection String Format

```
postgres://user:password@host:port/database?param=value
```

**Common parameters:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `sslmode` | `disable`, `require`, `verify-full` | SSL/TLS encryption |
| `connect_timeout` | Seconds | Connection timeout |
| `application_name` | String | App identifier in logs |

**Secure connection string (production):**
```env
DATABASE_URL="postgres://ironclaw:secure_pass@localhost:5432/ironclaw?sslmode=require&application_name=ironclaw"
```

#### PostgreSQL Hardening

**1. Restrict network access:**

Edit `postgresql.conf`:
```conf
listen_addresses = 'localhost'  # Only local connections
# OR for remote access with SSL:
# listen_addresses = '*'
```

Edit `pg_hba.conf`:
```conf
# Local connections only (most secure)
local   ironclaw   ironclaw   md5

# Remote connections with SSL required
hostssl ironclaw   ironclaw   0.0.0.0/0   md5
```

**2. Create read-only user for backups:**

```sql
CREATE USER ironclaw_readonly WITH PASSWORD 'backup_password';
GRANT CONNECT ON DATABASE ironclaw TO ironclaw_readonly;
GRANT USAGE ON SCHEMA public TO ironclaw_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ironclaw_readonly;
```

**3. Enable audit logging:**

In `postgresql.conf`:
```conf
log_statement = 'all'           # Log all statements
log_duration = on               # Log query duration
log_connections = on            # Log connection attempts
log_disconnections = on         # Log disconnections
```

**4. Set connection limits:**

```sql
ALTER USER ironclaw CONNECTION LIMIT 20;
```

#### PostgreSQL Pool Configuration

IronClaw uses connection pooling. Configure based on expected load:

```env
DATABASE_POOL_SIZE="20"         # Max connections (default: 10)
DATABASE_TIMEOUT_SECS="30"      # Connection timeout
```

**Sizing guidelines:**

| Workload | Pool Size | Notes |
|----------|-----------|-------|
| Single user | `5-10` | Default is fine |
| Multi-user (5-10) | `20-30` | Increase for concurrent channels |
| High throughput | `50+` | Monitor PostgreSQL `max_connections` |

### libSQL Configuration

#### Local Database Setup

libSQL requires no setup — the database file is created automatically:

```env
DATABASE_BACKEND="libsql"
LIBSQL_PATH="/home/user/.ironclaw/ironclaw.db"
```

**Default location:** `~/.ironclaw/ironclaw.db`

#### File Permissions (Linux/macOS)

Secure the database file:

```bash
# Set restrictive permissions (owner read/write only)
chmod 600 ~/.ironclaw/ironclaw.db

# Ensure directory is also protected
chmod 700 ~/.ironclaw/
```

#### Turso Remote Sync (Optional)

For backup and multi-device sync:

1. Create database at [Turso](https://turso.tech)
2. Get database URL and auth token
3. Configure:

```env
DATABASE_BACKEND="libsql"
LIBSQL_PATH="/home/user/.ironclaw/ironclaw.db"
LIBSQL_URL="libsql://your-db.turso.io"
LIBSQL_AUTH_TOKEN="your-auth-token"
```

**Sync behavior:**
- Local reads/writes are immediate
- Sync to Turso happens asynchronously
- Conflicts resolved with last-write-wins

#### libSQL Hardening

**1. Filesystem encryption:**

Use full-disk encryption (LUKS on Linux, FileVault on macOS, BitLocker on Windows) to protect the database file at rest.

**2. Backup encryption:**

```bash
# Create encrypted backup
sqlite3 ~/.ironclaw/ironclaw.db ".backup '/backup/ironclaw.db'"
gpg -c /backup/ironclaw.db  # Encrypt with passphrase
rm /backup/ironclaw.db      # Remove unencrypted copy
```

### Database Migrations

IronClaw runs migrations automatically on first connection. Migrations create:

- `settings` table — Configuration key-value store
- `secrets` table — Encrypted secrets storage
- `workspace` tables — Memory and file storage
- `jobs` table — Job queue and execution history
- `channels` table — Channel state and metadata
- `pairing_codes` table — DM pairing allowlist

**Manual migration check:**
```bash
ironclaw db migrate --check
```

### Database Backup Strategies

#### PostgreSQL Backup

**Logical backup (pg_dump):**
```bash
# Full database backup
pg_dump -U ironclaw ironclaw > ironclaw_backup.sql

# Compressed backup
pg_dump -U ironclaw ironclaw | gzip > ironclaw_backup.sql.gz

# Restore
psql -U ironclaw ironclaw < ironclaw_backup.sql
```

**Point-in-time recovery (PITR):**

Enable WAL archiving in `postgresql.conf`:
```conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

#### libSQL Backup

**File copy (while IronClaw is stopped):**
```bash
# Stop IronClaw first!
cp ~/.ironclaw/ironclaw.db /backup/ironclaw-$(date +%F).db
```

**Turso automatic backups:**

Enable in Turso dashboard — daily snapshots retained for 30 days.

### Database Monitoring

#### PostgreSQL

**Check connection count:**
```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'ironclaw';
```

**Check table sizes:**
```sql
SELECT
  relname AS table_name,
  pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

**Check pgvector index size:**
```sql
SELECT
  indexname,
  pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'workspace_embeddings';
```

#### libSQL

**Check file size:**
```bash
ls -lh ~/.ironclaw/ironclaw.db
```

**Query stats (via SQLite):**
```bash
sqlite3 ~/.ironclaw/ironclaw.db "SELECT * FROM pragma_database_list;"
```

### Troubleshooting

#### PostgreSQL Connection Errors

**"Connection refused":**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check listening port
sudo netstat -tlnp | grep 5432
```

**"Authentication failed":**
```bash
# Verify credentials
psql "postgres://ironclaw:password@localhost/ironclaw"

# Check pg_hba.conf allows your connection type
```

**"pgvector extension not found":**
```bash
# Install pgvector
sudo apt install postgresql-15-pgvector

# Enable extension
psql ironclaw -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### libSQL Errors

**"Database locked":**
- Ensure only one IronClaw instance is running
- Check for stale lock files in `~/.ironclaw/`

**"Disk I/O error":**
- Check disk space: `df -h ~/.ironclaw/`
- Verify file permissions: `ls -la ~/.ironclaw/`

#### Migration Failures

**Reset and re-run migrations:**
```bash
# PostgreSQL: drop and recreate database (WARNING: deletes all data!)
psql -c "DROP DATABASE ironclaw;"
psql -c "CREATE DATABASE ironclaw;"
psql ironclaw -c "CREATE EXTENSION vector;"

# Then re-run onboarding
ironclaw onboard
```

---

## Part 4: Channel Security Configuration

### Overview

Channels are input/output endpoints for IronClaw. Each has distinct security considerations:

| Channel | Transport | Auth Method | Risk Level |
|---------|-----------|-------------|------------|
| HTTP | Webhook (HTTPS) | Secret token | Medium |
| Telegram | Bot API | Bot token + pairing | Medium |
| Slack | Socket/Webhook | App token + signing secret | Medium |
| Gateway | HTTP API | Bearer token | High |
| CLI | Local stdin/stdout | OS user | Low |

---

### HTTP Channel

The HTTP channel receives webhooks from external services.

#### Configuration

```bash
HTTP_HOST=0.0.0.0          # Bind address
HTTP_PORT=8080             # Port
HTTP_WEBHOOK_SECRET=<secret>  # Shared secret for validation
HTTP_USER_ID=http          # User ID for messages
```

#### Webhook Secret Setup

**Via onboarding wizard** (recommended):
```bash
ironclaw onboard
# Step 6: Enable HTTP channel
# → Auto-generates secure random secret
# → Stores and uses automatically
```

**Manual setup** (environment variable):
```bash
# Generate a secure secret
openssl rand -hex 32
# Example output: a3f8b2c1d4e5f6789012345678901234567890abcdef1234567890abcdef

# Add to .env file (persistent across restarts)
echo "HTTP_WEBHOOK_SECRET=$(openssl rand -hex 32)" >> .env

# Or export for current session only
export HTTP_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

> **Note**: There is no `ironclaw secrets` CLI command. Secrets are managed via:
> - Onboarding wizard (recommended for initial setup)
> - Environment variables (for manual configuration)
> - OS keychain (automatic for master key)

#### Tunnel Configuration (for external webhooks)

Webhooks require HTTPS. Use a tunnel for development:

**ngrok**:
```bash
ngrok http 8080
# Copy the HTTPS URL (e.g., https://abc123.ngrok.io)
```

**Cloudflare Tunnel**:
```bash
cloudflared tunnel --url http://localhost:8080
```

**Set tunnel URL**:
```bash
# In settings or via env
TUNNEL_URL=https://abc123.ngrok.io
```

#### Security Checklist

- [ ] `HTTP_WEBHOOK_SECRET` is 32+ characters, randomly generated
- [ ] Tunnel URL starts with `https://`
- [ ] Secret is stored in secrets store, not `.env` (production)
- [ ] Firewall restricts `HTTP_PORT` to tunnel provider IPs (production)

#### Example: GitHub Webhook

```json
// GitHub webhook payload → HTTP channel
POST /webhook
X-Webhook-Secret: <your-secret>
Content-Type: application/json

{
  "action": "opened",
  "issue": { ... }
}
```

The host validates `X-Webhook-Secret` before forwarding to the channel.

---

### Telegram Channel

Telegram provides instant messaging via bot DMs and groups.

#### Configuration

```bash
TELEGRAM_OWNER_ID=123456789  # Restrict to this user (optional)
# Bot token stored in secrets store as `telegram_bot_token`
# Webhook secret stored as `telegram_webhook_secret`
```

#### Bot Token Setup

1. **Create bot via [@BotFather](https://t.me/BotFather)**:
   ```
   /newbot
   → Choose name and username
   → Copy token: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

2. **Store securely**:
   ```bash
   # Via wizard (recommended)
   ironclaw onboard  # Step 6: Enable Telegram, paste token

   # Or via environment variable
   export TELEGRAM_BOT_TOKEN=<your-token>
   # Or add to .env file (persistent)
   echo "TELEGRAM_BOT_TOKEN=<your-token>" >> .env
   ```

#### DM Pairing (Access Control)

When an unknown user DMs your bot, they receive a pairing code. You must approve them:

```
User: Hello!
Bot:  To pair with this bot, run: ironclaw pairing approve telegram ABC12345
```

**Commands**:
```bash
# List pending requests
ironclaw pairing list telegram

# Approve a user
ironclaw pairing approve telegram ABC12345

# List approved users
ironclaw pairing list telegram --json
```

#### DM Policy Modes

Configure in `~/.ironclaw/channels/telegram.capabilities.json`:

| Mode | Behavior |
|------|----------|
| `pairing` (default) | Unknown users get pairing code; approved users can message |
| `allowlist` | Only pre-approved IDs can message; no pairing reply |
| `open` | All users can message (not recommended) |

```json
{
  "config": {
    "dm_policy": "pairing",
    "allow_from": ["123456789", "@username"],
    "owner_id": "123456789"
  }
}
```

#### Owner Restriction

For maximum security, set `owner_id` to allow only one user:

```json
{
  "config": {
    "owner_id": "123456789"  // Only this Telegram user ID can interact
  }
}
```

#### Group Mentions

To use IronClaw in groups:

1. **Add bot to group** and grant necessary permissions
2. **Set bot username** in config:
   ```json
   {
     "config": {
       "bot_username": "YourBotName"
     }
   }
   ```
3. **Trigger via mention**: `@YourBotName what's the weather?`
4. **Or via command**: `/command what's the weather?`

**Respond to all messages** (not just mentions):
```json
{
  "config": {
    "respond_to_all_group_messages": true
  }
}
```

#### Security Checklist

- [ ] Bot token stored in secrets store
- [ ] `dm_policy` set to `pairing` or `allowlist`
- [ ] `owner_id` configured for single-user bots
- [ ] Webhook secret set (if using webhook mode)
- [ ] Bot removed from unnecessary groups

---

### Slack Channel

Slack integration uses the Socket Mode API for real-time messaging.

#### Configuration

Credentials stored in secrets store:
- `slack_bot_token` — Bot User OAuth Token (`xoxb-...`)
- `slack_app_token` — App-Level Token (`xapp-...`)
- `slack_signing_secret` — Signing Secret (for webhooks)

#### Slack App Setup

1. **Create app** at [api.slack.com/apps](https://api.slack.com/apps)
2. **Enable Socket Mode**:
   - Settings → Socket Mode → Enable
   - Generate App-Level Token (`xapp-...`) with `connections:write` scope
3. **Install to workspace**:
   - OAuth & Permissions → Install to Workspace
   - Copy Bot User OAuth Token (`xoxb-...`)
4. **Subscribe to events**:
   - Event Subscriptions → Subscribe to bot events
   - Add: `message.im`, `message.groups`, `message.channels`
5. **Get signing secret**:
   - Settings → Basic Information → App Credentials → Signing Secret

#### Store Credentials

```bash
# Via wizard (recommended)
ironclaw onboard  # Step 6: Enable Slack, paste tokens

# Or via environment variables
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
export SLACK_SIGNING_SECRET=<secret>
# Or add to .env file (persistent)
echo "SLACK_BOT_TOKEN=xoxb-..." >> .env
echo "SLACK_APP_TOKEN=xapp-..." >> .env
echo "SLACK_SIGNING_SECRET=<secret>" >> .env
```

#### Security Checklist

- [ ] All three tokens stored in secrets store
- [ ] App installed only to necessary workspaces
- [ ] Bot permissions follow least privilege
- [ ] Signing secret rotated periodically

---

### Gateway Channel

The Gateway channel exposes an HTTP API for programmatic access.

#### Configuration

```bash
GATEWAY_ENABLED=true
GATEWAY_HOST=127.0.0.1      # Bind to localhost only (recommended)
GATEWAY_PORT=3000
GATEWAY_AUTH_TOKEN=<token>  # Auto-generated if not set
GATEWAY_USER_ID=default
```

#### Auth Token

**Auto-generated** (default behavior):
```bash
# Leave GATEWAY_AUTH_TOKEN unset
# IronClaw generates a secure random token on startup
# Token printed to logs: "Web gateway enabled on 127.0.0.1:3003/?token=..."
# Copy and save securely for API calls
```

**Manual setup** (persistent across restarts):
```bash
# Generate secure token
openssl rand -hex 32

# Set via environment variable
export GATEWAY_AUTH_TOKEN=<your-token>
# Or add to .env file (persistent)
echo "GATEWAY_AUTH_TOKEN=<your-token>" >> .env
```

#### Network Isolation

**Development**:
```bash
GATEWAY_HOST=127.0.0.1  # Localhost only
```

**Production (behind reverse proxy)**:
```bash
GATEWAY_HOST=0.0.0.0    # All interfaces
# Firewall restricts to reverse proxy IP
# Reverse proxy handles TLS and rate limiting
```

#### API Usage

```bash
# Send message to IronClaw
curl -X POST http://localhost:3000/messages \
  -H "Authorization: Bearer <GATEWAY_AUTH_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"content": "What is the status?", "thread_id": "abc123"}'

# Response
{
  "message_id": "msg_123",
  "status": "queued"
}
```

#### Security Checklist

- [ ] `GATEWAY_HOST=127.0.0.1` unless behind reverse proxy
- [ ] `GATEWAY_AUTH_TOKEN` is 32+ characters, randomly generated
- [ ] Token stored in secrets store
- [ ] Firewall restricts `GATEWAY_PORT` (production)
- [ ] TLS termination at reverse proxy (production)

---

### CLI Channel

The CLI channel provides local terminal interaction.

#### Configuration

```bash
CLI_ENABLED=true  # Default: enabled
```

#### Security Model

- **No network exposure** — stdin/stdout only
- **OS user isolation** — Only the running user can interact
- **No auth token needed** — Relies on OS permissions

#### Usage

```bash
# Start IronClaw with CLI channel
ironclaw run

# Interact via terminal
> What's the weather?
< The weather is sunny...
```

#### Security Checklist

- [ ] Terminal session is locked when unattended
- [ ] No sensitive data in shell history (if logged)
- [ ] File permissions on `~/.ironclaw/` restrict access

---

## Part 5: Secrets Management

### Architecture

IronClaw uses a layered secrets system:

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│  (channels, LLM providers, database)    │
├─────────────────────────────────────────┤
│    SecretsContext (encrypted access)    │
├─────────────────────────────────────────┤
│         SecretsCrypto (AES-256-GCM)     │
│         Master Key (32 bytes)           │
├─────────────────────────────────────────┤
│         Secrets Store (database)        │
│         Encrypted key-value pairs       │
└─────────────────────────────────────────┘
```

### Master Key Management

The master key encrypts all secrets. It must be secured first.

#### Option 1: OS Keychain (Recommended)

**How it works**:
- Key stored in OS-managed secure storage
- Auto-unlocks on user login
- No environment variables needed

**Platform locations**:
| OS | Storage |
|----|---------|
| macOS | Keychain (login keychain) |
| Windows | Credential Manager |
| Linux | GNOME Keyring / KWallet (via libsecret) |

**Setup**:
```bash
# Automatic during onboarding
ironclaw onboard
# Step 2: Choose "OS Keychain"
# → Generates 32-byte key
# → Stores in keychain
```

**Verify**:
```bash
# Check if keychain key exists (may trigger system dialog)
ironclaw status
# Output: "Master key: OS keychain (configured)"
```

#### Option 2: Environment Variable

**How it works**:
- Key set via `SECRETS_MASTER_KEY` env var
- Must be loaded before IronClaw starts
- Not persisted; must be set each session

**Generate**:
```bash
export SECRETS_MASTER_KEY=$(openssl rand -hex 32)
# Example: SECRETS_MASTER_KEY=a1b2c3d4... (64 hex chars)
```

**Persist** (development only):
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export SECRETS_MASTER_KEY="<your-key>"' >> ~/.zshrc
source ~/.zshrc
```

**Security warning**: Environment variables can leak via:
- Process listings (`ps aux`)
- Core dumps
- Log files
- Child process inheritance

#### Option 3: Skip (Not Recommended)

```bash
# During onboarding, choose "Skip"
# Secrets features disabled
# All credentials stored unencrypted (insecure!)
```

### Secret Lifecycle

#### Creating Secrets

**Via onboarding wizard** (recommended for initial setup):
```bash
ironclaw onboard
# Step 2: Security setup (master key)
# Step 6: Channel configuration
# → Prompts for required secrets (bot tokens, webhook secrets)
# → Stores encrypted automatically in database
```

**Via environment variables** (for manual configuration):
```bash
# Add to .env file (persistent across restarts)
echo "MY_SECRET_NAME=<value>" >> .env

# Or export for current session
export MY_SECRET_NAME=<value>
```

> **Note**: IronClaw does **not** have a `secrets` CLI subcommand. Secrets are managed through:
> - Onboarding wizard (initial setup)
> - Environment variables (manual configuration)
> - Direct database operations (advanced, not covered here)

#### Reading Secrets

Secrets are automatically loaded at runtime from:
1. Environment variables (highest priority)
2. Encrypted secrets store (database)
3. Default values (lowest priority)

To view configured secrets:
```bash
# Check environment variables
cat .env | grep -v "^#" | grep -v "^$"

# Query database directly (advanced)
psql "$DATABASE_URL" -c "SELECT name, created_at FROM secrets WHERE user_id='default';"
```

#### Rotating Secrets

**Best practice**: Rotate every 90 days or after suspected exposure.

**For channel tokens** (Telegram, Slack):
```bash
# 1. Generate new token
NEW_TOKEN=$(openssl rand -hex 32)

# 2. Update in external system
# → Telegram: regenerate via @BotFather
# → GitHub: update webhook secret in repo settings

# 3. Update environment variable
# Edit .env file manually:
# HTTP_WEBHOOK_SECRET=<new-token>

# 4. Restart IronClaw
./target/release/ironclaw restart
```

**For master key** (requires re-onboarding):
```bash
# Backup existing secrets first!
# Then re-run onboarding with new master key
ironclaw onboard
```

#### Revoking Secrets

```bash
# Remove from .env file
sed -i '/MY_SECRET_NAME/d' .env

# Or delete from database directly (advanced)
psql "$DATABASE_URL" -c "DELETE FROM secrets WHERE name='my_secret_name' AND user_id='default';"
```

**Warning**: Revoking a secret breaks channels/features that depend on it.

### Secret Names by Category

#### LLM Providers

| Secret Name | Provider | Required |
|-------------|----------|----------|
| `nearai_api_key` | NEAR AI | No (session auth default) |
| `openai_api_key` | OpenAI | Yes (if backend=openai) |
| `anthropic_api_key` | Anthropic | Yes (if backend=anthropic) |
| `llm_compatible_api_key` | OpenAI-compatible | No |

#### Channels

| Secret Name | Channel | Required |
|-------------|---------|----------|
| `telegram_bot_token` | Telegram | Yes |
| `telegram_webhook_secret` | Telegram | No (recommended) |
| `slack_bot_token` | Slack | Yes |
| `slack_app_token` | Slack | Yes |
| `slack_signing_secret` | Slack | Yes |
| `http_webhook_secret` | HTTP | No (recommended) |
| `gateway_auth_token` | Gateway | No (auto-generated) |

#### Database

| Secret Name | Backend | Required |
|-------------|---------|----------|
| `database_url` | PostgreSQL | Yes (production) |
| `libsql_auth_token` | libSQL (Turso) | Yes (if using cloud sync) |

### Best Practices

#### 1. Use OS Keychain

```bash
# Recommended setup
ironclaw onboard
# → Choose "OS Keychain" for master key
# → All secrets encrypted and stored securely
```

#### 2. Never Commit Secrets

```bash
# .gitignore should include:
.env
*.secret
secrets.json
~/.ironclaw/
```

#### 3. Use Minimal Environment Variables

**Development**:
```bash
# ~/.ironclaw/.env (not committed)
DATABASE_BACKEND=libsql
LLM_BACKEND=nearai
# Secrets stored in secrets store, not here
```

**Production**:
```bash
# Systemd service or container env
DATABASE_BACKEND=postgres
# DATABASE_URL loaded from secrets store
# No API keys in env vars
```

#### 4. Rotate Regularly

```bash
# Quarterly rotation reminder
# Add to calendar: "Rotate IronClaw secrets"
# → Regenerate all channel tokens
# → Update in secrets store
# → Restart services
```

#### 5. Audit Access

```bash
# List all configured secrets
ironclaw secrets list

# Check which channels are enabled
ironclaw status

# Review logs for auth failures
journalctl -u ironclaw | grep -i "auth\|secret\|token"
```

#### 6. Backup Secrets

```bash
# Export secrets (for backup; store encrypted!)
ironclaw secrets export > secrets-backup.enc

# Import after restore
ironclaw secrets import < secrets-backup.enc
```

**Warning**: Exported secrets are sensitive. Encrypt the backup file:
```bash
# Encrypt with age (recommended)
age -o secrets-backup.age -R recipients.txt secrets-backup

# Decrypt
age -d -o secrets-backup -i key.txt secrets-backup.age
```

### Troubleshooting

#### "Master key not found"

**Symptoms**:
```
Error: Secrets master key not configured
```

**Solutions**:
1. **Check keychain**:
   ```bash
   # macOS: Open Keychain Access, search "ironclaw"
   # Linux: Check GNOME Keyring is running
   ```

2. **Set env var**:
   ```bash
   export SECRETS_MASTER_KEY=<your-key>
   ```

3. **Re-run onboarding**:
   ```bash
   ironclaw onboard --channels-only
   ```

#### "Secret not found"

**Symptoms**:
```
Error: Secret 'telegram_bot_token' not found
```

**Solutions**:
1. **Check environment variables**:
   ```bash
   cat .env | grep TELEGRAM
   # Or
   echo $TELEGRAM_BOT_TOKEN
   ```

2. **Set the missing secret**:
   ```bash
   # Add to .env file
   echo "TELEGRAM_BOT_TOKEN=<token>" >> .env
   
   # Or export for current session
   export TELEGRAM_BOT_TOKEN=<token>
   ```

3. **Re-run channel setup**:
   ```bash
   ironclaw onboard --channels-only
   ```

4. **Check database directly** (advanced):
   ```bash
   psql "$DATABASE_URL" -c "SELECT name FROM secrets WHERE user_id='default';"
   ```

#### "Keychain dialog keeps appearing"

**Symptoms**: macOS shows keychain unlock dialog repeatedly.

**Cause**: Keychain is locked or per-app authorization not granted.

**Solutions**:
1. **Unlock keychain**:
   - Open Keychain Access
   - Right-click "login" keychain → Unlock

2. **Grant permanent access**:
   - When dialog appears, click "Always Allow"
   - Or: Keychain Access → ironclaw item → Access Control → "Allow all applications"

3. **Use env var instead** (less secure):
   ```bash
   export SECRETS_MASTER_KEY=<key>
   ```

---

## Appendix A: Quick Reference

### Environment Variables (Minimal Production)

```bash
# Database
DATABASE_BACKEND=postgres
DATABASE_URL=<from-secrets-store>

# LLM
LLM_BACKEND=nearai

# Tunnel (if using webhooks)
TUNNEL_URL=https://your-tunnel.ngrok.io

# Secrets
# SECRETS_MASTER_KEY from OS keychain (not set as env var)
```

### Secret Names Quick List

```
# LLM
nearai_api_key
openai_api_key
anthropic_api_key
llm_compatible_api_key

# Channels
telegram_bot_token
telegram_webhook_secret
slack_bot_token
slack_app_token
slack_signing_secret
http_webhook_secret
gateway_auth_token

# Database
database_url
libsql_auth_token
```

### Commands

```bash
# Setup
ironclaw onboard
ironclaw onboard --channels-only
ironclaw onboard --skip-auth

# Environment variables (for secrets)
export SECRET_NAME=value
echo "SECRET_NAME=value" >> .env

# Database inspection (advanced)
psql "$DATABASE_URL" -c "SELECT name FROM secrets WHERE user_id='default';"

# Pairing (Telegram)
ironclaw pairing list telegram
ironclaw pairing approve telegram <code>

# Status
ironclaw status
```

---

## Appendix B: Security Checklist

### Initial Setup

- [ ] Master key configured (OS keychain preferred)
- [ ] Database credentials secured
- [ ] LLM API keys in secrets store
- [ ] `.env` files not committed to git

### Channel Security

- [ ] HTTP: Webhook secret set, tunnel uses HTTPS
- [ ] Telegram: DM pairing enabled, owner ID set
- [ ] Slack: All three tokens stored securely
- [ ] Gateway: Auth token generated, localhost-only binding
- [ ] CLI: Terminal session secured

### Ongoing Maintenance

- [ ] Secrets rotated every 90 days
- [ ] Audit logs reviewed weekly
- [ ] Unused channels disabled
- [ ] Backups encrypted and tested

---

## Part 6: Production Hardening

### Port Configuration

IronClaw uses two main ports:

| Port | Env Variable | Default | Purpose |
|------|--------------|---------|---------|
| HTTP | `HTTP_PORT` | `8080` | Webhook endpoints (Telegram, Slack, HTTP channels) |
| Gateway | `GATEWAY_PORT` | `3000` | Local API gateway for CLI and integrations |

#### Configure Ports

```bash
# In .env file
HTTP_PORT=8080
GATEWAY_PORT=3000

# Or export before running
export HTTP_PORT=8080
export GATEWAY_PORT=3000
ironclaw start
```

#### Port Conflict Resolution

```bash
# Check if port is in use
lsof -i :8080
netstat -tlnp | grep 8080

# Kill process using port (Linux/macOS)
kill -9 $(lsof -t -i:8080)

# Use alternative port
export HTTP_PORT=8090
ironclaw start
```

---

### Host Binding

Control which network interfaces IronClaw binds to:

| Binding | Address | Use Case | Security |
|---------|---------|----------|----------|
| Localhost | `127.0.0.1` | Single-user, local dev | ✅ Highest |
| All interfaces | `0.0.0.0` | Multi-user, container | ⚠️ Requires firewall |

#### Configure Host Binding

```bash
# Localhost only (recommended for single-user)
export HOST=127.0.0.1
ironclaw start

# All interfaces (container/VPS deployments)
export HOST=0.0.0.0
ironclaw start

# IPv6 localhost
export HOST=::1
ironclaw start
```

#### Firewall Rules (When Using 0.0.0.0)

```bash
# UFW (Ubuntu/Debian)
ufw allow 8080/tcp
ufw allow 8081/tcp
ufw enable

# firewalld (RHEL/CentOS)
firewall-cmd --permanent --add-port=8080/tcp
firewall-cmd --permanent --add-port=8081/tcp
firewall-cmd --reload

# AWS Security Group
# Inbound: TCP 8080, 8081 from trusted IPs only
```

---

### TLS/HTTPS Requirements

**Production deployments MUST use TLS**. Webhook providers (Telegram, Slack) require HTTPS endpoints.

#### Option 1: Tunnel Provider (Recommended)

```bash
# ngrok
ngrok http 8080
# Provides: https://abc123.ngrok.io

# Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8080
# Provides: https://your-subdomain.trycloudflare.com

# LocalXpose
loclx tunnel http --to localhost:8080
```

#### Option 2: Reverse Proxy with Let's Encrypt

```nginx
# /etc/nginx/sites-available/ironclaw
server {
    listen 80;
    server_name ironclaw.example.com;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name ironclaw.example.com;
    
    ssl_certificate /etc/letsencrypt/live/ironclaw.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ironclaw.example.com/privkey.pem;
    
    # Modern TLS settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Generate certificate
certbot certonly --webroot -w /var/www/certbot -d ironclaw.example.com

# Auto-renewal (add to crontab)
0 3 * * * certbot renew --quiet
```

#### Option 3: Self-Signed (Development Only)

```bash
# Generate self-signed cert
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ironclaw.key -out ironclaw.crt \
  -subj "/CN=localhost"

# Configure IronClaw to use TLS
export TLS_CERT_PATH=/path/to/ironclaw.crt
export TLS_KEY_PATH=/path/to/ironclaw.key
ironclaw start
```

---

### Rate Limiting

Protect against abuse and API quota exhaustion.

#### Built-in Rate Limiting

```bash
# In .env or config
export RATE_LIMIT_REQUESTS=100      # requests per window
export RATE_LIMIT_WINDOW=60         # window in seconds
export RATE_LIMIT_BURST=20          # burst allowance
```

#### Per-Channel Rate Limits

```toml
# In config/channels.toml
[telegram.rate_limit]
requests_per_minute = 30
burst = 5

[slack.rate_limit]
requests_per_minute = 50
burst = 10

[http.rate_limit]
requests_per_minute = 100
burst = 20
```

#### Nginx Rate Limiting (Additional Layer)

```nginx
# In nginx http block
limit_req_zone $binary_remote_addr zone=ironclaw_limit:10m rate=10r/s;
limit_conn_zone $binary_remote_addr zone=ironclaw_conn:10m;

server {
    location / {
        limit_req zone=ironclaw_limit burst=20 nodelay;
        limit_conn ironclaw_conn 10;
        
        proxy_pass http://127.0.0.1:8080;
    }
}
```

#### LLM API Rate Limit Protection

```bash
# Add delays between LLM requests
export LLM_REQUEST_DELAY_MS=100

# Max concurrent LLM requests
export LLM_MAX_CONCURRENT=5

# Circuit breaker (fail fast after N failures)
export LLM_CIRCUIT_BREAKER_THRESHOLD=5
export LLM_CIRCUIT_BREAKER_TIMEOUT_SEC=60
```

---

### Logging Configuration

#### Log Levels

```bash
# RUST_LOG format: [module::path=level]
export RUST_LOG=info                    # Default
export RUST_LOG=debug                   # Development
export RUST_LOG=ironclaw=debug,warn     # Specific modules
export RUST_LOG=error                   # Production (errors only)
```

#### Log Format

```bash
# Structured JSON (for log aggregation)
export LOG_FORMAT=json

# Human-readable (for development)
export LOG_FORMAT=pretty

# Include timestamps
export LOG_TIMESTAMP=true
```

#### Log Output

```bash
# Console (default)
export LOG_OUTPUT=stdout

# File logging
export LOG_OUTPUT=file
export LOG_FILE=/var/log/ironclaw/ironclaw.log

# Both
export LOG_OUTPUT=both
```

#### Log Rotation

```bash
# Install logrotate
sudo apt install logrotate  # Debian/Ubuntu
sudo yum install logrotate  # RHEL/CentOS

# /etc/logrotate.d/ironclaw
/var/log/ironclaw/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ironclaw ironclaw
    postrotate
        systemctl reload ironclaw > /dev/null 2>&1 || true
    endscript
}
```

#### Log Aggregation

```yaml
# docker-compose.yml (with Loki)
services:
  ironclaw:
    logging:
      driver: loki
      options:
        loki-url: "http://loki:3100/loki/api/v1/push"
        loki-pipeline-stages: |
          - json:
              expressions:
                level: level
                message: message
          - labels:
              level:
```

---

### Resource Tuning

#### Memory Limits

```bash
# Max memory for IronClaw process
export MAX_MEMORY_MB=512

# Garbage collection tuning
export GC_HEAP_GROWTH_FACTOR=2.0

# Vector store cache size
export VECTOR_CACHE_SIZE_MB=128
```

#### Connection Pooling

```bash
# Database connection pool
export DB_POOL_SIZE=10
export DB_MAX_OVERFLOW=20
export DB_POOL_TIMEOUT=30

# HTTP client connections
export HTTP_CLIENT_POOL_SIZE=20
export HTTP_CLIENT_KEEP_ALIVE_SEC=90
```

#### Worker Threads

```bash
# Tokio runtime workers
export TOKIO_WORKER_THREADS=4

# Channel processing workers
export CHANNEL_WORKERS=2

# Background task workers
export BACKGROUND_WORKERS=2
```

#### Container Resource Limits

```yaml
# docker-compose.yml
services:
  ironclaw:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 256M
```

---

### Process Management

#### Systemd Service (Linux)

```ini
# /etc/systemd/system/ironclaw.service
[Unit]
Description=IronClaw AI Assistant
After=network.target postgresql.service

[Service]
Type=exec
User=ironclaw
Group=ironclaw
WorkingDirectory=/opt/ironclaw
Environment=PATH=/opt/ironclaw/bin:/usr/bin
EnvironmentFile=/opt/ironclaw/.env
ExecStart=/opt/ironclaw/bin/ironclaw start
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/ironclaw/data /var/log/ironclaw
PrivateTmp=true

# Resource limits
LimitNOFILE=65536
LimitNPROC=64

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable ironclaw
sudo systemctl start ironclaw
sudo systemctl status ironclaw

# View logs
journalctl -u ironclaw -f
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ironclaw:
    image: ironclaw/ironclaw:latest
    container_name: ironclaw
    restart: unless-stopped
    ports:
      - "127.0.0.1:8080:8080"
      - "127.0.0.1:8081:8081"
    environment:
      - RUST_LOG=info
      - DATABASE_URL=postgresql://user:pass@db:5432/ironclaw
    env_file:
      - .env
    volumes:
      - ironclaw_data:/var/ironclaw/data
      - ./config:/var/ironclaw/config:ro
      - /var/log/ironclaw:/var/log/ironclaw
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: pgvector/pgvector:pg15
    container_name: ironclaw-db
    restart: unless-stopped
    environment:
      - POSTGRES_USER=ironclaw
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
      - POSTGRES_DB=ironclaw
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ironclaw"]
      interval: 10s
      timeout: 5s
      retries: 5
    secrets:
      - db_password

volumes:
  ironclaw_data:
  postgres_data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

#### Process Monitoring

```bash
# Check process status
ps aux | grep ironclaw
systemctl status ironclaw

# Memory usage
ps -o pid,rss,command -p $(pgrep ironclaw)

# Open file descriptors
lsof -p $(pgrep ironclaw) | wc -l

# Network connections
ss -tlnp | grep ironclaw
```

#### Automatic Restarts

```bash
# Systemd handles this automatically
# For manual supervision, use a watchdog script

#!/bin/bash
# /opt/ironclaw/watchdog.sh

HEALTH_URL="http://localhost:8080/health"
RESTART_THRESHOLD=3
CHECK_INTERVAL=60

failures=0

while true; do
    if ! curl -sf "$HEALTH_URL" > /dev/null; then
        ((failures++))
        if [ $failures -ge $RESTART_THRESHOLD ]; then
            echo "$(date): Restarting IronClaw after $failures failures"
            systemctl restart ironclaw
            failures=0
        fi
    else
        failures=0
    fi
    sleep $CHECK_INTERVAL
done
```

---

## Part 7: Verification & Troubleshooting

### Startup Checklist

Run through this checklist on every deployment:

#### Pre-Start Checks

```bash
# 1. Verify installation
ironclaw --version

# 2. Check configuration files exist
ls -la ~/.config/ironclaw/
ls -la /opt/ironclaw/config/  # Production

# 3. Validate .env file
cat .env | grep -v "^#" | grep -v "^$"

# 4. Test database connectivity
psql "$DATABASE_URL" -c "SELECT 1"

# 5. Check required ports are free
lsof -i :8080
lsof -i :8081

# 6. Verify secrets store
ironclaw secrets list

# 7. Check disk space
df -h /var/lib/ironclaw
df -h /var/log/ironclaw
```

#### Post-Start Verification

```bash
# 1. Check service status
systemctl status ironclaw
# OR
docker ps | grep ironclaw

# 2. Health endpoint
curl http://localhost:8080/health

# 3. Check listening ports
ss -tlnp | grep ironclaw

# 4. Verify logs are flowing
journalctl -u ironclaw -n 20
# OR
docker logs ironclaw --tail 20

# 5. Test channel connectivity
ironclaw status

# 6. Verify LLM connection
ironclaw llm test
```

---

### Verification Commands

#### System Health

```bash
# Full health check
ironclaw status

# Detailed system info
ironclaw system info

# Database health
ironclaw db health

# Channel status
ironclaw channels status
```

#### Connectivity Tests

```bash
# Test LLM API
curl -X POST http://localhost:8080/llm/test \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "model": "gpt-4"}'

# Test webhook endpoint
curl -X POST http://localhost:8080/webhook/test \
  -H "X-Webhook-Secret: your-secret" \
  -d '{"test": true}'

# Test gateway
curl http://localhost:8081/health
```

#### Log Analysis

```bash
# Search for errors
journalctl -u ironclaw | grep -i error

# Search for warnings
journalctl -u ironclaw | grep -i warn

# Time-based filtering
journalctl -u ironclaw --since "1 hour ago"
journalctl -u ironclaw --since today --until "1 hour ago"

# Follow logs in real-time
journalctl -u ironclaw -f

# Export logs
journalctl -u ironclaw --since yesterday > ironclaw-logs.txt
```

---

### Common Errors Table

| Error | Cause | Solution |
|-------|-------|----------|
| `Address already in use` | Port 8080/8081 occupied | `lsof -i :8080` then kill process or change port |
| `Connection refused` | Database not running | Start PostgreSQL: `systemctl start postgresql` |
| `pgvector extension not found` | Missing pgvector | `CREATE EXTENSION IF NOT EXISTS vector;` |
| `Invalid webhook secret` | Secret mismatch | Verify `WEBHOOK_SECRET` matches provider config |
| `TLS handshake failed` | Certificate issue | Check cert paths, regenerate if expired |
| `Out of memory` | Memory limit exceeded | Increase `MAX_MEMORY_MB` or add swap |
| `Too many open files` | File descriptor limit | `ulimit -n 65536` or update systemd `LimitNOFILE` |
| `Secret not found` | Missing secret in env or store | Set env var or re-run onboarding |
| `Rate limit exceeded` | Too many requests | Increase `RATE_LIMIT_REQUESTS` or add delay |
| `Database pool exhausted` | Connection pool too small | Increase `DB_POOL_SIZE` |
| `Permission denied` | File/directory permissions | `chown -R ironclaw:ironclaw /opt/ironclaw` |
| `Certificate verify failed` | SSL verification issue | Update CA certificates or check cert chain |

---

### Security Audit

#### Automated Security Scan

```bash
# Check for exposed secrets in config
grep -r "api_key\|token\|secret\|password" ~/.config/ironclaw/ --include="*.toml" --include="*.env"

# Verify file permissions
find /opt/ironclaw -type f -perm /o+rwx  # World-readable/writable files
find /opt/ironclaw -type d -perm /o+rx   # World-accessible directories

# Check for hardcoded credentials
grep -r "sk-\|Bearer \|Basic " /opt/ironclaw/config/

# Audit network exposure
ss -tlnp | grep ironclaw
netstat -tlnp | grep ironclaw
```

#### Security Checklist

```bash
# Run security audit script
ironclaw audit security

# Manual checklist:
# [ ] No secrets in .env files (use secrets store)
# [ ] Database user has minimal permissions
# [ ] TLS enabled for all external endpoints
# [ ] Webhook secrets configured for all channels
# [ ] Rate limiting enabled
# [ ] Logs don't contain sensitive data
# [ ] File permissions restricted (600 for secrets, 644 for configs)
# [ ] Service running as non-root user
# [ ] Firewall rules in place
# [ ] Regular backups configured
```

#### Penetration Testing

```bash
# Test webhook endpoint without auth
curl -X POST http://localhost:8080/webhook/telegram \
  -d '{"invalid": true}'
# Should return 401 Unauthorized

# Test rate limiting
for i in {1..150}; do curl -s http://localhost:8080/health; done
# Should see 429 Too Many Requests after threshold

# Test SQL injection (should be handled by ORM)
curl "http://localhost:8080/api/messages?filter='; DROP TABLE messages;--"
# Should return error, not execute SQL
```

---

### Performance Troubleshooting

#### Slow Response Times

```bash
# Check database query performance
export RUST_LOG=ironclaw::db=debug
ironclaw start

# Profile with tokio-console
cargo install tokio-console
export RUST_LOG=tokio=trace
tokio-console

# Check LLM API latency
curl -w "@curl-format.txt" -o /dev/null -s \
  -X POST http://localhost:8080/llm/test \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'

# curl-format.txt contents:
# time_namelookup:  %{time_namelookup}\n
# time_connect:     %{time_connect}\n
# time_appconnect:  %{time_appconnect}\n
# time_pretransfer: %{time_pretransfer}\n
# time_starttransfer: %{time_starttransfer}\n
# ----------\n
# time_total:       %{time_total}\n
```

#### High Memory Usage

```bash
# Check memory breakdown
ps -o pid,rss,vsz,command -p $(pgrep ironclaw)

# Profile heap allocations
cargo install heaptrack
heaptrack ironclaw start

# Check for memory leaks
watch -n 5 'ps -o pid,rss,command -p $(pgrep ironclaw)'
```

#### High CPU Usage

```bash
# Profile CPU usage
cargo install flamegraph
sudo perf record -F 99 -p $(pgrep ironclaw) -g -- sleep 30
sudo perf script | stackcollapse-perf.pl | flamegraph.pl > cpu.svg

# Check for hot loops
top -H -p $(pgrep ironclaw)
```

#### Database Performance

```bash
# Check slow queries
psql "$DATABASE_URL" -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check connection pool
psql "$DATABASE_URL" -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'ironclaw';"

# Vacuum and analyze
psql "$DATABASE_URL" -c "VACUUM ANALYZE;"

# Check table sizes
psql "$DATABASE_URL" -c "SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```

#### Network Issues

```bash
# Check network latency
ping -c 10 api.openai.com
mtr api.openai.com

# Test DNS resolution
dig api.openai.com
nslookup api.openai.com

# Check for packet loss
mtr -rwc 100 api.openai.com

# Verify tunnel connectivity
curl -I https://your-tunnel-url.ngrok.io/health
```

---

### Quick Reference: Emergency Commands

```bash
# Stop IronClaw immediately
systemctl stop ironclaw
# OR
docker stop ironclaw

# Emergency restart
systemctl restart ironclaw

# Clear all caches
ironclaw cache clear --all

# Reset database (WARNING: destructive)
ironclaw db reset --force

# Export all secrets (for backup)
ironclaw secrets export > secrets-backup.json

# Import secrets (for restore)
ironclaw secrets import < secrets-backup.json

# Generate diagnostic report
ironclaw diagnostic > diagnostic-report.txt

# Safe mode (minimal features)
ironclaw start --safe-mode
```

---

**End of Secure Setup Guide**
