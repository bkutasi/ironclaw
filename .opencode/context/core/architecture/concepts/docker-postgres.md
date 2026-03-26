<!-- Context: architecture/concepts | Priority: critical | Version: 1.0 | Updated: 2026-03-19 -->
# Concept: Docker PostgreSQL

**Purpose**: PostgreSQL database setup for IronClaw using Docker

**Last Updated**: 2026-03-19

## Core Idea

IronClaw uses PostgreSQL running in Docker for persistent storage of agent state, conversation history, and channel configurations. Database runs on port 5433 (non-standard to avoid conflicts).

## Key Points

- **Docker container**: Isolated, reproducible database environment
- **Port 5433**: Non-standard port avoids conflicts with system PostgreSQL
- **Database name**: `ironclaw`
- **Credentials**: Stored in environment variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`)
- **Persistence**: Volume mount preserves data across container restarts

## Quick Example

```bash
# Start PostgreSQL with Docker
docker run -d \
  --name ironclaw-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=yourpass \
  -e POSTGRES_DB=ironclaw \
  -p 5433:5432 \
  -v ironclaw-data:/var/lib/postgresql/data \
  postgres:16-alpine
```

## 📂 Codebase References

**Database Schema**:
- `src/database/migrations/` - SQL migration files
- `src/database/schema.rs` - Rust schema definitions

**Connection Management**:
- `src/database/pool.rs` - Connection pool setup
- `src/database/mod.rs` - Database module

**Configuration**:
- `.env.example` - Environment variable template
- `scripts/start-postgres.sh` - Docker startup script

**Tests**:
- `src/database/__tests__/` - Database integration tests

## Deep Dive

**Reference**: See `guides/build-process.md` for full setup workflow

## Related

- concepts/wasm-channels.md
- guides/env-config.md
- lookup/env-variables.md
