<!-- Context: development/lookup | Priority: medium | Version: 1.0 | Updated: 2026-02-13 -->

# Lookup: CI Workflow Patterns

**Purpose**: Quick reference for common CI/CD workflow configurations

**Last Updated**: 2026-02-13

---

## GitHub Actions with PostgreSQL

### Service Configuration

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    env:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ironclaw
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
    ports:
      - 5432:5432
```

### Setup Steps Order

| Step | Purpose | Example |
|------|---------|---------|
| 1. Wait for postgres | Ensure service is ready | `pg_isready -h localhost` |
| 2. Create database | Create if not exists | `createdb -h localhost test_db` |
| 3. Run migrations | Apply schema changes | `psql -f migrations/V1__*.sql` |
| 4. Run tests | Execute test suite | `cargo test` |

---

## Common Migration Commands

```bash
# Check postgres is ready
pg_isready -h localhost -p 5432

# Run single SQL file
psql -h localhost -U postgres -d ironclaw -f migrations/V1__init.sql

# Run all SQL files in order
for f in migrations/V*.sql; do
  psql -h localhost -U postgres -d ironclaw -f "$f"
done

# Using refinery CLI
refinery migrate -e DATABASE_URL -p migrations
```

---

## Environment Variables

| Variable | Example | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgres://user:pass@localhost:5432/db` | Connection string |
| `POSTGRES_USER` | `postgres` | Default username |
| `POSTGRES_PASSWORD` | `password` | Default password |
| `POSTGRES_DB` | `ironclaw` | Default database |

---

## Health Check Options

```yaml
options: >-
  --health-cmd pg_isready
  --health-interval 10s
  --health-timeout 5s
  --health-retries 5
```

---

**Related**:
- errors/postgres-setup.md (common postgres CI issues)
- guides/github-actions-setup.md (full setup guide)

**Codebase References**:
- `.github/workflows/test.yml`
- `migrations/` folder
