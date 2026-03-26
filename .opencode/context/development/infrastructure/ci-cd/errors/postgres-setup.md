<!-- Context: development/errors | Priority: high | Version: 1.0 | Updated: 2026-02-13 -->

# Error: PostgreSQL CI Setup Failure

**Purpose**: PostgreSQL starts but tests fail due to missing database migrations in CI

**Last Updated**: 2026-02-13

---

## Symptom

CI workflow `test.yml` starts PostgreSQL service successfully, but tests fail with:
```
test_workspace_fts_search ... FAILED
Failed to write: SearchFailed { reason: "Query failed: db error" }
```

10 `workspace_integration` tests fail with database errors.

---

## Root Cause

PostgreSQL container starts, but **database migrations are never run**. The integration tests require tables to exist before they can execute queries.

**Key Insight**: CI starts postgres with pgvector image but doesn't initialize schema. Tests pass locally because developers have already run migrations on their machines.

---

## Affected Files

| File | Issue |
|------|-------|
| `.github/workflows/test.yml` | Missing migration step before test execution |
| `tests/workspace_integration.rs` | 10 tests fail without database schema |

---

## Solution

Add database setup step to CI workflow **before** running tests:

### Option 1: Run Migration SQL Files

```yaml
- name: Setup database
  run: |
    # Wait for postgres to be ready
    until pg_isready -h localhost -p 5432; do sleep 1; done
    # Run migrations in order
    psql -h localhost -U postgres -d ironclaw -f migrations/V1__create_tables.sql
    psql -h localhost -U postgres -d ironclaw -f migrations/V2__add_indexes.sql
    # ... continue for V3 through V8
```

### Option 2: Use Refinery CLI

```yaml
- name: Run migrations
  run: |
    cargo install refinery_cli
    refinery migrate -e DATABASE_URL -p migrations
```

---

## Prevention Checklist

- [ ] Add migration step after postgres service starts
- [ ] Verify all migration files (V1-V8) are in repository
- [ ] Use same postgres image version as production
- [ ] Test CI workflow on clean environment

---

**Related**:
- lookup/workflow-patterns.md (CI workflow patterns)
- concepts/ci-pipeline-design.md (CI/CD pipeline concepts)

**Codebase References**:
- `.github/workflows/test.yml` (needs migration step)
- `migrations/` folder (SQL migration files)
- `tests/workspace_integration.rs` (failing tests)
