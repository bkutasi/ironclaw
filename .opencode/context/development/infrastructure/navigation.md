<!-- Context: development/navigation | Priority: high | Version: 1.1 | Updated: 2026-02-13 -->

# Infrastructure Navigation

**Purpose**: DevOps, CI/CD, and deployment patterns

**Last Updated**: 2026-02-13

---

## Structure

```
infrastructure/
├── navigation.md
│
├── ci-cd/
│   ├── errors/
│   │   └── postgres-setup.md    # PostgreSQL CI setup issues
│   └── lookup/
│       └── workflow-patterns.md # CI workflow reference
│
└── docker/
    └── (future content)
```

---

## Quick Navigation

### Errors
| File | Description | Priority |
|------|-------------|----------|
| [postgres-setup.md](ci-cd/errors/postgres-setup.md) | PostgreSQL CI migration failures | high |

### Lookup
| File | Description | Priority |
|------|-------------|----------|
| [workflow-patterns.md](ci-cd/lookup/workflow-patterns.md) | CI workflow configurations | medium |

---

## Loading Strategy

**For CI/CD work**:
1. Check errors/postgres-setup.md for common issues
2. Reference lookup/workflow-patterns.md for configuration examples

---

## Related Context

- **Core Standards** → `../../core/standards/code-quality.md`
- **Testing** → `../../core/standards/test-coverage.md`
