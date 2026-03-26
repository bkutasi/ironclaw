<!-- Context: project-intelligence/errors | Priority: high | Version: 1.0 | Updated: 2026-03-26 -->
# Error: Web-Search Freshness Parameter

**Symptom**: Tool execution fails with validation error:
```
Tool error: Invalid 'freshness': expected 'pd', 'pw', 'pm', 'py', or 'YYYY-MM-DDtoYYYY-MM-DD', got 'week'
```

## Root Cause

The `web-search` tool validates the `freshness` parameter against specific allowed values. Passing arbitrary strings like `'week'` causes immediate validation failure.

## Valid Freshness Values

| Value | Meaning | Use Case |
|-------|---------|----------|
| `pd` | Past day | Recent news, latest updates |
| `pw` | Past week | Weekly reports, recent events |
| `pm` | Past month | Monthly summaries |
| `py` | Past year | Annual reviews |
| `YYYY-MM-DDtoYYYY-MM-DD` | Date range | Specific time periods |

## Common Mistakes

| ❌ Invalid | ✅ Correct | Notes |
|-----------|-----------|-------|
| `'week'` | `'pw'` | Use abbreviation, not full word |
| `'month'` | `'pm'` | |
| `'year'` | `'py'` | |
| `'today'` | `'pd'` | |
| `'last 7 days'` | `'pw'` | |

## Fix

**Find the caller**:
```bash
grep -rn "freshness.*week\|freshness.*month\|freshness.*year" --include="*.rs" --include="*.ts" --include="*.json" .
```

**Update to use valid value**:
```typescript
// ❌ Before
await webSearch({ query: "latest AI news", freshness: "week" })

// ✅ After
await webSearch({ query: "latest AI news", freshness: "pw" })
```

## Diagnosis Query

Check web-search failures in database:
```sql
SELECT error_message, error_count, last_failure 
FROM tool_failures 
WHERE tool_name = 'web-search' 
ORDER BY last_failure DESC;
```

## 📂 Codebase References

**Tool Implementation**:
- `tools-src/web-search/src/lib.rs` — Freshness parameter validation logic
- `tools-src/web-search/src/params.rs` — Parameter parsing and validation

**Configuration**:
- `registry/tools/web-search.json` — Tool registry entry with schema
- `tools-src/web-search/web-search-tool.capabilities.json` — Tool capabilities

**Usage Examples**:
- Search codebase for `webSearch({` to find all callers

## Related

- errors/tool-repair-loops.md — Infinite repair loop from repeated failures
- concepts/tool-protection.md — Why web-search is protected from self-repair
- lookup/tool-parameters.md — Complete tool parameter reference
