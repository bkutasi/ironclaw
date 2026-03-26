# Guides Navigation

**Purpose**: Procedural guides for routine operations and debugging

---

## Structure

```
core/guides/
├── navigation.md
├── routine-debugging.md        # Systematic debugging workflow
└── security-checklist.md       # Security verification checklist
```

---

## Quick Routes

| Task | Path | Priority |
|------|------|----------|
| **Debug routine errors** | `routine-debugging.md` | ⭐⭐⭐⭐⭐ |
| **Error classification** | `routine-debugging.md#step-1-error-classification` | ⭐⭐⭐⭐⭐ |
| **Log analysis** | `routine-debugging.md#step-2-log-analysis` | ⭐⭐⭐⭐⭐ |
| **State inspection** | `routine-debugging.md#step-3-state-inspection` | ⭐⭐⭐⭐ |
| **Security checklist** | `security-checklist.md` | ⭐⭐⭐⭐ |

---

## Debug Workflow

1. **Classification** (5 min) → Identify error category
2. **Log Analysis** (10 min) → Extract correlation IDs, error chains
3. **State Inspection** (10 min) → Check pools, memory, queries
4. **Reproduction** (15 min) → Isolate and reproduce
5. **Root Cause** (10 min) → Identify and document

---

## Common Scenarios

**Intermittent Timeouts** → Query optimization, pool sizing  
**Configuration Drift** → Config management, validation  
**Cascade Failures** → Circuit breakers, isolation

---

## Related Context

- **Error catalog** → `../errors/routine-errors.md`
- **Recovery concepts** → `../concepts/routine-recovery.md`
- **Pattern examples** → `../examples/error-handling-patterns.md`
- **Core navigation** → `../navigation.md`
