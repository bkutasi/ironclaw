# Errors Navigation

**Purpose**: Error catalogs and resolution strategies for routine operations

---

## Structure

```
core/errors/
├── navigation.md
├── routine-errors.md           # Common routine operation errors
├── telegram-webhook-errors.md  # Telegram-specific errors
└── telegram-proactive-message-fix.md
```

---

## Quick Routes

| Task | Path | Priority |
|------|------|----------|
| **Routine error catalog** | `routine-errors.md` | ⭐⭐⭐⭐⭐ |
| **Error codes reference** | `routine-errors.md#error-codes-quick-reference` | ⭐⭐⭐⭐⭐ |
| **Retry strategies** | `routine-errors.md#error-handling-patterns` | ⭐⭐⭐⭐ |
| **Telegram webhook errors** | `telegram-webhook-errors.md` | ⭐⭐⭐ |
| **Proactive message fixes** | `telegram-proactive-message-fix.md` | ⭐⭐ |

---

## Error Categories

**Initialization Errors** (E001-E002) → Database connections, configuration  
**Runtime Errors** (E003-E004) → Query timeouts, serialization failures  
**Recovery Errors** (E005-E006) → Retry exhausted, state corruption

---

## Related Context

- **Debug procedures** → `../guides/routine-debugging.md`
- **Recovery concepts** → `../concepts/routine-recovery.md`
- **Pattern examples** → `../examples/error-handling-patterns.md`
- **Core navigation** → `../navigation.md`
