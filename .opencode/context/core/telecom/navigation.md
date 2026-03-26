# Telecom Navigation

**Purpose**: Telegram integration, communication channels, and messaging patterns

---

## Structure

```
telecom/
├── navigation.md
├── telegram-integration.md
└── guides/
    └── telegram-mode-selection.md
```

---

## Quick Routes

| Task | Path |
|------|------|
| **Understand Telegram integration** | `telegram-integration.md` |
| **Choose webhook vs polling** | `guides/telegram-mode-selection.md` |
| **Fix bot not responding** | `../integrations/errors/bot-no-response.md` |
| **Fix webhook errors** | `../errors/telegram-webhook-errors.md` |
| **Fix tunnel errors** | `../integrations/errors/tunnel-errors.md` |

---

## By Type

**Concepts** → Telegram integration architecture  
**Guides** → Mode selection (webhook vs polling)  
**Errors** → Webhook issues, tunnel problems, bot response issues

---

## Loading Strategy

**For Telegram setup**:
1. Load `telegram-integration.md` (overview)
2. Load `guides/telegram-mode-selection.md` (choose mode)
3. Reference error files if issues occur

**For troubleshooting**:
1. Identify error type (webhook, tunnel, bot response)
2. Load corresponding error file
3. Reference telegram-integration.md for context

---

## Related Context

- **Integrations** → `../integrations/navigation.md`
- **Architecture** → `../architecture/navigation.md`
- **Errors** → `../errors/navigation.md`
