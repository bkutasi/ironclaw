# Core Context Navigation

**Purpose**: Universal standards and workflows for all development

---

## Structure

```
core/
├── navigation.md
├── context-system.md
├── essential-patterns.md
│
├── architecture/              # NEW: System architecture
│   ├── navigation.md
│   ├── concepts/
│   ├── guides/
│   ├── lookup/
│   └── errors/
│
├── integrations/              # NEW: External services
│   ├── navigation.md
│   ├── concepts/
│   ├── guides/
│   ├── lookup/
│   └── errors/
│
├── telecom/                   # Telegram integration
│   └── telegram-integration.md
│
├── standards/
│   ├── navigation.md
│   ├── code-quality.md
│   ├── test-coverage.md
│   ├── documentation.md
│   ├── security-patterns.md
│   └── code-analysis.md
│
├── workflows/
│   ├── navigation.md
│   ├── code-review.md
│   ├── task-delegation.md
│   ├── feature-breakdown.md
│   ├── session-management.md
│   └── design-iteration.md
│
├── guides/
│   ├── resuming-sessions.md
│   └── security-checklist.md
│
├── errors/                    # Common errors
│   └── telegram-webhook-errors.md
│
├── task-management/
│   ├── navigation.md
│   ├── standards/
│   │   └── task-schema.md
│   ├── guides/
│   │   ├── splitting-tasks.md
│   │   └── managing-tasks.md
│   └── lookup/
│       └── task-commands.md
│
├── system/
│   └── context-guide.md
│
└── context-system/
    ├── guides/
    ├── examples/
    ├── standards/
    └── operations/
```

---

## Quick Routes

| Task | Path |
|------|------|
| **System architecture** | `architecture/navigation.md` |
| **Configuration precedence** | `architecture/concepts/config-precedence.md` |
| **WASM channels** | `architecture/concepts/wasm-channels.md` |
| **Docker PostgreSQL** | `architecture/concepts/docker-postgres.md` |
| **Build process** | `architecture/guides/build-process.md` |
| **Environment config** | `architecture/lookup/env-variables.md` |
| **LLM model errors** | `architecture/errors/llm-model-mismatch.md` |
| **Cloudflare tunnels** | `integrations/navigation.md` |
| **Ephemeral tunnel setup** | `integrations/guides/ephemeral-tunnel-setup.md` |
| **Persistent tunnel setup** | `integrations/guides/persistent-tunnel-setup.md` |
| **Tunnel errors** | `integrations/errors/tunnel-errors.md` |
| **Bot no response** | `integrations/errors/bot-no-response.md` |
| **Telegram integration** | `telecom/telegram-integration.md` |
| **Telegram mode selection** | `telecom/guides/telegram-mode-selection.md` |
| **Telegram errors** | `errors/telegram-webhook-errors.md` |
| **Security checklist** | `guides/security-checklist.md` |
| **Write code** | `standards/code-quality.md` |
| **Write tests** | `standards/test-coverage.md` |
| **Write docs** | `standards/documentation.md` |
| **Security patterns** | `standards/security-patterns.md` |
| **Review code** | `workflows/code-review.md` |
| **Delegate task** | `workflows/task-delegation.md` |
| **Break down feature** | `workflows/feature-breakdown.md` |
| **Resume session** | `guides/resuming-sessions.md` |
| **Manage tasks** | `task-management/navigation.md` |
| **Task CLI commands** | `task-management/lookup/task-commands.md` |
| **Context system** | `context-system.md` |

---

## By Type

**Architecture** → System design, WASM, Docker, build processes (critical)  
**Integrations** → Cloudflare tunnels, external services (critical)  
**Telecom** → Telegram integration patterns (high)  
**Errors** → Common errors and solutions (high)  
**Standards** → Code quality, testing, docs, security (critical)  
**Workflows** → Review, delegation, task breakdown (high)  
**Task Management** → JSON-driven task tracking with CLI (high)  
**System** → Context management and guides (medium)

---

## Related Context

- **Development** → `../development/navigation.md`
- **OpenAgents Control Repo** → `../openagents-repo/navigation.md`
