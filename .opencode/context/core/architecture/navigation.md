# Architecture Navigation

**Purpose**: System architecture, infrastructure, and build processes for IronClaw

---

## Structure

```
architecture/
├── navigation.md
├── concepts/
│   ├── config-precedence.md
│   ├── docker-postgres.md
│   └── wasm-channels.md
├── guides/
│   ├── build-process.md
│   └── env-config.md
├── lookup/
│   └── env-variables.md
└── errors/
    ├── common-build-issues.md
    └── llm-model-mismatch.md
```

---

## Quick Routes

| Task | Path |
|------|------|
| **Configuration precedence** | `concepts/config-precedence.md` |
| **Understand WASM channels** | `concepts/wasm-channels.md` |
| **Docker PostgreSQL setup** | `concepts/docker-postgres.md` |
| **Build Rust + WASM** | `guides/build-process.md` |
| **Environment variables** | `lookup/env-variables.md` |
| **Fix build errors** | `errors/common-build-issues.md` |
| **Fix LLM model errors** | `errors/llm-model-mismatch.md` |

---

## By Type

**Concepts** → Core architecture (WASM, Docker, database)  
**Guides** → Build processes, configuration  
**Lookup** → Environment variable reference  
**Errors** → Build and compilation issues

---

## Related Context

- **Integrations** → `../integrations/navigation.md`
- **Telecom** → `../telecom/navigation.md`
- **Core Standards** → `../standards/navigation.md`
