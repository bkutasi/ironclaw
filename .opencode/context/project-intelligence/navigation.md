<!-- Context: project-intelligence/nav | Priority: high | Version: 1.3 | Updated: 2026-03-26 -->

# Project Intelligence

> Start here for quick project understanding. These files bridge business and technical domains.

## Structure

```
.opencode/context/project-intelligence/
├── navigation.md              # This file - quick overview
├── business-domain.md         # Business context and problem statement
├── technical-domain.md        # Stack, architecture, technical decisions
├── business-tech-bridge.md    # How business needs map to solutions
├── decisions-log.md           # Major decisions with rationale
├── living-notes.md            # Active issues, debt, open questions
├── concepts/                  # Core concepts (what it is)
│   └── tool-protection.md     # PROTECTED_TOOL_NAMES mechanism
└── errors/                    # Common errors (what goes wrong)
    ├── routine-job-errors.md  # Job iteration, broken tools
    ├── tool-repair-loops.md   # Infinite repair loops
    └── web-search-errors.md   # Web search validation errors
```

## Quick Routes

| What You Need | File | Description |
|---------------|------|-------------|
| Understand the "why" | `business-domain.md` | Problem, users, value proposition |
| Understand the "how" | `technical-domain.md` | Stack, architecture, technical decisions |
| See the connection | `business-tech-bridge.md` | Business → technical mapping |
| Know the context | `decisions-log.md` | Why decisions were made |
| Current state | `living-notes.md` | Active issues and open questions |
| All of the above | Read all files in order | Full project intelligence |

## Concepts

- **[tool-protection.md](concepts/tool-protection.md)** — PROTECTED_TOOL_NAMES mechanism, why builtin tools are excluded from self-repair tracking

## Agent Tools

### done Tool — Job Completion Signaling

- **[ADR-003](decisions-log.md#adr-003-always-register-done-tool)** — Architectural decision (always available as builtin)
- **[ADR-001](decisions-log.md#adr-001-tool-based-completion-detection-done)** — Original tool-based completion design
- **[Living Notes](living-notes.md#done-tool)** — Behavioral changes and usage patterns
- **[Agent Notice](../docs/DONE_TOOL_NOTICE.md)** — Agent-facing documentation and usage guide

**Related**: [Job Errors](errors/routine-job-errors.md#error-job-maximum-iterations-exceeded-despite-completing-work) — "Maximum iterations exceeded" prevention

## Errors

- **[routine-job-errors.md](errors/routine-job-errors.md)** — Job iteration, broken tools, routine validation errors
- **[tool-repair-loops.md](errors/tool-repair-loops.md)** — Infinite tool repair loop diagnosis and fixes
- **[web-search-errors.md](errors/web-search-errors.md)** — Web search freshness parameter validation errors

## Usage

**New Team Member / Agent**:
1. Start with `navigation.md` (this file)
2. Read all files in order for complete understanding
3. Follow onboarding checklist in each file

**Quick Reference**:
- Business focus → `business-domain.md`
- Technical focus → `technical-domain.md`
- Decision context → `decisions-log.md`

## Integration

This folder is referenced from:
- `.opencode/context/core/standards/project-intelligence.md` (standards and patterns)
- `.opencode/context/core/system/context-guide.md` (context loading)

See `.opencode/context/core/context-system.md` for the broader context architecture.

## Maintenance

Keep this folder current:
- Update when business direction changes
- Document decisions as they're made
- Review `living-notes.md` regularly
- Archive resolved items from decisions-log.md

**Management Guide**: See `.opencode/context/core/standards/project-intelligence-management.md` for complete lifecycle management including:
- How to update, add, and remove files
- How to create new subfolders
- Version tracking and frontmatter standards
- Quality checklists and anti-patterns
- Governance and ownership

See `.opencode/context/core/standards/project-intelligence.md` for the standard itself.
