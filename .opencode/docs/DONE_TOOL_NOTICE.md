# Agent Notice: `done` Tool Now Always Available

**Effective:** Immediate  
**Scope:** All job contexts and deployment configurations

---

## What is the `done` Tool?

The `done` tool is your primary signal for job completion. Use it when:

- ✅ All assigned work is finished
- ✅ All acceptance criteria are met
- ✅ You have a summary ready to report

**What it does:** Ends the agentic loop immediately and marks the job as completed with your provided summary.

**When NOT to use it:** Mid-job, for partial updates, or before self-review is complete.

---

## What Was the Problem?

In certain deployment configurations, calling the `done` tool would fail with an error: **"Tool done not found"**.

**From your perspective, this meant:**

- You would attempt to signal completion normally
- Instead of completing, you'd receive an error response
- The job would remain in an incomplete state
- You'd need to find alternative (non-standard) ways to signal completion
- Behavior was inconsistent across different configurations

**Root cause:** The `done` tool was not registered when `allow_local_tools = false` in the agent configuration.

---

## What Changed?

The `done` tool is now **always available**, regardless of configuration.

**Key changes:**

- `done` is treated as a **core orchestrator tool** (same category as `echo`, `time`, `json`, `http`)
- It bypasses the `allow_local_tools` restriction entirely
- Registration is guaranteed in ALL deployment contexts
- No configuration can prevent it from being available

---

## What You Will Experience Differently

| Before | After |
|--------|-------|
| `done` might fail in some configs | `done` **always** works |
| Inconsistent completion behavior | Consistent behavior everywhere |
| Had to detect config before using `done` | Use `done` confidently in any context |
| Alternative completion signals needed | `done` is the single source of truth |

**Your new guarantee:** You can rely on `done` as the primary completion signal in every job, every time.

---

## Usage Reminder

### Correct Usage

```
Call done when:
├─ All subtasks completed
├─ Self-review passed
├─ Deliverables created/updated
└─ Ready to report summary

Example: done("Completed: Implemented JWT auth with refresh tokens and middleware")
```

### Incorrect Usage

```
Do NOT call done when:
├─ Job is partially complete
├─ Still waiting on external dependencies
├─ Haven't run self-review yet
└─ Need to continue working after signaling
```

---

## Summary

The `done` tool is now a guaranteed, always-available core tool. You can confidently use it to signal job completion in any deployment configuration without fear of errors. This restores consistent, predictable behavior across all agent contexts.

**Action required:** None. Continue using `done` as your standard completion signal. The tool will now work reliably in all scenarios.
