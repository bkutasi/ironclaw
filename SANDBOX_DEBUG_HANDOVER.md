# Sandbox Job Debug Handover

**Date:** 2026-03-23
**Status:** ✅ COMPLETE — All root causes fixed, force_text defense added, tested with llama.cpp local LLM.

---

## Problem Summary

Sandbox jobs (Docker container workers) get stuck and never show as completed in the web UI. Two root causes were identified and fixed; a third area needs investigation.

---

## Root Causes Found & Fixed

### 1. Missing `done` tool (FIXED)

**File:** `src/tools/builtin/done.rs` (NEW)
**Files modified:** `src/tools/builtin/mod.rs`, `src/tools/registry.rs`, `src/worker/container.rs`, `src/worker/job.rs`

The system relied on text-based completion detection (`llm_signals_completion()` in `src/util.rs`) which scans LLM text responses for phrases like "job is complete". This is fragile — different LLMs phrase things differently (e.g., "completed successfully" vs "successfully completed").

**Fix:** Added a `done` tool (originally called `complete_job`) that the LLM explicitly calls to signal completion. Both `ContainerDelegate` and `JobDelegate` intercept this tool call and break the agentic loop immediately.

### 2. Database status never updated on completion (FIXED)

**File:** `src/orchestrator/api.rs` — `report_complete` handler (line ~220)

When the container called `report_complete`, the orchestrator updated in-memory state (`ContainerJobManager`) but **never wrote to the database**. The web UI reads from `agent_jobs` table, so it always showed `"in_progress"`.

**Fix:** Added `store.update_sandbox_job_status()` call in the `report_complete` handler to persist `"completed"` or `"failed"` status.

### 3. Debug logging added to worker container (DONE)

**File:** `Dockerfile.worker` — added `ENV RUST_LOG=ironclaw=debug`

The agentic loop logs (LLM responses, tool calls, iteration tracking) were at `debug` level but the container defaulted to `info`. Now shows full LLM interaction logs.

---

## What Still Needs Work

### ~~The `force_text` mechanism is missing from container/job delegates~~ ✅ DONE

Both `ContainerDelegate` and `JobDelegate` now implement the 3-phase defense:
1. At `max_iterations - 1`: injects nudge message
2. At `max_iterations`: sets `force_text = true`, swaps to no-tools system prompt
3. At `max_iterations + 1`: hard ceiling → `MaxIterations`

See `src/worker/container.rs` and `src/worker/job.rs` for implementations.

### ~~Text-based completion detection could be improved~~ ✅ DONE (as fallback)

Added 8 phrases to `llm_signals_completion()` in `src/util.rs` as a secondary safety net.
Primary completion path is now the `done` tool.

### The `sandboxed: false` field in shell output

Still present — low priority UX clarification. Working as designed.

---

## Key Files Reference

| File | Role |
|------|------|
| `src/worker/container.rs` | Container runtime — `ContainerDelegate` implements `LoopDelegate` |
| `src/worker/job.rs` | Job worker — `JobDelegate` implements `LoopDelegate` |
| `src/agent/agentic_loop.rs` | Shared agentic loop engine (`run_agentic_loop`) |
| `src/agent/dispatcher.rs` | Chat delegate — has `force_text` mechanism (reference impl) |
| `src/orchestrator/api.rs` | Orchestrator HTTP API — `report_complete` handler |
| `src/orchestrator/job_manager.rs` | Container lifecycle management |
| `src/tools/builtin/done.rs` | New completion tool (originally `complete_job`) |
| `src/tools/registry.rs` | Tool registration |
| `src/util.rs` | `llm_signals_completion()` text detection |
| `src/history/store.rs` | `update_sandbox_job_status()` DB method |
| `src/channels/web/handlers/jobs.rs` | Web UI job list/detail handlers |
| `Dockerfile.worker` | Worker container image |

---

## How to Debug

### Watch container logs live
```bash
while true; do
  cid=$(docker ps -q --filter "ancestor=ironclaw-worker:latest" | head -1)
  if [ -n "$cid" ]; then
    echo "Found: $cid"
    docker logs -f "$cid"
    break
  fi
  sleep 0.1
done
```

### Rebuild after code changes
```bash
cargo build --release
docker build -f Dockerfile.worker -t ironclaw-worker .
```

### Key log lines to look for
```
DEBUG ironclaw::agent::agentic_loop: LLM tool_calls response iteration=N tools=[...]
DEBUG ironclaw::agent::agentic_loop: LLM text response iteration=N len=... response=...
INFO ironclaw::worker::container: Worker completed job ... successfully
WARN ironclaw::worker::container: Worker failed for job ...: max iterations (50) exceeded
```

### Test a sandbox job
Use the main agent (via Telegram/gateway/REPL) to create a job:
```
Create a sandbox test job that just runs echo "hello"
```

---

## Architecture Notes

- The container runs the same `ironclaw` binary in `worker` mode
- LLM calls are proxied through the orchestrator via `ProxyLlmProvider` (`src/worker/proxy_llm.rs`)
- The container registers only container-safe tools: shell, read_file, write_file, list_dir, apply_patch, done
- Job events are streamed to the orchestrator via HTTP (`/worker/{job_id}/events`) and persisted to `job_events` table
- The web UI reads job status from `agent_jobs` table (source='sandbox'), NOT from in-memory state
