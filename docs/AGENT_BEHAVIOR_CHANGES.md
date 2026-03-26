# Agent Behavior Changes — 2026-03-25

This document describes behavioral updates you will experience as an Ironclaw agent. These are not code changes — they are changes to how your execution environment behaves.

## Job Completion No Longer Requires Hitting Iteration Limits

When you run as a `full_job` (background scheduled routine), you will no longer fail from exhausting iterations without producing output.

As you approach the iteration limit, you will notice two things:

- At `max_iterations - 1`, you will receive a system message prompting you to wrap up and provide your final answer.
- At `max_iterations`, your tools will be removed from the system prompt. You will no longer see or call tools — you must produce a text response.

You can also call the `done` tool at any time to signal completion immediately with a summary of what you accomplished. This is the preferred way to finish.

Jobs that previously failed with "Maximum iterations exceeded" will now complete successfully.

## Builtin Tools No Longer Flagged as "Broken"

When you call builtin tools like `memory_read`, `message`, `job_status`, or `routine_update` and they return errors (e.g., reading a nonexistent memory path, providing an invalid job ID), those errors will no longer trigger the self-repair system.

Previously, each builtin tool error incremented a failure counter. After five failures, the system flagged those tools as "broken" every 60 seconds and attempted repairs it could never complete — causing perpetual log spam.

Now, only WASM tool failures are tracked for self-repair. Your builtin tools can fail naturally from invalid parameters or missing paths without generating false "broken" alerts. The recurring log noise will stop.

## Empty Timezone No Longer Blocks Routine Updates

When you call `routine_update` to change a routine's prompt or description, you will no longer encounter timezone validation errors from empty strings.

Previously, including `"timezone": ""` in your parameters — even unintentionally — caused the update to fail with "invalid IANA timezone: ''". This was especially problematic for self-modifying routines that only wanted to update the prompt.

Now, empty timezone strings are treated as "not provided." The field is simply skipped. You can update prompts, descriptions, and other fields freely without worrying about timezone validation.

## Duplicate Routine Names Give Clear Errors

When you call `routine_create` with a name that already exists, you will now receive a clear, actionable error message: "Routine 'X' already exists. Use routine_update to modify it."

Previously, the database rejected the insert with a raw unique constraint violation — a confusing PostgreSQL error that gave no guidance on what to do next. This was especially unhelpful for self-healing routines trying to recreate themselves.

The new error makes it immediately obvious what went wrong and what action to take instead.
