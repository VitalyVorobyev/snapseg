# Orchestrate: Run a milestone of independent tasks

You are the orchestrator. Given a milestone or a list of task IDs,
dispatch each task to its agent in dependency order, in parallel
where possible, and aggregate the results.

## Input

Milestone or task list: `$ARGUMENTS`

- `M<n>` — orchestrate every task under milestone `n` in
  `docs/ROADMAP.md` that isn't marked done.
- `M<n>-T01, M<n>-T02, …` — orchestrate the listed tasks.

## Process

1. **Read the task list.** From `docs/ROADMAP.md`, pull the spec for
   every task in scope. If any task is underspecified, stop and call
   `/architect` first.

2. **Compute the dependency graph.** Tasks declare dependencies in
   their `**Dependencies**` line. Build a topological order. Tasks at
   the same level can run in parallel.

3. **Dispatch in waves.** For each wave:
   - Read each task's suggested agent.
   - Launch one Agent call per task. Use `run_in_background: true`
     when the wave has more than one task so they actually run in
     parallel.
   - Wait for the wave to complete.

4. **After each wave.**
   - Run the pre-commit gate (`fmt`, `clippy`, `check`, `test`).
   - If any task's deviations matter to a later task, surface them
     before kicking off the next wave.
   - Commit each completed task with a message referencing the ID.

5. **Final report.**

   ```
   ## Orchestration — <milestone or task list>

   | Task | Agent | Status | Commit |
   |---|---|---|---|
   | M2-T01 | quick-implementer | done | abc1234 |
   | M2-T02 | model-adapter-integrator | blocked | — |
   | M2-T03 | deep-implementer | done | def5678 |

   ## Gate
   <pre-commit gate result for the final state>

   ## Blocked / deviations
   Per-task notes for anything not green.

   ## Recommended next step
   /gate-check | /architect <area> | merge
   ```

## Rules

- One task per agent dispatch. Do not bundle tasks into a single
  brief.
- Never silently widen a task's scope. If an agent surfaces a needed
  related change, treat it as a deviation and let the architect
  decide whether to spin off a new task.
- Never push without explicit authorisation in the current
  conversation.
