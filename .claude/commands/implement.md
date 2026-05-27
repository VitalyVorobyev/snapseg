# Implement: Execute a Task Specification

You are the dispatcher. Read the task spec, decide which subagent
should execute it per CLAUDE.md's dispatch table, and hand it off.

## Input

Task ID or inline specification: `$ARGUMENTS`

## Process

1. **Load the task.**
   - If `$ARGUMENTS` matches `M<n>-T<nn>`, find the spec in
     `docs/ROADMAP.md`.
   - Otherwise treat `$ARGUMENTS` as the inline spec.

2. **Choose the agent.** From CLAUDE.md's dispatch table:

   | Work shape | Agent |
   |---|---|
   | Mechanical, fully specifiable in the brief | `quick-implementer` |
   | Judgement, debugging, multi-crate API changes, design prose | `deep-implementer` |
   | Convention compliance / style audit / layering audit / doc audit | `rust-qa-officer` (use `/review` instead — it dispatches automatically) |
   | Adapter shape work, ONNX I/O mismatches, anything touching `snapseg-models/src/<family>.rs` or the registry contract | `model-adapter-integrator` |

   If the task is borderline, prefer `deep-implementer` over
   `quick-implementer`. If it touches an adapter, always prefer
   `model-adapter-integrator`.

3. **Dispatch via the Agent tool.** The brief you pass MUST include:
   - The task title + ID.
   - The acceptance criteria (verbatim).
   - The files in scope.
   - The verification command (default: the pre-commit gate from
     CLAUDE.md).
   - The expected report shape (the agent's own report format
     applies; don't override).
   - A pointer to CLAUDE.md and AGENTS.md so the agent can read
     them fresh.

4. **After the agent reports back.**
   - Read the diff, not just the summary.
   - Re-run the pre-commit gate yourself to confirm.
   - If the agent reported deviations, decide whether to accept,
     fix-forward, or revert.
   - If accepted: commit on the user's behalf with a message
     referencing the task ID. Push only if the user has previously
     authorised pushes in this conversation.

## Rules

- Do **not** edit code yourself. The dispatcher dispatches.
- Do **not** silently mutate the task spec to fit what the agent
  produced. If the spec was wrong, surface it and update the spec.
- Trust but verify: the agent's report describes what they *intended*
  to do; the diff is what *happened*.
- Never push without explicit authorisation in the current
  conversation.
