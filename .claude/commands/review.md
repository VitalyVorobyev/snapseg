# Review: Audit code against the conventions

You are the dispatcher for the `rust-qa-officer` agent. Hand the
target scope to that agent for a convention-compliance audit and
report back the punch list.

## Input

Target: `$ARGUMENTS`

Conventions for resolving `$ARGUMENTS`:
- Empty: audit files changed in the most recent commit
  (`git diff HEAD~1 --name-only`).
- A path: audit that file or directory.
- A crate name (e.g. `snapseg-models`): audit `crates/<name>/`.
- The literal `workspace`: audit every `.rs` file under `crates/`.

## Process

1. **Dispatch via the Agent tool** to `rust-qa-officer` with the
   resolved scope in the brief. The agent's own report format applies.

2. **Read the punch list.** The officer reports severity tiers (P0
   through P3). Do not auto-fix; the architect decides which to fix.

3. **Summarize back to the user.**
   - Headline counts per severity.
   - Top three findings inline.
   - A recommended next step: `/gate-check` if all green, or a list
     of follow-up task IDs (mapped to the findings) the user can
     accept or refine.

## Rules

- Do **not** edit code from this command.
- If the gate (`fmt` / `clippy` / `check` / `test` / `doc`) fails
  during the audit, surface that as the top finding and stop —
  conventions don't matter if the workspace doesn't build.
