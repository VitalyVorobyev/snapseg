---
name: snapseg-workspace-review
description: >
  Pre-release workspace audit for snapseg, tuned to its specific layering
  (snapseg-core at the base, snapseg-app at the apex), its trait contract
  (InteractiveSegmenter), its registry rules, and its data-flywheel
  feature axis. Invoke for "review snapseg workspace", "pre-release
  check", "audit snapseg before tagging", "is snapseg ready to ship",
  "snapseg workspace review", "snapseg release readiness". Produces a
  REVIEW.md artefact like rust-workspace-review but with snapseg-specific
  findings: adapter contract compliance, label-export readiness,
  registry sha256 hygiene, EP-selection robustness, classical-edges
  status.
---

# snapseg: workspace review

You are auditing the snapseg workspace for release readiness. This is
a snapseg-specific specialization of the generic `rust-workspace-review`
skill — keep the broad checklist from that skill in mind, but layer in
the snapseg-specific axes below.

Produce a `REVIEW.md` at the repo root with findings organized by
severity (P0 / P1 / P2 / P3). Then break P0/P1 findings into task
specifications the architect can hand to implementers.

## Read first

- `CLAUDE.md` and `AGENTS.md` — convention ground truth.
- `docs/ROADMAP.md` — what milestone we're auditing against.
- The latest 20 commits on `main` (`git log --oneline -20`).

## Audit axes

### 1. Layering (P0 if violated)

Verify every crate's `Cargo.toml` against AGENTS.md §1:

| Crate | May depend on (workspace) |
|---|---|
| `snapseg-core` | nothing |
| `snapseg-runtime` | `snapseg-core` |
| `snapseg-registry` | nothing in workspace |
| `snapseg-edges` | `snapseg-core` |
| `snapseg-models` | `snapseg-core`, `snapseg-runtime` |
| `snapseg-app` | every other |

Also: `snapseg-registry` does **not** pull `ort`. `snapseg-edges`
does **not** pull `ort` or ML deps. Nothing depends on `snapseg-app`.

### 2. Adapter contract (P0)

For every adapter in `crates/snapseg-models/src/<family>.rs`:

- Module-level rustdoc with the ONNX I/O schema in the same shape
  as `mobile_sam.rs` (input names + shapes + dtypes; output names +
  shapes).
- `from_parts(name, parts, input_size, config)` constructor.
- `Capabilities` honestly declared (false for what the export
  doesn't support).
- `set_image` returns `Ok(())` only when an embedding state can be
  produced.
- `segment()` is a thin orchestrator (no body over 30 LOC) calling
  named helpers.
- No `.unwrap()` / `.expect()` outside `#[cfg(test)]`.
- ort errors are mapped to `SegError::Backend(String)` with
  contextual prefixes.

### 3. Registry hygiene (P1)

- Every `[[model]]` in `models.toml` either has a real sha256 OR an
  all-zeros sha256 explicitly marked as "trust cache".
- URLs that are placeholders (`example.invalid`, `huggingface.co/snapseg/...`)
  are not in models intended to ship. They're fine in development.
- `ModelEntry::parts` declares at least one part per model.
- `family` slugs match adapter modules (no orphan slugs, no orphan
  modules).

### 4. EP selection (P1)

`snapseg-runtime::select_execution_providers` (once extracted) must
gracefully skip EPs that aren't built in via their feature flag.
Default `RuntimeConfig` falls back to CPU.

### 5. UI freeze risk (P2, P1 if shipped to operators)

- Inference still runs on the UI thread? If yes, that's a P2 in
  alpha, P1 in beta — but it's a known M5 item, so flag as P2 with
  M5 reference.

### 6. Subpixel edges status (P2)

`snapseg-edges` is M3. If still a placeholder body returning the
naive boundary, note as a known gap.

### 7. Label export readiness (P0 once M2 lands)

Once M2 is in scope:
- `snapseg-labels` crate (or module) exists and matches the schema
  in the `snapseg-label-export` skill.
- "Save label…" button present in the app.
- Atomic write semantics (write to `.partial/`, `mv` to final).
- COCO CLI subcommand exists and roundtrips a known fixture.

### 8. Standard Rust hygiene (rolled up from rust-workspace-review)

- `cargo fmt --all --check` clean.
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- `cargo check --workspace` clean.
- `cargo test --workspace` passes.
- `cargo doc --workspace --no-deps` no missing-docs warnings on
  public items.
- `cargo audit` no unhandled advisories.
- `LICENSE`, `README.md`, `CLAUDE.md`, `AGENTS.md` exist and are
  up to date.

### 9. Convention drift

Run the `rust-qa-officer` agent on the full workspace. Roll its P0
and P1 findings into the review.

## Output format

Write to `REVIEW.md` at the repo root:

```
# snapseg workspace review — <date>

## Gate
| Step | Result |
|---|---|
| layering | ok / fail |
| adapter contract | ok / N issues |
| registry | ok / N issues |
| EP selection | ok / N issues |
| UI freeze | known (M5) |
| subpixel edges | known (M3) |
| labels | N/A (M2 not yet started) |
| fmt | ok |
| clippy | ok |
| test | ok |
| doc | ok |
| audit | ok |

## Findings

### P0
- [crate/file.rs:LINE] short title — description, recommended fix.

### P1
- ...

### P2
- ...

### P3
- ...

## Recommended task specs
For each P0/P1, output a task spec in the M<n>-T<nn> format the
`/architect` command produces, ready for the user to accept.

## Recommendation
Ship | Hold for M<n> | Hold with list
```

Then post a summary back to the user with headline counts and the
top three findings.
