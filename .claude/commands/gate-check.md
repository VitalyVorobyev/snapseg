# Gate-check: Pre-release readiness audit

Run the full pre-release gate. This is the audit that must pass before
tagging a release or merging a milestone branch.

## Input

Optional milestone tag: `$ARGUMENTS`

If supplied, the report headline references that milestone (e.g.
`M2 — Label flywheel: gate report`).

## Steps

Run each step. Stop on the first failure unless the user explicitly
asked to continue.

1. **Workspace builds clean.**
   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo check --workspace
   ```

2. **Tests pass.**
   ```bash
   cargo test --workspace
   ```

3. **Doc coverage.**
   ```bash
   cargo doc --workspace --no-deps
   ```
   Inspect the output for `warning: missing documentation`. The bar is
   zero on any `pub` item.

4. **Security audit.**
   ```bash
   cargo audit
   ```
   Tolerate yanked dependencies if they're transitive and there's no
   fix upstream; flag anything else.

5. **Smoke launch.** Build the app, launch with a short timeout to
   confirm it starts, kill cleanly.
   ```bash
   cargo build -p snapseg-app
   ./target/debug/snapseg &
   APP=$!
   sleep 3
   kill -0 $APP && kill $APP && echo "smoke ok"
   ```

6. **Convention audit.** Dispatch `rust-qa-officer` on `workspace`.
   Report any P0/P1 findings as gate failures.

## Report format

```
## snapseg gate report — <milestone or HEAD>

| Step | Result |
|---|---|
| fmt | ok / fail |
| clippy | ok / fail |
| check | ok / fail |
| test | ok / fail (N passed, M failed) |
| doc | ok / N missing-doc warnings |
| audit | ok / N advisories |
| smoke | ok / fail |
| qa-officer | ok / P0=n P1=n P2=n |

## Findings
<top 5 issues if any>

## Recommendation
Ship / hold / hold-with-list
```
