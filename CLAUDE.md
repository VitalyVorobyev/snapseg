# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**snapseg** is a Rust workspace for **interactive deep-model image segmentation** of industrial parts, built around a deliberate data-flywheel framing: smart promptable models (MobileSAM, eventually RITM / FocalClick / SAM2-Tiny) produce labels with operator clicks; smaller, faster, domain-specialized models are then fine-tuned on those labels. The current product is the operator-facing tool that turns clicks into datasets.

The runtime story is **all-Rust, no Python**: `ort` 2.x (load-dynamic) wraps onnxruntime; `egui` drives the desktop UI; `ndarray` is the tensor type; classical subpixel edge refinement sharpens the boundaries that go into labels. macOS dev expects `brew install onnxruntime`; Linux dev expects the distribution's onnxruntime package.

## Build & test commands

```bash
# Build
cargo build --workspace
cargo build -p snapseg-app --release

# Type-check only
cargo check --workspace

# Test
cargo test --workspace

# Lint
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# Doc coverage
cargo doc --workspace --no-deps

# Smoke launch (UI; needs onnxruntime + MobileSAM ONNX files for full loop)
./target/debug/snapseg
```

`ort` is pinned to `=2.0.0-rc.10` in the workspace `Cargo.toml`. rc.12 fails to compile in load-dynamic mode (missing VitisAI symbol in OrtApi); do not bump without verifying the fix lands upstream.

## Workspace architecture

Six crates with strict layering:

```
snapseg-app          (egui binary `snapseg`, single user-facing entry)
       ↓
snapseg-models       (per-family adapters: mobile_sam, ritm, focalclick, ...)
       ↓
snapseg-runtime      snapseg-registry     snapseg-edges
(ort wrapper,        (TOML model zoo +    (classical
 EP selection,       sha256 download +    subpixel edge
 preprocessing)      cache resolution)    refinement)
       ↓                    ↓                    ↓
                    snapseg-core
                (types, traits, no I/O,
                 no ort, no GUI)
```

**Dependency rules**
- `snapseg-core` depends on nothing in the workspace. It defines `Prompt`, `PromptSession`, `Capabilities`, `GrayImage`, `SegmentationResult`, and the `InteractiveSegmenter` trait. No `ort`, no `egui`, no `image`.
- `snapseg-runtime` may depend on `snapseg-core` and `ort`. Nothing else from the workspace.
- `snapseg-registry` may depend on `serde`, `toml`, `sha2`, `directories`, optionally `reqwest`. No `ort`. No `snapseg-runtime`.
- `snapseg-edges` may depend on `snapseg-core` and `ndarray`. No `ort`. No ML.
- `snapseg-models` may depend on `snapseg-core`, `snapseg-runtime`, and `ort`. One module per family.
- `snapseg-app` is the apex: depends on every other workspace crate. No other crate may depend on it.

**Where code goes**
- New model family → `snapseg-models/src/<family>.rs`. Trigger `model-adapter-integrator`.
- Tensor / preprocessing helpers shared across adapters → `snapseg-runtime::preprocess`.
- Registry / cache / download logic → `snapseg-registry`.
- Classical CV (edges, contours, polygon ops) → `snapseg-edges`.
- UI widgets, dialogs, panels, canvas → `snapseg-app/src/<area>.rs`.
- Cross-cutting types (everything an adapter author needs) → `snapseg-core`.

## Code style and conventions

These are the rules `rust-qa-officer` audits. They override taste; if a rule below conflicts with personal style, the rule wins.

### Naming

- Trait-implementing types: `<Family>Segmenter` (`MobileSamSegmenter`, `RitmSegmenter`, `FocalClickSegmenter`).
- Constructors from a registry-resolved part map: `from_parts(name, parts, input_size, config)`.
- Registry types: `Registry`, `ModelEntry`, `ModelPart`. Family slug is lowercase snake (`mobile_sam`, `ritm`, `focalclick`).
- Coordinate-bearing types use `Point2 { x, y }` from `snapseg-core`, never bare `(f32, f32)` in public signatures.
- Errors: `SegError` for the segmenter contract; `BackendError` for runtime; `RegistryError` for registry. Domain crates re-throw via `From` rather than wrapping in `anyhow` at API boundaries.

### Module + function size

- Modules over ~200 LOC must be either data declarations (a long `enum`, a registry of constants) or be split.
- Functions over 60 LOC must carry a `// why this is long:` comment justifying it, or be split.
- One responsibility per module; the file name is the responsibility, not a noun (`canvas.rs`, not `widgets.rs`).

### Documentation

- Every `pub` item has at least one rustdoc line. `pub fn` whose behaviour isn't obvious from the name carries a `# Errors` block (when fallible) and/or a `# Examples` block.
- Tensor-bearing functions document layout at the point the tensor enters or leaves: `// input: NCHW [1, 3, S, S]`, `// output: HxW bool mask`.
- New adapter files must lead with a module-level rustdoc describing the ONNX I/O contract (input names + shapes + dtypes; output names + shapes).

### Tensors

- ndarray dimensions: `Array4<f32>` for `NCHW`, `Array3<f32>` for `CHW`, `Array2<bool>` / `Array2<f32>` for masks / logits. Don't introduce `ArrayD` in public signatures unless the rank is genuinely dynamic.
- SAM-style pixel-space normalization (`SAM_PIXEL_MEAN` / `SAM_PIXEL_STD`) lives in `snapseg-runtime::preprocess` and is shared, not duplicated.
- Float comparisons in tests use approximate equality with a stated tolerance, never `==`.

### Error handling

- No `.unwrap()` / `.expect()` in library code outside `#[cfg(test)]` and `examples/`. The compiler must guarantee the invariant, or it must be expressed as an `Err` variant.
- Adapter `segment()` returns `SegError::NoImage` before `set_image`; `SegError::UnsupportedPrompt` when the model's `Capabilities` doesn't cover the user's prompt; `SegError::Backend(String)` for ort or shape errors.

## Subagent dispatch rules

The architect (Claude in the main conversation) plans, defines task specs, and dispatches a subagent for implementation. The subagent commits code; the architect reviews and integrates.

Choose the subagent by the shape of the work, not the area:

| Work shape | Agent |
|---|---|
| Mechanical, fully specifiable in the brief (rename, add field, mirror a pattern across N files, port a known transformation) | `quick-implementer` (Sonnet) |
| Judgement, debugging, multi-crate API shape changes, design-y prose where the value is in the categorisation | `deep-implementer` (Opus) |
| Convention compliance check, style audit, doc coverage audit, layering audit | `rust-qa-officer` |
| Anything touching `snapseg-models/src/<family>.rs` or the contract between `models.toml` / `ModelEntry` / `Backend` / `InteractiveSegmenter` | `model-adapter-integrator` (Opus) |

Each agent definition is in `.claude/agents/<name>.md`. Slash commands in `.claude/commands/` drive the workflow: `/architect`, `/implement <task-id>`, `/review [target]`, `/gate-check`, `/orchestrate`.

## Pre-commit quality gates

Before committing on any branch:

1. `cargo fmt --all`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo check --workspace`
4. `cargo test --workspace`

Before tagging a release:

5. `cargo doc --workspace --no-deps` — no missing-docs warnings on public items.
6. `cargo audit`
7. `/gate-check` — a smoke launch + the above all green.

## Reference layout

```
snapseg/
├── Cargo.toml              # workspace
├── LICENSE
├── README.md               # user-facing
├── CLAUDE.md               # this file
├── AGENTS.md               # public agent roster
├── docs/
│   ├── ROADMAP.md          # milestone-level plan
│   └── deep-research-report.md   # background research
├── crates/
│   ├── snapseg-core/
│   ├── snapseg-runtime/
│   ├── snapseg-models/
│   ├── snapseg-registry/
│   ├── snapseg-edges/
│   └── snapseg-app/
├── models.toml             # default model registry
├── .claude/
│   ├── agents/             # quick-implementer, deep-implementer, rust-qa-officer, model-adapter-integrator
│   ├── commands/           # architect, implement, review, gate-check, orchestrate
│   ├── skills/             # snapseg-add-model, snapseg-label-export, snapseg-finetune-recipe, snapseg-workspace-review
│   └── settings.local.json
└── .github/workflows/      # ci.yml, audit.yml
```
