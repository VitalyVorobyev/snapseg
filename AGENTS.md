# AGENTS.md — snapseg

This repository is a Rust workspace for **interactive deep-model image segmentation** with a data-flywheel framing: smart models produce labels via operator clicks, smaller models are fine-tuned on those labels. The published surface is currently a desktop binary (`snapseg`); the workspace crates are not yet released to crates.io.

This document is the agent-facing roster. The convention rules an agent enforces or follows are in [`CLAUDE.md`](./CLAUDE.md). Per-agent prompts live in [`.claude/agents/`](./.claude/agents/).

The codebase prioritizes:

- **No Python at runtime.** `ort` (load-dynamic) wraps onnxruntime; `egui` is the UI; `ndarray` is the tensor type.
- **Model-family-agnostic core.** The `InteractiveSegmenter` trait in `snapseg-core` is the only contract; adapters live in `snapseg-models` and dispatch on `ModelEntry::family`.
- **Determinism inside an adapter.** Same image + same `PromptSession` → same `SegmentationResult` (modulo float-EP non-determinism that ort itself owns).
- **Label-first workflow.** Outputs are exportable artifacts (mask PNG, logits, subpixel polygon, prompt history) consumable by downstream fine-tuning. The UI exists to feed this pipeline.

---

## 1) Layering rules

```
snapseg-app   ┐
              ├── may depend on every other crate
snapseg-models│
              ├── may depend on snapseg-core, snapseg-runtime, ort
snapseg-runtime, snapseg-registry, snapseg-edges
              ├── snapseg-runtime: snapseg-core, ort
              │   snapseg-registry: serde, toml, sha2, directories, optional reqwest — no ort
              │   snapseg-edges: snapseg-core, ndarray — no ort, no ML
snapseg-core  │   no workspace deps, no ort, no egui, no I/O
```

Forbidden inversions:

- `snapseg-core` depending on anything in the workspace (it's the contract).
- `snapseg-registry` pulling `ort` (registry is a metadata + download crate; ort lives in -runtime).
- Any crate depending on `snapseg-app`.

Where code goes:

| Concern | Crate / module |
|---|---|
| Trait, prompt types, capabilities, mask/logits, errors | `snapseg-core` |
| Adapter for a new model family | `snapseg-models/src/<family>.rs` |
| Tensor preprocessing shared across adapters | `snapseg-runtime::preprocess` |
| ort session creation, EP selection | `snapseg-runtime` |
| `models.toml` schema, cache resolution, sha256 download | `snapseg-registry` |
| Classical CV (contours, subpixel edges, polygon ops) | `snapseg-edges` |
| UI (panel, canvas, dialogs, mask textures) | `snapseg-app/src/<area>.rs` |
| Label export / fine-tune ergonomics (future) | TBD: likely a new `snapseg-labels` crate |

---

## 2) Goals and non-goals

### Goals

- Interactive segmentation that produces high-quality labels with as few clicks as the model allows.
- Multiple model families behind one trait; adapters interchangeable per image.
- Classical subpixel boundary refinement on top of model masks.
- Label-export and fine-tune feedback loop as a first-class feature (see `docs/ROADMAP.md` M2 / M7).
- Industrial-deployment friendliness: single static binary, optional accelerators, sealed (no Python).

### Non-goals (unless explicitly requested)

- Online / streaming inference. Single image, click-driven.
- Video / multi-frame propagation.
- Multi-class semantic segmentation. Binary (foreground / background) is the contract.
- Cloud-side inference. Everything is local.
- A Python runtime dependency. Training tooling **is** Python (we accept that for fine-tuning), but the shipped binary is not.

---

## 3) Workflow contract

The user is the human-in-the-loop. Claude in the main conversation is the **architect** — plans, defines task specs, dispatches an implementation subagent, reviews, integrates. Each commit on `main` ideally maps to one task brief.

Slash commands in `.claude/commands/` drive the workflow:

| Command | What it does |
|---|---|
| `/architect [area]` | Architect produces or refines task specifications in the `M<n>-T<nn>` format. |
| `/implement <task-id>` | Dispatcher reads CLAUDE.md's dispatch table and hands the task to `quick-implementer`, `deep-implementer`, or `model-adapter-integrator`. |
| `/review [target]` | Dispatches `rust-qa-officer` for a convention / style / doc / layering audit. |
| `/gate-check` | Pre-tag readiness: fmt + clippy + check + test + audit + doc + smoke. |
| `/orchestrate` | Multi-task runner for a milestone of independent tasks. |

---

## 4) Agent roster

Full prompts in `.claude/agents/`; this is the elevator-pitch view.

| Agent | Model | Use for |
|---|---|---|
| `quick-implementer` | Sonnet | Mechanical, fully specifiable: rename, add field, port a known shape, mirror a pattern across N files. |
| `deep-implementer` | Opus | Judgement work: new logic, debugging mask-quality failures with evidence, multi-crate API changes, design prose. |
| `rust-qa-officer` | inherit | Convention compliance against CLAUDE.md and this file. Layering, naming, doc coverage, function length, tensor-layout docs. |
| `model-adapter-integrator` | Opus | Anything touching `snapseg-models/src/<family>.rs` or the contract between `models.toml`, `ModelEntry`, `Backend`, and `InteractiveSegmenter`. Knows SAM / RITM / FocalClick I/O conventions. |

Project-local skills the architect can pull into a task brief:

| Skill | Purpose |
|---|---|
| `snapseg-add-model` | End-to-end checklist for adding a new model family adapter. |
| `snapseg-label-export` | Spec + recipe for capturing labels from a finished `PromptSession`. |
| `snapseg-finetune-recipe` | Fine-tune a chosen model family on snapseg-exported labels and re-export ONNX. |
| `snapseg-workspace-review` | snapseg-tuned variant of `rust-workspace-review`. |

---

## 5) Build, test, and quality gates

Before opening a PR:

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --workspace`
- `cargo test --workspace`

Before tagging:

- `cargo doc --workspace --no-deps` — no missing-docs warnings.
- `cargo audit`
- `/gate-check` — smoke launch the binary.

`ort` is pinned to `=2.0.0-rc.10` in the workspace `Cargo.toml`. Do not bump without verifying that the `OrtApi` VitisAI symbol issue in rc.12 is fixed upstream.

---

## 6) Conventions cheatsheet

The full ruleset is in [`CLAUDE.md`](./CLAUDE.md). The non-negotiables:

- One responsibility per module; the file name is the responsibility.
- No `.unwrap()` / `.expect()` in library code outside tests.
- Every `pub` item has rustdoc.
- Tensor layouts are documented at the function boundary they cross.
- Functions over 60 LOC carry a `// why this is long:` justification or get split.
- `snapseg-core` has zero workspace deps. Period.
