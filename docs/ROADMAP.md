# snapseg roadmap

Milestone-level plan. Tasks within each milestone are spelled out via
`/architect <area>` and tracked as `M<n>-T<nn>` IDs (see the
[architect command](../.claude/commands/architect.md)). The `/orchestrate`
command runs a milestone of independent tasks.

The data-flywheel framing — *smart models produce labels, small models
learn from them* — is the load-bearing product axis. Milestones M2 and
M7 are the two halves of that loop; everything else exists to make M2
trustworthy and M7 fast.

## Status

Current milestone: **M1 — Pipeline polish.** Workspace builds clean,
the binary boots, MobileSAM runs end-to-end, agents and skills are in
place. We are not yet producing reusable labels.

| Milestone | Theme | Status |
|---|---|---|
| M1 | Pipeline polish, agents, skills, CI | in progress |
| M2 | Label flywheel (P0) | next |
| M3 | Subpixel edge refinement | scheduled |
| M4 | Multi-mask output + IoU selection | scheduled |
| M5 | Threaded inference, cancellation | scheduled |
| M6 | Second model family (RITM or FocalClick) | scheduled |
| M7 | Fine-tune loop closure | scheduled |

---

## M1 — Pipeline polish

Goal: a workspace that any contributor (human or subagent) can land
work in without rediscovering conventions. The /architect → /implement →
/review workflow is exercised on real tasks at least once before M1
ships.

- [x] Workspace scaffold (six crates) and MobileSAM end-to-end loop.
- [x] LICENSE, README, CLAUDE.md, AGENTS.md.
- [x] `.claude/{agents,commands,skills}` kit and `settings.local.json`.
- [x] Code-hygiene sweep — module split, helper extraction, rustdoc.
- [x] CI (`ci.yml`, `audit.yml`).
- [x] This roadmap.
- [ ] Workflow dogfooding: drive *one* task end-to-end through
      `/architect → /implement → /review` to validate the dispatch
      plumbing in a real session.

## M2 — Label flywheel (P0)

Goal: every successful interactive segmentation can be saved to disk
in a stable, dataset-shaped format. Without this, every other downstream
plan is on hold.

Spec is in the [`snapseg-label-export`](../.claude/skills/snapseg-label-export/SKILL.md)
skill. Decomposition:

- **M2-T01** — `snapseg-labels` crate scaffold (`Cargo.toml`, `lib.rs`).
- **M2-T02** — `Label` / `Provenance` / `LabelDir` types.
- **M2-T03** — `LabelDir::save()` with atomic write (`<sha>.partial/` →
  `<sha>/` move).
- **M2-T04** — `Provenance` builder in `snapseg-app`, wired to the
  active model + the latest `SegmentationResult`.
- **M2-T05** — "Save label…" side-panel button.
- **M2-T06** — `snapseg-labels` binary with `coco` subcommand.
- **M2-T07** — Unit + roundtrip tests on the COCO conversion.

Suggested agents: M2-T01, M2-T02, M2-T04, M2-T05, M2-T07 →
`quick-implementer`. M2-T03, M2-T06 → `deep-implementer`.

## M3 — Subpixel edge refinement

Goal: the polygon written into a label is precise to ~0.1 px on a
well-lit step edge, not the blocky mask boundary.

- **M3-T01** — Marching-squares contour extraction in `snapseg-edges`.
- **M3-T02** — Gaussian-smoothed first derivative along the local
  normal; parabolic fit; subpixel offset per vertex.
- **M3-T03** — `refine_polygon()` API contract finalised; placeholder
  body in `snapseg-edges/src/lib.rs` replaced.
- **M3-T04** — Synthetic-image test: step edge with known subpixel
  location, refined polygon within 0.2 px.
- **M3-T05** — Side-panel toggle "Refine subpixel edges"; refined
  polygon overlaid on the canvas.
- **M3-T06** — Subpixel polygon written to `labels/.../polygon.json`.

All deep-implementer (algorithmic) except M3-T05 (quick-implementer).

## M4 — Multi-mask output + IoU selection

Goal: when the SAM decoder returns K masks, snapseg picks the one with
the highest IoU prediction instead of always the first.

- **M4-T01** — Decode all K masks instead of just mask 0.
- **M4-T02** — Read `iou_predictions` output; pick argmax K.
- **M4-T03** — Side-panel toggle to expose all K masks for hand
  inspection (debug).

deep-implementer (the boundary between "right mask" and "wrong mask"
is judgement-shaped).

## M5 — Threaded inference + cancellation

Goal: clicking no longer freezes the UI. The encoder pass runs on a
worker thread; subsequent decoder passes are cancelable so a fast
clicker doesn't queue up stale predictions.

- **M5-T01** — Worker thread + bounded channel between UI and segmenter.
- **M5-T02** — Cancellation token honored at the segmenter boundary
  (best-effort — ort itself isn't cancelable mid-op, but we can drop
  stale results before they reach the UI).
- **M5-T03** — Side-panel state machine: idle / embedding / decoding;
  visible spinner.

deep-implementer.

## M6 — Second model family

Goal: prove the `InteractiveSegmenter` trait is general by adding a
non-SAM adapter. Most operational benefit on industrial images comes
from a RITM-style model that's small enough to fine-tune in M7.

- **M6-T01** — RITM adapter implementing `set_image` / `segment`
  against the canonical RITM ONNX schema. Reuses the existing
  stub in `crates/snapseg-models/src/ritm.rs`.
- **M6-T02** — Click-map rasterizer (Gaussian disks) in
  `snapseg-runtime::preprocess`, shared between RITM and FocalClick.
- **M6-T03** — Model picker in the side panel that surfaces every
  family present in `models.toml`.

model-adapter-integrator owns M6-T01 and M6-T02.

## M7 — Fine-tune loop closure

Goal: a user with ≥200 labels produces a domain-specialized model and
deploys it via `models.toml`.

Spec is in the
[`snapseg-finetune-recipe`](../.claude/skills/snapseg-finetune-recipe/SKILL.md)
skill. Decomposition:

- **M7-T01** — Python tooling repo (`tools/finetune/`) outside the
  Rust workspace. Dataset adapter from snapseg label format to the
  family's training loader.
- **M7-T02** — RITM fine-tune script (default first family).
- **M7-T03** — ONNX re-export with the snapseg adapter's expected
  I/O signature.
- **M7-T04** — Documentation on adding the new model to `models.toml`
  with the right sha256.
- **M7-T05** — Eval harness: IoU @ N clicks on the holdout set;
  side-by-side base vs fine-tuned.

Owner mix: M7-T01, T02, T03 are Python work; deep-implementer with
extra context on the chosen training library. T04, T05 are
quick-implementer.

---

## Out of scope

- Video / multi-frame propagation.
- Multi-class semantic segmentation. snapseg is binary (foreground /
  background) by design; multi-class is a different product.
- Cloud-side inference. Everything is local.
- Python *runtime* dependency in the shipped binary. Training tooling
  (M7) is Python; that's fine because it's offline.
- Colour image input. snapseg is grayscale-only — `snapseg-core::GrayImage`
  is the workspace contract and RGB-pretrained models replicate the
  channel inside the adapter. See `CLAUDE.md` "Image colour" for the
  full reasoning.

## Decisions parking lot

Things we've punted; revisit when reaching the listed milestone:

- **CoreML EP at runtime.** Today: build supports the feature flag,
  default `RuntimeConfig` falls back to CPU. Revisit at M5 when
  threaded inference is the differentiator.
- **WGPU / wonnx backend.** Interesting for sealed deployments
  without onnxruntime; deferred. Revisit if `load-dynamic` becomes a
  deployment burden.
- **Mask-export resolution.** Today: mask PNG is at original image
  resolution. SAM emits at original resolution natively; some
  community exports leave masks at low-res. Revisit if we ship the
  RITM adapter (M6) — its mask resolution is configurable.
- **Label-export schema versioning.** Today: `schema_version = "1.0"`.
  Bump deliberately; older labels must remain readable.
