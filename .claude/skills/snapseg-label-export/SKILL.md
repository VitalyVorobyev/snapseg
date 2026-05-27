---
name: snapseg-label-export
description: >
  Specification and recipe for capturing labels from a finished snapseg
  PromptSession so they feed back into model training. Invoke whenever
  the user wants to export labels, save a labeled dataset, build a
  COCO-shaped dataset, capture clicks + masks for later fine-tuning,
  add a "save label" feature, or design the label format. Also invoke
  for: "how do I save labels", "export a dataset", "build a label set
  from snapseg", "convert to COCO", "what's in a snapseg label". Covers
  on-disk layout, provenance, COCO conversion, and the architect's task
  spec for the `snapseg-label-export` module.
---

# snapseg: label export — capturing labels from a session

You are guiding the design or implementation of label export in
snapseg. This is M2 in `docs/ROADMAP.md` and is the highest-leverage
feature on the path from "click-driven mask viewer" to "label factory
that feeds a domain-specialized model".

Read `CLAUDE.md` and `AGENTS.md` first; this skill is the *what* and
*how*, not the *whether*.

## Why this is M2 (and not M4)

snapseg's framing is a **data flywheel**: smart models produce labels
with operator clicks; smaller, domain-specialized models are then
fine-tuned on those labels. Without a label-export workflow, the
flywheel doesn't spin. Subpixel edges (M3) and multi-mask refinement
(M4) make the labels *sharper*, but a not-yet-sharp label is more
useful than no label at all.

## What a label is

One label = one segmented region on one image. Stored as a directory
under `labels/`:

```
labels/<image-sha8>/
├── image.png         # the source image, byte-identical copy
├── prompts.json      # the PromptSession that produced the mask
├── mask.png          # binary PNG, 0 / 255
├── logits.png        # 16-bit grayscale PNG of the logits (optional)
├── polygon.json      # subpixel polygon (M3+)
└── meta.toml         # provenance
```

`<image-sha8>` is the first 8 hex chars of the SHA-256 of `image.png`.
Collisions are handled by appending `-2`, `-3`, etc. If the operator
labels the same image twice (two different regions), append `-r2`,
`-r3` for *regions* on the same source.

### `prompts.json`

```json
{
  "session": [
    {"type": "click", "x": 412.5, "y": 287.0, "polarity": "positive", "t_ms": 0},
    {"type": "click", "x": 460.0, "y": 290.0, "polarity": "negative", "t_ms": 1247},
    {"type": "box",   "x0": 380.0, "y0": 260.0, "x1": 520.0, "y1": 320.0, "t_ms": 2155}
  ]
}
```

Coordinates are in the source image pixel frame. `t_ms` is offset
from the first prompt (useful for ergonomics / fatigue analysis later).

### `meta.toml`

```toml
schema_version = "1.0"
created_at = "2026-05-27T14:23:11Z"
source_image_sha256 = "..."
source_image_path = "/Volumes/data/glue_runs/2026-05-27/IMG_0042.tif"
source_image_width = 1920
source_image_height = 1080

[model]
registry_name = "mobile-sam"
family = "mobile_sam"
encoder_sha256 = "..."
decoder_sha256 = "..."

[runtime]
execution_provider = "CoreML"
encoder_ms = 1247
decoder_ms = 38
snapseg_commit = "abc1234"

[operator]
note = "glue bead on rear bracket, well-lit"
quality = "good"   # good | needs_review | reject
```

### `mask.png`

Binary PNG (single-channel u8), `0` background, `255` foreground.
**Identical resolution to `image.png`.** No upscaling, no downsampling
in the export.

### `logits.png`

Optional. 16-bit grayscale PNG. The decoder's raw float logits
shifted and scaled to `[0, 65535]`; preserves the soft boundary.
Useful for active-learning sample selection later.

## Implementation surface

A new module (probably eventually a new crate `snapseg-labels`) owns:

```rust
pub struct Label { /* paths to the artifacts above */ }
pub struct LabelDir { /* labels/ root */ }

impl LabelDir {
    pub fn new(root: PathBuf) -> Self;

    /// Capture a label from the current session.
    pub fn save(
        &mut self,
        image: &GrayImage,
        source_path: Option<&Path>,
        session: &PromptSession,
        result: &SegmentationResult,
        provenance: &Provenance,
    ) -> Result<PathBuf, LabelError>;
}
```

The app gets a "Save label…" button in the side panel that builds a
`Provenance` and calls `LabelDir::save()`. The button is enabled
only when both an image and a result-bearing mask are present.

## COCO conversion

A `snapseg-labels` CLI subcommand rolls a `labels/` directory into a
single COCO JSON for handoff to a training pipeline:

```bash
snapseg-labels coco --in ./labels --out dataset_2026-05.json
```

The COCO conversion preserves provenance in the `image.note` field
so a downstream training tool can filter by `quality`, model family,
or commit hash.

## Pitfalls

- **Don't re-encode the source image.** The label is meaningless if
  the operator can't reproduce it. Copy bytes; don't read-and-write
  through `image` crate's encoder. Use `std::fs::copy`.
- **Don't store mask edges from subpixel refinement.** Until M3
  lands, the `polygon.json` field is absent; downstream tooling
  must tolerate that.
- **Sha-bucketing.** Many operators on the same image (two ops
  labeling two regions on the same part) need unique directories.
  Use `<sha>-r2`, `-r3` for multiple regions on the same source.
- **Atomicity.** Write to a `labels/<sha>.partial/`, then `mv` to
  `labels/<sha>/` once all files are present. A label that exists
  but is missing `mask.png` is worse than no label.

## Architect's task spec for M2

Use the `/architect` flow to break label export into:

- **M2-T01.** `snapseg-labels` crate scaffold (Cargo.toml, lib.rs).
- **M2-T02.** Label / Provenance / LabelDir types in `snapseg-labels`.
- **M2-T03.** `LabelDir::save` implementation (atomic dir mv).
- **M2-T04.** `Provenance` builder in `snapseg-app`, hooked to the
  active model and the latest `SegmentationResult`.
- **M2-T05.** "Save label…" button in the side panel.
- **M2-T06.** `snapseg-labels` binary with `coco` subcommand.
- **M2-T07.** Unit tests on the COCO roundtrip.

Each task is implementer-shaped; M2-T03 and M2-T06 are
`deep-implementer`, the rest are `quick-implementer`.
