---
name: snapseg-add-model
description: >
  End-to-end checklist for adding a new model family adapter to snapseg.
  Invoke whenever the user wants to add a new ONNX model (RITM, FocalClick,
  EfficientSAM, SAM2-Tiny, or anything else), wire a new family into the
  registry, or debug an I/O-shape mismatch between an ONNX export and the
  snapseg adapter. Also invoke for: "add support for X", "wire up Y model",
  "how do I add an adapter for Z", "register a new model family", "extend
  the InteractiveSegmenter trait for a new prompt type". Covers file
  layout, ONNX I/O contract, click-map / box-prompt rasterization, registry
  entry shape, and smoke-test recipe.
---

# snapseg: add a new model family adapter

You are guiding the implementation of a new per-family ONNX adapter
in the snapseg workspace. Read `CLAUDE.md` and `AGENTS.md` before
starting; this skill is a checklist, not a substitute.

## When this skill applies

- Adding RITM, FocalClick, EfficientSAM, SAM2-Tiny, or a custom
  fine-tuned model.
- Repairing an existing adapter against a new ONNX export.
- Extending the `InteractiveSegmenter` trait for a new prompt
  modality (rare — needs architect sign-off first).

For routine work, dispatch the `model-adapter-integrator` agent and
hand them this skill as context.

## Step-by-step checklist

### 1. Pick the slug and the name

- **Family slug** (lowercase snake): `mobile_sam`, `ritm`,
  `focalclick`, `efficient_sam`, `sam2_tiny`. Used in `models.toml`
  `family = "..."` and as the source file name.
- **Display name** (kebab): `mobile-sam`, `ritm-hrnet18-cocolvis`.
  Used in `models.toml` `name = "..."` and in the app's model picker.

### 2. Identify the ONNX I/O schema

Inspect the ONNX file(s) outside the binary (Python, one-off):

```python
import onnx
m = onnx.load("model.onnx")
for x in m.graph.input:
    dims = [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]
    print(x.name, dims)
for y in m.graph.output:
    dims = [d.dim_value or d.dim_param for d in y.type.tensor_type.shape.dim]
    print(y.name, dims)
```

Categorise the family:

| Family | Shape |
|---|---|
| SAM-like | Two files: encoder + decoder. Encoder runs once per image; decoder runs per prompt. Click + box. |
| RITM-like | Single file. Inputs: image, click-map (2-channel Gaussian disks), prev-mask. Clicks + scribbles. |
| FocalClick | Single file (global) + optional second file (local refinement). Clicks + scribbles. |
| Other | Document explicitly in the new adapter's rustdoc. |

### 3. Add the registry entry

Edit `models.toml` (use `models-extra.toml` for personal pins).

Single-file model:

```toml
[[model]]
name = "ritm-hrnet18-cocolvis"
family = "ritm"
input_size = [1024, 1024]
license = "MIT"
notes = "RITM HRNet-18 on COCO+LVIS. Click-driven."

[[model.parts]]
name = "model"
url = "https://..."
sha256 = "<digest or all zeros to trust cache>"
size_mb = 41
```

Two-file model:

```toml
[[model]]
name = "mobile-sam"
family = "mobile_sam"
input_size = [1024, 1024]
license = "Apache-2.0"
notes = "Distilled SAM with TinyViT encoder."

[[model.parts]]
name = "encoder"
url = "https://..."
sha256 = "..."
size_mb = 27

[[model.parts]]
name = "decoder"
url = "https://..."
sha256 = "..."
size_mb = 16
```

The `all-zeros sha256` sentinel means "trust whatever is cached" —
use during early bring-up; replace with the real digest before
shipping.

### 4. Write the adapter

Create `crates/snapseg-models/src/<family>.rs`. Lead with a module-level
rustdoc that documents the full ONNX I/O schema in the same format
the existing `mobile_sam.rs` uses. The architect and qa-officer rely on
this comment to validate the adapter without re-inspecting the ONNX.

Required structure:

```rust
pub struct <Family>Segmenter {
    name: String,
    input_size: u32,
    /* one Backend per ONNX file */
    state: Option<EncodedImage /* or similar per-family state */>,
}

impl <Family>Segmenter {
    pub fn from_parts(
        name: String,
        parts: &HashMap<String, PathBuf>,
        input_size: u32,
        config: &RuntimeConfig,
    ) -> Result<Self, SegError> { /* load every Backend */ }
}

impl InteractiveSegmenter for <Family>Segmenter {
    fn name(&self) -> &str { ... }
    fn capabilities(&self) -> Capabilities { /* honest! */ }
    fn set_image(&mut self, image: &GrayImage) -> Result<(), SegError> { ... }
    fn segment(&mut self, session: &PromptSession) -> Result<SegmentationResult, SegError> {
        // Orchestrator: encode_prompts -> build_inputs -> run -> decode -> cache.
    }
}

// Named helpers, one responsibility each.
fn encode_prompts(...) -> ...
fn build_decoder_inputs(...) -> ...
fn decode_masks_output(...) -> ...
fn cache_low_res(...)
```

Mirror `mobile_sam.rs` for SAM-like families. For RITM-like, write a
`rasterize_click_map(session, h, w, sigma)` helper that builds the
2-channel Gaussian-disk input. For FocalClick, document the
global-vs-local mode switch in the file's rustdoc.

### 5. Register the family

Add to the dispatch in `crates/snapseg-models/src/lib.rs`:

```rust
pub mod <family>;
// ...
pub fn family_label(family: &str) -> Result<&'static str, SegError> {
    Ok(match family {
        "ritm" => "RITM",
        "focalclick" => "FocalClick",
        "mobile_sam" => "MobileSAM",
        "<family>" => "<Display Name>",   // ← add here
        ...
    })
}
```

### 6. Declare honest `Capabilities`

| Family | positive_clicks | negative_clicks | bbox | scribble | mask_input |
|---|---|---|---|---|---|
| SAM family | true | true | true | false | true |
| RITM | true | true | false | true | true |
| FocalClick | true | true | false | true | true |
| EfficientSAM | true | true | true | false | true |

If a particular export removes a capability (e.g. some MobileSAM
exports drop `has_mask_input` so iterative refinement is off), flag
the capability as false and note it in the rustdoc.

### 7. Smoke-test

Hand-test before reporting done:

1. `cargo build -p snapseg-app`
2. Launch `./target/debug/snapseg`.
3. Open a test image.
4. Use "Load <Family>…" or hand-edit the dialog flow if multi-part.
5. Click positive on an object; mask appears.
6. Right-click negative on background; mask shrinks.

Record observed encoder ms + decoder ms + mask coverage in the report.

### 8. Gate

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace
cargo test --workspace
cargo doc --workspace --no-deps
```

All five must be green before reporting done.

## Common pitfalls

- **Input name variance.** Community exports rename `input_image` →
  `image`, `point_coords` → `points`, etc. Use name-substring lookup
  with a fallback to the first input, but record the export's exact
  names in the adapter's rustdoc.
- **Pixel range mismatch.** SAM expects `[0, 255]` after subtracting
  the SAM pixel mean (123.675…). Don't divide by 255 first.
- **Click rasterization radius.** RITM's default Gaussian disk
  radius is **σ = 5** at the resized canvas. Smaller → ambiguous;
  larger → blurs adjacent clicks.
- **`has_mask_input` semantics.** Pass `0.0` on the first call,
  `1.0` on subsequent calls. If the model isn't iterative, hardcode
  `0.0` and leave `mask_input` as zeros.
- **Multi-mask outputs.** SAM decoder K=4 with `multimask_output=True`
  is a quality win but requires picking by IoU prediction. snapseg
  currently takes the first mask; document if a particular export
  doesn't behave that way.
