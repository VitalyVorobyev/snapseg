//! RITM adapter (Reviving Iterative Training with Mask guidance,
//! Sofiiuk et al. 2021). HRNet-backboned, click-driven interactive
//! segmenter: each user click triggers exactly one network call that
//! consumes the full image, the accumulated click history rasterised
//! into a 2-channel positive/negative Gaussian disk map, and the
//! previous-iteration mask as a refinement prior.
//!
//! ## ONNX I/O contract
//!
//! Built against the `saic-vul/ritm_interactive_segmentation` upstream
//! ONNX export convention. Single network file; encoder + decoder are
//! fused into one graph.
//!
//! ```text
//! in   `image`       : float32 [1, 3, H, W]   ImageNet-normalised RGB,
//!                                              letterbox-padded to (H, W)
//! in   `click_map`   : float32 [1, 2, H, W]   channel 0 = positive Gaussian disks,
//!                                              channel 1 = negative Gaussian disks
//!                                              (σ = 5 px on the resized canvas)
//! in   `prev_mask`   : float32 [1, 1, H, W]   previous logits (zeros on first call)
//! out  `instances`   : float32 [1, 1, H, W]   logits at the resized canvas;
//!                                              threshold at 0 for the mask
//! ```
//!
//! H and W default to the registry-declared `input_size` (1024×1024 in
//! the bundled entry). Community RITM exports occasionally substitute
//! `images` / `points` / `pred` for the input/output names; the
//! adapter resolves those by case-insensitive substring matching with
//! a fallback to ordinal slot lookup.
//!
//! ### Preprocessing convention
//!
//! Letterbox-with-pad (RITM upstream): resize so the longer side hits
//! the target edge, pad the shorter side with zeros. The
//! pre-normalisation pad in `imagenet_letterbox_gray` keeps padded
//! regions at 0.0 in the model's input space — matching the official
//! RITM `isegm/inference/predictors/base.py` pipeline. The same
//! [`snapseg_runtime::preprocess::LetterboxGeometry`] info struct that drives
//! `point_to_encoder` is reused for the click-map rasterizer's
//! `scale` argument.
//!
//! ### Output decoding
//!
//! `instances` is at the **resized canvas** resolution (not the
//! original image), so the adapter:
//!   1. slices off the bottom-right pad region (`[.., ..new_h, ..new_w]`),
//!   2. bilinearly resizes the unpadded crop back to `(orig_h, orig_w)`,
//!   3. thresholds at 0 to produce the boolean mask.
//!
//! The full-canvas logits (including the still-padded region) are
//! cached as `prev_mask` for the next call; RITM was trained with
//! iterative refinement against this exact tensor layout.
//!
//! ### Scribble handling
//!
//! Scribbles are rasterised as a sequence of positive (or negative)
//! disks, matching what the rasterizer does for individual clicks —
//! the model itself doesn't distinguish a polyline stroke from N
//! discrete clicks once it's projected into the click-map.
//!
//! ### Fused4ch variant
//!
//! Some community RITM exports (notably the "ritm32"/HRNet-32
//! distillations) fuse the image and prev_mask into a single
//! 4-channel input and take the prompt list as a tensor instead of a
//! pre-rasterised click_map:
//!
//! ```text
//! in   `image`   : float32 [1, 4, H_net, W_net]   image[:3] + prev_mask[3:4]
//! in   `points`  : float32 [1, 2*M, 3]            (y, x, click_indx) at
//!                                                  network-canvas pixels.
//!                                                  Rows 0..M = positive
//!                                                  clicks, rows M..2M =
//!                                                  negative clicks. Unused
//!                                                  slots are sentinel
//!                                                  (-1, -1, -1).
//! in   `size`    : int64   [2]                    (H_net, W_net) — the
//!                                                  network's output target.
//! out  `output`  : float32 [1, 1, size[0], size[1]] logits.
//! ```
//!
//! The schema is detected at `from_parts` time from the first input's
//! channel count and dispatched separately in `segment`.
//!
//! Status: **wired** (M6-T01). `from_parts` loads an ort session,
//! detects schema, `set_image` letterboxes the image into
//! ImageNet-normalised RGB on a `(H_net, W_net)` canvas, `segment`
//! either rasterises a 2-channel click_map (Separate variant) or
//! builds the points tensor and the fused 4-channel image (Fused4ch
//! variant), then resizes the mask back to the source image
//! resolution. The Separate variant has been written against the
//! upstream `saic-vul` contract but not exercised live (no ONNX
//! shipped); the Fused4ch variant is exercised against the bundled
//! `ritm32.onnx` export.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2, Array3, Array4, s};

use snapseg_core::{
    Capabilities, GrayImage, InteractiveSegmenter, MaskCandidate, Polarity, Prompt, PromptSession,
    SegError, SegmentationResult,
};
use snapseg_runtime::preprocess::{
    LetterboxGeometry, imagenet_letterbox_gray, rasterize_click_map, resize_bilinear_chw,
};
use snapseg_runtime::{Backend, RuntimeConfig};

/// `(shape, data)` of an extracted `f32` tensor. Shape uses `usize` so
/// it slots straight into ndarray reshape calls; data is owned so the
/// caller can outlive the ort `SessionOutputs` borrow.
type ExtractedTensor = (Vec<usize>, Vec<f32>);

/// Click rasterisation σ (Gaussian std-dev) in resized-canvas pixels.
/// Matches RITM upstream (see
/// `isegm.inference.clicker.Clicker` and the dataloader-side
/// `DistMaps` generator with `norm_radius=5`).
const RITM_CLICK_SIGMA: f32 = 5.0;

/// Which of the two community RITM ONNX conventions this session
/// implements. Detected at construction from the declared input names
/// and the channel count of the first input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RitmSchema {
    /// `saic-vul/ritm_interactive_segmentation` reference export.
    /// Three explicit tensor inputs:
    /// ```text
    /// image      [1, 3, H, W]
    /// click_map  [1, 2, H, W]
    /// prev_mask  [1, 1, H, W]
    /// ```
    Separate,
    /// "iseg-onnx" / community ONNX export that fuses image and
    /// prev_mask into a single 4-channel tensor and takes the prompt
    /// list directly. Schema:
    /// ```text
    /// image   [1, 4, H, W]     image[3 channels] || prev_mask[1 channel]
    /// points  [1, N, 3]        each row = (y, x, indx); positive clicks
    ///                          have `indx ∈ [0, N+)`, negative clicks
    ///                          have `indx ∈ [N+, N+ + N-)` — the model
    ///                          uses the index to split polarity.
    /// size    [2]              i64 (orig_h, orig_w) for output rescaling.
    /// ```
    Fused4ch,
}

/// RITM `InteractiveSegmenter`. Owns one ort session (the fused
/// encoder + decoder graph) and caches the preprocessed image plus
/// the previous-call logits for iterative refinement.
pub struct RitmSegmenter {
    name: String,
    /// Network input height. RITM upstream uses 1024.
    input_h: u32,
    /// Network input width. RITM upstream uses 1024.
    input_w: u32,
    /// Which ONNX I/O convention this session implements. Detected at
    /// load time from input names and shapes.
    schema: RitmSchema,
    backend: Backend,
    state: Option<EncodedImage>,
}

/// Per-image state cached by `set_image`. The pre-padded image tensor
/// is identical for every subsequent `segment()` call; only the
/// click-map and prev-mask change.
struct EncodedImage {
    /// Letterbox-padded `[1, 3, H, W]` ImageNet-normalised image.
    image: Array4<f32>,
    /// Geometry of the letterbox: original size, resized size, scale,
    /// canvas size.
    resize_info: LetterboxGeometry,
    /// Previous logits at the resized canvas resolution
    /// (`[1, 1, H, W]`). `None` before the first `segment()` call.
    prev_logits: Option<Array4<f32>>,
}

impl RitmSegmenter {
    /// Construct from a part map that supplies a single `"model"`
    /// part path. `input_size` is the square edge length the network
    /// was exported for (RITM is typically 1024).
    ///
    /// # Errors
    ///
    /// - [`SegError::InvalidInput`] if the `"model"` part is missing
    ///   from `parts`.
    /// - [`SegError::Backend`] if the underlying ort session fails
    ///   to load.
    pub fn from_parts(
        name: String,
        parts: &HashMap<String, PathBuf>,
        input_size: u32,
        config: &RuntimeConfig,
    ) -> Result<Self, SegError> {
        Self::from_parts_with_shape(name, parts, (input_size, input_size), config)
    }

    /// Construct with an explicit non-square canvas `(H, W)`. Some
    /// custom RITM exports use rectangular canvases; pass
    /// `(1024, 1024)` for the canonical export.
    ///
    /// # Errors
    ///
    /// Same set as [`RitmSegmenter::from_parts`].
    pub fn from_parts_with_shape(
        name: String,
        parts: &HashMap<String, PathBuf>,
        input_shape: (u32, u32),
        config: &RuntimeConfig,
    ) -> Result<Self, SegError> {
        let model_path = parts
            .get("model")
            .ok_or_else(|| SegError::InvalidInput("RITM needs a 'model' part".into()))?;
        let mut backend = Backend::load(model_path.clone(), config)
            .map_err(|e| SegError::Backend(format!("RITM model load: {e}")))?;
        let schema = detect_schema(&mut backend)?;
        tracing::info!(schema = ?schema, "RITM session schema detected");
        Ok(Self {
            name,
            input_h: input_shape.0,
            input_w: input_shape.1,
            schema,
            backend,
            state: None,
        })
    }
}

/// Pick the I/O convention from the session's declared inputs. Falls
/// back to `Separate` only when the first input is genuinely 3-channel
/// and a `click_map`-like input is present; anything else with a
/// 4-channel first input is the fused variant.
fn detect_schema(backend: &mut Backend) -> Result<RitmSchema, SegError> {
    let inputs = &backend.session_mut().inputs;
    if inputs.is_empty() {
        return Err(SegError::Backend("RITM ONNX declares no inputs".into()));
    }
    let first = &inputs[0];
    let first_shape: Vec<i64> = first
        .input_type
        .tensor_shape()
        .map(|s| s.iter().copied().collect())
        .unwrap_or_default();
    // Look for the channels dimension. Both common conventions are
    // NCHW with channels at index 1.
    let channels = first_shape.get(1).copied().unwrap_or(-1);
    if channels == 4 {
        Ok(RitmSchema::Fused4ch)
    } else if channels == 3
        && inputs
            .iter()
            .any(|i| i.name.to_lowercase().contains("click") || i.name.to_lowercase() == "points")
    {
        // 3-channel first input *and* a click_map / points sibling — only
        // the saic-vul reference export looks like this.
        Ok(RitmSchema::Separate)
    } else {
        // Default to Fused4ch — most community exports we've seen.
        Ok(RitmSchema::Fused4ch)
    }
}

impl InteractiveSegmenter for RitmSegmenter {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            positive_clicks: true,
            negative_clicks: true,
            bbox: false,
            scribble: true,
            mask_input: true,
            recommended_input_size: (self.input_w, self.input_h),
        }
    }

    fn set_image(&mut self, image: &GrayImage) -> Result<(), SegError> {
        let (input, resize_info) =
            imagenet_letterbox_gray(image, self.input_h as usize, self.input_w as usize);
        self.state = Some(EncodedImage {
            image: input,
            resize_info,
            prev_logits: None,
        });
        Ok(())
    }

    fn reset_prompt_state(&mut self) {
        // Drop the previous-iteration logits so the next `segment` call
        // feeds a zero `prev_mask` to the network. The cached
        // letterboxed image tensor and its resize info stay — they
        // depend only on the source image, not on the prompt history.
        if let Some(state) = self.state.as_mut() {
            state.prev_logits = None;
        }
    }

    fn segment(&mut self, session: &PromptSession) -> Result<SegmentationResult, SegError> {
        let started = Instant::now();
        let state = self.state.as_mut().ok_or(SegError::NoImage)?;

        let target_h = self.input_h as usize;
        let target_w = self.input_w as usize;

        // Prev-mask tensor (the only state shared across schemas):
        // zeros on the first call, previous logits afterwards.
        let prev_mask = state
            .prev_logits
            .clone()
            .unwrap_or_else(|| Array4::<f32>::zeros((1, 1, target_h, target_w)));

        let (logits_shape, logits_data) = match self.schema {
            RitmSchema::Separate => {
                let click_map = rasterize_click_map(
                    &session.prompts,
                    target_h,
                    target_w,
                    state.resize_info.scale,
                    RITM_CLICK_SIGMA,
                );
                run_ritm_separate(&mut self.backend, state.image.clone(), click_map, prev_mask)?
            }
            RitmSchema::Fused4ch => {
                // Points and size both live in network-canvas
                // coordinates (1024×1024 for the canonical RITM
                // export). The model rescales its mask back to the
                // original image resolution downstream via the
                // `decode_instances` resize.
                let points = build_points_tensor(&session.prompts, &state.resize_info);
                run_ritm_fused(
                    &mut self.backend,
                    state.image.clone(),
                    prev_mask,
                    points,
                    self.input_h,
                    self.input_w,
                )?
            }
        };

        tracing::debug!(raw_output_shape = ?logits_shape, "ritm output (pre-decode)");
        let (mask, logits) = decode_instances(&logits_shape, &logits_data, &state.resize_info)?;

        // Cache the full-canvas logits as prev_mask for the next call.
        state.prev_logits = into_prev_logits(logits_shape, logits_data, target_h, target_w);

        let elapsed = started.elapsed();
        tracing::debug!(
            ms = elapsed.as_millis() as u64,
            n_prompts = session.prompts.len(),
            schema = ?self.schema,
            "ritm segment"
        );

        // RITM is single-mask: surface the prediction as a one-element
        // `candidates` vec so the side-panel cycler iterates uniformly
        // across families. iou is sentinel-1.0 (no alternative to rank
        // against).
        let candidate = MaskCandidate {
            mask: mask.clone(),
            logits: logits.clone(),
            iou: 1.0,
        };

        Ok(SegmentationResult {
            mask,
            logits,
            inference_time: elapsed,
            candidates: vec![candidate],
        })
    }
}

/// Run the saic-vul-style RITM network on
/// `(image, click_map, prev_mask)` and return the `instances` output as
/// `(shape, data)`.
///
/// Input names are resolved against the session's declared inputs by
/// case-insensitive substring matching, with a positional fallback so
/// community exports with different names (`images` / `points` /
/// `pred` instead of `image` / `click_map` / `prev_mask`) still load.
fn run_ritm_separate(
    backend: &mut Backend,
    image: Array4<f32>,
    click_map: Array4<f32>,
    prev_mask: Array4<f32>,
) -> Result<ExtractedTensor, SegError> {
    // Dump the declared input/output schema once per call when debug is
    // on — community RITM exports rename and reshape these (some fuse
    // image+prev_mask into a single 4-channel input). The dump pairs
    // well with the name resolution below.
    {
        let session = backend.session_mut();
        for inp in session.inputs.iter() {
            let dims: Vec<i64> = inp
                .input_type
                .tensor_shape()
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            tracing::debug!(name = %inp.name, shape = ?dims, "ritm declared input");
        }
        for out in session.outputs.iter() {
            let dims: Vec<i64> = out
                .output_type
                .tensor_shape()
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            tracing::debug!(name = %out.name, shape = ?dims, "ritm declared output");
        }
    }
    let names = resolve_input_names(backend)?;

    let v_image = into_dyn_value(image, "image")?;
    let v_clicks = into_dyn_value(click_map, "click_map")?;
    let v_prev = into_dyn_value(prev_mask, "prev_mask")?;

    let outputs = backend
        .session_mut()
        .run(ort::inputs![
            names.image.as_str()      => v_image,
            names.click_map.as_str()  => v_clicks,
            names.prev_mask.as_str()  => v_prev,
        ])
        .map_err(|e| SegError::Backend(format!("RITM run: {e}")))?;

    // Find the instances/pred output by substring match, fall back
    // to the first output otherwise.
    let pick = pick_logits_output(&outputs)?;
    pick.ok_or_else(|| SegError::Backend("RITM returned no outputs".into()))
}

/// Convenience alias: resolved ONNX input names. The canonical RITM
/// export uses `image` / `click_map` / `prev_mask`; community exports
/// sometimes use `images` / `points` / `pred`.
struct ResolvedInputs {
    image: String,
    click_map: String,
    prev_mask: String,
}

/// Run the Fused4ch RITM variant. Concatenates `image_3ch` and
/// `prev_mask` along the channel axis to produce the 4-channel
/// `image` tensor the export expects, builds the `points` list-of-
/// coordinates tensor, and passes `size = [orig_h, orig_w]` so the
/// model can rescale its mask output back to the source resolution.
///
/// Input names are matched by substring (`image|input`, `point|click`,
/// `size|shape`) and fall back to positional slots 0 / 1 / 2.
fn run_ritm_fused(
    backend: &mut Backend,
    image_3ch: Array4<f32>,
    prev_mask: Array4<f32>,
    points: Array3<f32>,
    orig_h: u32,
    orig_w: u32,
) -> Result<ExtractedTensor, SegError> {
    let (_, _, h_net, w_net) = image_3ch.dim();
    let mut fused = Array4::<f32>::zeros((1, 4, h_net, w_net));
    fused
        .slice_mut(s![.., ..3, .., ..])
        .assign(&image_3ch.slice(s![.., .., .., ..]));
    fused
        .slice_mut(s![.., 3..4, .., ..])
        .assign(&prev_mask.slice(s![.., .., .., ..]));

    let size = Array1::from_vec(vec![orig_h as i64, orig_w as i64]);

    let declared: Vec<String> = backend
        .session_mut()
        .inputs
        .iter()
        .map(|i| i.name.clone())
        .collect();
    if declared.len() < 3 {
        return Err(SegError::Backend(format!(
            "Fused4ch RITM expected ≥3 inputs, declared {} ({declared:?})",
            declared.len()
        )));
    }
    let image_name =
        find_input(&declared, &["image", "input"]).unwrap_or_else(|| declared[0].clone());
    let points_name =
        find_input(&declared, &["point", "click"]).unwrap_or_else(|| declared[1].clone());
    let size_name =
        find_input(&declared, &["size", "shape"]).unwrap_or_else(|| declared[2].clone());

    let v_image = into_dyn_value(fused, "image")?;
    let v_points = into_dyn_value(points, "points")?;
    let v_size = into_dyn_i64_value(size, "size")?;

    let outputs = backend
        .session_mut()
        .run(ort::inputs![
            image_name.as_str()  => v_image,
            points_name.as_str() => v_points,
            size_name.as_str()   => v_size,
        ])
        .map_err(|e| SegError::Backend(format!("RITM run (fused): {e}")))?;

    pick_logits_output(&outputs)?
        .ok_or_else(|| SegError::Backend("RITM returned no outputs".into()))
}

/// Build the Fused4ch `points` tensor `[1, 2*M, 3]` from a prompt
/// session.
///
/// The Fused4ch ONNX export follows the upstream
/// `isegm.model.is_model::get_coord_features` convention: the points
/// tensor is **split in half** along the row axis, with the first `M`
/// rows reserved for positive clicks and the next `M` rows reserved
/// for negative clicks. Unused slots are sentinel `(-1, -1, -1)` rows
/// — the model's internal `DistMaps` builder masks those out before
/// rasterising into the 2-channel click map.
///
/// Row layout is `(y, x, click_indx)` in the **resized network canvas**
/// (the same coordinate system the image tensor lives in). `click_indx`
/// is a per-row monotonic counter starting at 0 in each half — the
/// model uses it only for click-order ranking, not for polarity.
///
/// Boxes are ignored — RITM doesn't take box prompts. Scribbles fan
/// out into one click per sample point.
fn build_points_tensor(prompts: &[Prompt], resize: &LetterboxGeometry) -> Array3<f32> {
    // Points layout matches the upstream `isegm` `DistMaps` rasteriser:
    // each row is `(y_net, x_net, click_indx)` in **network-canvas
    // pixel coordinates** (post-letterbox). The model's internal
    // rasteriser uses the same canvas as the image tensor, so we
    // pre-scale image-space clicks by `resize.scale` here. The fill
    // order within each half is positive-clicks-first; the third
    // column is a monotonic counter (click order), unused by the
    // rasteriser but preserved for exports that read it.
    let scale = resize.scale;
    let mut positives: Vec<(f32, f32)> = Vec::new();
    let mut negatives: Vec<(f32, f32)> = Vec::new();
    for p in prompts {
        match p {
            Prompt::Click { point, polarity } => {
                let xy = (point.x * scale, point.y * scale);
                match polarity {
                    Polarity::Positive => positives.push(xy),
                    Polarity::Negative => negatives.push(xy),
                }
            }
            Prompt::Scribble { points, polarity } => {
                for sp in points {
                    let xy = (sp.x * scale, sp.y * scale);
                    match polarity {
                        Polarity::Positive => positives.push(xy),
                        Polarity::Negative => negatives.push(xy),
                    }
                }
            }
            Prompt::Box(_) => {}
        }
    }

    // Try the simple convention first: rows = (x, y, polarity_flag)
    // where polarity_flag = 0 for positive, 1 for negative. Layout is
    // a flat list, with positives first then negatives. Pad to an
    // even row count so the model's `Reshape -> [-1, 2, H, W]` split
    // doesn't trip.
    let half = positives.len().max(negatives.len()).max(1);
    let total = 2 * half;
    let mut data: Vec<f32> = Vec::with_capacity(total * 3);

    // First half: positive clicks (then sentinel rows). Row layout is
    // `(y, x, click_indx)`.
    for (i, (x, y)) in positives.iter().enumerate() {
        data.push(*y);
        data.push(*x);
        data.push(i as f32);
    }
    for _ in positives.len()..half {
        data.push(-1.0);
        data.push(-1.0);
        data.push(-1.0);
    }
    // Second half: negative clicks (then sentinel rows).
    let pos_count = positives.len();
    for (i, (x, y)) in negatives.iter().enumerate() {
        data.push(*y);
        data.push(*x);
        data.push((pos_count + i) as f32);
    }
    for _ in negatives.len()..half {
        data.push(-1.0);
        data.push(-1.0);
        data.push(-1.0);
    }

    Array3::from_shape_vec((1, total, 3), data).expect("points tensor shape")
}

/// Match the session's declared input names to the three RITM inputs
/// by case-insensitive substring matching, with a positional
/// fallback (input 0 → image, input 1 → click_map, input 2 →
/// prev_mask). Returns the actual names from the session so
/// `ort::inputs!` references them verbatim.
fn resolve_input_names(backend: &mut Backend) -> Result<ResolvedInputs, SegError> {
    let declared: Vec<String> = backend
        .session_mut()
        .inputs
        .iter()
        .map(|i| i.name.clone())
        .collect();
    if declared.len() < 3 {
        return Err(SegError::Backend(format!(
            "RITM ONNX expected ≥3 inputs, declared {} ({declared:?})",
            declared.len()
        )));
    }
    let image = find_input(&declared, &["image", "input"]).unwrap_or_else(|| declared[0].clone());
    let click_map =
        find_input(&declared, &["click", "point"]).unwrap_or_else(|| declared[1].clone());
    let prev_mask =
        find_input(&declared, &["prev", "mask", "pred_in"]).unwrap_or_else(|| declared[2].clone());
    Ok(ResolvedInputs {
        image,
        click_map,
        prev_mask,
    })
}

/// Find the first declared input whose name (lowercased) contains
/// any of `needles`. Returns `None` if none match — caller falls
/// back to positional lookup.
fn find_input(declared: &[String], needles: &[&str]) -> Option<String> {
    for name in declared {
        let lc = name.to_lowercase();
        for needle in needles {
            if lc.contains(needle) {
                return Some(name.clone());
            }
        }
    }
    None
}

/// Pick the logits output from a RITM run. Looks for `instances` /
/// `pred` / `mask` substrings first, then falls back to the first
/// output. Extraction is inlined because `SessionOutputs::iter()`
/// yields `ValueRef<'_>` whose lifetime is tied to `outputs`, so
/// threading it through a helper signature is awkward.
fn pick_logits_output<'r>(
    outputs: &ort::session::SessionOutputs<'r>,
) -> Result<Option<ExtractedTensor>, SegError> {
    let preferred = ["instances", "pred", "mask", "logits", "output"];
    for (name, value) in outputs.iter() {
        let lc = name.to_lowercase();
        if preferred.iter().any(|p| lc.contains(p)) {
            let (shape_ref, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| SegError::Backend(format!("{name} extract: {e}")))?;
            return Ok(Some((
                shape_ref.iter().map(|&d| d as usize).collect(),
                data.to_vec(),
            )));
        }
    }
    if let Some((name, value)) = outputs.iter().next() {
        let (shape_ref, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| SegError::Backend(format!("{name} extract: {e}")))?;
        return Ok(Some((
            shape_ref.iter().map(|&d| d as usize).collect(),
            data.to_vec(),
        )));
    }
    Ok(None)
}

/// Decode the RITM `instances` output into a mask + logits at the
/// **original image** resolution. The output arrives at the resized
/// canvas (`[1, 1, target_h, target_w]`); we crop the pad region and
/// bilinearly resize back to `(orig_h, orig_w)`.
///
/// # Errors
///
/// Returns [`SegError::Backend`] when the shape is not 4-D, when batch
/// ≠ 1 or channels ≠ 1, when `(h, w)` doesn't match the canvas
/// dimensions declared in `resize_info`, or when the buffer length is
/// smaller than expected.
fn decode_instances(
    shape: &[usize],
    data: &[f32],
    resize_info: &LetterboxGeometry,
) -> Result<(Array2<bool>, Array2<f32>), SegError> {
    if shape.len() != 4 {
        return Err(SegError::Backend(format!(
            "expected 4-D instances, got {shape:?}"
        )));
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    if n != 1 || c != 1 {
        return Err(SegError::Backend(format!(
            "unexpected instances shape {shape:?} (want [1, 1, H, W])"
        )));
    }
    let expected_len = n * c * h * w;
    if data.len() < expected_len {
        return Err(SegError::Backend(format!(
            "instances buffer length {} < expected {expected_len} for shape {shape:?}",
            data.len()
        )));
    }
    let target_h = resize_info.target_h as usize;
    let target_w = resize_info.target_w as usize;
    let orig_h = resize_info.orig_h as usize;
    let orig_w = resize_info.orig_w as usize;

    // The Separate variant emits logits at the **network canvas**
    // resolution (including the bottom-right pad). The Fused4ch
    // variant emits logits already at the **original image** size
    // (because we passed `size` into the model). Detect which by
    // matching the returned spatial dims.
    let full = Array3::from_shape_vec((1, h, w), data[..expected_len].to_vec())
        .map_err(|e| SegError::Backend(format!("instances reshape: {e}")))?;
    let resized = if h == orig_h && w == orig_w {
        full
    } else if h == target_h && w == target_w {
        let new_h = resize_info.new_h as usize;
        let new_w = resize_info.new_w as usize;
        let cropped = full.slice(s![.., ..new_h, ..new_w]).to_owned();
        resize_bilinear_chw(&cropped, orig_h, orig_w)
    } else {
        return Err(SegError::Backend(format!(
            "instances returned at {h}x{w}; expected canvas {target_h}x{target_w} or original {orig_h}x{orig_w}"
        )));
    };

    let mut logits = Array2::<f32>::zeros((orig_h, orig_w));
    let mut mask = Array2::<bool>::from_elem((orig_h, orig_w), false);
    for y in 0..orig_h {
        for x in 0..orig_w {
            let v = resized[(0, y, x)];
            logits[(y, x)] = v;
            mask[(y, x)] = v > 0.0;
        }
    }
    Ok((mask, logits))
}

/// Try to reshape the raw `instances` buffer into a `[1, 1, H, W]`
/// tensor for the prev_mask cache. Returns `None` if the shape is
/// unexpected so the caller can silently skip the cache update (the
/// next iteration will then run with a zero prev_mask, which is
/// still valid input even if less informative).
fn into_prev_logits(
    shape: Vec<usize>,
    data: Vec<f32>,
    target_h: usize,
    target_w: usize,
) -> Option<Array4<f32>> {
    if shape != [1, 1, target_h, target_w] {
        return None;
    }
    Array4::from_shape_vec((1, 1, target_h, target_w), data).ok()
}

/// Build an `ort` `DynValue` from a strongly-typed `ndarray::Array`.
/// Mirrors the same helper in `mobile_sam.rs`.
fn into_dyn_value<D: ndarray::Dimension>(
    arr: ndarray::Array<f32, D>,
    what: &str,
) -> Result<ort::value::DynValue, SegError> {
    ort::value::Value::from_array(arr.into_dyn())
        .map_err(|e| SegError::Backend(format!("{what} value: {e}")))
        .map(|v| v.into_dyn())
}

/// Same as [`into_dyn_value`] but for `i64`-typed tensors (the
/// `size` input on the Fused4ch RITM export is `int64[2]`).
fn into_dyn_i64_value<D: ndarray::Dimension>(
    arr: ndarray::Array<i64, D>,
    what: &str,
) -> Result<ort::value::DynValue, SegError> {
    ort::value::Value::from_array(arr.into_dyn())
        .map_err(|e| SegError::Backend(format!("{what} value: {e}")))
        .map(|v| v.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_parts` rejects an empty part map with `InvalidInput` — no
    /// `"model"` part means there's nothing to load.
    #[test]
    fn from_parts_rejects_missing_model_part() {
        let parts: HashMap<String, PathBuf> = HashMap::new();
        let cfg = RuntimeConfig::default();
        let err = RitmSegmenter::from_parts("ritm-test".into(), &parts, 1024, &cfg)
            .err()
            .expect("missing 'model' part must fail");
        match err {
            SegError::InvalidInput(msg) => assert!(msg.contains("model"), "msg: {msg}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// `from_parts` surfaces a `Backend` error when the supplied
    /// `"model"` path doesn't exist on disk. We don't need a real ONNX
    /// to confirm this — the runtime fails fast before it tries to
    /// parse the file.
    #[test]
    fn from_parts_fails_on_missing_file() {
        let mut parts: HashMap<String, PathBuf> = HashMap::new();
        parts.insert(
            "model".to_string(),
            PathBuf::from("/definitely/not/a/real/path/ritm.onnx"),
        );
        let cfg = RuntimeConfig::default();
        let err = RitmSegmenter::from_parts("ritm-test".into(), &parts, 1024, &cfg)
            .err()
            .expect("non-existent ONNX path must fail");
        match err {
            SegError::Backend(msg) => assert!(
                msg.contains("RITM model load") || msg.contains("not found"),
                "msg: {msg}"
            ),
            other => panic!("expected Backend, got {other:?}"),
        }
    }
}
