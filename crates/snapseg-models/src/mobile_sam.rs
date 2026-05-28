//! MobileSAM adapter (Zhang et al. 2023). Distilled SAM with a tiny image
//! encoder (~10 MB) plus the SAM mask decoder. Click + box prompts; the
//! image embedding is computed once per image, then each prompt is a
//! cheap decoder pass.
//!
//! ## ONNX I/O contract
//!
//! Built against the canonical SAM ONNX schema used by the official
//! Meta export and the most common community MobileSAM exports.
//!
//! Encoder:
//! ```text
//! in   `input_image`      : float32 [1, 3, S, S]            (S typically 1024)
//! out  *first output*     : float32 [1, 256, S/16, S/16]    image embedding
//! ```
//!
//! The adapter also tolerates three common community-export variants
//! by introspecting the session's declared input shape: unbatched
//! CHW `[3, S, S]`, batched channels-last NHWC `[1, S, S, 3]`, and
//! unbatched HWC `[S, S, 3]` (the layout produced by some of the
//! 2023 MobileSAM exports). The output is unsqueezed back to 4-D
//! when the encoder dropped the batch dim, so downstream code can
//! keep assuming `[1, 256, h, w]`.
//!
//! Decoder:
//! ```text
//! in   `image_embeddings` : float32 [1, 256, h, w]
//! in   `point_coords`     : float32 [1, N, 2]               encoder-space pixels
//! in   `point_labels`     : float32 [1, N]                  1/0/2/3/-1
//! in   `mask_input`       : float32 [1, 1, 256, 256]
//! in   `has_mask_input`   : float32 [1]
//! in   `orig_im_size`     : float32 [2]                     (orig_h, orig_w)
//! out  `masks`            : float32 [1, K, orig_h, orig_w]
//! out  `iou_predictions`  : float32 [1, K]
//! out  `low_res_masks`    : float32 [1, K, 256, 256]
//! ```
//!
//! Output name lookups use case-insensitive substring matching with a
//! fallback to the first output tensor, so minor naming variances in
//! community exports are tolerated.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ndarray::{Array2, Array3, Array4};

use snapseg_core::{
    Capabilities, GrayImage, InteractiveSegmenter, MaskCandidate, Polarity, Prompt, PromptSession,
    SegError, SegmentationResult,
};
use snapseg_runtime::preprocess::{LetterboxGeometry, sam_preprocess_gray};
use snapseg_runtime::{Backend, RuntimeConfig};

/// Convenience alias: a tensor extracted as `(shape, data)` of `f32`,
/// or `None` if the named output isn't present.
type NamedTensor = Option<(Vec<usize>, Vec<f32>)>;

/// MobileSAM `InteractiveSegmenter`. Owns the encoder + decoder sessions
/// and caches the image embedding produced by `set_image`.
pub struct MobileSamSegmenter {
    name: String,
    /// Encoder canvas height. Canonical SAM is 1024; some community
    /// exports (e.g. the 2023-06-29 MobileSAM) declare 682.
    input_h: u32,
    /// Encoder canvas width. Canonical SAM is 1024.
    input_w: u32,
    encoder: Backend,
    decoder: Backend,
    state: Option<EncodedImage>,
}

/// Per-image state: the encoder's output embedding plus the
/// preprocessing transform we'll need to map prompt points from image
/// space into encoder space, plus a cache for iterative refinement.
struct EncodedImage {
    embedding: Array4<f32>,
    resize_info: LetterboxGeometry,
    /// `low_res_masks` from the previous decoder call. Fed back as
    /// `mask_input` so SAM can iteratively refine.
    prev_low_res: Option<Array4<f32>>,
}

impl MobileSamSegmenter {
    /// Construct from a part map keyed by `"encoder"` and `"decoder"`.
    ///
    /// # Errors
    ///
    /// Returns [`SegError::InvalidInput`] if either part is missing,
    /// and [`SegError::Backend`] if the underlying ort session fails
    /// to load.
    pub fn from_parts(
        name: String,
        parts: &HashMap<String, PathBuf>,
        input_size: u32,
        config: &RuntimeConfig,
    ) -> Result<Self, SegError> {
        Self::from_parts_with_shape(name, parts, (input_size, input_size), config)
    }

    /// Construct with an explicit non-square encoder canvas `(H, W)`.
    /// Use for exports whose encoder declares a non-square input (the
    /// 2023-06-29 MobileSAM export wants 682×1024); pass `(1024, 1024)`
    /// for canonical SAM.
    ///
    /// # Errors
    ///
    /// Same set as [`MobileSamSegmenter::from_parts`].
    pub fn from_parts_with_shape(
        name: String,
        parts: &HashMap<String, PathBuf>,
        input_shape: (u32, u32),
        config: &RuntimeConfig,
    ) -> Result<Self, SegError> {
        let enc_path = parts
            .get("encoder")
            .ok_or_else(|| SegError::InvalidInput("MobileSAM needs an 'encoder' part".into()))?;
        let dec_path = parts
            .get("decoder")
            .ok_or_else(|| SegError::InvalidInput("MobileSAM needs a 'decoder' part".into()))?;
        let encoder = Backend::load(enc_path.clone(), config)
            .map_err(|e| SegError::Backend(format!("encoder load: {e}")))?;
        let decoder = Backend::load(dec_path.clone(), config)
            .map_err(|e| SegError::Backend(format!("decoder load: {e}")))?;
        Ok(Self {
            name,
            input_h: input_shape.0,
            input_w: input_shape.1,
            encoder,
            decoder,
            state: None,
        })
    }
}

impl InteractiveSegmenter for MobileSamSegmenter {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            positive_clicks: true,
            negative_clicks: true,
            bbox: true,
            scribble: false,
            mask_input: true,
            recommended_input_size: (self.input_w, self.input_h),
        }
    }

    fn set_image(&mut self, image: &GrayImage) -> Result<(), SegError> {
        let (input, resize_info) =
            sam_preprocess_gray(image, self.input_h as usize, self.input_w as usize);
        let embedding = run_encoder(&mut self.encoder, input)?;
        self.state = Some(EncodedImage {
            embedding,
            resize_info,
            prev_low_res: None,
        });
        Ok(())
    }

    fn segment(&mut self, session: &PromptSession) -> Result<SegmentationResult, SegError> {
        let started = Instant::now();
        let state = self.state.as_mut().ok_or(SegError::NoImage)?;

        let (point_coords, point_labels) = encode_prompts(&session.prompts, &state.resize_info)?;
        if point_labels.is_empty() {
            return Ok(empty_result(state, started.elapsed()));
        }

        // Build the six decoder inputs, run the decoder, and decode +
        // cache its outputs. Kept inline because `SessionOutputs<'r>`
        // borrows from the session and is awkward to thread through a
        // helper signature.
        let mask_input = state
            .prev_low_res
            .clone()
            .unwrap_or_else(|| Array4::<f32>::zeros((1, 1, 256, 256)));
        let has_mask = if state.prev_low_res.is_some() {
            1.0_f32
        } else {
            0.0
        };
        let has_mask_input = ndarray::arr1(&[has_mask]);
        let orig_im_size = ndarray::arr1(&[
            state.resize_info.orig_h as f32,
            state.resize_info.orig_w as f32,
        ]);

        let v_emb = into_dyn_value(state.embedding.clone(), "image_embeddings")?;
        let v_coords = into_dyn_value(point_coords, "point_coords")?;
        let v_labels = into_dyn_value(point_labels, "point_labels")?;
        let v_mask = into_dyn_value(mask_input, "mask_input")?;
        let v_has = into_dyn_value(has_mask_input, "has_mask_input")?;
        let v_orig = into_dyn_value(orig_im_size, "orig_im_size")?;

        let outputs = self
            .decoder
            .session_mut()
            .run(ort::inputs![
                "image_embeddings" => v_emb,
                "point_coords"     => v_coords,
                "point_labels"     => v_labels,
                "mask_input"       => v_mask,
                "has_mask_input"   => v_has,
                "orig_im_size"     => v_orig,
            ])
            .map_err(|e| SegError::Backend(format!("decoder run: {e}")))?;

        let (mask_shape, mask_data) = extract_named(&outputs, "masks")?
            .ok_or_else(|| SegError::Backend("decoder missing 'masks' output".into()))?;

        // The decoder reports a per-candidate IoU prediction; we keep
        // them so the side panel can rank candidates. If the output
        // isn't present (a stripped community export) we fall back to
        // uniform 0.0 per slot, which collapses argmax to the
        // first-mask default.
        let iou_per_k: Vec<f32> = match extract_named(&outputs, "iou_predictions")? {
            Some((iou_shape, iou_data)) => {
                tracing::debug!(shape = ?iou_shape, ious = ?iou_data, "mask iou_predictions");
                iou_data
            }
            None => Vec::new(),
        };

        let candidates = decode_all_masks(
            mask_shape.clone(),
            &mask_data,
            &iou_per_k,
            state.resize_info.orig_h as usize,
            state.resize_info.orig_w as usize,
        )?;

        // Log per-candidate foreground pixel count too — handy when
        // the IoU prediction looks unreasonable.
        for (k, c) in candidates.iter().enumerate() {
            let fg = c.mask.iter().filter(|b| **b).count();
            tracing::debug!(k, fg_pixels = fg, iou = c.iou, "mask candidate");
        }

        if let Some((lr_shape, lr_data)) = extract_named(&outputs, "low_res_masks")? {
            if let Some(lr) = into_low_res(lr_shape, lr_data) {
                state.prev_low_res = Some(lr);
            }
        }

        // Headline mask/logits = argmax-IoU candidate (matches the
        // reference SAM `select_best_mask` heuristic). When IoUs are
        // missing or tied, this picks the first.
        let best_idx = argmax_iou(&candidates);
        let (mask, logits) = {
            // `candidates` is non-empty by construction (decode_all_masks
            // returns `Err` for K = 0), so indexing is safe.
            let c = &candidates[best_idx];
            (c.mask.clone(), c.logits.clone())
        };

        let elapsed = started.elapsed();
        tracing::debug!(
            ms = elapsed.as_millis() as u64,
            n_prompts = session.prompts.len(),
            k = candidates.len(),
            best = best_idx,
            "mobile_sam segment"
        );

        Ok(SegmentationResult {
            mask,
            logits,
            inference_time: elapsed,
            candidates,
        })
    }
}

/// Return the index of the highest-IoU candidate. Falls back to 0 when
/// the vec is non-empty but all IoUs tie or are NaN.
fn argmax_iou(candidates: &[MaskCandidate]) -> usize {
    let mut best = 0usize;
    let mut best_iou = f32::NEG_INFINITY;
    for (i, c) in candidates.iter().enumerate() {
        if c.iou > best_iou {
            best_iou = c.iou;
            best = i;
        }
    }
    best
}

/// Run the encoder and reshape its first 4-D output into the cached
/// `[1, 256, S/16, S/16]` embedding tensor.
///
/// Handles four common export variants of the MobileSAM encoder by
/// introspecting the session's declared input shape:
///
/// - canonical batched NCHW: `[1, 3, S, S]`
/// - unbatched CHW:          `[3, S, S]`
/// - batched NHWC:           `[1, S, S, 3]`
/// - unbatched HWC:          `[S, S, 3]`
///
/// The input arrives here as NCHW; we permute / squeeze as needed
/// based on the rank and the position of the channels-3 dim. The
/// output is unsqueezed back to 4-D `[1, 256, h, w]` for the cache
/// regardless of which variant the encoder used.
fn run_encoder(encoder: &mut Backend, input: Array4<f32>) -> Result<Array4<f32>, SegError> {
    // Capture expected shape as owned ints so the borrow on session
    // metadata ends before we call session_mut().run() below.
    let expected_shape: Vec<i64> = encoder
        .session_mut()
        .inputs
        .first()
        .and_then(|i| i.input_type.tensor_shape())
        .map(|s| s.iter().copied().collect())
        .unwrap_or_default();
    tracing::debug!(shape = ?expected_shape, "encoder input shape");

    let layout = classify_encoder_layout(&expected_shape);
    let outputs = match layout {
        EncoderLayout::Nchw => {
            let v = ort::value::Value::from_array(input)
                .map_err(|e| SegError::Backend(format!("encoder input value: {e}")))?;
            encoder
                .session_mut()
                .run(ort::inputs!["input_image" => v])
                .map_err(|e| SegError::Backend(format!("encoder run: {e}")))?
        }
        EncoderLayout::Chw => {
            let chw: Array3<f32> = input.index_axis_move(ndarray::Axis(0), 0);
            let v = ort::value::Value::from_array(chw)
                .map_err(|e| SegError::Backend(format!("encoder input value: {e}")))?;
            encoder
                .session_mut()
                .run(ort::inputs!["input_image" => v])
                .map_err(|e| SegError::Backend(format!("encoder run: {e}")))?
        }
        EncoderLayout::Nhwc => {
            // NCHW [N, 3, H, W] -> NHWC [N, H, W, 3] then materialise
            // a contiguous buffer (permuted_axes returns a view with
            // rearranged strides; ort wants a standard-layout array).
            let nhwc = input.permuted_axes([0, 2, 3, 1]);
            let nhwc = nhwc.as_standard_layout().to_owned();
            let v = ort::value::Value::from_array(nhwc)
                .map_err(|e| SegError::Backend(format!("encoder input value: {e}")))?;
            encoder
                .session_mut()
                .run(ort::inputs!["input_image" => v])
                .map_err(|e| SegError::Backend(format!("encoder run: {e}")))?
        }
        EncoderLayout::Hwc => {
            let chw: Array3<f32> = input.index_axis_move(ndarray::Axis(0), 0);
            let hwc = chw.permuted_axes([1, 2, 0]);
            let hwc = hwc.as_standard_layout().to_owned();
            let v = ort::value::Value::from_array(hwc)
                .map_err(|e| SegError::Backend(format!("encoder input value: {e}")))?;
            encoder
                .session_mut()
                .run(ort::inputs!["input_image" => v])
                .map_err(|e| SegError::Backend(format!("encoder run: {e}")))?
        }
    };

    // Pick the first output whose name contains "embedding", else fall
    // back to the first output regardless of name.
    let pick: Option<(Vec<usize>, Vec<f32>)> = {
        let mut chosen = None;
        for (name, value) in outputs.iter() {
            if name.to_lowercase().contains("embedding") {
                let (shape_ref, data) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| SegError::Backend(format!("embedding extract: {e}")))?;
                chosen = Some((
                    shape_ref.iter().map(|&d| d as usize).collect(),
                    data.to_vec(),
                ));
                break;
            }
        }
        if chosen.is_none() {
            if let Some((_, value)) = outputs.iter().next() {
                let (shape_ref, data) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| SegError::Backend(format!("embedding extract: {e}")))?;
                chosen = Some((
                    shape_ref.iter().map(|&d| d as usize).collect(),
                    data.to_vec(),
                ));
            }
        }
        chosen
    };

    let (shape, data) =
        pick.ok_or_else(|| SegError::Backend("encoder returned no outputs".into()))?;
    tracing::debug!(shape = ?shape, "encoder output shape");
    // Unbatched encoder exports return `[256, S/16, S/16]`; prepend
    // an implicit batch-1 dim so the rest of the adapter can keep
    // assuming 4-D `[1, 256, h, w]`.
    let (n, c, h, w) = match shape.len() {
        4 => (shape[0], shape[1], shape[2], shape[3]),
        3 => (1, shape[0], shape[1], shape[2]),
        _ => {
            return Err(SegError::Backend(format!(
                "expected 3-D or 4-D embedding, got shape {shape:?}"
            )));
        }
    };
    Array4::from_shape_vec((n, c, h, w), data)
        .map_err(|e| SegError::Backend(format!("embedding reshape: {e}")))
}

/// Memory layout the encoder expects, inferred from the session's
/// declared input shape. Channels-3 position tells channels-first vs
/// channels-last; presence of the leading batch dim distinguishes
/// batched from unbatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderLayout {
    /// `[1, 3, S, S]` — canonical SAM / MobileSAM export.
    Nchw,
    /// `[3, S, S]` — community CHW export, no batch dim.
    Chw,
    /// `[1, S, S, 3]` — channels-last with batch.
    Nhwc,
    /// `[S, S, 3]` — channels-last, no batch.
    Hwc,
}

/// Decide the encoder layout from the session's declared input shape.
/// Defaults to `Nchw` when the shape is empty or ambiguous (the
/// adapter's historical assumption, which matches the canonical
/// export).
fn classify_encoder_layout(shape: &[i64]) -> EncoderLayout {
    match shape.len() {
        4 if shape[3] == 3 => EncoderLayout::Nhwc,
        4 => EncoderLayout::Nchw,
        3 if shape[2] == 3 => EncoderLayout::Hwc,
        3 => EncoderLayout::Chw,
        _ => EncoderLayout::Nchw,
    }
}

/// Build the SAM `point_coords` / `point_labels` tensors from a prompt
/// session in image-space coordinates.
///
/// Label convention: 1 = positive click, 0 = negative click,
/// 2 = box top-left, 3 = box bottom-right, −1 = padding (unused here).
/// Scribbles are ignored — the canonical MobileSAM export doesn't
/// ingest them.
fn encode_prompts(
    prompts: &[Prompt],
    resize_info: &LetterboxGeometry,
) -> Result<(Array3<f32>, Array2<f32>), SegError> {
    let mut coords: Vec<f32> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    let mut has_box = false;
    for prompt in prompts {
        match prompt {
            Prompt::Click { point, polarity } => {
                let (ex, ey) = resize_info.point_to_encoder(point.x, point.y);
                coords.push(ex);
                coords.push(ey);
                labels.push(match polarity {
                    Polarity::Positive => 1.0,
                    Polarity::Negative => 0.0,
                });
            }
            Prompt::Box(b) => {
                has_box = true;
                let (x0, y0) = resize_info.point_to_encoder(b.x0, b.y0);
                let (x1, y1) = resize_info.point_to_encoder(b.x1, b.y1);
                coords.extend_from_slice(&[x0, y0, x1, y1]);
                labels.push(2.0);
                labels.push(3.0);
            }
            Prompt::Scribble { .. } => {
                // MobileSAM doesn't consume scribbles in this export.
            }
        }
    }
    if labels.is_empty() {
        return Ok((
            Array3::<f32>::zeros((1, 0, 2)),
            Array2::<f32>::zeros((1, 0)),
        ));
    }
    // SAM convention: when there is no box, append a dummy `[0, 0]`
    // point with label `-1`. The model's `not_a_point_embed` zeroes
    // the positional encoding for label==-1 and adds a special learned
    // embedding instead; SAM was trained with this padding so the
    // distribution at inference matches. Without it, the click-only
    // case is out-of-distribution and the mask drifts (typically
    // landing one feature stride away from the click).
    if !has_box {
        coords.push(0.0);
        coords.push(0.0);
        labels.push(-1.0);
    }
    let n = labels.len();
    let point_coords = Array3::from_shape_vec((1, n, 2), coords)
        .map_err(|e| SegError::Backend(format!("point_coords shape: {e}")))?;
    let point_labels = Array2::from_shape_vec((1, n), labels)
        .map_err(|e| SegError::Backend(format!("point_labels shape: {e}")))?;
    Ok((point_coords, point_labels))
}

/// Decode all K predicted masks into a vector of [`MaskCandidate`] at
/// the source image resolution. Each candidate carries the model's
/// self-reported IoU when available (uniform 0.0 otherwise, so argmax
/// falls back to slot order). Single-candidate exports return a
/// one-element vec.
///
/// # Errors
///
/// Returns [`SegError::Backend`] when the tensor shape is not 4-D,
/// when batch ≠ 1 or K = 0, when the buffer length doesn't match
/// `n * k * h * w` (defence against ort returning a partially-filled
/// extract), or when `(h, w)` does not match the expected
/// `(orig_h, orig_w)` (the decoder is contracted to resize back to
/// the original image dimensions; any other shape is a silent
/// upstream change we want to fail loudly on).
fn decode_all_masks(
    shape: Vec<usize>,
    data: &[f32],
    iou_per_k: &[f32],
    expected_h: usize,
    expected_w: usize,
) -> Result<Vec<MaskCandidate>, SegError> {
    if shape.len() != 4 {
        return Err(SegError::Backend(format!(
            "expected 4-D masks, got {shape:?}"
        )));
    }
    let (n, k, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    if n != 1 || k == 0 {
        return Err(SegError::Backend(format!(
            "unexpected masks shape {shape:?}"
        )));
    }
    if h != expected_h || w != expected_w {
        return Err(SegError::Backend(format!(
            "decoder returned masks at {h}x{w}, expected {expected_h}x{expected_w} (orig_im_size)"
        )));
    }
    let expected_len = n * k * h * w;
    if data.len() < expected_len {
        return Err(SegError::Backend(format!(
            "masks buffer length {} < expected {expected_len} for shape {shape:?}",
            data.len()
        )));
    }
    let mut out: Vec<MaskCandidate> = Vec::with_capacity(k);
    let hw = h * w;
    for ki in 0..k {
        let base = ki * hw;
        let mut logits = Array2::<f32>::zeros((h, w));
        let mut mask = Array2::<bool>::from_elem((h, w), false);
        for y in 0..h {
            for x in 0..w {
                let v = data[base + y * w + x];
                logits[(y, x)] = v;
                mask[(y, x)] = v > 0.0;
            }
        }
        let iou = iou_per_k.get(ki).copied().unwrap_or(0.0);
        out.push(MaskCandidate { mask, logits, iou });
    }
    Ok(out)
}

/// Build the cached `low_res_masks` tensor for the next iteration.
/// Returns `None` if the shape doesn't match the expected
/// `[1, 1, 256, 256]` so the caller can silently skip the update.
fn into_low_res(shape: Vec<usize>, data: Vec<f32>) -> Option<Array4<f32>> {
    if shape.len() != 4 {
        return None;
    }
    Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), data).ok()
}

/// All-false mask at the source resolution. Returned when the prompt
/// session is empty after filtering — SAM requires at least one prompt
/// to produce anything useful.
fn empty_result(state: &EncodedImage, elapsed: Duration) -> SegmentationResult {
    let h = state.resize_info.orig_h as usize;
    let w = state.resize_info.orig_w as usize;
    let empty_mask = Array2::<bool>::from_elem((h, w), false);
    let empty_logits = Array2::<f32>::zeros((h, w));
    let candidate = MaskCandidate {
        mask: empty_mask.clone(),
        logits: empty_logits.clone(),
        iou: 0.0,
    };
    SegmentationResult {
        mask: empty_mask,
        logits: empty_logits,
        inference_time: elapsed,
        candidates: vec![candidate],
    }
}

/// Find a named output, extract it as `(shape, data)` of `f32`. Returns
/// `Ok(None)` if the name isn't present; only errors when extraction
/// itself fails.
fn extract_named<'r>(
    outputs: &ort::session::SessionOutputs<'r>,
    name: &str,
) -> Result<NamedTensor, SegError> {
    let lc = name.to_lowercase();
    for (out_name, value) in outputs.iter() {
        if out_name.to_lowercase() == lc {
            let (shape_ref, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| SegError::Backend(format!("{name} extract: {e}")))?;
            return Ok(Some((
                shape_ref.iter().map(|&d| d as usize).collect(),
                data.to_vec(),
            )));
        }
    }
    Ok(None)
}

/// Build an `ort` `DynValue` from a strongly-typed `ndarray::Array`.
/// `into_dyn_value` erases the typed `Value<TensorValueType<f32>>` into
/// the dyn-typed value `ort::inputs!` expects.
fn into_dyn_value<D: ndarray::Dimension>(
    arr: ndarray::Array<f32, D>,
    what: &str,
) -> Result<ort::value::DynValue, SegError> {
    ort::value::Value::from_array(arr.into_dyn())
        .map_err(|e| SegError::Backend(format!("{what} value: {e}")))
        .map(|v| v.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_canonical_nchw() {
        assert_eq!(
            classify_encoder_layout(&[1, 3, 1024, 1024]),
            EncoderLayout::Nchw
        );
        // Dynamic spatial dims still resolve to NCHW (channels-3 at index 1).
        assert_eq!(
            classify_encoder_layout(&[-1, 3, -1, -1]),
            EncoderLayout::Nchw
        );
    }

    #[test]
    fn layout_chw_no_batch() {
        assert_eq!(
            classify_encoder_layout(&[3, 1024, 1024]),
            EncoderLayout::Chw
        );
    }

    #[test]
    fn layout_nhwc_channels_last_batched() {
        assert_eq!(
            classify_encoder_layout(&[1, 1024, 1024, 3]),
            EncoderLayout::Nhwc
        );
    }

    #[test]
    fn layout_hwc_no_batch() {
        // The variant produced by the 2023-06-29 MobileSAM export the
        // smoke test exercises: `[-1, -1, 3]`.
        assert_eq!(classify_encoder_layout(&[-1, -1, 3]), EncoderLayout::Hwc);
        assert_eq!(
            classify_encoder_layout(&[1024, 1024, 3]),
            EncoderLayout::Hwc
        );
    }

    #[test]
    fn layout_empty_defaults_to_nchw() {
        assert_eq!(classify_encoder_layout(&[]), EncoderLayout::Nchw);
    }

    /// Three 2x2 candidates with distinct IoU predictions. Verifies the
    /// candidate vec is built in slot order, every mask carries its
    /// per-slot logits, and `argmax_iou` picks the highest-IoU slot.
    #[test]
    fn decode_all_masks_three_candidates() {
        // Shape [1, 3, 2, 2]; data laid out as N, K, H, W.
        // Slot 0: all -1 (mask all false)
        // Slot 1: all  1 (mask all true)
        // Slot 2: diagonal positive
        let data = vec![
            // K=0
            -1.0, -1.0, -1.0, -1.0, // K=1
            1.0, 1.0, 1.0, 1.0, // K=2 (diagonal)
            1.0, -1.0, -1.0, 1.0,
        ];
        let shape = vec![1, 3, 2, 2];
        let ious = vec![0.1, 0.5, 0.9];

        let candidates = decode_all_masks(shape, &data, &ious, 2, 2).expect("decode");
        assert_eq!(candidates.len(), 3);

        // Slot 0: all false.
        assert!(candidates[0].mask.iter().all(|b| !*b));
        assert!((candidates[0].iou - 0.1).abs() < 1e-6);

        // Slot 1: all true.
        assert!(candidates[1].mask.iter().all(|b| *b));
        assert!((candidates[1].iou - 0.5).abs() < 1e-6);

        // Slot 2: diagonal.
        assert!(candidates[2].mask[(0, 0)]);
        assert!(!candidates[2].mask[(0, 1)]);
        assert!(!candidates[2].mask[(1, 0)]);
        assert!(candidates[2].mask[(1, 1)]);
        assert!((candidates[2].iou - 0.9).abs() < 1e-6);

        // Logits round-trip too.
        assert!((candidates[2].logits[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((candidates[2].logits[(0, 1)] - -1.0).abs() < 1e-6);

        // argmax picks slot 2 (highest IoU).
        assert_eq!(argmax_iou(&candidates), 2);
    }

    /// K = 1 collapses to a one-element vec; argmax is trivially 0 and
    /// missing IoUs fall back to 0.0 without erroring.
    #[test]
    fn decode_all_masks_single_candidate_missing_ious() {
        let data = vec![1.0, -1.0, -1.0, 1.0];
        let shape = vec![1, 1, 2, 2];
        let ious: Vec<f32> = vec![]; // missing entirely

        let candidates = decode_all_masks(shape, &data, &ious, 2, 2).expect("decode");
        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].iou - 0.0).abs() < 1e-6);
        assert_eq!(argmax_iou(&candidates), 0);
    }

    /// Tensor with wrong number of dims is rejected with `Backend`.
    #[test]
    fn decode_all_masks_rejects_non_4d() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let shape = vec![3, 2, 2]; // 3-D, not 4-D
        let err = decode_all_masks(shape, &data, &[], 2, 2).expect_err("must fail");
        match err {
            SegError::Backend(msg) => assert!(msg.contains("4-D"), "msg: {msg}"),
            other => panic!("expected Backend, got {other:?}"),
        }
    }

    /// Mismatch with the registry-declared `(orig_h, orig_w)` is fatal —
    /// catches silent upstream contract changes in community exports.
    #[test]
    fn decode_all_masks_rejects_wrong_spatial_dims() {
        let data = vec![1.0; 4];
        let shape = vec![1, 1, 2, 2];
        // expected 3x3, got 2x2 — must reject.
        let err = decode_all_masks(shape, &data, &[], 3, 3).expect_err("must fail");
        match err {
            SegError::Backend(msg) => assert!(msg.contains("orig_im_size"), "msg: {msg}"),
            other => panic!("expected Backend, got {other:?}"),
        }
    }
}
