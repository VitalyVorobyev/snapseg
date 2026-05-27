//! MobileSAM adapter (Zhang et al. 2023). Distilled SAM with a tiny image
//! encoder (~10MB) plus the SAM mask decoder. Click + box prompts; image
//! embedding is computed once per image, then each prompt is a cheap
//! decoder pass.
//!
//! Built against the canonical SAM ONNX I/O schema used by the official
//! Meta export and the most common community MobileSAM exports:
//!
//! Encoder:
//!   in   `input_image`      : float32 [1, 3, S, S]
//!   out  *first*             : float32 [1, 256, S/16, S/16]   (image embedding)
//!
//! Decoder:
//!   in   `image_embeddings` : float32 [1, 256, h, w]
//!   in   `point_coords`     : float32 [1, N, 2]
//!   in   `point_labels`     : float32 [1, N]
//!   in   `mask_input`       : float32 [1, 1, 256, 256]
//!   in   `has_mask_input`   : float32 [1]
//!   in   `orig_im_size`     : float32 [2]   (orig_h, orig_w)
//!   out  `masks`            : float32 [1, K, orig_h, orig_w]
//!   out  `iou_predictions`  : float32 [1, K]
//!   out  `low_res_masks`    : float32 [1, K, 256, 256]
//!
//! If a particular export uses different I/O names, the lookups in
//! `set_image` / `segment` are the only thing that needs to move.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array2, Array3, Array4};

use snapseg_core::{
    Capabilities, GrayImage, InteractiveSegmenter, Polarity, Prompt, PromptSession, SegError,
    SegmentationResult,
};
use snapseg_runtime::preprocess::{SamResize, sam_preprocess_gray};
use snapseg_runtime::{Backend, RuntimeConfig};

pub struct MobileSamSegmenter {
    name: String,
    input_size: u32,
    encoder: Backend,
    decoder: Backend,
    state: Option<EncodedImage>,
}

struct EncodedImage {
    embedding: Array4<f32>,
    resize_info: SamResize,
    /// `low_res_masks` from the previous decoder call, fed back as
    /// `mask_input` to let SAM iteratively refine.
    prev_low_res: Option<Array4<f32>>,
}

impl MobileSamSegmenter {
    pub fn from_parts(
        name: String,
        parts: &HashMap<String, PathBuf>,
        input_size: u32,
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
            input_size,
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
            recommended_input_size: (self.input_size, self.input_size),
        }
    }

    fn set_image(&mut self, image: &GrayImage) -> Result<(), SegError> {
        let (input, resize_info) = sam_preprocess_gray(image, self.input_size as usize);

        let v_input = ort::value::Value::from_array(input)
            .map_err(|e| SegError::Backend(format!("encoder input value: {e}")))?;
        let outputs = self
            .encoder
            .session_mut()
            .run(ort::inputs!["input_image" => v_input])
            .map_err(|e| SegError::Backend(format!("encoder run: {e}")))?;

        // Most exports name the output something with "embedding"; if not,
        // fall back to the first output tensor.
        let (_, value) = outputs
            .iter()
            .find(|(name, _)| name.to_lowercase().contains("embedding"))
            .or_else(|| outputs.iter().next())
            .ok_or_else(|| SegError::Backend("encoder returned no outputs".into()))?;

        let (shape_ref, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| SegError::Backend(format!("embedding extract: {e}")))?;
        let shape: Vec<usize> = shape_ref.iter().map(|&d| d as usize).collect();
        if shape.len() != 4 {
            return Err(SegError::Backend(format!(
                "expected 4-D embedding, got shape {shape:?}"
            )));
        }
        let embedding = Array4::from_shape_vec(
            (shape[0], shape[1], shape[2], shape[3]),
            data.to_vec(),
        )
        .map_err(|e| SegError::Backend(format!("embedding reshape: {e}")))?;

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

        // SAM prompt-label convention:
        //   1 = positive click, 0 = negative click,
        //   2 = box top-left,   3 = box bottom-right,
        //  -1 = padding / "no prompt" slot.
        let mut coords: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        for prompt in &session.prompts {
            match prompt {
                Prompt::Click { point, polarity } => {
                    let (ex, ey) = state.resize_info.point_to_encoder(point.x, point.y);
                    coords.push(ex);
                    coords.push(ey);
                    labels.push(match polarity {
                        Polarity::Positive => 1.0,
                        Polarity::Negative => 0.0,
                    });
                }
                Prompt::Box(b) => {
                    let (x0, y0) = state.resize_info.point_to_encoder(b.x0, b.y0);
                    let (x1, y1) = state.resize_info.point_to_encoder(b.x1, b.y1);
                    coords.extend_from_slice(&[x0, y0, x1, y1]);
                    labels.push(2.0);
                    labels.push(3.0);
                }
                Prompt::Scribble { .. } => {
                    // MobileSAM does not ingest scribbles via this export.
                }
            }
        }

        if labels.is_empty() {
            let h = state.resize_info.orig_h as usize;
            let w = state.resize_info.orig_w as usize;
            return Ok(SegmentationResult {
                mask: Array2::from_elem((h, w), false),
                logits: Array2::zeros((h, w)),
                inference_time: started.elapsed(),
            });
        }

        let n_points = labels.len();
        let point_coords = Array3::from_shape_vec((1, n_points, 2), coords)
            .map_err(|e| SegError::Backend(e.to_string()))?;
        let point_labels = Array2::from_shape_vec((1, n_points), labels)
            .map_err(|e| SegError::Backend(e.to_string()))?;

        let mask_input = state
            .prev_low_res
            .clone()
            .unwrap_or_else(|| Array4::<f32>::zeros((1, 1, 256, 256)));
        let has_mask = if state.prev_low_res.is_some() { 1.0_f32 } else { 0.0 };
        let has_mask_input = ndarray::arr1(&[has_mask]);
        let orig_im_size = ndarray::arr1(&[
            state.resize_info.orig_h as f32,
            state.resize_info.orig_w as f32,
        ]);

        let mkval = |arr: ndarray::ArrayD<f32>, what: &str| {
            ort::value::Value::from_array(arr)
                .map_err(|e| SegError::Backend(format!("{what} value: {e}")))
        };
        let v_emb = mkval(state.embedding.clone().into_dyn(), "image_embeddings")?;
        let v_coords = mkval(point_coords.into_dyn(), "point_coords")?;
        let v_labels = mkval(point_labels.into_dyn(), "point_labels")?;
        let v_mask = mkval(mask_input.into_dyn(), "mask_input")?;
        let v_has = mkval(has_mask_input.into_dyn(), "has_mask_input")?;
        let v_orig = mkval(orig_im_size.into_dyn(), "orig_im_size")?;

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

        let masks_value = outputs
            .iter()
            .find(|(name, _)| name.to_lowercase() == "masks")
            .map(|(_, v)| v)
            .ok_or_else(|| SegError::Backend("decoder missing 'masks' output".into()))?;
        let (m_shape_ref, m_data) = masks_value
            .try_extract_tensor::<f32>()
            .map_err(|e| SegError::Backend(format!("masks extract: {e}")))?;
        let m_shape: Vec<usize> = m_shape_ref.iter().map(|&d| d as usize).collect();
        if m_shape.len() != 4 {
            return Err(SegError::Backend(format!(
                "expected 4-D masks, got {m_shape:?}"
            )));
        }
        let (n, k, h, w) = (m_shape[0], m_shape[1], m_shape[2], m_shape[3]);
        if n != 1 || k == 0 {
            return Err(SegError::Backend(format!(
                "unexpected masks shape {m_shape:?}"
            )));
        }

        // First of K predicted masks. Multi-mask selection (by IoU
        // prediction) is a refinement for later.
        let stride_per_mask = h * w;
        let mut logits = Array2::<f32>::zeros((h, w));
        let mut mask = Array2::<bool>::from_elem((h, w), false);
        for y in 0..h {
            for x in 0..w {
                let v = m_data[y * w + x];
                logits[(y, x)] = v;
                mask[(y, x)] = v > 0.0;
            }
        }
        let _ = stride_per_mask; // currently only mask 0 consumed; kept for the next iteration.

        if let Some((_, lr_value)) = outputs
            .iter()
            .find(|(name, _)| name.to_lowercase() == "low_res_masks")
        {
            if let Ok((lr_shape_ref, lr_data)) = lr_value.try_extract_tensor::<f32>() {
                let lr_shape: Vec<usize> = lr_shape_ref.iter().map(|&d| d as usize).collect();
                if lr_shape.len() == 4 {
                    if let Ok(lr_owned) = Array4::from_shape_vec(
                        (lr_shape[0], lr_shape[1], lr_shape[2], lr_shape[3]),
                        lr_data.to_vec(),
                    ) {
                        state.prev_low_res = Some(lr_owned);
                    }
                }
            }
        }

        let elapsed = started.elapsed();
        tracing::debug!(
            ms = elapsed.as_millis() as u64,
            n_prompts = session.prompts.len(),
            "mobile_sam segment"
        );

        Ok(SegmentationResult {
            mask,
            logits,
            inference_time: elapsed,
        })
    }
}
