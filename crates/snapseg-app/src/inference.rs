//! Synchronous inference plumbing.
//!
//! This is the seam where threaded inference will land (M5). Today
//! both `run_set_image` and `run_segment` block the UI thread; the
//! signatures and surrounding state are kept narrow so the threaded
//! refactor can replace bodies without churning the call sites.

use std::time::Instant;

use eframe::egui;

use crate::app::SnapsegApp;
use crate::textures::mask_to_texture;

impl SnapsegApp {
    /// Run the segmenter's `set_image` on the currently loaded image.
    /// No-ops if either image or segmenter is missing. Updates
    /// `embedding_ready` on success; surfaces errors via `self.error`.
    pub(crate) fn run_set_image(&mut self) {
        // Clear stale per-image state at the start so a successful
        // encoder pass on a new image leaves nothing behind from the
        // previous one.
        self.last_mask = None;
        self.last_logits = None;
        self.refined_polygon = None;
        self.first_prompt_at = None;
        self.prompt_t_ms.clear();
        self.last_candidates.clear();
        self.selected_mask_idx = 0;
        self.selected_vertex_idx = None;
        self.hover_pixel = None;

        let (Some(img), Some(seg)) = (&self.image, self.segmenter.as_mut()) else {
            return;
        };
        let started = Instant::now();
        match seg.set_image(&img.gray) {
            Ok(()) => {
                self.embedding_ready = true;
                let elapsed = started.elapsed();
                self.last_encoder_ms = Some(elapsed.as_millis() as u64);
                tracing::info!(ms = elapsed.as_millis() as u64, "encoder pass complete");
            }
            Err(e) => {
                tracing::error!("set_image failed: {e}");
                self.error = Some(format!("set_image: {e}"));
                self.embedding_ready = false;
            }
        }
    }

    /// Run the segmenter on the current `PromptSession`, upload the
    /// resulting mask to a texture, and surface latency in the panel.
    /// No-ops when no segmenter is loaded, the embedding isn't ready,
    /// or the prompt session is empty (overlay cleared in that case).
    // why this is long: one inference call updates ten correlated
    // pieces of UI state (mask, logits, candidates, refined polygon,
    // mask texture, latency, selected indices, error). Splitting them
    // into per-field helpers would lose locality without reducing
    // coupling — they all key off the same `Result<SegmentationResult>`.
    pub(crate) fn run_segment(&mut self, ctx: &egui::Context) {
        if !self.embedding_ready {
            return;
        }
        let Some(seg) = self.segmenter.as_mut() else {
            return;
        };
        if self.session.is_empty() {
            self.mask_texture = None;
            self.last_inference_ms = None;
            self.refined_polygon = None;
            self.last_candidates.clear();
            self.selected_mask_idx = 0;
            self.selected_vertex_idx = None;
            return;
        }
        match seg.segment(&self.session) {
            Ok(res) => {
                self.last_inference_ms = Some(res.inference_time.as_millis() as u64);
                // Pick argmax-IoU candidate as the headline. The
                // adapter already populated `res.mask` / `res.logits`
                // from that slot, but we recompute here so the
                // UI-visible index stays consistent if the adapter
                // contract ever drifts.
                let selected = res.selected_index();
                self.selected_mask_idx = selected;
                self.last_candidates = res.candidates;
                let mask_for_paint = self
                    .last_candidates
                    .get(selected)
                    .map(|c| c.mask.clone())
                    .unwrap_or(res.mask);
                let logits_for_paint = self
                    .last_candidates
                    .get(selected)
                    .map(|c| c.logits.clone())
                    .unwrap_or(res.logits);
                self.mask_texture = Some(mask_to_texture(ctx, &mask_for_paint));
                if self.refine_edges {
                    if let Some(img) = self.image.as_ref() {
                        self.refined_polygon = Some(snapseg_edges::refine_polygon(
                            &mask_for_paint,
                            &img.gray,
                            self.refine_params,
                        ));
                    } else {
                        self.refined_polygon = None;
                    }
                } else {
                    self.refined_polygon = None;
                }
                self.selected_vertex_idx = None;
                self.last_mask = Some(mask_for_paint);
                self.last_logits = Some(logits_for_paint);
                self.error = None;
            }
            Err(e) => {
                tracing::error!("segment failed: {e}");
                self.error = Some(format!("segment: {e}"));
                self.last_mask = None;
                self.last_logits = None;
                self.refined_polygon = None;
                self.last_candidates.clear();
                self.selected_mask_idx = 0;
                self.selected_vertex_idx = None;
            }
        }
    }

    /// Switch the currently displayed mask to `idx`, refresh the
    /// overlay texture, and re-run subpixel refinement against the new
    /// mask if the toggle is on. Out-of-range / no-image / no-candidate
    /// inputs no-op silently — the operator can hammer arrow keys
    /// without crashing.
    pub(crate) fn select_mask(&mut self, idx: usize, ctx: &egui::Context) {
        if self.last_candidates.is_empty() || idx >= self.last_candidates.len() {
            return;
        }
        // `idx` was bounds-checked above, so indexing is safe.
        let candidate = self.last_candidates[idx].clone();
        self.selected_mask_idx = idx;
        self.mask_texture = Some(mask_to_texture(ctx, &candidate.mask));
        if self.refine_edges {
            if let Some(img) = self.image.as_ref() {
                self.refined_polygon = Some(snapseg_edges::refine_polygon(
                    &candidate.mask,
                    &img.gray,
                    self.refine_params,
                ));
                // Polygon vertex count changed; reset the cycler.
                self.selected_vertex_idx = None;
            }
        }
        self.last_mask = Some(candidate.mask);
        self.last_logits = Some(candidate.logits);
    }

    /// Run the subpixel refinement once against the latest cached mask.
    /// No-op when there's no mask or no image.
    pub(crate) fn refine_now(&mut self) {
        let (Some(mask), Some(img)) = (self.last_mask.as_ref(), self.image.as_ref()) else {
            return;
        };
        self.refined_polygon = Some(snapseg_edges::refine_polygon(
            mask,
            &img.gray,
            self.refine_params,
        ));
    }
}
