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
        self.first_prompt_at = None;
        self.prompt_t_ms.clear();

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
            return;
        }
        match seg.segment(&self.session) {
            Ok(res) => {
                self.last_inference_ms = Some(res.inference_time.as_millis() as u64);
                self.mask_texture = Some(mask_to_texture(ctx, &res.mask));
                self.last_mask = Some(res.mask.clone());
                self.last_logits = Some(res.logits.clone());
                self.error = None;
            }
            Err(e) => {
                tracing::error!("segment failed: {e}");
                self.error = Some(format!("segment: {e}"));
                self.last_mask = None;
                self.last_logits = None;
            }
        }
    }
}
