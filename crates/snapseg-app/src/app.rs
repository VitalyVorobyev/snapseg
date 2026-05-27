//! Top-level application state and the `eframe::App::update` wiring.
//!
//! Methods that *do* work — opening dialogs, running inference, drawing
//! the canvas, drawing the panel — live in dedicated modules and extend
//! `SnapsegApp` via `impl` blocks there. This module owns only the
//! struct definition, the default constructor, and the per-frame
//! orchestration.

use std::path::PathBuf;

use eframe::egui;

use snapseg_core::{GrayImage, InteractiveSegmenter, Polarity, PromptSession};

/// Per-frame application state.
///
/// One image, one segmenter, one prompt session. The segmenter is held
/// behind a trait object so adapter swapping at runtime is one
/// reassignment.
pub struct SnapsegApp {
    /// The image currently being segmented, if any.
    pub(crate) image: Option<LoadedImage>,
    /// Polarity attached to a primary-button click; secondary-button click
    /// uses the opposite. Keeps the workflow single-handed.
    pub(crate) current_polarity: Polarity,
    /// Accumulated prompts fed to the segmenter on every `segment()` call.
    pub(crate) session: PromptSession,
    /// Last error to surface in the status bar; cleared on the next
    /// successful operation.
    pub(crate) error: Option<String>,
    /// Active segmenter. `None` means "no model loaded".
    pub(crate) segmenter: Option<Box<dyn InteractiveSegmenter>>,
    /// Display label for the active model (shown in the side panel).
    pub(crate) segmenter_label: Option<String>,
    /// `true` once `set_image` has succeeded on the current image, so the
    /// next click can call `segment` profitably.
    pub(crate) embedding_ready: bool,
    /// Most recent mask uploaded to the GPU for overlay rendering.
    pub(crate) mask_texture: Option<egui::TextureHandle>,
    /// Wall-clock duration of the last `segment` call, surfaced in the panel.
    pub(crate) last_inference_ms: Option<u64>,
}

impl Default for SnapsegApp {
    fn default() -> Self {
        Self {
            image: None,
            current_polarity: Polarity::Positive,
            session: PromptSession::new(),
            error: None,
            segmenter: None,
            segmenter_label: None,
            embedding_ready: false,
            mask_texture: None,
            last_inference_ms: None,
        }
    }
}

/// An image loaded from disk plus its GPU-side texture for display.
pub(crate) struct LoadedImage {
    pub(crate) path: PathBuf,
    pub(crate) gray: GrayImage,
    pub(crate) texture: egui::TextureHandle,
}

impl eframe::App for SnapsegApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::right("controls")
            .default_width(280.0)
            .show(ctx, |ui| self.draw_controls(ui, ctx));

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| self.draw_status(ui));

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.image.is_some() {
                self.draw_canvas(ui, ctx);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Open an image (right panel) to begin.");
                });
            }
        });
    }
}
