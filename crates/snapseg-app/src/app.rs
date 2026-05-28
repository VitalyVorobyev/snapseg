//! Top-level application state and the `eframe::App::update` wiring.
//!
//! Methods that *do* work — opening dialogs, running inference, drawing
//! the canvas, drawing the panel — live in dedicated modules and extend
//! `SnapsegApp` via `impl` blocks there. This module owns only the
//! struct definition, the default constructor, and the per-frame
//! orchestration.

use std::path::PathBuf;

use eframe::egui;

use snapseg_core::{GrayImage, InteractiveSegmenter, MaskCandidate, Polarity, PromptSession};

use crate::coords::ViewState;

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
    /// Where labels are persisted on disk. Defaults to `./labels/`.
    pub(crate) label_dir: snapseg_labels::LabelDir,
    /// Per-prompt timestamp offsets, parallel to `session.prompts`. Zero
    /// for the first prompt; ms since the first prompt for subsequent
    /// ones. Reset when the prompt session is cleared.
    pub(crate) prompt_t_ms: Vec<u64>,
    /// Instant of the first prompt of the current session, used to
    /// compute `prompt_t_ms`.
    pub(crate) first_prompt_at: Option<std::time::Instant>,
    /// Family slug of the active segmenter, e.g. `"mobile_sam"`. Set
    /// when a model is loaded; needed for provenance.
    pub(crate) segmenter_family: Option<String>,
    /// Registry name of the active segmenter, e.g. `"mobile-sam"`. Set
    /// when a model is loaded.
    pub(crate) segmenter_registry_name: Option<String>,
    /// Cached SHA-256 of the encoder ONNX, if any. Set when loading.
    /// Populated for SAM-family models that ship encoder + decoder
    /// parts; `None` for single-network families.
    pub(crate) encoder_sha256: Option<String>,
    /// Cached SHA-256 of the decoder ONNX, if any. Set when loading.
    /// Same population rule as [`Self::encoder_sha256`].
    pub(crate) decoder_sha256: Option<String>,
    /// Cached SHA-256 of the single-network ONNX, if any. Set when
    /// loading. Populated for single-network families (RITM,
    /// FocalClick) where the registry's `parts["model"]` is the whole
    /// graph; `None` for SAM-family models.
    pub(crate) model_sha256: Option<String>,
    /// Most recent encoder pass duration (ms). Set by `run_set_image`.
    pub(crate) last_encoder_ms: Option<u64>,
    /// Most recent mask. Held so the user can save it as a label long
    /// after the segment call. Replaced on every successful segment.
    pub(crate) last_mask: Option<ndarray::Array2<bool>>,
    /// Most recent logits (for `logits.png` in the saved label).
    pub(crate) last_logits: Option<ndarray::Array2<f32>>,
    /// Status line of the last save attempt; surfaced in the side panel.
    pub(crate) last_save_status: Option<String>,
    /// Quality flag the operator will attach to the next saved label.
    /// Reset to `Good` after each successful save.
    pub(crate) pending_quality: snapseg_labels::LabelQuality,
    /// Free-form note the operator will attach to the next saved label.
    /// Cleared after each successful save.
    pub(crate) pending_note: String,
    /// Whether to run `snapseg_edges::refine_polygon` after every successful
    /// `segment()` call. The operator can toggle this from the side panel.
    pub(crate) refine_edges: bool,
    /// Per-vertex subpixel refinement of the latest mask, if
    /// `refine_edges` is on and the last segment succeeded.
    pub(crate) refined_polygon: Option<snapseg_edges::RefinedPolygon>,
    /// Knobs for the refinement pass. Defaults from
    /// [`snapseg_edges::RefineParams::default`].
    pub(crate) refine_params: snapseg_edges::RefineParams,
    /// Model to auto-load on the first frame after construction.
    /// Cleared after the first `tick_pending_autoload` call. Kept on
    /// `SnapsegApp` rather than `with_config`'s scope so the
    /// auto-load runs inside the egui event loop (where `ctx` exists
    /// for texture uploads).
    pub(crate) pending_autoload: Option<crate::config::ModelConfig>,
    /// Zoom + pan applied to the canvas. Reset to identity on new
    /// image / new model / double-click on the canvas.
    pub(crate) view: ViewState,
    /// Most recent hover position in image-pixel coordinates, if the
    /// cursor is over the image rect. Cleared when the cursor leaves
    /// the canvas; surfaced in the side panel's status section.
    pub(crate) hover_pixel: Option<(u32, u32)>,
    /// All K candidate masks from the latest `segment` call. Held so
    /// the side-panel cycler can switch between them without re-running
    /// inference. Cleared on new image / new prompts / model load.
    pub(crate) last_candidates: Vec<MaskCandidate>,
    /// Index into `last_candidates` of the currently selected mask.
    /// Defaults to `argmax-IoU` after each `segment`; arrow keys /
    /// radio buttons let the operator override. Out-of-range values
    /// are tolerated (clamped at paint time).
    pub(crate) selected_mask_idx: usize,
    /// Index into `refined_polygon.vertices` of the currently selected
    /// vertex, surfaced in the side panel and rendered as a filled dot
    /// on top of the gold contour. `None` when no polygon is available
    /// or the operator hasn't started cycling.
    pub(crate) selected_vertex_idx: Option<usize>,
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
            label_dir: snapseg_labels::LabelDir::new(std::path::PathBuf::from("./labels")),
            prompt_t_ms: Vec::new(),
            first_prompt_at: None,
            segmenter_family: None,
            segmenter_registry_name: None,
            encoder_sha256: None,
            decoder_sha256: None,
            model_sha256: None,
            last_encoder_ms: None,
            last_mask: None,
            last_logits: None,
            last_save_status: None,
            pending_quality: snapseg_labels::LabelQuality::Good,
            pending_note: String::new(),
            refine_edges: false,
            refined_polygon: None,
            refine_params: snapseg_edges::RefineParams::default(),
            pending_autoload: None,
            view: ViewState::default(),
            hover_pixel: None,
            last_candidates: Vec::new(),
            selected_mask_idx: 0,
            selected_vertex_idx: None,
        }
    }
}

impl SnapsegApp {
    /// Build from a loaded operator config. Applies overrides from the
    /// config to the default-constructed app and schedules a deferred
    /// auto-load of `default_model` on the first frame (see
    /// [`SnapsegApp::tick_pending_autoload`]).
    pub fn with_config(cfg: crate::config::AppConfig) -> Self {
        let mut app = Self::default();
        if let Some(dir) = cfg.label_dir {
            app.label_dir = snapseg_labels::LabelDir::new(dir);
        }
        if let Some(refine) = cfg.refine_edges {
            app.refine_edges = refine;
        }
        app.pending_autoload = cfg.default_model;
        app
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
        self.tick_pending_autoload(ctx);

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
