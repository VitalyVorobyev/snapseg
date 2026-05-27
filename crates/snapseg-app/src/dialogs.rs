//! File-picker plumbing.
//!
//! `rfd` opens platform-native dialogs; these methods translate user
//! file selections into app state mutations and trigger the encoder
//! pass synchronously. Async file dialogs are a future refactor.

use std::collections::HashMap;
use std::path::PathBuf;

use eframe::egui;

use snapseg_core::InteractiveSegmenter;
use snapseg_models::mobile_sam::MobileSamSegmenter;
use snapseg_runtime::RuntimeConfig;

use crate::app::SnapsegApp;
use crate::textures::load_image;

impl SnapsegApp {
    /// Open a file dialog, load the chosen image, reset the prompt
    /// session, and trigger the encoder pass if a segmenter is loaded.
    /// Errors surface via `self.error`; nothing panics.
    pub(crate) fn open_image_dialog(&mut self, ctx: &egui::Context) {
        let Some(path) = pick_image() else { return };
        match load_image(&path, ctx) {
            Ok(loaded) => {
                tracing::info!(
                    path = %loaded.path.display(),
                    w = loaded.gray.width,
                    h = loaded.gray.height,
                    "image loaded"
                );
                self.image = Some(loaded);
                self.session.clear();
                self.mask_texture = None;
                self.last_inference_ms = None;
                self.embedding_ready = false;
                self.error = None;
                self.run_set_image();
            }
            Err(e) => {
                tracing::error!("failed to load image: {e:#}");
                self.error = Some(format!("Load failed: {e}"));
            }
        }
    }

    /// Open two file dialogs (encoder, then decoder), build a
    /// [`MobileSamSegmenter`], and run the encoder if an image is
    /// already loaded.
    pub(crate) fn load_mobile_sam_dialog(&mut self) {
        let Some(enc) = pick_onnx("Select MobileSAM encoder.onnx") else {
            return;
        };
        let Some(dec) = pick_onnx("Select MobileSAM decoder.onnx") else {
            return;
        };

        let mut parts: HashMap<String, PathBuf> = HashMap::new();
        parts.insert("encoder".to_string(), enc);
        parts.insert("decoder".to_string(), dec);

        let config = RuntimeConfig::default();
        match MobileSamSegmenter::from_parts("mobile-sam".to_string(), &parts, 1024, &config) {
            Ok(seg) => {
                tracing::info!("MobileSAM loaded");
                self.segmenter_label = Some(format!("{} ({})", seg.name(), "CPU"));
                self.segmenter = Some(Box::new(seg));
                self.embedding_ready = false;
                self.error = None;
                self.run_set_image();
            }
            Err(e) => {
                tracing::error!("MobileSAM load failed: {e}");
                self.error = Some(format!("MobileSAM load: {e}"));
            }
        }
    }
}

fn pick_image() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        .pick_file()
}

fn pick_onnx(title: &str) -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("ONNX", &["onnx"])
        .set_title(title)
        .pick_file()
}
