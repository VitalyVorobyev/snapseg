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

        // Hash the ONNX files now, before moving them into `parts`, so
        // we can record encoder/decoder identity in label provenance
        // without re-reading the file later.
        let encoder_sha256 = compute_sha256(&enc).ok();
        let decoder_sha256 = compute_sha256(&dec).ok();

        let mut parts: HashMap<String, PathBuf> = HashMap::new();
        parts.insert("encoder".to_string(), enc);
        parts.insert("decoder".to_string(), dec);

        let config = RuntimeConfig::default();
        match MobileSamSegmenter::from_parts("mobile-sam".to_string(), &parts, 1024, &config) {
            Ok(seg) => {
                tracing::info!("MobileSAM loaded");
                self.segmenter_label = Some(format!("{} ({})", seg.name(), "CPU"));
                self.segmenter = Some(Box::new(seg));
                self.segmenter_family = Some("mobile_sam".to_string());
                self.segmenter_registry_name = Some("mobile-sam".to_string());
                self.encoder_sha256 = encoder_sha256;
                self.decoder_sha256 = decoder_sha256;
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

    /// File-picker for the label root directory. On success, updates
    /// `self.label_dir` and clears any stale save status from the
    /// previous root.
    pub(crate) fn pick_label_dir_dialog(&mut self) {
        let picked = rfd::FileDialog::new()
            .set_title("Pick labels directory…")
            .pick_folder();
        if let Some(p) = picked {
            self.label_dir = snapseg_labels::LabelDir::new(p);
            self.last_save_status = None;
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

/// Stream-hash a file with SHA-256, returning the lowercase hex digest.
/// Used to record encoder/decoder ONNX identity in label provenance so
/// labels can be traced back to the exact model weights that produced
/// them.
fn compute_sha256(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}
