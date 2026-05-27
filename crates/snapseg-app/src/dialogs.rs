//! File-picker plumbing and model-load dispatch.
//!
//! `rfd` opens platform-native dialogs; these methods translate user
//! file selections into app state mutations and trigger the encoder
//! pass synchronously. Async file dialogs are a future refactor.
//!
//! [`try_load_model`] is the shared dispatch point used by both the
//! file-dialog path and the config auto-load path.

use std::collections::HashMap;
use std::path::PathBuf;

use eframe::egui;

use snapseg_core::InteractiveSegmenter;
use snapseg_models::focalclick::FocalClickSegmenter;
use snapseg_models::mobile_sam::MobileSamSegmenter;
use snapseg_models::ritm::RitmSegmenter;
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

        try_load_model(
            self,
            "mobile-sam".to_string(),
            "mobile_sam".to_string(),
            1024,
            parts,
        );
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

    /// Run the pending auto-load scheduled by [`SnapsegApp::with_config`].
    /// Called at the top of every `update` frame; the `take()` ensures it
    /// executes at most once per construction. The encoder pass also runs
    /// here if an image was already loaded before this tick.
    pub(crate) fn tick_pending_autoload(&mut self, _ctx: &egui::Context) {
        let Some(cfg) = self.pending_autoload.take() else {
            return;
        };
        try_load_model(self, cfg.name, cfg.family, cfg.input_size, cfg.parts);
    }
}

/// Build a segmenter for the given `family`, wire it into `app`, and run
/// the encoder pass if an image is already loaded.
///
/// `parts` maps part names (e.g. `"encoder"`, `"decoder"`, `"model"`) to
/// on-disk ONNX paths. SHA-256 digests of the encoder and decoder parts
/// are computed and stored for label provenance.
///
/// Unknown family slugs are surfaced via `app.error`; this function never
/// panics.
// why this is long: arms per family + sha256 bookkeeping + error wiring;
// splitting per-family would scatter context without reducing coupling.
pub(crate) fn try_load_model(
    app: &mut SnapsegApp,
    name: String,
    family: String,
    input_size: u32,
    parts: HashMap<String, PathBuf>,
) {
    // Hash whichever part files are present for label provenance.
    let encoder_sha256 = parts.get("encoder").and_then(|p| compute_sha256(p).ok());
    let decoder_sha256 = parts.get("decoder").and_then(|p| compute_sha256(p).ok());

    let config = RuntimeConfig::default();

    let result: Result<Box<dyn InteractiveSegmenter>, String> = match family.as_str() {
        "mobile_sam" => MobileSamSegmenter::from_parts(name.clone(), &parts, input_size, &config)
            .map(|s| Box::new(s) as Box<dyn InteractiveSegmenter>)
            .map_err(|e| format!("MobileSAM load: {e}")),
        "ritm" => {
            let seg = RitmSegmenter::new(name.clone(), (input_size, input_size));
            Ok(Box::new(seg) as Box<dyn InteractiveSegmenter>)
        }
        "focalclick" => {
            let seg = FocalClickSegmenter::new(name.clone(), (input_size, input_size));
            Ok(Box::new(seg) as Box<dyn InteractiveSegmenter>)
        }
        other => Err(format!("unknown family '{other}'")),
    };

    match result {
        Ok(seg) => {
            tracing::info!(family = %family, name = %name, "model loaded");
            app.segmenter_label = Some(format!("{} (CPU)", seg.name()));
            app.segmenter_family = Some(family);
            app.segmenter_registry_name = Some(name);
            app.encoder_sha256 = encoder_sha256;
            app.decoder_sha256 = decoder_sha256;
            app.embedding_ready = false;
            app.error = None;
            app.segmenter = Some(seg);
            app.run_set_image();
        }
        Err(e) => {
            tracing::error!("model load failed: {e}");
            app.error = Some(e);
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
