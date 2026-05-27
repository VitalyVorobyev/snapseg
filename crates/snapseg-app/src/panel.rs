//! Right-side controls panel and bottom status bar. Pure rendering of
//! app state; no inference, no I/O.

use eframe::egui;
use snapseg_core::Polarity;

use crate::app::SnapsegApp;

impl SnapsegApp {
    /// Right side panel: open / load / polarity toggle / prompt counter
    /// / model + embedding status / latency.
    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("snapseg");
        ui.separator();

        if ui.button("Open image…").clicked() {
            self.open_image_dialog(ctx);
        }
        if ui.button("Load MobileSAM…").clicked() {
            self.load_mobile_sam_dialog();
        }

        ui.add_space(8.0);
        ui.label("Click tool:");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.current_polarity, Polarity::Positive, "● Positive");
            ui.selectable_value(&mut self.current_polarity, Polarity::Negative, "✕ Negative");
        });
        ui.label(
            egui::RichText::new("Right-click = opposite polarity")
                .small()
                .weak(),
        );

        ui.add_space(8.0);
        ui.separator();
        ui.label(format!("Prompts: {}", self.session.prompts.len()));
        if ui.button("Clear prompts").clicked() {
            self.session.clear();
            self.mask_texture = None;
            self.last_inference_ms = None;
        }

        ui.add_space(8.0);
        ui.separator();
        match &self.segmenter_label {
            Some(s) => ui.label(format!("Model: {s}")),
            None => ui.label("Model: (none loaded)"),
        };
        ui.label(format!(
            "Embedding: {}",
            if self.embedding_ready {
                "ready"
            } else {
                "pending"
            }
        ));
        if let Some(ms) = self.last_inference_ms {
            ui.label(format!("Last segment: {ms} ms"));
        }
    }

    /// Bottom status bar: error message (if any), otherwise filename +
    /// dimensions + prompt count, otherwise idle hint.
    pub(crate) fn draw_status(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, err);
            } else if let Some(img) = &self.image {
                ui.label(format!(
                    "{}  •  {}×{}  •  prompts: {}",
                    img.path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                    img.gray.width,
                    img.gray.height,
                    self.session.prompts.len(),
                ));
            } else {
                ui.label("idle — open an image to begin");
            }
        });
    }
}
