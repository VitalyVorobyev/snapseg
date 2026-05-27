//! Right-side controls panel and bottom status bar. Pure rendering of
//! app state; no inference, no I/O.

use eframe::egui;
use snapseg_core::Polarity;

use crate::app::SnapsegApp;
use crate::labels;

impl SnapsegApp {
    /// Right side panel: open / load / polarity toggle / prompt counter
    /// / model + embedding status / latency.
    // why this is long: UI-rendering with section discipline; reads top
    // to bottom matching on-screen order, which is clearer than
    // fragmenting it into per-section helpers that each take `&mut self`.
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
            self.prompt_t_ms.clear();
            self.first_prompt_at = None;
            self.mask_texture = None;
            self.last_inference_ms = None;
            self.last_mask = None;
            self.last_logits = None;
            self.refined_polygon = None;
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

        ui.add_space(8.0);
        ui.separator();
        ui.label(format!("Labels: {}", self.label_dir.root.display()));
        if ui.button("Change…").clicked() {
            self.pick_label_dir_dialog();
        }

        ui.add_space(4.0);
        ui.label("Quality:");
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.pending_quality,
                snapseg_labels::LabelQuality::Good,
                "Good",
            );
            ui.selectable_value(
                &mut self.pending_quality,
                snapseg_labels::LabelQuality::NeedsReview,
                "Needs review",
            );
            ui.selectable_value(
                &mut self.pending_quality,
                snapseg_labels::LabelQuality::Reject,
                "Reject",
            );
        });

        ui.add_space(2.0);
        ui.label("Note:");
        ui.text_edit_singleline(&mut self.pending_note);

        let save_enabled =
            self.image.is_some() && self.last_mask.is_some() && self.segmenter_family.is_some();
        ui.add_space(4.0);
        if ui
            .add_enabled(save_enabled, egui::Button::new("Save label"))
            .clicked()
        {
            match labels::save_current_label(self) {
                Ok(label) => {
                    tracing::info!(id = %label.id, dir = %label.dir.display(), "label saved");
                    self.last_save_status = Some(format!("Saved {}", label.id));
                    self.pending_note.clear();
                    self.pending_quality = snapseg_labels::LabelQuality::Good;
                }
                Err(e) => {
                    tracing::error!("save label failed: {e}");
                    self.last_save_status = Some(format!("Save failed: {e}"));
                }
            }
        }
        match &self.last_save_status {
            Some(s) => ui.label(format!("Last save: {s}")),
            None => ui.label("Last save: (none yet)"),
        };

        ui.add_space(8.0);
        ui.separator();
        let prev_refine = self.refine_edges;
        ui.checkbox(&mut self.refine_edges, "Refine subpixel edges");
        let edge_on = !prev_refine && self.refine_edges;
        let edge_off = prev_refine && !self.refine_edges;
        if edge_on {
            self.refine_now();
        }
        if edge_off {
            self.refined_polygon = None;
        }
        if self.refine_edges {
            if let Some(p) = &self.refined_polygon {
                ui.label(format!("Polygon: {} vertices", p.vertices.len()));
            } else {
                ui.label("Polygon: (pending next segment)");
            }
        }

        if let Some(ms) = self.last_inference_ms {
            ui.add_space(8.0);
            ui.separator();
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
