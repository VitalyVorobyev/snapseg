//! "Labels" section of the side panel: label-dir picker, quality
//! radio, note text edit, save button, and the last-save status line.
//!
//! Owned by the [`crate::panel`] module; called as a single helper out
//! of [`crate::app::SnapsegApp::draw_controls`] so the top-to-bottom
//! reading order of the panel stays linear.

use eframe::egui;

use crate::app::SnapsegApp;
use crate::labels;

impl SnapsegApp {
    /// Render the labels block. Mutates `self` because the save button
    /// triggers [`labels::save_current_label`] and updates the
    /// last-save status line.
    pub(super) fn draw_labels_section(&mut self, ui: &mut egui::Ui) {
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
    }
}
