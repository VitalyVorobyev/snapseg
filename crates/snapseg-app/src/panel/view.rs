//! "View" expander section of the side panel: cursor pixel + grayscale
//! readout, multi-mask cycler (with IoU display), and subpixel-edge
//! vertex stepper. Also owns the arrow-key shortcut handler that drives
//! the cycler and stepper from the keyboard.
//!
//! All three widgets read `&mut self` because the cycler and stepper
//! trigger texture re-uploads via
//! [`crate::app::SnapsegApp::select_mask`]; the pixel readout is the
//! only purely read-only widget but stays here for cohesion.

use eframe::egui;

use crate::app::SnapsegApp;

impl SnapsegApp {
    /// Render the contents of the "View" `CollapsingHeader`. Called
    /// from [`crate::app::SnapsegApp::draw_controls`] inside the
    /// header's body closure.
    pub(super) fn draw_view_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.draw_pixel_readout(ui);
        ui.add_space(4.0);
        self.draw_mask_cycler(ui, ctx);
        ui.add_space(4.0);
        self.draw_vertex_stepper(ui);
        // Arrow keys handled after the widgets so any focused widget
        // (e.g. the note text edit) gets first dibs at consuming them.
        self.handle_arrow_keys(ctx);
    }

    fn draw_pixel_readout(&self, ui: &mut egui::Ui) {
        match self.hover_pixel.and_then(|(x, y)| {
            self.image
                .as_ref()
                .map(|img| (x, y, img.gray.data[(y as usize, x as usize)]))
        }) {
            Some((x, y, gray)) => {
                ui.label(format!("Cursor: ({x}, {y})  gray = {gray}"));
            }
            None => {
                ui.label(
                    egui::RichText::new("Cursor: (hover over image)")
                        .small()
                        .weak(),
                );
            }
        }
    }

    fn draw_mask_cycler(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let k = self.last_candidates.len();
        if k == 0 {
            ui.label(egui::RichText::new("Mask: (no result)").small().weak());
            return;
        }
        if k == 1 {
            // Single-candidate exports (RITM / single-mask MobileSAM)
            // — skip the cycler chrome but show the IoU so the
            // operator sees the model's confidence.
            let iou = self.last_candidates[0].iou;
            ui.label(format!("Mask: 1 / 1  •  IoU {iou:.2}"));
            return;
        }
        let idx = self.selected_mask_idx.min(k - 1);
        let iou = self.last_candidates[idx].iou;
        ui.label(format!("Mask: {} / {k}  •  IoU {iou:.2}", idx + 1));
        ui.horizontal(|ui| {
            let prev_clicked = ui.button("◀").clicked();
            let next_clicked = ui.button("▶").clicked();
            if prev_clicked && idx > 0 {
                self.select_mask(idx - 1, ctx);
            }
            if next_clicked && idx + 1 < k {
                self.select_mask(idx + 1, ctx);
            }
        });
    }

    fn draw_vertex_stepper(&mut self, ui: &mut egui::Ui) {
        let Some(polygon) = self.refined_polygon.as_ref() else {
            ui.label(egui::RichText::new("Vertex: (no polygon)").small().weak());
            return;
        };
        let n = polygon.vertices.len();
        if n == 0 {
            ui.label(
                egui::RichText::new("Vertex: (polygon is empty)")
                    .small()
                    .weak(),
            );
            return;
        }
        let idx = match self.selected_vertex_idx {
            Some(i) if i < n => i,
            _ => 0,
        };
        let v = polygon.vertices[idx];
        let conf = polygon.confidence.get(idx).copied().unwrap_or(0.0);
        ui.label(format!(
            "Vertex {} / {n}  •  ({:.2}, {:.2})  conf={conf:.1}",
            idx + 1,
            v.x,
            v.y
        ));
        ui.horizontal(|ui| {
            let prev_clicked = ui.button("◀").clicked();
            let next_clicked = ui.button("▶").clicked();
            if prev_clicked {
                self.selected_vertex_idx = Some((idx + n - 1) % n);
            }
            if next_clicked {
                self.selected_vertex_idx = Some((idx + 1) % n);
            }
        });
        if self.selected_vertex_idx.is_none() {
            self.selected_vertex_idx = Some(idx);
        }
    }

    /// Map keyboard arrows to mask / vertex cycling. Plain ←/→ cycles
    /// mask; Shift+←/→ cycles vertex. Skipped entirely when egui
    /// reports a focused widget (so the note text edit keeps its
    /// arrow-key cursor motion).
    fn handle_arrow_keys(&mut self, ctx: &egui::Context) {
        if ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (left, right, shift) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.modifiers.shift,
            )
        });
        if !left && !right {
            return;
        }
        if shift {
            // Vertex stepper.
            if let Some(polygon) = self.refined_polygon.as_ref() {
                let n = polygon.vertices.len();
                if n == 0 {
                    return;
                }
                let cur = self.selected_vertex_idx.unwrap_or(0);
                let next = if left {
                    (cur + n - 1) % n
                } else {
                    (cur + 1) % n
                };
                self.selected_vertex_idx = Some(next);
            }
            return;
        }
        // Plain arrows → mask cycler.
        let k = self.last_candidates.len();
        if k < 2 {
            return;
        }
        let cur = self.selected_mask_idx.min(k - 1);
        let next = if left {
            cur.saturating_sub(1)
        } else {
            (cur + 1).min(k - 1)
        };
        if next != cur {
            self.select_mask(next, ctx);
        }
    }
}
