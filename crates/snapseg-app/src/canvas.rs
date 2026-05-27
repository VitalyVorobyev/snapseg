//! Central image canvas: fits the source image into the available
//! area, paints the segmentation overlay, captures clicks, and draws
//! the prompt markers.
//!
//! `draw_canvas` is a method on `SnapsegApp` because every step pokes
//! at app state; the geometry-only helpers live in [`crate::coords`]
//! and the texture builders in [`crate::textures`].

use eframe::egui;
use snapseg_core::{Polarity, Prompt};

use crate::app::SnapsegApp;
use crate::coords::{fit_rect, image_to_screen, opposite, screen_to_image};

impl SnapsegApp {
    /// Render the image, the mask overlay, and the prompt markers,
    /// and capture clicks. Primary mouse contributes a click of the
    /// current polarity; secondary mouse contributes the opposite.
    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(img) = &self.image else { return };

        let avail = ui.available_size();
        let img_size = egui::vec2(img.gray.width as f32, img.gray.height as f32);
        let display_rect = fit_rect(img_size, avail, ui.cursor().min);

        let response = ui.allocate_rect(display_rect, egui::Sense::click());
        let painter = ui.painter_at(display_rect);

        paint_image(&painter, img.texture.id(), display_rect);
        if let Some(mask_tex) = &self.mask_texture {
            paint_image(&painter, mask_tex.id(), display_rect);
        }

        let clicked = capture_click(self, &response, display_rect, img_size);
        paint_prompts(&painter, &self.session.prompts, display_rect, img_size);

        if clicked {
            self.run_segment(ctx);
        }
    }
}

fn paint_image(painter: &egui::Painter, id: egui::TextureId, rect: egui::Rect) {
    painter.image(
        id,
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}

/// Returns true if a new click was appended to `app.session`.
fn capture_click(
    app: &mut SnapsegApp,
    response: &egui::Response,
    display_rect: egui::Rect,
    img_size: egui::Vec2,
) -> bool {
    if !(response.clicked() || response.secondary_clicked()) {
        return false;
    }
    let secondary = response.secondary_clicked();
    let Some(pos) = response.interact_pointer_pos() else {
        return false;
    };
    let img_pt = screen_to_image(pos, display_rect, img_size);
    let polarity = if secondary {
        opposite(app.current_polarity)
    } else {
        app.current_polarity
    };
    // Track per-prompt timing in lockstep with `session.prompts` so the
    // `t_ms` field on the saved label matches the order of clicks.
    let now = std::time::Instant::now();
    let t_ms = match app.first_prompt_at {
        Some(start) => (now - start).as_millis() as u64,
        None => {
            app.first_prompt_at = Some(now);
            0
        }
    };
    app.prompt_t_ms.push(t_ms);
    app.session.push(Prompt::Click {
        point: img_pt,
        polarity,
    });
    true
}

fn paint_prompts(
    painter: &egui::Painter,
    prompts: &[Prompt],
    display_rect: egui::Rect,
    img_size: egui::Vec2,
) {
    for prompt in prompts {
        if let Prompt::Click { point, polarity } = prompt {
            let screen = image_to_screen(*point, display_rect, img_size);
            let (fill, stroke) = match polarity {
                Polarity::Positive => (egui::Color32::from_rgb(80, 220, 100), egui::Color32::BLACK),
                Polarity::Negative => (egui::Color32::from_rgb(230, 90, 90), egui::Color32::BLACK),
            };
            painter.circle_filled(screen, 6.0, fill);
            painter.circle_stroke(screen, 6.0, egui::Stroke::new(1.5, stroke));
        }
    }
}
