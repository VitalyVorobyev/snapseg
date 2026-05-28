//! Central image canvas: fits the source image into the available
//! area, paints the segmentation overlay, captures clicks, draws the
//! prompt markers, and tracks zoom/pan + hover state.
//!
//! `draw_canvas` is a method on `SnapsegApp` because every step pokes
//! at app state; the geometry-only helpers live in [`crate::coords`]
//! and the texture builders in [`crate::textures`].
//!
//! ## Input bindings
//!
//! - Primary click: append a prompt of the current polarity.
//! - Secondary (right) click: append the opposite polarity.
//! - Mouse wheel (vertical scroll): zoom around the cursor.
//! - Middle-button drag: pan the view.
//! - Double-click (primary): reset zoom + pan to the fit baseline.
//!
//! Why middle-button drag instead of space-drag: middle-button is a
//! single-input gesture and doesn't conflict with potential
//! keyboard-driven mask/vertex cycling on the side panel. Space-drag
//! would have stolen `Space` from any future keyboard shortcut.

use eframe::egui;
use snapseg_core::{Point2, Polarity, Prompt};

use crate::app::SnapsegApp;
use crate::coords::{
    ViewState, fit_rect, image_to_screen, opposite, screen_to_image, view_rect, zoom_around,
};

impl SnapsegApp {
    /// Render the image, the mask overlay, the prompt markers, and
    /// the polygon, captures clicks, and updates `view_state` for any
    /// mouse-wheel / drag / double-click gestures.
    // why this is long: the canvas owns six concerns (allocation,
    // gesture handling, mask paint, click capture, polygon paint,
    // prompt paint) that pass the same `display_rect` / `img_size`
    // pair around. Splitting them would force every helper to take a
    // 5-arg geometry tuple, which is worse than one method per
    // gesture class.
    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // Extract image-derived data up front so the rest of the
        // method can take `&mut self` freely without fighting the
        // image borrow.
        let (img_w, img_h, img_tex_id) = match self.image.as_ref() {
            Some(img) => (img.gray.width, img.gray.height, img.texture.id()),
            None => return,
        };
        let img_size = egui::vec2(img_w as f32, img_h as f32);

        let avail = ui.available_size();
        let origin = ui.cursor().min;
        let fit = fit_rect(img_size, avail, origin);
        let allocated = egui::Rect::from_min_size(origin, avail);

        // Allocate the *whole* available area so wheel/drag/hover work
        // even when the cursor is outside the (possibly shrunk) image
        // rect. Clicks are gated to inside the image rect below.
        let response = ui.allocate_rect(allocated, egui::Sense::click_and_drag());
        let painter = ui.painter_at(allocated);

        // Gesture handling. Run before paint so zoom/pan from this
        // frame's input affects this frame's image position (avoids a
        // one-frame lag).
        self.view = compute_new_view(self.view, &response, ctx, fit, allocated);
        // Re-derive display rect after gestures so paint + click use
        // the same transform the gesture just applied.
        let display_rect = view_rect(self.view, fit, allocated);

        // Update hover readout. Off-image hover clears the readout
        // (the side panel decides whether to render it).
        self.hover_pixel = compute_hover_pixel(&response, display_rect, img_size, img_w, img_h);

        paint_image(&painter, img_tex_id, display_rect);
        if let Some(mask_tex) = &self.mask_texture {
            paint_image(&painter, mask_tex.id(), display_rect);
        }

        let clicked = capture_click(self, &response, display_rect, img_size);
        if let Some(polygon) = &self.refined_polygon {
            paint_polygon(&painter, &polygon.vertices, display_rect, img_size);
        }
        // Selected-vertex dot drawn LAST so it sits on top of the gold
        // contour and the prompt markers (the contour is more important
        // than the prompt markers for vertex editing).
        if let Some(idx) = self.selected_vertex_idx {
            if let Some(polygon) = &self.refined_polygon {
                if let Some(v) = polygon.vertices.get(idx) {
                    let screen = image_to_screen(*v, display_rect, img_size);
                    painter.circle_filled(screen, 5.0, egui::Color32::from_rgb(255, 80, 220));
                    painter.circle_stroke(
                        screen,
                        5.0,
                        egui::Stroke::new(1.5, egui::Color32::BLACK),
                    );
                }
            }
        }
        paint_prompts(&painter, &self.session.prompts, display_rect, img_size);

        if clicked {
            self.run_segment(ctx);
        }
    }
}

/// Fold mouse-wheel zoom, middle-button drag pan, and double-click
/// reset into a new [`ViewState`]. Pure on the inputs (`view`,
/// `response`, `ctx.input`), so it composes cleanly without sharing
/// `&mut self` with the rest of the frame.
fn compute_new_view(
    view: ViewState,
    response: &egui::Response,
    ctx: &egui::Context,
    fit: egui::Rect,
    allocated: egui::Rect,
) -> ViewState {
    // Double-click takes precedence — if the operator just
    // double-clicked to reset, we want the identity view regardless of
    // any drag or wheel input on the same frame.
    if response.double_clicked() {
        return ViewState::default();
    }

    let mut next = view;

    // Mouse wheel → zoom around the cursor. egui delivers wheel ticks
    // as `smooth_scroll_delta.y`; positive = scroll up = zoom in.
    // Trackpads deliver fractional values, which the `zoom_around`
    // clamp absorbs.
    if response.hovered() {
        let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.5 {
            if let Some(cursor) = response.hover_pos() {
                // Each ~50 px of scroll ≈ exp(1) ≈ 2.72× zoom step;
                // halve the rate to make trackpad scrolls feel
                // proportional to mouse-wheel ticks.
                let factor = (scroll / 120.0).exp();
                next = zoom_around(next, factor, cursor, fit, allocated);
            }
        }
    }

    // Middle-button drag → pan. egui doesn't expose a per-button drag
    // delta, so we sample `pointer.delta()` while the middle button is
    // held.
    let middle_drag =
        ctx.input(|i| i.pointer.middle_down() && (i.pointer.delta() != egui::Vec2::ZERO));
    if middle_drag && response.hovered() {
        let delta = ctx.input(|i| i.pointer.delta());
        next.pan += delta;
        // Re-clamp by reading back the clamp offset from view_rect.
        let unclamped = view_rect(next, fit, egui::Rect::EVERYTHING);
        let clamped = view_rect(next, fit, allocated);
        next.pan += clamped.min - unclamped.min;
    }

    next
}

/// Map the hover position from screen space into an integer image
/// pixel `(x, y)`. Returns `None` if the cursor is off-image or
/// outside the integer image bounds.
fn compute_hover_pixel(
    response: &egui::Response,
    display_rect: egui::Rect,
    img_size: egui::Vec2,
    img_w: u32,
    img_h: u32,
) -> Option<(u32, u32)> {
    if !response.hovered() {
        return None;
    }
    let pos = response.hover_pos()?;
    if !display_rect.contains(pos) {
        return None;
    }
    let pt = screen_to_image(pos, display_rect, img_size);
    if pt.x < 0.0 || pt.y < 0.0 {
        return None;
    }
    let x = pt.x.floor() as i64;
    let y = pt.y.floor() as i64;
    if x < 0 || y < 0 || x >= img_w as i64 || y >= img_h as i64 {
        return None;
    }
    Some((x as u32, y as u32))
}

fn paint_image(painter: &egui::Painter, id: egui::TextureId, rect: egui::Rect) {
    painter.image(
        id,
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}

/// Returns true if a new click was appended to `app.session`. Only
/// clicks inside the displayed image rect count; clicks in the
/// surrounding allocated area (e.g. the dark margin around a non-
/// fullscreen image) are ignored so panning and zooming don't
/// accidentally inject prompts.
fn capture_click(
    app: &mut SnapsegApp,
    response: &egui::Response,
    display_rect: egui::Rect,
    img_size: egui::Vec2,
) -> bool {
    if !(response.clicked() || response.secondary_clicked()) {
        return false;
    }
    // Drop the click if a drag is in progress so middle-button-drag
    // pan doesn't end with a stray prompt.
    if response.dragged() {
        return false;
    }
    let secondary = response.secondary_clicked();
    let Some(pos) = response.interact_pointer_pos() else {
        return false;
    };
    if !display_rect.contains(pos) {
        return false;
    }
    let img_pt = screen_to_image(pos, display_rect, img_size);
    // Discard clicks just outside the integer image bounds — the user
    // probably caught the canvas margin.
    if img_pt.x < 0.0 || img_pt.y < 0.0 || img_pt.x >= img_size.x || img_pt.y >= img_size.y {
        return false;
    }
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

fn paint_polygon(
    painter: &egui::Painter,
    vertices: &[Point2],
    display_rect: egui::Rect,
    img_size: egui::Vec2,
) {
    if vertices.len() < 2 {
        return;
    }
    let stroke = egui::Stroke::new(
        1.5,
        egui::Color32::from_rgba_premultiplied(255, 215, 0, 220),
    );
    for i in 0..vertices.len() {
        let a = image_to_screen(vertices[i], display_rect, img_size);
        let b = image_to_screen(vertices[(i + 1) % vertices.len()], display_rect, img_size);
        painter.line_segment([a, b], stroke);
    }
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
