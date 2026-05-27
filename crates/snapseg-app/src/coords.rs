//! Pure coordinate transforms between screen space (egui) and image
//! space (the source pixel frame). No app state; no side effects.

use eframe::egui;
use snapseg_core::{Point2, Polarity};

/// Largest rect with the same aspect ratio as `inner_size` that fits
/// inside `available`, centered within it and anchored at `origin`.
pub(crate) fn fit_rect(
    inner_size: egui::Vec2,
    available: egui::Vec2,
    origin: egui::Pos2,
) -> egui::Rect {
    let scale = (available.x / inner_size.x).min(available.y / inner_size.y);
    let scaled = inner_size * scale;
    let offset = (available - scaled) * 0.5;
    egui::Rect::from_min_size(origin + offset, scaled)
}

/// Map a screen-space point inside the displayed image rect into image
/// pixel coordinates. The mapping is affine — no clamping is performed;
/// callers can decide whether off-image points are interesting.
pub(crate) fn screen_to_image(
    p: egui::Pos2,
    display_rect: egui::Rect,
    img_size: egui::Vec2,
) -> Point2 {
    let u = (p.x - display_rect.min.x) / display_rect.width();
    let v = (p.y - display_rect.min.y) / display_rect.height();
    Point2::new(u * img_size.x, v * img_size.y)
}

/// Inverse of [`screen_to_image`].
pub(crate) fn image_to_screen(
    pt: Point2,
    display_rect: egui::Rect,
    img_size: egui::Vec2,
) -> egui::Pos2 {
    let u = pt.x / img_size.x;
    let v = pt.y / img_size.y;
    egui::pos2(
        display_rect.min.x + u * display_rect.width(),
        display_rect.min.y + v * display_rect.height(),
    )
}

/// Flip the polarity. Used to make secondary mouse clicks contribute the
/// opposite of the selected click tool.
pub(crate) fn opposite(p: Polarity) -> Polarity {
    match p {
        Polarity::Positive => Polarity::Negative,
        Polarity::Negative => Polarity::Positive,
    }
}
