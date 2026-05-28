//! Pure coordinate transforms between screen space (egui) and image
//! space (the source pixel frame).
//!
//! The transforms are an affine `screen ↔ image` map driven by a
//! display rect (the on-screen position of the image's bounding box).
//! Zoom and pan live in [`ViewState`]; [`view_rect`] applies them on top
//! of the canvas's natural fit-to-window rect to produce the actual
//! display rect that the painter and the click-capture code share.
//!
//! Pixel-readout, click capture, prompt-marker placement, polygon
//! drawing, and selected-vertex paint all go through the same two
//! functions ([`screen_to_image`] / [`image_to_screen`]) so a zoom or
//! pan change can't desync those code paths.

use eframe::egui;
use snapseg_core::{Point2, Polarity};

/// Zoom + pan applied on top of the natural "fit image into the
/// available rect" baseline. The identity transform is `zoom = 1.0`,
/// `pan = (0, 0)`, which renders exactly as the pre-zoom version of the
/// canvas did.
///
/// Coordinate convention:
/// - `zoom` is a multiplier on the fit rect's size. Values above 1.0
///   magnify; values below shrink. Clamped to `[ZOOM_MIN, ZOOM_MAX]`.
/// - `pan` is a screen-space translation in pixels, applied **after**
///   scaling. Pan is clamped on-the-fly inside [`view_rect`] so at
///   least `MIN_VISIBLE_PX` of the image stays inside the allocated rect.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewState {
    /// Multiplicative zoom factor relative to the fit-to-rect baseline.
    pub zoom: f32,
    /// Screen-space pan vector applied after zoom.
    pub pan: egui::Vec2,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }
}

/// Minimum zoom multiplier. Below this, the image becomes hard to click.
pub(crate) const ZOOM_MIN: f32 = 0.1;
/// Maximum zoom multiplier. Above this, scrolling becomes useless
/// because each click moves the same image-pixel by less than 1 px.
pub(crate) const ZOOM_MAX: f32 = 16.0;
/// Minimum number of pixels of the displayed image rect that must
/// remain inside the allocated canvas after panning. Prevents the
/// operator from accidentally panning the image entirely off-screen.
pub(crate) const MIN_VISIBLE_PX: f32 = 32.0;

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

/// Apply [`ViewState`] on top of the fit rect to produce the actual
/// display rect. The fit rect's centre stays fixed at `zoom = 1.0`,
/// `pan = 0`; zoom dilates around that centre, then `pan` translates.
///
/// Pan is clamped so at least [`MIN_VISIBLE_PX`] of the resulting rect
/// stays inside `allocated`, preventing the image from disappearing.
pub(crate) fn view_rect(view: ViewState, fit: egui::Rect, allocated: egui::Rect) -> egui::Rect {
    let centre = fit.center();
    let half = fit.size() * 0.5 * view.zoom;
    let unclamped = egui::Rect::from_min_max(centre - half, centre + half).translate(view.pan);
    clamp_to_visible(unclamped, allocated)
}

/// Clamp `rect` so at least [`MIN_VISIBLE_PX`] of it lies inside
/// `bounds` on each axis. The rect's size is preserved; only its
/// position changes.
fn clamp_to_visible(rect: egui::Rect, bounds: egui::Rect) -> egui::Rect {
    // The image must overlap `bounds` by at least MIN_VISIBLE_PX on
    // each axis; equivalently, the image rect cannot move so far that
    // its overlap with `bounds` drops below that floor on either side.
    let min_x = bounds.min.x - rect.width() + MIN_VISIBLE_PX;
    let max_x = bounds.max.x - MIN_VISIBLE_PX;
    let min_y = bounds.min.y - rect.height() + MIN_VISIBLE_PX;
    let max_y = bounds.max.y - MIN_VISIBLE_PX;
    // The clamp bounds can invert when the image is smaller than the
    // overlap floor (degenerate `bounds` or tiny rect); in that case we
    // skip the clamp on that axis to avoid producing a NaN-shaped rect.
    let x = if min_x <= max_x {
        rect.min.x.clamp(min_x, max_x)
    } else {
        rect.min.x
    };
    let y = if min_y <= max_y {
        rect.min.y.clamp(min_y, max_y)
    } else {
        rect.min.y
    };
    egui::Rect::from_min_size(egui::pos2(x, y), rect.size())
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

/// Update `view` so that zooming by `delta` (multiplicative) keeps the
/// image-space point currently under `cursor` pinned to the same screen
/// pixel. The returned `ViewState` has both `zoom` and `pan` updated;
/// the caller is responsible for re-rendering with it.
///
/// `delta < 1.0` zooms out; `delta > 1.0` zooms in. The result is
/// clamped to `[ZOOM_MIN, ZOOM_MAX]`.
pub(crate) fn zoom_around(
    view: ViewState,
    delta: f32,
    cursor: egui::Pos2,
    fit: egui::Rect,
    allocated: egui::Rect,
) -> ViewState {
    let new_zoom = (view.zoom * delta).clamp(ZOOM_MIN, ZOOM_MAX);
    let effective_delta = new_zoom / view.zoom;
    if (effective_delta - 1.0).abs() < f32::EPSILON {
        return view;
    }

    // Image-space point under the cursor at the current view — must
    // land at the same screen position after the zoom. To make this
    // identity hold, the pan adjusts by the cursor-relative offset
    // weighted by `(1 - effective_delta)`. Derivation:
    //
    //   pre  : screen = fit_centre + zoom * (img_norm - 0.5) * fit_size + pan
    //   post : screen = fit_centre + new_zoom * (img_norm - 0.5) * fit_size + pan'
    //
    // Setting `pre == post` at the cursor (`screen = cursor`) and
    // solving for `pan'`:
    //
    //   pan' = pan + (cursor - fit_centre - pan) * (1 - new_zoom/zoom)
    //
    // (the `pan` inside the parenthesis is the screen-space rect
    // centre's *current* offset, not the post-zoom one).
    let current_centre = fit.center() + view.pan;
    let new_pan = view.pan + (cursor - current_centre) * (1.0 - effective_delta);

    let candidate = ViewState {
        zoom: new_zoom,
        pan: new_pan,
    };
    // Re-derive the clamped rect to back-compute the post-clamp pan
    // (otherwise the cursor-anchor invariant can drift by the clamp).
    let unclamped = view_rect(
        ViewState {
            zoom: new_zoom,
            pan: new_pan,
        },
        fit,
        // Use an `infinite` clip bounds: we want the unclamped pan back.
        egui::Rect::EVERYTHING,
    );
    let clamped = clamp_to_visible(unclamped, allocated);
    let pan_correction = clamped.min - unclamped.min;
    ViewState {
        zoom: candidate.zoom,
        pan: candidate.pan + pan_correction,
    }
}

/// Flip the polarity. Used to make secondary mouse clicks contribute the
/// opposite of the selected click tool.
pub(crate) fn opposite(p: Polarity) -> Polarity {
    match p {
        Polarity::Positive => Polarity::Negative,
        Polarity::Negative => Polarity::Positive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity-view round-trip: a known canvas-space point in the
    /// middle of the display rect maps to the centre of the image.
    #[test]
    fn screen_to_image_identity_centre() {
        let display = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(640.0, 480.0));
        let img_size = egui::vec2(1280.0, 960.0);
        let centre = display.center();
        let pt = screen_to_image(centre, display, img_size);
        assert!((pt.x - 640.0).abs() < 1e-3);
        assert!((pt.y - 480.0).abs() < 1e-3);
    }

    /// Round-trip: image → screen → image returns the original point
    /// under non-identity zoom + pan. Establishes the post-transform
    /// invariant the click-capture code relies on.
    #[test]
    fn round_trip_under_zoom_and_pan() {
        let img_size = egui::vec2(800.0, 600.0);
        let allocated = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
        let fit = fit_rect(img_size, allocated.size(), allocated.min);
        let view = ViewState {
            zoom: 2.5,
            pan: egui::vec2(-40.0, 17.0),
        };
        let display = view_rect(view, fit, allocated);

        // Pick a representative grid of image points; round-trip each.
        for &(ix, iy) in &[
            (0.0_f32, 0.0_f32),
            (123.4, 45.6),
            (400.0, 300.0),
            (799.0, 599.0),
        ] {
            let img_pt = Point2::new(ix, iy);
            let screen = image_to_screen(img_pt, display, img_size);
            let back = screen_to_image(screen, display, img_size);
            assert!(
                (back.x - ix).abs() < 1e-3,
                "x round-trip drifted: {ix} -> {} -> {} (display={display:?})",
                screen.x,
                back.x
            );
            assert!(
                (back.y - iy).abs() < 1e-3,
                "y round-trip drifted: {iy} -> {} -> {} (display={display:?})",
                screen.y,
                back.y
            );
        }
    }

    /// Critical contract: a click at a known screen position under
    /// non-identity (zoom, pan) lands at the right image pixel. This is
    /// the exact path the canvas takes in `capture_click`.
    #[test]
    fn click_at_known_screen_lands_at_expected_image_pixel() {
        let img_size = egui::vec2(800.0, 600.0); // image is 800x600
        let allocated = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
        let fit = fit_rect(img_size, allocated.size(), allocated.min);
        // Pre-conditions: fit rect occupies most of the allocated area.
        assert!(fit.width() > 100.0);

        // At zoom = 2.0 the image rect is twice as wide as fit. With
        // pan = (0,0) it stays centred. The fit centre lies at the
        // allocated centre by construction, so screen-centre still maps
        // to image centre (400, 300) regardless of zoom.
        let view = ViewState {
            zoom: 2.0,
            pan: egui::Vec2::ZERO,
        };
        let display = view_rect(view, fit, allocated);
        let pt = screen_to_image(allocated.center(), display, img_size);
        assert!(
            (pt.x - 400.0).abs() < 1e-3,
            "centre x = {}, expected 400.0",
            pt.x
        );
        assert!(
            (pt.y - 300.0).abs() < 1e-3,
            "centre y = {}, expected 300.0",
            pt.y
        );

        // Pan moves the screen-centre image-space target by the inverse
        // of pan/display_width * img_width.
        let view = ViewState {
            zoom: 2.0,
            pan: egui::vec2(-50.0, 30.0),
        };
        let display = view_rect(view, fit, allocated);
        let pt = screen_to_image(allocated.center(), display, img_size);
        // Shifting the display rect by `pan` in screen space shifts the
        // image-space point under any fixed screen pixel by
        // `-pan / display_size * img_size`.
        let expected_x = 400.0 - (-50.0) / display.width() * img_size.x;
        let expected_y = 300.0 - (30.0) / display.height() * img_size.y;
        assert!(
            (pt.x - expected_x).abs() < 1e-2,
            "x = {pt:?}, expected {expected_x}"
        );
        assert!(
            (pt.y - expected_y).abs() < 1e-2,
            "y = {pt:?}, expected {expected_y}"
        );
    }

    /// Zooming around a cursor pins the image-space point under that
    /// cursor: the cursor maps to the same image pixel before and
    /// after the zoom step.
    #[test]
    fn zoom_around_pins_cursor_point() {
        let img_size = egui::vec2(800.0, 600.0);
        let allocated = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
        let fit = fit_rect(img_size, allocated.size(), allocated.min);
        let view = ViewState::default();
        let display = view_rect(view, fit, allocated);

        // Cursor at an off-centre but inside-image position.
        let cursor = egui::pos2(220.0, 410.0);
        let before = screen_to_image(cursor, display, img_size);

        let view2 = zoom_around(view, 1.4, cursor, fit, allocated);
        let display2 = view_rect(view2, fit, allocated);
        let after = screen_to_image(cursor, display2, img_size);

        assert!(
            (after.x - before.x).abs() < 0.1,
            "x drift: {} -> {}",
            before.x,
            after.x
        );
        assert!(
            (after.y - before.y).abs() < 0.1,
            "y drift: {} -> {}",
            before.y,
            after.y
        );
    }

    /// Zoom clamps at the limits and pan-clamping never returns NaN.
    #[test]
    fn zoom_clamps_and_pan_stays_finite() {
        let img_size = egui::vec2(800.0, 600.0);
        let allocated = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
        let fit = fit_rect(img_size, allocated.size(), allocated.min);
        let view = ViewState::default();
        let cursor = allocated.center();

        // Zoom in until we hit the cap.
        let mut v = view;
        for _ in 0..50 {
            v = zoom_around(v, 2.0, cursor, fit, allocated);
        }
        assert!(v.zoom <= ZOOM_MAX + 1e-6);

        // Zoom out the other way.
        v = ViewState::default();
        for _ in 0..50 {
            v = zoom_around(v, 0.5, cursor, fit, allocated);
        }
        assert!(v.zoom >= ZOOM_MIN - 1e-6);

        // Pan stays finite throughout.
        assert!(v.pan.x.is_finite() && v.pan.y.is_finite());
    }

    /// Pan clamping keeps at least MIN_VISIBLE_PX of the image inside
    /// the allocated rect — the image cannot be panned to oblivion.
    #[test]
    fn pan_clamps_to_min_visible() {
        let img_size = egui::vec2(800.0, 600.0);
        let allocated = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
        let fit = fit_rect(img_size, allocated.size(), allocated.min);
        // Massive pan that would push the image fully off-screen.
        let view = ViewState {
            zoom: 1.0,
            pan: egui::vec2(10_000.0, -10_000.0),
        };
        let display = view_rect(view, fit, allocated);
        let overlap = display.intersect(allocated);
        assert!(
            overlap.width() >= MIN_VISIBLE_PX - 1e-3,
            "overlap.width = {}, want ≥ {MIN_VISIBLE_PX}",
            overlap.width()
        );
        assert!(
            overlap.height() >= MIN_VISIBLE_PX - 1e-3,
            "overlap.height = {}, want ≥ {MIN_VISIBLE_PX}",
            overlap.height()
        );
    }
}
