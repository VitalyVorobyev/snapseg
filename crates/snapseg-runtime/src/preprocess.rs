//! Input-tensor preprocessing shared by every adapter:
//!   - grayscale → 3-channel replication for RGB-only pretrained networks,
//!   - SAM-style "resize longest side + pad" with SAM's pixel-space stats,
//!   - bilinear resize building block,
//!   - ImageNet stats for click-based segmenters that follow that recipe.
//!
//! Adapters compose these into their own preprocessing pipelines.

use ndarray::{Array3, Array4, Axis, s};
use snapseg_core::{GrayImage, Polarity, Prompt};

/// Per-channel ImageNet mean (R, G, B) on pixels in the `[0, 1]` range.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Per-channel ImageNet standard deviation (R, G, B) on pixels in the
/// `[0, 1]` range. Used together with [`IMAGENET_MEAN`].
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Per-channel SAM / MobileSAM mean (R, G, B) on pixels in the
/// `[0, 255]` range. SAM does *not* rescale to `[0, 1]` before
/// normalization; pass the byte-valued tensor straight through.
pub const SAM_PIXEL_MEAN: [f32; 3] = [123.675, 116.28, 103.53];

/// Per-channel SAM / MobileSAM standard deviation (R, G, B) on pixels in
/// the `[0, 255]` range. Used together with [`SAM_PIXEL_MEAN`].
pub const SAM_PIXEL_STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Replicate a `[H, W]` u8 grayscale image into a `[3, H, W]` float tensor
/// in `[0, 1]`. Adapters that need [0, 255] just scale back up afterward.
pub fn gray_to_rgb_chw(image: &GrayImage) -> Array3<f32> {
    let h = image.height as usize;
    let w = image.width as usize;
    let mut out = Array3::<f32>::zeros((3, h, w));
    for ((y, x), &v) in image.data.indexed_iter() {
        let f = v as f32 / 255.0;
        out[(0, y, x)] = f;
        out[(1, y, x)] = f;
        out[(2, y, x)] = f;
    }
    out
}

/// Replicate a `[H, W]` u8 grayscale image into a `[3, H, W]` float tensor
/// in `[0, 255]` (no rescaling). SAM family expects this scale.
pub fn gray_to_rgb_chw_byte(image: &GrayImage) -> Array3<f32> {
    let h = image.height as usize;
    let w = image.width as usize;
    let mut out = Array3::<f32>::zeros((3, h, w));
    for ((y, x), &v) in image.data.indexed_iter() {
        let f = v as f32;
        out[(0, y, x)] = f;
        out[(1, y, x)] = f;
        out[(2, y, x)] = f;
    }
    out
}

/// In-place per-channel normalization on a `[3, H, W]` tensor.
pub fn normalize_chw(chw: &mut Array3<f32>, mean: [f32; 3], std: [f32; 3]) {
    for c in 0..3 {
        let m = mean[c];
        let s = std[c];
        let mut channel = chw.slice_mut(s![c, .., ..]);
        channel.mapv_inplace(|v| (v - m) / s);
    }
}

/// Apply [`IMAGENET_MEAN`] / [`IMAGENET_STD`] in place. Convenience
/// wrapper over [`normalize_chw`] for the most common per-channel
/// recipe used by click-based segmenters.
pub fn normalize_imagenet(chw: &mut Array3<f32>) {
    normalize_chw(chw, IMAGENET_MEAN, IMAGENET_STD);
}

/// Apply [`SAM_PIXEL_MEAN`] / [`SAM_PIXEL_STD`] in place. Convenience
/// wrapper for SAM-family adapters.
pub fn normalize_sam(chw: &mut Array3<f32>) {
    normalize_chw(chw, SAM_PIXEL_MEAN, SAM_PIXEL_STD);
}

/// Wrap a `[3, H, W]` tensor into a `[1, 3, H, W]` batch.
pub fn add_batch_dim(chw: Array3<f32>) -> Array4<f32> {
    chw.insert_axis(Axis(0))
}

/// Nearest-neighbor resize on a `[C, H, W]` tensor. Cheap and exact for
/// integer scale factors; used for click-map rasterization where
/// interpolation would smear delta peaks.
pub fn resize_nearest_chw(src: &Array3<f32>, target_h: usize, target_w: usize) -> Array3<f32> {
    let (c, sh, sw) = src.dim();
    let mut out = Array3::<f32>::zeros((c, target_h, target_w));
    for cc in 0..c {
        for y in 0..target_h {
            let sy = (y * sh) / target_h;
            for x in 0..target_w {
                let sx = (x * sw) / target_w;
                out[(cc, y, x)] = src[(cc, sy, sx)];
            }
        }
    }
    out
}

/// Bilinear resize on a `[C, H, W]` tensor.
pub fn resize_bilinear_chw(src: &Array3<f32>, target_h: usize, target_w: usize) -> Array3<f32> {
    let (c, sh, sw) = src.dim();
    let mut out = Array3::<f32>::zeros((c, target_h, target_w));
    if sh == 0 || sw == 0 || target_h == 0 || target_w == 0 {
        return out;
    }
    let scale_y = sh as f32 / target_h as f32;
    let scale_x = sw as f32 / target_w as f32;
    for y in 0..target_h {
        let fy = (y as f32 + 0.5) * scale_y - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - fy.floor();
        for x in 0..target_w {
            let fx = (x as f32 + 0.5) * scale_x - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - fx.floor();
            for cc in 0..c {
                let v00 = src[(cc, y0, x0)];
                let v01 = src[(cc, y0, x1)];
                let v10 = src[(cc, y1, x0)];
                let v11 = src[(cc, y1, x1)];
                let top = v00 * (1.0 - wx) + v01 * wx;
                let bot = v10 * (1.0 - wx) + v11 * wx;
                out[(cc, y, x)] = top * (1.0 - wy) + bot * wy;
            }
        }
    }
    out
}

/// SAM preprocessing: resize so the image fits within `(target_h, target_w)`
/// preserving aspect ratio, zero-pad the shorter axis to fill the canvas,
/// normalize with SAM's pixel-space mean/std. Returns the
/// `[1, 3, target_h, target_w]` tensor and a [`LetterboxGeometry`] describing the
/// transform so prompt points can be mapped back into encoder space.
///
/// For canonical SAM exports use `target_h == target_w == 1024`. Some
/// community MobileSAM exports declare a non-square canvas (e.g. the
/// 2023-06-29 export uses `1024 × 682`) — pass those dimensions instead.
pub fn sam_preprocess_gray(
    image: &GrayImage,
    target_h: usize,
    target_w: usize,
) -> (Array4<f32>, LetterboxGeometry) {
    let h = image.height as usize;
    let w = image.width as usize;
    // Aspect-preserving scale that lets the image fit inside both
    // dimensions of the canvas.
    let scale_h = target_h as f32 / h as f32;
    let scale_w = target_w as f32 / w as f32;
    let scale = scale_h.min(scale_w);
    let new_h = ((h as f32) * scale).round() as usize;
    let new_w = ((w as f32) * scale).round() as usize;

    // 1) gray → 3xHxW in [0, 255]
    let rgb = gray_to_rgb_chw_byte(image);

    // 2) bilinear resize to (3, new_h, new_w)
    let mut resized = resize_bilinear_chw(&rgb, new_h, new_w);

    // 3) SAM normalize the resized image BEFORE padding. The official
    //    SAM preprocess (segment_anything/modeling/sam.py::preprocess)
    //    normalises first then pads with literal zeros — so padded
    //    regions are 0.0 in the encoder's input space. If we pad first
    //    and normalise after, the pad becomes (-mean/std) ≈ -2.0 per
    //    channel, which is out-of-distribution for the encoder and the
    //    MobileSAM 2023-06-29 export shows spatially unstable masks as
    //    a result (clicks far from the centre drift several patch rows).
    normalize_sam(&mut resized);

    // 4) paste the normalised image into a zero-filled canvas.
    let mut canvas = Array3::<f32>::zeros((3, target_h, target_w));
    canvas.slice_mut(s![.., ..new_h, ..new_w]).assign(&resized);

    let info = LetterboxGeometry {
        orig_h: h as u32,
        orig_w: w as u32,
        new_h: new_h as u32,
        new_w: new_w as u32,
        scale,
        target_h: target_h as u32,
        target_w: target_w as u32,
    };
    (canvas.insert_axis(Axis(0)), info)
}

/// Geometric info from an aspect-preserving letterbox-with-pad
/// preprocess. Lets adapters translate prompt points from image space
/// into the model's resized+padded space.
///
/// Shared by every click-based family — MobileSAM (`sam_preprocess_gray`),
/// RITM and FocalClick (`imagenet_letterbox_gray`) — because they all
/// use the same uniform-scale letterbox transform; the per-family
/// difference is just the pixel-normalisation stats.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxGeometry {
    pub orig_h: u32,
    pub orig_w: u32,
    pub new_h: u32,
    pub new_w: u32,
    pub scale: f32,
    pub target_h: u32,
    pub target_w: u32,
}

impl LetterboxGeometry {
    /// Map a point from original image coordinates to encoder
    /// (resized+padded) coordinates.
    pub fn point_to_encoder(&self, x: f32, y: f32) -> (f32, f32) {
        (x * self.scale, y * self.scale)
    }
}

/// ImageNet-normalised letterbox preprocess for RITM / FocalClick.
/// Resize so the image fits within `(target_h, target_w)` preserving
/// aspect ratio, normalise with [`IMAGENET_MEAN`] / [`IMAGENET_STD`]
/// in `[0, 1]` pixel space, then zero-pad the bottom-right of the
/// canvas. Returns the `[1, 3, target_h, target_w]` tensor and a
/// [`LetterboxGeometry`] describing the transform so prompt points can be
/// mapped back into the resized canvas (use [`LetterboxGeometry::scale`]
/// when calling [`rasterize_click_map`]).
///
/// Normalisation happens *before* padding so the padded region stays
/// at zero in the model's input space rather than `-mean/std`
/// (≈ -2.0 per channel), which would be far out-of-distribution and
/// destabilise predictions near the canvas edges. This matches the
/// official RITM `isegm/inference/predictors/base.py` preprocessing.
pub fn imagenet_letterbox_gray(
    image: &GrayImage,
    target_h: usize,
    target_w: usize,
) -> (Array4<f32>, LetterboxGeometry) {
    let h = image.height as usize;
    let w = image.width as usize;
    let scale_h = target_h as f32 / h as f32;
    let scale_w = target_w as f32 / w as f32;
    let scale = scale_h.min(scale_w);
    let new_h = ((h as f32) * scale).round() as usize;
    let new_w = ((w as f32) * scale).round() as usize;

    // gray → 3xHxW in [0, 1] (RITM expects ImageNet stats applied to
    // [0, 1] pixels, not [0, 255]).
    let rgb = gray_to_rgb_chw(image);
    let mut resized = resize_bilinear_chw(&rgb, new_h, new_w);
    normalize_imagenet(&mut resized);

    let mut canvas = Array3::<f32>::zeros((3, target_h, target_w));
    canvas.slice_mut(s![.., ..new_h, ..new_w]).assign(&resized);

    let info = LetterboxGeometry {
        orig_h: h as u32,
        orig_w: w as u32,
        new_h: new_h as u32,
        new_w: new_w as u32,
        scale,
        target_h: target_h as u32,
        target_w: target_w as u32,
    };
    (canvas.insert_axis(Axis(0)), info)
}

/// Rasterise a [`PromptSession`](snapseg_core::PromptSession)'s prompts
/// into a 2-channel Gaussian-disk click map.
///
/// Output layout: NCHW `[1, 2, target_h, target_w]`. Channel 0 carries
/// positive (foreground) clicks; channel 1 carries negative
/// (background) clicks. Each click contributes a Gaussian disk
/// `exp(-(dx² + dy²) / (2σ²))` centred at the click's resized-canvas
/// coordinates, peaking at `1.0`. Overlapping clicks of the same
/// polarity combine by max so close clicks don't sum past 1.0.
///
/// `scale` is the same uniform aspect-preserving factor used to map
/// image-space coordinates into the model's resized canvas (e.g.
/// [`LetterboxGeometry::scale`] — RITM / FocalClick use the same letterbox
/// convention). Click points outside `[0, target_h) × [0, target_w)`
/// after scaling are clamped through the per-pixel range loop, so
/// only the intersecting half of the disk shows up on the canvas.
///
/// `sigma` is the Gaussian standard deviation in **resized-canvas
/// pixels**. RITM's reference implementation uses σ = 5 px; the
/// rasterising window covers `±3σ` (≈ 99.7 % of the Gaussian mass).
///
/// Both [`Prompt::Click`] and [`Prompt::Scribble`] contribute disks;
/// scribbles are treated as a sequence of clicks with the scribble's
/// polarity. [`Prompt::Box`] prompts are silently ignored — RITM and
/// FocalClick don't consume them; the adapter's
/// [`InteractiveSegmenter::capabilities`](snapseg_core::InteractiveSegmenter::capabilities)
/// should already advertise `bbox = false` for those families.
///
/// # Panics
///
/// Does not panic. `sigma <= 0` is treated as a no-op (no disk
/// painted). Zero-sized targets return a correctly-shaped zero
/// tensor.
pub fn rasterize_click_map(
    prompts: &[Prompt],
    target_h: usize,
    target_w: usize,
    scale: f32,
    sigma: f32,
) -> Array4<f32> {
    let mut canvas = Array4::<f32>::zeros((1, 2, target_h, target_w));
    if sigma <= 0.0 || target_h == 0 || target_w == 0 {
        return canvas;
    }
    let kernel = DiskKernel {
        two_sigma_sq: 2.0 * sigma * sigma,
        // ±3σ covers ~99.7 % of the Gaussian; outside that disks
        // contribute < 0.012 to the peak, well below the model's
        // signal floor.
        radius: (3.0 * sigma).ceil() as i32,
    };

    for prompt in prompts {
        match prompt {
            Prompt::Click { point, polarity } => {
                paint_disk(
                    &mut canvas,
                    channel_for_polarity(*polarity),
                    point.x * scale,
                    point.y * scale,
                    kernel,
                );
            }
            Prompt::Scribble { points, polarity } => {
                let ch = channel_for_polarity(*polarity);
                for p in points {
                    paint_disk(&mut canvas, ch, p.x * scale, p.y * scale, kernel);
                }
            }
            Prompt::Box(_) => {
                // Click-map families (RITM, FocalClick) don't consume
                // box prompts; adapter capabilities advertise that.
            }
        }
    }
    canvas
}

/// Parameters of a single Gaussian disk: precomputed `2σ²` for the
/// inner exponent and the integer half-window in pixels.
#[derive(Debug, Clone, Copy)]
struct DiskKernel {
    two_sigma_sq: f32,
    radius: i32,
}

/// Channel index in the `[1, 2, H, W]` click-map for a given
/// polarity. Convention: channel 0 = positive (foreground), channel
/// 1 = negative (background). Matches the RITM upstream and the
/// FocalClick ONNX export it inherits from.
fn channel_for_polarity(p: Polarity) -> usize {
    match p {
        Polarity::Positive => 0,
        Polarity::Negative => 1,
    }
}

/// Paint a single Gaussian disk onto `canvas[0, ch, .., ..]`, combining
/// with whatever is already there by `max` (so overlapping same-polarity
/// clicks don't sum above 1.0). Canvas dimensions are read from the
/// tensor itself; out-of-bounds iteration is clipped, so partial disks
/// at the edge of the canvas render the visible half.
fn paint_disk(canvas: &mut Array4<f32>, ch: usize, cx: f32, cy: f32, kernel: DiskKernel) {
    let (_, _, target_h, target_w) = canvas.dim();
    let cx_i = cx.round() as i32;
    let cy_i = cy.round() as i32;
    let h_i = target_h as i32;
    let w_i = target_w as i32;
    // Clip the per-pixel iteration window so we never read or write
    // out of bounds; partial disks at the edge of the canvas are fine.
    let y0 = (cy_i - kernel.radius).max(0);
    let y1 = (cy_i + kernel.radius).min(h_i - 1);
    let x0 = (cx_i - kernel.radius).max(0);
    let x1 = (cx_i + kernel.radius).min(w_i - 1);
    if y0 > y1 || x0 > x1 {
        return;
    }
    for y in y0..=y1 {
        let dy = y as f32 - cy;
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            let r2 = dx * dx + dy * dy;
            let v = (-r2 / kernel.two_sigma_sq).exp();
            let slot = &mut canvas[(0, ch, y as usize, x as usize)];
            if v > *slot {
                *slot = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snapseg_core::Point2;

    /// A single positive click at an integer location lands a disk
    /// with peak value 1.0 exactly at that pixel, and the negative
    /// channel stays untouched.
    #[test]
    fn rasterize_single_positive_click_peaks_at_centre() {
        let prompts = vec![Prompt::Click {
            point: Point2::new(10.0, 12.0),
            polarity: Polarity::Positive,
        }];
        let map = rasterize_click_map(&prompts, 32, 32, 1.0, 5.0);
        assert_eq!(map.dim(), (1, 2, 32, 32));
        let peak = map[(0, 0, 12, 10)];
        assert!(
            (peak - 1.0).abs() < 1e-5,
            "peak at centre = {peak}, expected ~1.0"
        );
        // Negative channel must be all zeros.
        let neg_sum: f32 = map.slice(s![0, 1, .., ..]).iter().copied().sum();
        assert_eq!(neg_sum, 0.0);
        // Falloff: with radius = ceil(3σ) = 15 px the iteration window
        // around (10, 12) is roughly [y -3..27, x -5..25]; a far-corner
        // pixel (31, 31) is well outside the painted window and stays
        // at the canvas initial value.
        assert_eq!(map[(0, 0, 31, 31)], 0.0);
        // Symmetry check: a pixel one off-axis from the centre carries
        // the expected exp(-1/(2σ²)) value, which for σ=5 is ≈ 0.9802.
        let neighbour = map[(0, 0, 12, 11)];
        let expected = (-1.0_f32 / (2.0 * 5.0 * 5.0)).exp();
        assert!(
            (neighbour - expected).abs() < 1e-5,
            "neighbour = {neighbour}, expected {expected}"
        );
    }

    /// A scaled click: image-space (20, 24) at scale 0.5 should land
    /// at canvas (10, 12).
    #[test]
    fn rasterize_applies_scale_to_click_centre() {
        let prompts = vec![Prompt::Click {
            point: Point2::new(20.0, 24.0),
            polarity: Polarity::Positive,
        }];
        let map = rasterize_click_map(&prompts, 32, 32, 0.5, 5.0);
        let peak = map[(0, 0, 12, 10)];
        assert!((peak - 1.0).abs() < 1e-5);
    }

    /// Negative click goes to channel 1; positive channel stays zero.
    #[test]
    fn rasterize_negative_goes_to_channel_one() {
        let prompts = vec![Prompt::Click {
            point: Point2::new(8.0, 8.0),
            polarity: Polarity::Negative,
        }];
        let map = rasterize_click_map(&prompts, 16, 16, 1.0, 5.0);
        assert!((map[(0, 1, 8, 8)] - 1.0).abs() < 1e-5);
        let pos_sum: f32 = map.slice(s![0, 0, .., ..]).iter().copied().sum();
        assert_eq!(pos_sum, 0.0);
    }

    /// Box prompts are silently ignored by the rasterizer (RITM /
    /// FocalClick don't consume boxes).
    #[test]
    fn rasterize_ignores_box_prompts() {
        use snapseg_core::BBox;
        let prompts = vec![Prompt::Box(BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 10.0,
            y1: 10.0,
        })];
        let map = rasterize_click_map(&prompts, 16, 16, 1.0, 5.0);
        let total: f32 = map.iter().copied().sum();
        assert_eq!(total, 0.0);
    }

    /// Scribble polylines paint a disk at every stroke point with the
    /// scribble's polarity.
    #[test]
    fn rasterize_paints_scribble_points() {
        let prompts = vec![Prompt::Scribble {
            points: vec![Point2::new(5.0, 5.0), Point2::new(10.0, 10.0)],
            polarity: Polarity::Positive,
        }];
        let map = rasterize_click_map(&prompts, 16, 16, 1.0, 5.0);
        assert!((map[(0, 0, 5, 5)] - 1.0).abs() < 1e-5);
        assert!((map[(0, 0, 10, 10)] - 1.0).abs() < 1e-5);
    }
}
