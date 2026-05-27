//! Input-tensor preprocessing shared by every adapter:
//!   - grayscale → 3-channel replication for RGB-only pretrained networks,
//!   - SAM-style "resize longest side + pad" with SAM's pixel-space stats,
//!   - bilinear resize building block,
//!   - ImageNet stats for click-based segmenters that follow that recipe.
//!
//! Adapters compose these into their own preprocessing pipelines.

use ndarray::{Array3, Array4, Axis, s};
use snapseg_core::GrayImage;

/// Standard ImageNet normalization constants (in [0, 1] pixel range).
/// Most click-based segmenters were trained on COCO/LVIS/Pascal with these.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// SAM / MobileSAM pixel-space stats — applied to pixels in the [0, 255]
/// range, *not* [0, 1]. See SamPredictor.set_image in the official repo.
pub const SAM_PIXEL_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
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

pub fn normalize_imagenet(chw: &mut Array3<f32>) {
    normalize_chw(chw, IMAGENET_MEAN, IMAGENET_STD);
}

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

/// SAM preprocessing: resize the longest side to `target`, zero-pad the
/// shorter side to (target, target), normalize with SAM's pixel-space
/// mean/std. Returns the `[1, 3, target, target]` tensor and a
/// [`SamResize`] describing the transform so prompt points can be mapped
/// back into encoder space.
pub fn sam_preprocess_gray(image: &GrayImage, target: usize) -> (Array4<f32>, SamResize) {
    let h = image.height as usize;
    let w = image.width as usize;
    let scale = target as f32 / h.max(w) as f32;
    let new_h = ((h as f32) * scale).round() as usize;
    let new_w = ((w as f32) * scale).round() as usize;

    // 1) gray → 3xHxW in [0, 255]
    let rgb = gray_to_rgb_chw_byte(image);

    // 2) bilinear resize to (3, new_h, new_w)
    let resized = resize_bilinear_chw(&rgb, new_h, new_w);

    // 3) paste into a (3, target, target) zero-padded canvas
    let mut canvas = Array3::<f32>::zeros((3, target, target));
    canvas
        .slice_mut(s![.., ..new_h, ..new_w])
        .assign(&resized);

    // 4) SAM normalize in-place (still [0, 255] before)
    normalize_sam(&mut canvas);

    let info = SamResize {
        orig_h: h as u32,
        orig_w: w as u32,
        new_h: new_h as u32,
        new_w: new_w as u32,
        scale,
        target: target as u32,
    };
    (canvas.insert_axis(Axis(0)), info)
}

/// Geometric info from a SAM-style preprocess: lets adapters translate
/// prompt points from image space into the model's resized+padded space.
#[derive(Debug, Clone, Copy)]
pub struct SamResize {
    pub orig_h: u32,
    pub orig_w: u32,
    pub new_h: u32,
    pub new_w: u32,
    pub scale: f32,
    pub target: u32,
}

impl SamResize {
    /// Map a point from original image coordinates to encoder
    /// (resized+padded) coordinates.
    pub fn point_to_encoder(&self, x: f32, y: f32) -> (f32, f32) {
        (x * self.scale, y * self.scale)
    }
}
