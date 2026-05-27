//! Input-tensor preprocessing shared by every adapter:
//!   - grayscale-to-3-channel replication for RGB-only pretrained networks,
//!   - ImageNet mean/std normalization,
//!   - bilinear resize to a target shape.
//!
//! Adapters compose these into their own preprocessing pipelines.

use snapseg_core::GrayImage;
use ndarray::{Array3, Array4, s};

/// Standard ImageNet normalization constants. Most click-based segmenters
/// were trained on COCO/LVIS/Pascal with these stats.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Replicate a `[H, W]` u8 grayscale image into a `[3, H, W]` float tensor
/// in `[0, 1]`. Cheap channel-broadcast — we'll normalize separately.
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

/// In-place ImageNet normalization on a `[3, H, W]` tensor: subtract mean
/// per channel, divide by std.
pub fn normalize_imagenet(chw: &mut Array3<f32>) {
    for c in 0..3 {
        let mean = IMAGENET_MEAN[c];
        let std = IMAGENET_STD[c];
        let mut channel = chw.slice_mut(s![c, .., ..]);
        channel.mapv_inplace(|v| (v - mean) / std);
    }
}

/// Wrap a `[3, H, W]` tensor into a `[1, 3, H, W]` batch.
pub fn add_batch_dim(chw: Array3<f32>) -> Array4<f32> {
    chw.insert_axis(ndarray::Axis(0))
}

/// Nearest-neighbor placeholder for resize. Real bilinear resize comes in
/// the next iteration; nearest is fine for the skeleton API and for
/// click-map rasterization where bilinear would actually be wrong.
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
