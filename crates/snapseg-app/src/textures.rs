//! GPU texture building — egui-friendly `ColorImage` construction from
//! a grayscale image on disk or a boolean mask in memory. Both upload
//! the texture via `ctx.load_texture`; the returned handle is held by
//! the app for the rest of the frame's lifetime (or until replaced).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use eframe::egui;
use ndarray::Array2;
use snapseg_core::GrayImage;

use crate::app::LoadedImage;

/// Load a grayscale image from disk, build the corresponding egui
/// texture, and return both wrapped in a [`LoadedImage`].
///
/// # Errors
///
/// Fails if the file cannot be opened, decoded, or shape-converted
/// into an `ndarray::Array2<u8>`.
pub(crate) fn load_image(path: &PathBuf, ctx: &egui::Context) -> Result<LoadedImage> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();

    let gray = GrayImage::from_array(
        ndarray::Array2::from_shape_vec((h, w), raw.clone()).context("ndarray shape mismatch")?,
    );

    let texture = upload_gray_texture(ctx, path, w, h, &raw);

    Ok(LoadedImage {
        path: path.clone(),
        gray,
        texture,
    })
}

/// Build an egui `TextureHandle` from a binary mask, using a fixed
/// translucent blue for true pixels and transparent for false pixels.
/// Useful for the segmentation overlay drawn on top of the source
/// image in the canvas.
pub(crate) fn mask_to_texture(ctx: &egui::Context, mask: &Array2<bool>) -> egui::TextureHandle {
    let (h, w) = mask.dim();
    let mut pixels = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            if mask[(y, x)] {
                pixels.push(egui::Color32::from_rgba_premultiplied(40, 110, 200, 110));
            } else {
                pixels.push(egui::Color32::TRANSPARENT);
            }
        }
    }
    let img = egui::ColorImage {
        size: [w, h],
        pixels,
    };
    ctx.load_texture("mask", img, egui::TextureOptions::NEAREST)
}

fn upload_gray_texture(
    ctx: &egui::Context,
    path: &Path,
    w: usize,
    h: usize,
    raw: &[u8],
) -> egui::TextureHandle {
    let mut pixels = Vec::with_capacity(w * h);
    for &v in raw {
        pixels.push(egui::Color32::from_gray(v));
    }
    let color_image = egui::ColorImage {
        size: [w, h],
        pixels,
    };
    ctx.load_texture(
        path.file_name().and_then(|s| s.to_str()).unwrap_or("image"),
        color_image,
        egui::TextureOptions::LINEAR,
    )
}
