//! Classical subpixel edge refinement.
//!
//! Takes the raster mask from a deep segmenter plus the original grayscale
//! image, and returns a polygon whose vertices are snapped to the true
//! gradient ridge — typically to ~0.1 px on a step edge.
//!
//! Algorithm sketch (full implementation lands in Phase 2):
//!   1. Trace the mask boundary as an ordered polyline (marching squares).
//!   2. For each vertex, sample the grayscale image along the local normal
//!      in a band of `±band` px after Gaussian smoothing.
//!   3. Take the first derivative along the normal; parabolic-fit the three
//!      samples around its absolute maximum to get a subpixel offset.
//!   4. Replace the vertex with the refined position. Reject and keep the
//!      original if the peak prominence is below `min_strength`.

use snapseg_core::{GrayImage, Point2};
use ndarray::Array2;

/// Tunable knobs for the refinement pass. Defaults aim at ~1 MP grayscale
/// industrial photos with moderately sharp edges.
#[derive(Debug, Clone, Copy)]
pub struct RefineParams {
    /// Half-width of the search band along the normal, in pixels.
    pub band: u32,
    /// Sigma of the pre-smoothing Gaussian along the normal, in pixels.
    pub sigma: f32,
    /// Reject refinements whose peak gradient magnitude is below this.
    pub min_strength: f32,
    /// Reject refinements that move the vertex further than this along the
    /// normal — keeps weak/ambiguous samples close to the mask boundary.
    pub max_displacement: f32,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            band: 5,
            sigma: 1.0,
            min_strength: 4.0,
            max_displacement: 4.0,
        }
    }
}

/// A closed polygon refined against the image.
#[derive(Debug, Clone)]
pub struct RefinedPolygon {
    pub vertices: Vec<Point2>,
    /// Per-vertex peak gradient magnitude. Useful for QA.
    pub confidence: Vec<f32>,
}

/// Refine the boundary of a binary mask against its source image.
///
/// Phase-2 placeholder: returns the integer boundary as `f32` points with
/// uniform confidence so downstream code can be wired now. The real
/// parabolic-fit refinement replaces the body without changing the API.
pub fn refine_polygon(
    mask: &Array2<bool>,
    _image: &GrayImage,
    _params: RefineParams,
) -> RefinedPolygon {
    let boundary = naive_boundary(mask);
    let confidence = vec![0.0_f32; boundary.len()];
    RefinedPolygon {
        vertices: boundary,
        confidence,
    }
}

/// Crude boundary extraction: pick mask pixels that have at least one
/// background neighbor in the 4-connected sense. Order is row-major scan,
/// *not* a true contour traversal — sufficient as a placeholder so the
/// downstream pipeline can be tested before marching squares is plumbed in.
fn naive_boundary(mask: &Array2<bool>) -> Vec<Point2> {
    let (h, w) = mask.dim();
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !mask[(y, x)] {
                continue;
            }
            let mut is_boundary = false;
            if x == 0 || x + 1 == w || y == 0 || y + 1 == h {
                is_boundary = true;
            } else if !mask[(y, x - 1)]
                || !mask[(y, x + 1)]
                || !mask[(y - 1, x)]
                || !mask[(y + 1, x)]
            {
                is_boundary = true;
            }
            if is_boundary {
                out.push(Point2::new(x as f32 + 0.5, y as f32 + 0.5));
            }
        }
    }
    out
}
