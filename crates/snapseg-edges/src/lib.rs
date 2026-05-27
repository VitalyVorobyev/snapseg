//! Classical subpixel edge refinement.
//!
//! Takes the raster mask from a deep segmenter plus the original grayscale
//! image, and returns a polygon whose vertices are snapped to the true
//! gradient ridge — typically to ~0.1 px on a clean step edge.
//!
//! ## Pipeline
//!
//! 1. **Contour extraction** (marching squares): trace the
//!    foreground/background boundary as one or more ordered closed
//!    polylines at half-pixel positions.
//! 2. **Per-vertex refinement**: along each vertex's local normal, sample
//!    the image, Gaussian-smooth, take the first derivative,
//!    parabolic-fit the absolute peak.
//! 3. **Concatenation**: this MVP returns the **longest** polyline only;
//!    masks with holes or multiple foreground components have their
//!    secondary polylines discarded.
//!
//! ## Sign convention
//!
//! Tangent at vertex `v_i` is `(v_{i+1} - v_{i-1}) / 2`. Normal is
//! `(tangent.y, -tangent.x)` then unit-normalised. The refinement is
//! invariant to traversal direction, so the convention only matters when
//! interpreting raw confidence diagnostics.
//!
//! ## Failure modes
//!
//! `refine_polygon` is infallible: a degenerate input (empty mask, no
//! detectable contour, all vertices reject) returns an empty
//! [`RefinedPolygon`]. Per-vertex failures (out-of-band sampling, peak
//! below `min_strength`, offset above `max_displacement`) fall back to the
//! unrefined marching-squares position with `confidence = 0`.
//!
//! ## follow-up
//!
//! A future API should return `Vec<RefinedPolygon>` to preserve
//! multi-component / hole topology — track on the M3 milestone.

use ndarray::Array2;
use snapseg_core::{GrayImage, Point2};

mod contour;
mod refine;
mod sample;

/// Tunable knobs for the refinement pass. Defaults aim at ~1 MP grayscale
/// industrial photos with moderately sharp edges.
#[derive(Debug, Clone, Copy)]
pub struct RefineParams {
    /// Half-width of the search band along the normal, in pixels. Samples
    /// are taken at 0.5 px spacing across `[-band, +band]`.
    pub band: u32,
    /// Sigma of the pre-smoothing Gaussian along the normal, in pixels.
    pub sigma: f32,
    /// Reject refinements whose peak gradient magnitude (intensity per
    /// pixel) is below this. A pure 0→255 step gives peaks in the hundreds;
    /// `4.0` is a permissive industrial-photo floor.
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
///
/// `vertices.first() == vertices.last()` (within float tolerance) when the
/// polygon is non-empty: callers that draw the polygon do not need to
/// re-emit the closure segment.
#[derive(Debug, Clone, Default)]
pub struct RefinedPolygon {
    /// Polygon vertices in traversal order; the last vertex repeats the
    /// first to mark closure.
    pub vertices: Vec<Point2>,
    /// Per-vertex peak gradient magnitude (intensity per pixel). `0.0`
    /// flags a vertex that was rejected and left at its marching-squares
    /// position. The trailing closure vertex mirrors the first.
    pub confidence: Vec<f32>,
}

/// Refine the boundary of a binary mask against its source image.
///
/// # Inputs
///
/// - `mask`: row-major `Array2<bool>` of shape `[H, W]`. `true` is
///   foreground.
/// - `image`: grayscale source. Must have the same `[H, W]` as `mask`;
///   mismatched shapes are tolerated but produce no useful refinement
///   (vertices that sample outside the image fall back to their
///   marching-squares position).
/// - `params`: see [`RefineParams`].
///
/// # Returns
///
/// A single [`RefinedPolygon`] tracing the **longest** contour found in
/// the mask. An empty `RefinedPolygon` is returned when:
///   - the mask is empty / all-foreground (no contour),
///   - marching squares produces no segments,
///   - the image dimensions are smaller than `2x2`.
///
/// See the crate-level docs for the multi-component follow-up.
pub fn refine_polygon(
    mask: &Array2<bool>,
    image: &GrayImage,
    params: RefineParams,
) -> RefinedPolygon {
    let polylines = contour::marching_squares(mask);
    let longest = match polylines.into_iter().max_by_key(|p| p.len()) {
        Some(p) => p,
        None => return RefinedPolygon::default(),
    };
    let refined = refine::refine_polyline(&longest, image, params, true);
    let mut vertices = Vec::with_capacity(refined.len());
    let mut confidence = Vec::with_capacity(refined.len());
    for r in refined {
        vertices.push(r.point);
        confidence.push(r.confidence);
    }
    RefinedPolygon {
        vertices,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use snapseg_core::GrayImage;

    /// Abramowitz & Stegun 7.1.26 approximation of `erf`. Good to ~5e-4
    /// over all real `x`; ample for an 8-bit step-edge test.
    fn erf(x: f32) -> f32 {
        // A&S 7.1.26 coefficients, truncated to f32 precision.
        let a1: f32 = 0.254_829_6;
        let a2: f32 = -0.284_496_72;
        let a3: f32 = 1.421_413_8;
        let a4: f32 = -1.453_152_1;
        let a5: f32 = 1.061_405_4;
        let p: f32 = 0.327_591_1;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }

    /// Pixel-area-average of a 1-D Gaussian-CDF step centred at `edge` with
    /// width `sigma`, sampled at the centre of `[axis - 0.5, axis + 0.5]`
    /// via 16× supersampling. This yields a properly anti-aliased step
    /// whose discrete gradient peak coincides with the continuous peak to
    /// well under 0.05 px — i.e. the test asserts the algorithm, not the
    /// quantisation noise.
    fn anti_aliased_step(axis: f32, edge: f32, sigma: f32) -> f32 {
        let nss = 16;
        let mut acc = 0.0;
        let scale = sigma * std::f32::consts::SQRT_2;
        for i in 0..nss {
            let s = axis - 0.5 + (i as f32 + 0.5) / nss as f32;
            acc += 0.5 * (1.0 + erf((s - edge) / scale));
        }
        acc / nss as f32
    }

    fn build_vertical_step(width: u32, height: u32, edge_x: f32, sigma: f32) -> GrayImage {
        let mut arr = Array2::<u8>::zeros((height as usize, width as usize));
        for y in 0..height as usize {
            for x in 0..width as usize {
                let s = anti_aliased_step(x as f32, edge_x, sigma);
                arr[(y, x)] = (255.0 * s).round().clamp(0.0, 255.0) as u8;
            }
        }
        GrayImage::from_array(arr)
    }

    fn build_horizontal_step(width: u32, height: u32, edge_y: f32, sigma: f32) -> GrayImage {
        let mut arr = Array2::<u8>::zeros((height as usize, width as usize));
        for y in 0..height as usize {
            for x in 0..width as usize {
                let s = anti_aliased_step(y as f32, edge_y, sigma);
                arr[(y, x)] = (255.0 * s).round().clamp(0.0, 255.0) as u8;
            }
        }
        GrayImage::from_array(arr)
    }

    fn median(mut xs: Vec<f32>) -> f32 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = xs.len();
        if n == 0 {
            return f32::NAN;
        }
        if n % 2 == 1 {
            xs[n / 2]
        } else {
            0.5 * (xs[n / 2 - 1] + xs[n / 2])
        }
    }

    #[test]
    fn step_edge_horizontal() {
        // Vertical boundary at x = 32.3; mask is x >= 32.
        let w = 64u32;
        let h = 64u32;
        let image = build_vertical_step(w, h, 32.3, 2.0);
        let mut mask = Array2::<bool>::from_elem((h as usize, w as usize), false);
        for y in 0..h as usize {
            for x in 32..w as usize {
                mask[(y, x)] = true;
            }
        }
        let poly = refine_polygon(&mask, &image, RefineParams::default());
        assert!(!poly.vertices.is_empty(), "expected a refined polygon");

        // Collect x-coordinates of vertices on the vertical edge: those
        // with confidence > 0 and y away from the top/bottom (where the
        // polygon turns the corner).
        let xs: Vec<f32> = poly
            .vertices
            .iter()
            .zip(poly.confidence.iter())
            .filter(|(p, c)| {
                **c > RefineParams::default().min_strength && p.y >= 2.0 && p.y <= (h as f32) - 2.0
            })
            .map(|(p, _)| p.x)
            .collect();
        assert!(
            xs.len() > 10,
            "expected many edge vertices, got {}",
            xs.len()
        );
        let med = median(xs.clone());
        let max_dev = xs.iter().map(|x| (x - 32.3).abs()).fold(0.0_f32, f32::max);
        assert!(
            (med - 32.3).abs() < 0.1,
            "median refined x = {med}, expected ~32.3"
        );
        assert!(max_dev < 0.2, "max deviation = {max_dev}");
        eprintln!(
            "[evidence] step_edge_horizontal: vertices={} median_x={med:.4} max_dev={max_dev:.4}",
            xs.len()
        );
    }

    #[test]
    fn step_edge_vertical() {
        // Horizontal boundary at y = 32.3; mask is y >= 32.
        let w = 64u32;
        let h = 64u32;
        let image = build_horizontal_step(w, h, 32.3, 2.0);
        let mut mask = Array2::<bool>::from_elem((h as usize, w as usize), false);
        for y in 32..h as usize {
            for x in 0..w as usize {
                mask[(y, x)] = true;
            }
        }
        let poly = refine_polygon(&mask, &image, RefineParams::default());
        let ys: Vec<f32> = poly
            .vertices
            .iter()
            .zip(poly.confidence.iter())
            .filter(|(p, c)| {
                **c > RefineParams::default().min_strength && p.x >= 2.0 && p.x <= (w as f32) - 2.0
            })
            .map(|(p, _)| p.y)
            .collect();
        assert!(
            ys.len() > 10,
            "expected many edge vertices, got {}",
            ys.len()
        );
        let med = median(ys.clone());
        let max_dev = ys.iter().map(|y| (y - 32.3).abs()).fold(0.0_f32, f32::max);
        assert!(
            (med - 32.3).abs() < 0.1,
            "median refined y = {med}, expected ~32.3"
        );
        assert!(max_dev < 0.2, "max deviation = {max_dev}");
        eprintln!(
            "[evidence] step_edge_vertical: vertices={} median_y={med:.4} max_dev={max_dev:.4}",
            ys.len()
        );
    }

    #[test]
    fn weak_edge_rejected() {
        // Uniform image → zero gradient → every vertex rejected. Run
        // marching squares once and refine its output directly so the test
        // is invariant to HashMap-driven walk order.
        let w = 64u32;
        let h = 64u32;
        let arr = Array2::<u8>::from_elem((h as usize, w as usize), 128u8);
        let image = GrayImage::from_array(arr);
        let mut mask = Array2::<bool>::from_elem((h as usize, w as usize), false);
        for y in 20..40 {
            for x in 20..40 {
                mask[(y, x)] = true;
            }
        }
        let ms = super::contour::marching_squares(&mask);
        let longest = ms.into_iter().max_by_key(|p| p.len()).unwrap();
        let params = RefineParams {
            min_strength: 1.0,
            ..RefineParams::default()
        };
        let refined = super::refine::refine_polyline(&longest, &image, params, true);
        assert_eq!(refined.len(), longest.len());
        let mut max_move: f32 = 0.0;
        for (i, (r, b)) in refined.iter().zip(longest.iter()).enumerate() {
            let d = ((r.point.x - b.x).powi(2) + (r.point.y - b.y).powi(2)).sqrt();
            if d > max_move {
                max_move = d;
            }
            assert!(
                r.confidence == 0.0,
                "vertex {i} got confidence {}",
                r.confidence
            );
        }
        assert!(max_move == 0.0, "expected zero movement, got {max_move}");
        eprintln!(
            "[evidence] weak_edge_rejected: vertices={} max_move={max_move} max_conf=0",
            refined.len()
        );
    }

    #[test]
    fn displacement_clamped() {
        // Vertical step edge far from the mask boundary. MS boundary lies
        // at x = 31.5; image step is at x = 37.5 → wanted offset = +6.0 px.
        // With max_displacement = 4, every refinement must be rejected and
        // the vertex left at x = 31.5. Band is widened to 8 so the peak is
        // actually inside the search range (otherwise the test would pass
        // trivially by hitting the band boundary).
        let w = 64u32;
        let h = 64u32;
        let image = build_vertical_step(w, h, 37.5, 2.0);
        let mut mask = Array2::<bool>::from_elem((h as usize, w as usize), false);
        for y in 0..h as usize {
            for x in 32..w as usize {
                mask[(y, x)] = true;
            }
        }
        let params = RefineParams {
            band: 8,
            max_displacement: 4.0,
            ..RefineParams::default()
        };
        // Marching-squares walk order depends on HashMap iteration so we
        // refine the same polyline we read off, not a fresh public-API call.
        let ms = super::contour::marching_squares(&mask);
        let longest = ms.into_iter().max_by_key(|p| p.len()).unwrap();
        let refined = super::refine::refine_polyline(&longest, &image, params, true);
        assert!(!refined.is_empty());
        assert_eq!(refined.len(), longest.len());
        let mut max_move = 0.0_f32;
        for (r, b) in refined.iter().zip(longest.iter()) {
            let d = ((r.point.x - b.x).powi(2) + (r.point.y - b.y).powi(2)).sqrt();
            if d > max_move {
                max_move = d;
            }
        }
        assert!(
            max_move <= params.max_displacement + 1e-4,
            "max move = {max_move}, expected ≤ {}",
            params.max_displacement
        );
        eprintln!(
            "[evidence] displacement_clamped: vertices={} max_move={max_move:.4} (cap=4.0)",
            refined.len()
        );
    }

    #[test]
    fn closed_polygon_topology() {
        // 32x32 square inside a 64x64 uniform-gradient image.
        let w = 64u32;
        let h = 64u32;
        let mut arr = Array2::<u8>::zeros((h as usize, w as usize));
        for y in 0..h as usize {
            for x in 0..w as usize {
                arr[(y, x)] = ((x as f32 * 4.0).clamp(0.0, 255.0)) as u8;
            }
        }
        let image = GrayImage::from_array(arr);
        let mut mask = Array2::<bool>::from_elem((h as usize, w as usize), false);
        for y in 16..48 {
            for x in 16..48 {
                mask[(y, x)] = true;
            }
        }
        let poly = refine_polygon(&mask, &image, RefineParams::default());
        assert!(poly.vertices.len() >= 16, "got {}", poly.vertices.len());
        let first = poly.vertices.first().copied().unwrap();
        let last = poly.vertices.last().copied().unwrap();
        assert!(
            (first.x - last.x).abs() < 1e-5,
            "first.x={}, last.x={}",
            first.x,
            last.x
        );
        assert!(
            (first.y - last.y).abs() < 1e-5,
            "first.y={}, last.y={}",
            first.y,
            last.y
        );
        let dx = (first.x - last.x).abs();
        let dy = (first.y - last.y).abs();
        eprintln!(
            "[evidence] closed_polygon_topology: vertices={} closure_dx={dx} closure_dy={dy}",
            poly.vertices.len()
        );
    }
}
