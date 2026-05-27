//! Per-vertex subpixel refinement of a polyline against a grayscale image.
//!
//! Given an ordered polyline (typically from `contour::marching_squares`),
//! refine each vertex along the local normal: sample the image in a band
//! around the vertex, smooth, take the first derivative, locate the peak,
//! and parabolic-fit for subpixel accuracy.
//!
//! ## Sign convention
//!
//! For two adjacent vertices `prev` and `next`, the tangent is
//! `(next - prev) / 2` and the normal used here is
//! `(tangent.y, -tangent.x)` then unit-normalised. With image coordinates
//! where `+y` is downward and a polyline traversed clockwise as seen on
//! screen (= counter-clockwise in mathematical convention), this normal
//! points outward (away from foreground). The refinement is invariant to
//! traversal direction — see the design note in the crate root.

use snapseg_core::{GrayImage, Point2};

use crate::RefineParams;
use crate::sample::{
    bilinear_sample, central_diff, convolve_reflect, gaussian_kernel, parabolic_peak,
};

/// Outcome for one vertex.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VertexRefinement {
    pub point: Point2,
    pub confidence: f32,
}

/// Refine every vertex of `polyline` in place.
///
/// `polyline` is treated as closed iff `treat_as_closed` is `true` — when
/// closed, the prev/next of the first and last vertices wrap. When open,
/// the endpoints use their single neighbour as the tangent reference and
/// are typically left at confidence 0.
pub(crate) fn refine_polyline(
    polyline: &[Point2],
    image: &GrayImage,
    params: RefineParams,
    treat_as_closed: bool,
) -> Vec<VertexRefinement> {
    let n = polyline.len();
    if n < 2 {
        return polyline
            .iter()
            .map(|&p| VertexRefinement {
                point: p,
                confidence: 0.0,
            })
            .collect();
    }

    // If the polyline reports closure by repeating the first vertex at the
    // end, drop the duplicate for refinement and re-add it at the end so
    // the output stays closed without double-refining the same point.
    let (working, closed_marker) =
        if treat_as_closed && polyline.len() >= 2 && approx_eq(polyline[0], polyline[n - 1]) {
            (&polyline[..n - 1], true)
        } else {
            (polyline, false)
        };

    let kernel = gaussian_kernel(params.sigma / 0.5);
    let band = params.band.max(1) as i32;
    let band_steps = 2 * band; // sample step is 0.5 px

    let m = working.len();
    let mut out: Vec<VertexRefinement> = Vec::with_capacity(m + closed_marker as usize);
    for i in 0..m {
        let prev = neighbour(working, i, -1, treat_as_closed);
        let next = neighbour(working, i, 1, treat_as_closed);
        let here = working[i];
        let refined = refine_one(here, prev, next, band_steps, &kernel, image, params).unwrap_or(
            VertexRefinement {
                point: here,
                confidence: 0.0,
            },
        );
        out.push(refined);
    }
    if closed_marker {
        let first = out[0];
        out.push(first);
    }
    out
}

fn neighbour(poly: &[Point2], i: usize, delta: i32, closed: bool) -> Option<Point2> {
    let n = poly.len() as i32;
    let j = i as i32 + delta;
    if closed {
        Some(poly[j.rem_euclid(n) as usize])
    } else if (0..n).contains(&j) {
        Some(poly[j as usize])
    } else {
        None
    }
}

/// Refine a single vertex. Returns `None` if degenerate (no tangent, all
/// samples OOB, etc.); caller falls back to the unrefined vertex.
fn refine_one(
    here: Point2,
    prev: Option<Point2>,
    next: Option<Point2>,
    band_steps: i32,
    kernel: &[f32],
    image: &GrayImage,
    params: RefineParams,
) -> Option<VertexRefinement> {
    let (tx, ty) = tangent(prev, next, here)?;
    let len = (tx * tx + ty * ty).sqrt();
    if len <= 0.0 || !len.is_finite() {
        return None;
    }
    let utx = tx / len;
    let uty = ty / len;
    // normal = rotate tangent by -90deg in screen coords.
    let nx = uty;
    let ny = -utx;

    let n_samples = (2 * band_steps + 1) as usize;
    let mut samples: Vec<f32> = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let t = (k as i32 - band_steps) as f32 * 0.5;
        let px = here.x + t * nx;
        let py = here.y + t * ny;
        let s = bilinear_sample(image, px, py)?;
        samples.push(s);
    }

    let smoothed = convolve_reflect(&samples, kernel);
    let deriv = central_diff(&smoothed);

    // |derivative| times sample-step scale (0.5 px) so `min_strength` is in
    // intensity-per-pixel units, not intensity-per-sample-step.
    let mag: Vec<f32> = deriv.iter().map(|v| v.abs() * 2.0).collect();

    let (idx, &peak) =
        mag.iter().enumerate().fold(
            (0usize, &0.0_f32),
            |(bi, bv), (i, v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            },
        );

    if peak < params.min_strength {
        return None;
    }

    let subpix = if idx == 0 || idx + 1 >= mag.len() {
        0.0_f32
    } else {
        parabolic_peak(mag[idx - 1], mag[idx], mag[idx + 1]).unwrap_or(0.0)
    };

    // Convert from sample-step index to image-space offset along the
    // normal. Sample step is 0.5 px; index `band_steps` is offset 0.
    let offset_px = ((idx as i32 - band_steps) as f32 + subpix) * 0.5;

    if offset_px.abs() > params.max_displacement {
        return None;
    }

    Some(VertexRefinement {
        point: Point2::new(here.x + offset_px * nx, here.y + offset_px * ny),
        confidence: peak,
    })
}

/// Tangent at a vertex from its neighbours. Falls back to the single
/// available neighbour at an open endpoint. Returns `None` if both
/// neighbours are absent or coincide with the vertex.
fn tangent(prev: Option<Point2>, next: Option<Point2>, here: Point2) -> Option<(f32, f32)> {
    match (prev, next) {
        (Some(p), Some(n)) => Some((0.5 * (n.x - p.x), 0.5 * (n.y - p.y))),
        (Some(p), None) => Some((here.x - p.x, here.y - p.y)),
        (None, Some(n)) => Some((n.x - here.x, n.y - here.y)),
        (None, None) => None,
    }
}

fn approx_eq(a: Point2, b: Point2) -> bool {
    (a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6
}
