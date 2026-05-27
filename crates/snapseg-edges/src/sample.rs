//! Pixel-space sampling and 1-D signal helpers for subpixel refinement.
//!
//! Three small primitives, kept in one module because they share the same
//! responsibility: turn an image and a 1-D scan of positions into a smoothed
//! derivative signal whose argmax can be parabolic-fit to subpixel accuracy.
//!
//!   - [`bilinear_sample`]: read a `GrayImage` at non-integer `(x, y)`.
//!   - [`gaussian_kernel`]: build a normalised 1-D Gaussian.
//!   - [`convolve_reflect`]: 1-D convolution with mirror boundary handling.
//!   - [`central_diff`]: 1-D central differences.
//!   - [`parabolic_peak`]: subpixel argmax of three samples around an integer peak.

use snapseg_core::GrayImage;

/// Bilinear lookup into a grayscale image at floating coordinates.
///
/// `image.data` is row-major `Array2<u8>` with shape `[H, W]`, indexed as
/// `[row=y, col=x]`. Coordinates use the pixel-centre convention: integer
/// `(x, y)` lands exactly on the centre of pixel `[y, x]`.
///
/// Returns `None` when any of the four neighbouring samples would fall
/// outside `[0, W-1] x [0, H-1]`. Caller treats `None` as "do not refine
/// this vertex".
pub(crate) fn bilinear_sample(image: &GrayImage, x: f32, y: f32) -> Option<f32> {
    let w = image.width as i32;
    let h = image.height as i32;
    if w < 2 || h < 2 {
        return None;
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    if x0 < 0 || y0 < 0 || x1 >= w || y1 >= h {
        return None;
    }
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let data = &image.data;
    // Safe indexing: bounds checked above.
    let v00 = data[(y0 as usize, x0 as usize)] as f32;
    let v01 = data[(y0 as usize, x1 as usize)] as f32;
    let v10 = data[(y1 as usize, x0 as usize)] as f32;
    let v11 = data[(y1 as usize, x1 as usize)] as f32;
    let top = v00 * (1.0 - fx) + v01 * fx;
    let bot = v10 * (1.0 - fx) + v11 * fx;
    Some(top * (1.0 - fy) + bot * fy)
}

/// 1-D Gaussian kernel of standard deviation `sigma_steps`, expressed in
/// sample-step units. Half-width `ceil(3 * sigma_steps)` and the kernel is
/// L1-normalised so convolution preserves signal magnitude.
///
/// Returns an empty kernel for `sigma_steps <= 0` so the caller can treat it
/// as identity.
pub(crate) fn gaussian_kernel(sigma_steps: f32) -> Vec<f32> {
    if sigma_steps <= 0.0 || !sigma_steps.is_finite() {
        return Vec::new();
    }
    let half = (3.0 * sigma_steps).ceil() as i32;
    let two_sigma_sq = 2.0 * sigma_steps * sigma_steps;
    let mut kernel: Vec<f32> = (-half..=half)
        .map(|i| (-(i as f32).powi(2) / two_sigma_sq).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    if sum > 0.0 {
        for v in kernel.iter_mut() {
            *v /= sum;
        }
    }
    kernel
}

/// 1-D convolution of `signal` with `kernel`, mirror-reflecting at the
/// boundaries.
///
/// `kernel` is centred (its middle element is the zero-offset tap). For
/// out-of-bounds reads at index `i`, the sample at `|i|` is mirrored back
/// using `reflect(i, n)` so that index `-1` returns `signal[1]` and index
/// `n` returns `signal[n-2]`. This avoids the spurious zero-bias that
/// zero-padding would introduce at the band edges.
///
/// Returns `signal` unchanged when `kernel` is empty.
pub(crate) fn convolve_reflect(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    if kernel.is_empty() {
        return signal.to_vec();
    }
    let n = signal.len();
    let half = (kernel.len() / 2) as i32;
    let mut out = vec![0.0_f32; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for (k, &w) in kernel.iter().enumerate() {
            let offset = k as i32 - half;
            let idx = reflect(i as i32 + offset, n as i32);
            acc += w * signal[idx as usize];
        }
        *slot = acc;
    }
    out
}

/// Reflect `i` into `[0, n)` by mirroring at the boundaries (no repetition
/// of the edge sample).
fn reflect(i: i32, n: i32) -> i32 {
    if n <= 1 {
        return 0;
    }
    let mut i = i;
    let period = 2 * (n - 1);
    i = i.rem_euclid(period);
    if i >= n {
        i = period - i;
    }
    i
}

/// Central-difference first derivative. Endpoints use forward / backward
/// differences so the output has the same length as the input.
pub(crate) fn central_diff(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }
    let mut out = vec![0.0_f32; n];
    out[0] = signal[1] - signal[0];
    out[n - 1] = signal[n - 1] - signal[n - 2];
    for i in 1..n - 1 {
        out[i] = 0.5 * (signal[i + 1] - signal[i - 1]);
    }
    out
}

/// Parabolic fit around an integer peak. Given samples `y0, y1, y2` taken
/// at offsets `-1, 0, +1` from the peak index, returns the subpixel offset
/// of the true peak relative to the integer peak — in `[-0.5, 0.5]` for a
/// well-formed concave-down triple.
///
/// Returns `None` when the parabola is flat (denominator near zero), so the
/// caller can fall back to the integer peak.
pub(crate) fn parabolic_peak(y0: f32, y1: f32, y2: f32) -> Option<f32> {
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < f32::EPSILON {
        return None;
    }
    let offset = 0.5 * (y0 - y2) / denom;
    if !offset.is_finite() {
        return None;
    }
    Some(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use snapseg_core::GrayImage;

    #[test]
    fn bilinear_centre_of_pixel_equals_pixel() {
        let arr = Array2::from_shape_fn((4, 4), |(y, x)| (y * 10 + x) as u8);
        let img = GrayImage::from_array(arr);
        let v = bilinear_sample(&img, 2.0, 1.0).unwrap();
        assert!((v - 12.0).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn bilinear_half_step_is_average() {
        let arr = Array2::from_shape_fn((2, 2), |(_, x)| if x == 0 { 0u8 } else { 100u8 });
        let img = GrayImage::from_array(arr);
        let v = bilinear_sample(&img, 0.5, 0.5).unwrap();
        assert!((v - 50.0).abs() < 1e-5);
    }

    #[test]
    fn bilinear_oob_returns_none() {
        let arr = Array2::<u8>::zeros((2, 2));
        let img = GrayImage::from_array(arr);
        assert!(bilinear_sample(&img, -0.1, 0.0).is_none());
        assert!(bilinear_sample(&img, 1.0, 0.0).is_none());
    }

    #[test]
    fn gaussian_kernel_normalised() {
        let k = gaussian_kernel(1.0);
        let s: f32 = k.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(k.len() >= 7);
    }

    #[test]
    fn convolve_reflect_passes_identity_for_empty_kernel() {
        let s = vec![1.0, 2.0, 3.0];
        let out = convolve_reflect(&s, &[]);
        assert_eq!(out, s);
    }

    #[test]
    fn convolve_reflect_preserves_constant() {
        let s = vec![5.0; 10];
        let k = gaussian_kernel(1.5);
        let out = convolve_reflect(&s, &k);
        for v in out {
            assert!((v - 5.0).abs() < 1e-4);
        }
    }

    #[test]
    fn parabolic_peak_at_centre() {
        // y = -(x^2) — peak at x=0
        let off = parabolic_peak(-1.0, 0.0, -1.0).unwrap();
        assert!(off.abs() < 1e-6);
    }

    #[test]
    fn parabolic_peak_shifted() {
        // y = -(x - 0.3)^2 + c. y(-1)=-1.69+c, y(0)=-0.09+c, y(1)=-0.49+c
        let y0 = -1.69;
        let y1 = -0.09;
        let y2 = -0.49;
        let off = parabolic_peak(y0, y1, y2).unwrap();
        assert!((off - 0.3).abs() < 1e-5, "got {off}");
    }

    #[test]
    fn parabolic_peak_flat_returns_none() {
        assert!(parabolic_peak(1.0, 1.0, 1.0).is_none());
    }
}
