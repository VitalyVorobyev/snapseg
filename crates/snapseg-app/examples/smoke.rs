//! Headless diagnostic CLI for the MobileSAM pipeline.
//!
//! Mirrors `crates/snapseg-app/src/{config,main,dialogs}.rs` so a
//! regression in the live-UI pipeline shows up here too. Use this
//! to iterate on adapter and config bugs without the GUI.
//!
//! Examples (run from the workspace root, defaults to a single
//! positive click at the image centre):
//!
//! ```bash
//! # Smoke: default image + click.
//! cargo run -p snapseg-app --example smoke
//!
//! # Specific image, specific clicks, save a mask-overlay PNG.
//! cargo run -p snapseg-app --example smoke -- \
//!     --image assets/test_images/gptchess1.png \
//!     --click 340,285,pos \
//!     --click 360,290,neg \
//!     --out /tmp/chess_mask.png
//! ```
//!
//! Output:
//! - one log line per stage (config load, model load, encoder, decoder)
//! - mask stats: bbox + foreground pixel count + percent of frame
//! - PNG overlay at the path passed via `--out` (default
//!   `<image>.mask.png`)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ndarray::Array2;
use serde::Deserialize;

use snapseg_core::{GrayImage, InteractiveSegmenter, Point2, Polarity, Prompt, PromptSession};
use snapseg_models::mobile_sam::MobileSamSegmenter;
use snapseg_models::ritm::RitmSegmenter;
use snapseg_runtime::RuntimeConfig;

#[derive(Debug, Deserialize)]
struct SmokeConfig {
    onnxruntime_path: Option<PathBuf>,
    default_model: ModelCfg,
}

/// Accept the same `input_size = N` (scalar, square) or
/// `input_size = [H, W]` (array) that the live app accepts. Keeps the
/// smoke harness aligned with `snapseg-app::config::InputSize`.
#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(untagged)]
enum InputSize {
    Square(u32),
    HW([u32; 2]),
}

impl InputSize {
    fn as_hw(&self) -> (u32, u32) {
        match self {
            InputSize::Square(n) => (*n, *n),
            InputSize::HW([h, w]) => (*h, *w),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelCfg {
    name: String,
    family: String,
    input_size: InputSize,
    parts: HashMap<String, PathBuf>,
}

#[derive(Debug, Clone)]
struct Click {
    x: f32,
    y: f32,
    polarity: Polarity,
}

#[derive(Debug)]
struct Cli {
    image: Option<PathBuf>,
    clicks: Vec<Click>,
    out: Option<PathBuf>,
    canvas: Option<(u32, u32)>,
    /// `(cx, cy, radius)` in image pixels: assert the predicted mask's
    /// bbox centroid lies within `radius` of `(cx, cy)`. Non-zero exit
    /// on miss so this can run as a regression gate.
    assert_bbox_center: Option<(f32, f32, f32)>,
}

fn parse_cli() -> Result<Cli> {
    let mut image: Option<PathBuf> = None;
    let mut clicks: Vec<Click> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut canvas: Option<(u32, u32)> = None;
    let mut assert_bbox_center: Option<(f32, f32, f32)> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" | "-i" => {
                image = Some(args.next().context("--image needs a value")?.into());
            }
            "--click" | "-c" => {
                let raw = args.next().context("--click needs X,Y[,pos|neg]")?;
                clicks.push(parse_click(&raw)?);
            }
            "--out" | "-o" => {
                out = Some(args.next().context("--out needs a path")?.into());
            }
            "--canvas" => {
                let raw = args.next().context("--canvas needs HxW (e.g. 682x1024)")?;
                let (h, w) = raw.split_once('x').context("--canvas wants HxW")?;
                let h: u32 = h.trim().parse().context("canvas H")?;
                let w: u32 = w.trim().parse().context("canvas W")?;
                canvas = Some((h, w));
            }
            "--assert-bbox-center" => {
                let raw = args
                    .next()
                    .context("--assert-bbox-center needs CX,CY,RADIUS")?;
                let parts: Vec<&str> = raw.split(',').collect();
                if parts.len() != 3 {
                    bail!("--assert-bbox-center expects CX,CY,RADIUS; got '{raw}'");
                }
                let cx: f32 = parts[0].trim().parse().context("assert CX")?;
                let cy: f32 = parts[1].trim().parse().context("assert CY")?;
                let r: f32 = parts[2].trim().parse().context("assert RADIUS")?;
                assert_bbox_center = Some((cx, cy, r));
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: smoke [--image PATH] [--click X,Y[,pos|neg]]... [--out PATH] \
                     [--canvas HxW] [--assert-bbox-center CX,CY,RADIUS]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    Ok(Cli {
        image,
        clicks,
        out,
        canvas,
        assert_bbox_center,
    })
}

fn parse_click(raw: &str) -> Result<Click> {
    let parts: Vec<&str> = raw.split(',').collect();
    if parts.len() < 2 || parts.len() > 3 {
        bail!("--click expects X,Y or X,Y,pos|neg; got '{raw}'");
    }
    let x: f32 = parts[0].trim().parse().context("click X")?;
    let y: f32 = parts[1].trim().parse().context("click Y")?;
    let polarity = match parts.get(2).map(|s| s.trim().to_lowercase()).as_deref() {
        Some("pos") | Some("positive") | None => Polarity::Positive,
        Some("neg") | Some("negative") => Polarity::Negative,
        Some(other) => bail!("click polarity must be pos|neg; got '{other}'"),
    };
    Ok(Click { x, y, polarity })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = parse_cli()?;

    let cwd = std::env::current_dir()?;
    tracing::info!(cwd = %cwd.display(), "smoke start");

    let cfg_path = find_config_upward(&cwd).context("no snapseg.toml found walking up from cwd")?;
    let cfg_dir = cfg_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tracing::info!(config = %cfg_path.display(), "loaded snapseg.toml");

    let text = std::fs::read_to_string(&cfg_path)?;
    let cfg: SmokeConfig = toml::from_str(&text).context("parsing snapseg.toml")?;

    if let Some(p) = cfg.onnxruntime_path.as_ref() {
        let abs = resolve(&cfg_dir, p);
        if !abs.exists() {
            bail!("onnxruntime_path does not exist: {}", abs.display());
        }
        tracing::info!(path = %abs.display(), "setting ORT_DYLIB_PATH");
        // why this is unsafe: std::env::set_var is unsafe in the 2024
        // edition. Done before any ort call on the only thread that
        // exists at this point.
        unsafe { std::env::set_var("ORT_DYLIB_PATH", &abs) };
    }

    let family = cfg.default_model.family.clone();
    if !matches!(family.as_str(), "mobile_sam" | "ritm") {
        bail!(
            "smoke supports family ∈ {{mobile_sam, ritm}}; got '{}'",
            family
        );
    }

    let parts: HashMap<String, PathBuf> = cfg
        .default_model
        .parts
        .into_iter()
        .map(|(k, p)| (k, resolve(&cfg_dir, &p)))
        .collect();
    for (k, p) in &parts {
        if !p.exists() {
            bail!("model part '{k}' does not exist: {}", p.display());
        }
    }

    let image_path = cli
        .image
        .map(|p| resolve(&cfg_dir, &p))
        .unwrap_or_else(|| cfg_dir.join("assets/test_images/gptchess1.png"));
    if !image_path.exists() {
        bail!("image does not exist: {}", image_path.display());
    }
    tracing::info!(image = %image_path.display(), "loading image");

    // Load the source image as both grayscale (for the segmenter) and
    // RGB (for the overlay output PNG).
    let img_rgb = image::open(&image_path)?.to_rgb8();
    let (w, h) = (img_rgb.width(), img_rgb.height());
    let gray_buf: Vec<u8> = img_rgb
        .pixels()
        .map(|p| {
            let [r, g, b] = p.0;
            ((0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32).round() as u8).clamp(0, 255)
        })
        .collect();
    let gray = GrayImage::from_array(
        Array2::from_shape_vec((h as usize, w as usize), gray_buf)
            .context("gray shape mismatch")?,
    );
    tracing::info!(w, h, "image loaded");

    let clicks = if cli.clicks.is_empty() {
        // Default: one positive click at the image centre.
        let c = Click {
            x: w as f32 / 2.0,
            y: h as f32 / 2.0,
            polarity: Polarity::Positive,
        };
        tracing::info!(x = c.x, y = c.y, "default click at centre");
        vec![c]
    } else {
        cli.clicks
    };

    let shape = cli
        .canvas
        .unwrap_or_else(|| cfg.default_model.input_size.as_hw());
    tracing::info!(
        family = %family,
        canvas_h = shape.0,
        canvas_w = shape.1,
        "constructing segmenter"
    );
    let mut seg: Box<dyn InteractiveSegmenter> = match family.as_str() {
        "mobile_sam" => Box::new(
            MobileSamSegmenter::from_parts_with_shape(
                cfg.default_model.name.clone(),
                &parts,
                shape,
                &RuntimeConfig::default(),
            )
            .context("MobileSamSegmenter::from_parts_with_shape")?,
        ),
        "ritm" => Box::new(
            RitmSegmenter::from_parts_with_shape(
                cfg.default_model.name.clone(),
                &parts,
                shape,
                &RuntimeConfig::default(),
            )
            .context("RitmSegmenter::from_parts_with_shape")?,
        ),
        // Already gated above; unreachable in practice.
        other => bail!("unsupported family '{other}'"),
    };

    let t_enc = Instant::now();
    seg.set_image(&gray).context("set_image")?;
    let enc_ms = t_enc.elapsed().as_millis() as u64;
    tracing::info!(ms = enc_ms, "set_image complete");

    let mut session = PromptSession::new();
    for c in &clicks {
        session.push(Prompt::Click {
            point: Point2::new(c.x, c.y),
            polarity: c.polarity,
        });
    }

    let t_dec = Instant::now();
    let result = seg.segment(&session).context("segment")?;
    let dec_ms = t_dec.elapsed().as_millis() as u64;

    // Mask stats: bbox + fg count + mean logits.
    let (bbox, fg) = mask_bbox_and_count(&result.mask);
    let total = result.mask.len();
    let pct = 100.0 * fg as f64 / total as f64;
    let (mean_in, mean_out) = mean_logits_inside_outside(&result.logits, &bbox);
    tracing::info!(
        ms = dec_ms,
        mask_h = result.mask.shape()[0],
        mask_w = result.mask.shape()[1],
        fg_pixels = fg,
        fg_pct = format!("{pct:.2}%"),
        bbox = format!("{:?}", bbox),
        mean_logit_in = format!("{mean_in:.3}"),
        mean_logit_out = format!("{mean_out:.3}"),
        "decoder pass complete"
    );

    let out_path = cli
        .out
        .unwrap_or_else(|| image_path.with_extension("mask.png"));
    save_overlay(&img_rgb, &result.mask, &clicks, &out_path)
        .with_context(|| format!("saving overlay to {}", out_path.display()))?;
    tracing::info!(out = %out_path.display(), "wrote overlay PNG");

    println!(
        "OK encoder={enc_ms}ms decoder={dec_ms}ms mask={}x{} fg={fg}/{total} ({pct:.2}%) bbox={bbox:?} overlay={}",
        result.mask.shape()[0],
        result.mask.shape()[1],
        out_path.display(),
    );

    let exit_code = if let Some((cx, cy, radius)) = cli.assert_bbox_center {
        if fg == 0 {
            eprintln!("FAIL --assert-bbox-center: mask is empty");
            1
        } else {
            let (min_x, min_y, max_x, max_y) = bbox;
            let ccx = (min_x + max_x) as f32 * 0.5;
            let ccy = (min_y + max_y) as f32 * 0.5;
            let dist = ((ccx - cx).powi(2) + (ccy - cy).powi(2)).sqrt();
            if dist <= radius {
                println!(
                    "PASS --assert-bbox-center: mask centre ({ccx:.1}, {ccy:.1}) within {radius} of ({cx}, {cy}); dist={dist:.1}"
                );
                0
            } else {
                eprintln!(
                    "FAIL --assert-bbox-center: mask centre ({ccx:.1}, {ccy:.1}) is {dist:.1} px from ({cx}, {cy}), exceeds radius {radius}"
                );
                1
            }
        }
    } else {
        0
    };

    // why this is exit(): onnxruntime's C++ static destructors race
    // with the Rust runtime teardown on macOS and otherwise print
    // `libc++abi: terminating ... mutex lock failed`. We're done.
    std::process::exit(exit_code);
}

/// `(x0, y0, x1, y1)` inclusive bbox of true pixels, and the fg pixel
/// count. Returns `((0,0,0,0), 0)` when the mask is empty.
fn mask_bbox_and_count(mask: &Array2<bool>) -> ((usize, usize, usize, usize), usize) {
    let (h, w) = (mask.shape()[0], mask.shape()[1]);
    let mut min_x = usize::MAX;
    let mut min_y = usize::MAX;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut count = 0usize;
    for y in 0..h {
        for x in 0..w {
            if mask[(y, x)] {
                count += 1;
                if x < min_x {
                    min_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if count == 0 {
        return ((0, 0, 0, 0), 0);
    }
    ((min_x, min_y, max_x, max_y), count)
}

/// Mean logits inside and outside the bbox. Diagnostic: if mean_in is
/// only slightly above mean_out, the model isn't confident — that
/// often indicates the click landed in a region the encoder didn't
/// distinguish well.
fn mean_logits_inside_outside(
    logits: &Array2<f32>,
    bbox: &(usize, usize, usize, usize),
) -> (f32, f32) {
    let (h, w) = (logits.shape()[0], logits.shape()[1]);
    let (min_x, min_y, max_x, max_y) = *bbox;
    let mut sum_in = 0.0f64;
    let mut sum_out = 0.0f64;
    let mut count_in = 0usize;
    let mut count_out = 0usize;
    for y in 0..h {
        for x in 0..w {
            let v = logits[(y, x)] as f64;
            if y >= min_y && y <= max_y && x >= min_x && x <= max_x {
                sum_in += v;
                count_in += 1;
            } else {
                sum_out += v;
                count_out += 1;
            }
        }
    }
    let in_avg = if count_in > 0 {
        (sum_in / count_in as f64) as f32
    } else {
        0.0
    };
    let out_avg = if count_out > 0 {
        (sum_out / count_out as f64) as f32
    } else {
        0.0
    };
    (in_avg, out_avg)
}

/// Render the source RGB image with a translucent blue overlay where
/// `mask` is true and small colored dots at the click positions, then
/// save as PNG.
fn save_overlay(
    rgb: &image::RgbImage,
    mask: &Array2<bool>,
    clicks: &[Click],
    out: &Path,
) -> Result<()> {
    let (w, h) = (rgb.width(), rgb.height());
    let mut overlay = rgb.clone();
    let (mh, mw) = (mask.shape()[0], mask.shape()[1]);
    if mh != h as usize || mw != w as usize {
        bail!("mask shape {mh}x{mw} doesn't match image {h}x{w}; cannot overlay");
    }
    for y in 0..h {
        for x in 0..w {
            if mask[(y as usize, x as usize)] {
                let p = overlay.get_pixel_mut(x, y);
                // 60% original, 40% blue (RGB 64, 130, 240).
                let blend = |orig: u8, tint: u8| -> u8 {
                    ((orig as f32) * 0.6 + (tint as f32) * 0.4) as u8
                };
                p.0[0] = blend(p.0[0], 64);
                p.0[1] = blend(p.0[1], 130);
                p.0[2] = blend(p.0[2], 240);
            }
        }
    }
    // Draw clicks as 4-pixel-radius solid disks: green positive,
    // red negative.
    for c in clicks {
        let cx = c.x.round() as i32;
        let cy = c.y.round() as i32;
        let rgb_dot = match c.polarity {
            Polarity::Positive => [40u8, 220, 40],
            Polarity::Negative => [240u8, 60, 60],
        };
        for dy in -4..=4i32 {
            for dx in -4..=4i32 {
                if dx * dx + dy * dy > 16 {
                    continue;
                }
                let xx = cx + dx;
                let yy = cy + dy;
                if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                    continue;
                }
                let p = overlay.get_pixel_mut(xx as u32, yy as u32);
                p.0 = rgb_dot;
            }
        }
    }
    overlay.save(out)?;
    Ok(())
}

fn resolve(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

fn find_config_upward(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        let candidate = dir.join("snapseg.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}
