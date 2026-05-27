//! Headless smoke test for the MobileSAM pipeline.
//!
//! Run from the workspace root with:
//!
//! ```bash
//! cargo run -p snapseg-app --example smoke -- assets/test_images/gptchess1.png
//! ```
//!
//! Loads `snapseg.toml` (walks up from cwd), exports `ORT_DYLIB_PATH`,
//! constructs the [`MobileSamSegmenter`], runs one encoder pass and one
//! decoder pass with a single positive click at the image centre, and
//! prints mask statistics. Exits non-zero on any error.
//!
//! Mirrors `crates/snapseg-app/src/{config,main,dialogs}.rs` so a
//! regression in the live-UI pipeline shows up here too — this is
//! the unit of work to iterate on when the GUI panics inside `ort`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ndarray::Array2;
use serde::Deserialize;

use snapseg_core::{GrayImage, InteractiveSegmenter, Point2, Polarity, Prompt, PromptSession};
use snapseg_models::mobile_sam::MobileSamSegmenter;
use snapseg_runtime::RuntimeConfig;

/// Minimal subset of `snapseg-app`'s `AppConfig` — just the fields the
/// smoke needs. Lives here so the example stays standalone (no lib
/// target on `snapseg-app`).
#[derive(Debug, Deserialize)]
struct SmokeConfig {
    onnxruntime_path: Option<PathBuf>,
    default_model: ModelCfg,
}

#[derive(Debug, Deserialize)]
struct ModelCfg {
    name: String,
    family: String,
    input_size: u32,
    parts: HashMap<String, PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cwd = std::env::current_dir()?;
    tracing::info!(cwd = %cwd.display(), "smoke start");

    let cfg_path =
        find_config_upward(&cwd).context("no snapseg.toml found anywhere walking up from cwd")?;
    let cfg_dir = cfg_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tracing::info!(config = %cfg_path.display(), "loaded snapseg.toml");

    let text = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("reading {}", cfg_path.display()))?;
    let cfg: SmokeConfig = toml::from_str(&text).context("parsing snapseg.toml")?;

    if let Some(p) = cfg.onnxruntime_path.as_ref() {
        let abs = resolve(&cfg_dir, p);
        if !abs.exists() {
            bail!("onnxruntime_path does not exist: {}", abs.display());
        }
        tracing::info!(path = %abs.display(), "setting ORT_DYLIB_PATH");
        // why this is unsafe: std::env::set_var is unsafe in the 2024
        // edition. Done before any ort call on the only thread that
        // exists at this point, so the OnceLock inside ort sees the
        // right value.
        unsafe { std::env::set_var("ORT_DYLIB_PATH", &abs) };
    }

    if cfg.default_model.family != "mobile_sam" {
        bail!(
            "smoke supports family = 'mobile_sam' only, got '{}'",
            cfg.default_model.family
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

    let image_arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/test_images/gptchess1.png".to_string());
    let image_path = resolve(&cfg_dir, Path::new(&image_arg));
    if !image_path.exists() {
        bail!("image does not exist: {}", image_path.display());
    }
    tracing::info!(image = %image_path.display(), "loading image");
    let img = image::open(&image_path)?.to_luma8();
    let (w, h) = (img.width(), img.height());
    let gray = GrayImage::from_array(
        Array2::from_shape_vec((h as usize, w as usize), img.into_raw())
            .context("image bytes / shape mismatch")?,
    );
    tracing::info!(w, h, "image loaded");

    tracing::info!("constructing MobileSAM segmenter");
    let mut seg = MobileSamSegmenter::from_parts(
        cfg.default_model.name.clone(),
        &parts,
        cfg.default_model.input_size,
        &RuntimeConfig::default(),
    )
    .context("MobileSamSegmenter::from_parts")?;

    let t_enc = Instant::now();
    seg.set_image(&gray).context("set_image")?;
    let enc_ms = t_enc.elapsed().as_millis() as u64;
    tracing::info!(ms = enc_ms, "encoder pass complete");

    let mut session = PromptSession::new();
    session.push(Prompt::Click {
        point: Point2::new(w as f32 / 2.0, h as f32 / 2.0),
        polarity: Polarity::Positive,
    });

    let t_dec = Instant::now();
    let result = seg.segment(&session).context("segment")?;
    let dec_ms = t_dec.elapsed().as_millis() as u64;

    let fg: usize = result.mask.iter().filter(|&&b| b).count();
    let total = result.mask.len();
    let pct = 100.0 * fg as f64 / total as f64;
    tracing::info!(
        ms = dec_ms,
        mask_h = result.mask.shape()[0],
        mask_w = result.mask.shape()[1],
        fg_pixels = fg,
        fg_pct = format!("{pct:.1}%"),
        "decoder pass complete"
    );

    println!(
        "OK encoder={enc_ms}ms decoder={dec_ms}ms mask={}x{} fg={fg}/{total} ({pct:.1}%)",
        result.mask.shape()[0],
        result.mask.shape()[1],
    );

    // why this is exit(0): onnxruntime's C++ static destructors race
    // with the Rust runtime teardown on macOS and the process otherwise
    // prints `libc++abi: terminating ... mutex lock failed`. We're done
    // and the work succeeded; skip orderly teardown.
    std::process::exit(0);
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
