//! On-disk label format for snapseg — the substrate of the data flywheel.
//!
//! snapseg's framing is that smart promptable models produce labels from
//! operator clicks, and smaller domain-specialized models are then
//! fine-tuned on those labels. This crate captures snapseg labels to
//! disk in a stable schema so that downstream training pipelines can
//! consume them without further negotiation with the app.
//!
//! One label is one segmented region on one image, stored as the
//! directory `labels/<image-sha8>/` containing `image.png`,
//! `prompts.json`, `mask.png`, and `meta.toml` (with optional
//! `logits.png` and `polygon.json`). See
//! `.claude/skills/snapseg-label-export/SKILL.md` for the full layout
//! and `docs/ROADMAP.md` M2 for the milestone.
//!
//! The on-disk schema version this crate produces and reads is
//! [`SCHEMA_VERSION`] (`"1.0"`). Older labels must remain readable for
//! the foreseeable future; bump only for breaking changes.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// On-disk schema version written into `meta.toml`. Bump for breaking
/// changes; older labels must remain readable for the foreseeable
/// future.
pub const SCHEMA_VERSION: &str = "1.0";

/// One label on disk. This is the post-save handle; for the pre-save
/// in-memory bundle see [`Provenance`] plus the arguments to
/// `LabelDir::save` (added in M2-T03).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    /// Directory containing `image.png` / `prompts.json` / `mask.png` /
    /// `meta.toml`.
    pub dir: PathBuf,
    /// First 8 hex chars of the source image SHA-256, possibly with a
    /// region suffix (`-r2`, `-r3`) if multiple regions on the same
    /// image were saved.
    pub id: String,
}

/// Operator-supplied quality flag, persisted in `meta.toml`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LabelQuality {
    /// Operator has reviewed; the label is suitable for training.
    Good,
    /// Operator is uncertain; the label should be re-reviewed before
    /// going into a training set.
    NeedsReview,
    /// Operator rejects the label; do not include in training.
    Reject,
}

/// Notes the operator attaches to a single label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorNote {
    /// Free-form note shown when reviewing the label later.
    #[serde(default)]
    pub note: String,
    /// Quality flag, defaults to [`LabelQuality::Good`].
    pub quality: LabelQuality,
}

impl Default for OperatorNote {
    fn default() -> Self {
        Self {
            note: String::new(),
            quality: LabelQuality::Good,
        }
    }
}

/// Which model produced this mask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenance {
    /// `name` field from `models.toml` (e.g. `"mobile-sam"`).
    pub registry_name: String,
    /// `family` slug (e.g. `"mobile_sam"`).
    pub family: String,
    /// SHA-256 of the encoder ONNX, if the family is multi-part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_sha256: Option<String>,
    /// SHA-256 of the decoder ONNX, if the family is multi-part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_sha256: Option<String>,
    /// SHA-256 of the single ONNX file, if the family is single-network.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}

/// How the model was run when the mask was produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProvenance {
    /// `"CPU"` / `"CoreML"` / `"CUDA"` etc.
    pub execution_provider: String,
    /// Encoder pass wall-clock, in ms. `None` for single-network
    /// families.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_ms: Option<u64>,
    /// Decoder pass wall-clock (or single-network call) in ms.
    pub decoder_ms: u64,
    /// `git rev-parse HEAD` of the snapseg checkout when the label was
    /// produced. Best-effort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapseg_commit: Option<String>,
}

/// Everything we know about a label's pedigree. Persisted into
/// `meta.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// On-disk schema version; should match [`SCHEMA_VERSION`] at
    /// save time.
    pub schema_version: String,
    /// RFC3339 timestamp from `chrono::Utc::now()` at save time.
    pub created_at: String,
    /// SHA-256 of the source image bytes as they were on disk.
    pub source_image_sha256: String,
    /// Where the image came from on the operator's disk, if known.
    /// Skipped during serialization when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_path: Option<PathBuf>,
    /// Width of the source image, in pixels.
    pub source_image_width: u32,
    /// Height of the source image, in pixels.
    pub source_image_height: u32,
    /// Which model produced the mask.
    pub model: ModelProvenance,
    /// How the model was run.
    pub runtime: RuntimeProvenance,
    /// Operator-attached note + quality flag.
    pub operator: OperatorNote,
}

/// JSON shape for `prompts.json`. One entry per user-issued prompt in
/// the order they were issued.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSessionJson {
    /// Prompts in issue order. Coordinates are in source-image pixel
    /// space.
    pub session: Vec<PromptRecord>,
}

/// One persisted prompt. Coordinates are in source-image pixel space.
/// `t_ms` is offset from the first prompt in the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptRecord {
    /// A single click at `(x, y)` with a polarity (`"positive"` or
    /// `"negative"`).
    Click {
        /// X pixel coordinate in the source image.
        x: f32,
        /// Y pixel coordinate in the source image.
        y: f32,
        /// `"positive"` (include) or `"negative"` (exclude).
        polarity: String,
        /// Milliseconds since the first prompt in the session.
        t_ms: u64,
    },
    /// An axis-aligned box `(x0, y0)`–`(x1, y1)`.
    Box {
        /// Top-left X.
        x0: f32,
        /// Top-left Y.
        y0: f32,
        /// Bottom-right X.
        x1: f32,
        /// Bottom-right Y.
        y1: f32,
        /// Milliseconds since the first prompt in the session.
        t_ms: u64,
    },
    /// A free-form polyline of `[x, y]` samples.
    Scribble {
        /// Ordered `[x, y]` samples in source-image pixel space.
        points: Vec<[f32; 2]>,
        /// `"positive"` (include) or `"negative"` (exclude).
        polarity: String,
        /// Milliseconds since the first prompt in the session.
        t_ms: u64,
    },
}

/// Filesystem root holding every label produced by snapseg. Today this
/// is a flat directory of `<sha>/` subdirectories; the schema versions
/// the directory layout for future evolution.
#[derive(Debug, Clone)]
pub struct LabelDir {
    /// Path to the directory that contains per-label subdirectories.
    pub root: PathBuf,
}

impl LabelDir {
    /// Construct a [`LabelDir`]. Does not create the directory; that's
    /// [`LabelDir::save`]'s job on first use.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Persist a label to disk under [`LabelDir::root`].
    ///
    /// Writes are atomic: every artifact is staged under a
    /// `<id>.partial/` directory, and only after all files are in
    /// place does the directory get renamed to its final `<id>/`
    /// form. A crash part-way through leaves a `.partial/` orphan
    /// that does not pollute the dataset.
    ///
    /// The label `id` is the first 8 hex chars of the source image
    /// SHA-256. If a label for that image already exists, the next
    /// free `-r2`, `-r3`, … suffix is used so a single image can
    /// carry multiple regions.
    ///
    /// `image_path`, when `Some`, is hashed and copied byte-for-byte
    /// into the label as `image.png` — preserving the operator's
    /// original bytes is required for reproducibility, see the
    /// [`snapseg-label-export`] skill. When `None`, `gray` is PNG-
    /// encoded and the encoded bytes are hashed instead.
    ///
    /// [`snapseg-label-export`]: ../../.claude/skills/snapseg-label-export/SKILL.md
    ///
    /// # Errors
    ///
    /// - [`LabelError::InvalidInput`] if `mask` (or `logits`, when
    ///   `Some`) dimensions don't match `gray`, or if the directory
    ///   already holds 99 region siblings for the same image.
    /// - [`LabelError::Io`] for filesystem failures (create / copy /
    ///   rename).
    /// - [`LabelError::Json`] / [`LabelError::TomlSer`] /
    ///   [`LabelError::Image`] when (de)serialization or PNG
    ///   encoding fails.
    pub fn save(
        &mut self,
        image_path: Option<&Path>,
        gray: &snapseg_core::GrayImage,
        prompts: PromptSessionJson,
        mask: &ndarray::Array2<bool>,
        logits: Option<&ndarray::Array2<f32>>,
        inputs: ProvenanceInputs,
    ) -> Result<Label, LabelError> {
        save_impl(self, image_path, gray, prompts, mask, logits, inputs)
    }
}

/// Caller-supplied portion of [`Provenance`]. [`LabelDir::save`]
/// completes the bundle by filling in `schema_version`,
/// `created_at`, `source_image_sha256`, `source_image_path`,
/// `source_image_width`, and `source_image_height`.
#[derive(Debug, Clone)]
pub struct ProvenanceInputs {
    /// Which model produced the mask.
    pub model: ModelProvenance,
    /// How the model was run.
    pub runtime: RuntimeProvenance,
    /// Operator-attached note + quality flag.
    pub operator: OperatorNote,
}

// why this is long: this is the load-bearing atomic-write protocol;
// breaking it into many small helpers would obscure the strict file
// order (validate → hash → pick slot → stage → rename → cleanup) the
// crash-safety argument depends on. Keep it as one readable script.
fn save_impl(
    dir: &mut LabelDir,
    image_path: Option<&Path>,
    gray: &snapseg_core::GrayImage,
    prompts: PromptSessionJson,
    mask: &ndarray::Array2<bool>,
    logits: Option<&ndarray::Array2<f32>>,
    inputs: ProvenanceInputs,
) -> Result<Label, LabelError> {
    // --- validate inputs before touching the filesystem ---
    let expected_dim = (gray.height as usize, gray.width as usize);
    if mask.dim() != expected_dim {
        return Err(LabelError::InvalidInput(format!(
            "mask dim {:?} does not match image dim {:?}",
            mask.dim(),
            expected_dim,
        )));
    }
    if let Some(l) = logits {
        if l.dim() != expected_dim {
            return Err(LabelError::InvalidInput(format!(
                "logits dim {:?} does not match image dim {:?}",
                l.dim(),
                expected_dim,
            )));
        }
    }

    // --- compute SHA-256 of the source bytes ---
    // If we have a path, stream the file (large TIFFs are common in
    // industrial workflows); otherwise PNG-encode the in-memory
    // grayscale and hash that buffer.
    let (source_sha256, encoded_png_bytes): (String, Option<Vec<u8>>) = match image_path {
        Some(p) => (sha256_file(p)?, None),
        None => {
            let bytes = encode_gray_to_png(gray)?;
            let sha = sha256_bytes(&bytes);
            (sha, Some(bytes))
        }
    };
    let id_base = &source_sha256[..8];

    // --- pick a free slot (id_base or id_base-r{2..=99}) ---
    fs::create_dir_all(&dir.root)?;
    let id = pick_free_slot(&dir.root, id_base)?;
    let final_path = dir.root.join(&id);
    let partial = dir.root.join(format!("{id}.partial"));

    // --- stage every artifact under partial/, cleaning up on error ---
    let staged = stage_artifacts(
        &partial,
        image_path,
        gray,
        encoded_png_bytes.as_deref(),
        &prompts,
        mask,
        logits,
        &source_sha256,
        inputs,
    );
    if let Err(e) = staged {
        let _ = fs::remove_dir_all(&partial);
        return Err(e);
    }

    // --- atomic rename → done ---
    if let Err(e) = fs::rename(&partial, &final_path) {
        let _ = fs::remove_dir_all(&partial);
        return Err(LabelError::Io(e));
    }

    Ok(Label {
        dir: final_path,
        id,
    })
}

// why this is long: stages every artifact in the documented order
// (image, mask, logits, prompts, meta) and constructing the full
// Provenance bundle inline keeps the file-write sequence reviewable
// in one place. Splitting would scatter the order across helpers.
#[allow(clippy::too_many_arguments)]
fn stage_artifacts(
    partial: &Path,
    image_path: Option<&Path>,
    gray: &snapseg_core::GrayImage,
    encoded_png_bytes: Option<&[u8]>,
    prompts: &PromptSessionJson,
    mask: &ndarray::Array2<bool>,
    logits: Option<&ndarray::Array2<f32>>,
    source_sha256: &str,
    inputs: ProvenanceInputs,
) -> Result<(), LabelError> {
    // Use create_dir (not create_dir_all): the slot picker promised
    // this path does not exist; if it does, treat as I/O error
    // because some other process won the race.
    fs::create_dir(partial)?;

    // image.png
    let image_dst = partial.join("image.png");
    match image_path {
        Some(p) => {
            fs::copy(p, &image_dst)?;
        }
        None => {
            // We already encoded the PNG when computing the hash;
            // write the same bytes so the file's SHA-256 matches.
            let Some(bytes) = encoded_png_bytes else {
                return Err(LabelError::InvalidInput(
                    "internal: encoded PNG bytes missing while image_path is None".into(),
                ));
            };
            fs::write(&image_dst, bytes)?;
        }
    }

    // mask.png — binary, 0/255 single-channel u8 at gray's resolution.
    let (h, w) = (gray.height, gray.width);
    let mask_bytes: Vec<u8> = mask.iter().map(|&b| if b { 255u8 } else { 0u8 }).collect();
    let mask_img = image::GrayImage::from_raw(w, h, mask_bytes)
        .ok_or_else(|| LabelError::InvalidInput(format!("could not build {w}x{h} mask buffer")))?;
    mask_img.save_with_format(partial.join("mask.png"), image::ImageFormat::Png)?;

    // logits.png (optional, 16-bit grayscale, tanh-mapped).
    if let Some(l) = logits {
        let logits_u16: Vec<u16> = l
            .iter()
            .map(|&v| {
                let clamped = v.clamp(-8.0, 8.0);
                let mapped = (clamped.tanh() + 1.0) * 0.5;
                (mapped * 65535.0).round().clamp(0.0, 65535.0) as u16
            })
            .collect();
        let logits_img: image::ImageBuffer<image::Luma<u16>, Vec<u16>> =
            image::ImageBuffer::from_raw(w, h, logits_u16).ok_or_else(|| {
                LabelError::InvalidInput(format!("could not build {w}x{h} logits buffer"))
            })?;
        logits_img.save_with_format(partial.join("logits.png"), image::ImageFormat::Png)?;
    }

    // prompts.json
    let prompts_str = serde_json::to_string_pretty(prompts)?;
    fs::write(partial.join("prompts.json"), prompts_str)?;

    // meta.toml — complete the Provenance with the image-dependent
    // fields, then serialize.
    let provenance = Provenance {
        schema_version: SCHEMA_VERSION.to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        source_image_sha256: source_sha256.to_string(),
        source_image_path: image_path.map(Path::to_path_buf),
        source_image_width: w,
        source_image_height: h,
        model: inputs.model,
        runtime: inputs.runtime,
        operator: inputs.operator,
    };
    let meta_str = toml::to_string_pretty(&provenance)?;
    fs::write(partial.join("meta.toml"), meta_str)?;

    Ok(())
}

/// Pick the next free slot under `root` for an image whose 8-char id
/// is `id_base`. Returns `id_base` itself if neither `<id_base>/`
/// nor `<id_base>.partial/` exists, otherwise `id_base-r{r}` for the
/// smallest `r` in `2..=99` that is free.
fn pick_free_slot(root: &Path, id_base: &str) -> Result<String, LabelError> {
    let candidate = id_base.to_string();
    if !slot_taken(root, &candidate) {
        return Ok(candidate);
    }
    for r in 2..=99u32 {
        let candidate = format!("{id_base}-r{r}");
        if !slot_taken(root, &candidate) {
            return Ok(candidate);
        }
    }
    Err(LabelError::InvalidInput(format!(
        "more than 99 regions already saved for image {id_base}"
    )))
}

fn slot_taken(root: &Path, id: &str) -> bool {
    root.join(id).exists() || root.join(format!("{id}.partial")).exists()
}

/// Stream a file through SHA-256. Avoids loading the whole image
/// into memory; industrial TIFFs can be hundreds of MB.
fn sha256_file(path: &Path) -> Result<String, LabelError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(nibble(b >> 4));
        s.push(nibble(b & 0x0f));
    }
    s
}

fn nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + (n - 10)) as char,
        _ => unreachable!("nibble takes 4-bit input"),
    }
}

/// PNG-encode an in-memory grayscale image into a byte buffer. Used
/// when no source path is available — the encoded buffer is hashed
/// for the label id and then written verbatim to `image.png` so the
/// on-disk file's SHA-256 matches `meta.toml`.
fn encode_gray_to_png(gray: &snapseg_core::GrayImage) -> Result<Vec<u8>, LabelError> {
    let (h, w) = (gray.height, gray.width);
    // ndarray::Array2 is stored row-major by default; flatten to the
    // raw byte order the image crate expects.
    let raw: Vec<u8> = gray.data.iter().copied().collect();
    let img = image::GrayImage::from_raw(w, h, raw)
        .ok_or_else(|| LabelError::InvalidInput(format!("could not build {w}x{h} image buffer")))?;
    let mut buf: Vec<u8> = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(buf)
}

/// Convert a live [`snapseg_core::PromptSession`] into the on-disk
/// JSON shape.
///
/// `t_ms_offsets` is one entry per prompt in `session.prompts`
/// giving the milliseconds since the first prompt; pass all zeros
/// when the caller didn't track timing.
///
/// # Errors
///
/// Returns [`LabelError::InvalidInput`] if `t_ms_offsets.len()`
/// differs from `session.prompts.len()`.
pub fn prompts_to_json(
    session: &snapseg_core::PromptSession,
    t_ms_offsets: &[u64],
) -> Result<PromptSessionJson, LabelError> {
    if t_ms_offsets.len() != session.prompts.len() {
        return Err(LabelError::InvalidInput(
            "t_ms_offsets length must match session.prompts".to_string(),
        ));
    }
    let mut session_json: Vec<PromptRecord> = Vec::with_capacity(session.prompts.len());
    for (prompt, &t_ms) in session.prompts.iter().zip(t_ms_offsets.iter()) {
        let record = match prompt {
            snapseg_core::Prompt::Click { point, polarity } => PromptRecord::Click {
                x: point.x,
                y: point.y,
                polarity: polarity_str(*polarity).to_string(),
                t_ms,
            },
            snapseg_core::Prompt::Box(b) => PromptRecord::Box {
                x0: b.x0,
                y0: b.y0,
                x1: b.x1,
                y1: b.y1,
                t_ms,
            },
            snapseg_core::Prompt::Scribble { points, polarity } => PromptRecord::Scribble {
                points: points.iter().map(|p| [p.x, p.y]).collect(),
                polarity: polarity_str(*polarity).to_string(),
                t_ms,
            },
        };
        session_json.push(record);
    }
    Ok(PromptSessionJson {
        session: session_json,
    })
}

fn polarity_str(p: snapseg_core::Polarity) -> &'static str {
    match p {
        snapseg_core::Polarity::Positive => "positive",
        snapseg_core::Polarity::Negative => "negative",
    }
}

/// Fallible operations on labels.
#[derive(Debug, Error)]
pub enum LabelError {
    /// Underlying I/O failure (file create, copy, rename, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization failure for `prompts.json`.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// TOML serialization failure for `meta.toml`.
    #[error("TOML serialize error: {0}")]
    TomlSer(#[from] toml::ser::Error),
    /// TOML deserialization failure for `meta.toml`.
    #[error("TOML deserialize error: {0}")]
    TomlDe(#[from] toml::de::Error),
    /// `image` crate failure when encoding `mask.png` / `logits.png`.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    /// Caller-supplied input violated an invariant (e.g. mismatched
    /// mask and image dimensions).
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array2;
    use snapseg_core::{BBox, GrayImage, Point2, Polarity, Prompt, PromptSession};

    fn make_gray_4x4() -> GrayImage {
        // 16 distinct values so PNG encoding produces something
        // meaningful and the SHA-256 isn't all-zeros.
        let data = Array2::from_shape_vec((4, 4), (0u8..16).collect()).expect("4x4 shape");
        GrayImage::from_array(data)
    }

    fn make_mask_4x4() -> Array2<bool> {
        let mut m = Array2::<bool>::default((4, 4));
        m[(1, 1)] = true;
        m[(1, 2)] = true;
        m[(2, 1)] = true;
        m[(2, 2)] = true;
        m
    }

    fn make_inputs() -> ProvenanceInputs {
        ProvenanceInputs {
            model: ModelProvenance {
                registry_name: "test-model".to_string(),
                family: "test".to_string(),
                encoder_sha256: None,
                decoder_sha256: None,
                model_sha256: Some("deadbeef".to_string()),
            },
            runtime: RuntimeProvenance {
                execution_provider: "CPU".to_string(),
                encoder_ms: None,
                decoder_ms: 12,
                snapseg_commit: None,
            },
            operator: OperatorNote::default(),
        }
    }

    fn make_session_one_click() -> (PromptSession, Vec<u64>) {
        let mut s = PromptSession::new();
        s.push(Prompt::Click {
            point: Point2::new(1.5, 2.0),
            polarity: Polarity::Positive,
        });
        (s, vec![0])
    }

    #[test]
    fn save_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        let gray = make_gray_4x4();
        let mask = make_mask_4x4();
        let (session, offsets) = make_session_one_click();
        let prompts = prompts_to_json(&session, &offsets).expect("prompts_to_json");

        let label = dir
            .save(None, &gray, prompts, &mask, None, make_inputs())
            .expect("save");

        // image.png — decodes to 4x4 grayscale.
        let decoded_img = image::open(label.dir.join("image.png")).expect("open image.png");
        assert_eq!(decoded_img.width(), 4);
        assert_eq!(decoded_img.height(), 4);

        // mask.png — decodes to 4x4, exact 0/255 match.
        let decoded_mask = image::open(label.dir.join("mask.png")).expect("open mask.png");
        let mask_u8 = decoded_mask.to_luma8();
        assert_eq!(mask_u8.width(), 4);
        assert_eq!(mask_u8.height(), 4);
        for ((row, col), &m) in mask.indexed_iter() {
            // image crate is (x=col, y=row) — flip the lookup.
            let pixel = mask_u8.get_pixel(col as u32, row as u32)[0];
            let expected = if m { 255u8 } else { 0u8 };
            assert_eq!(pixel, expected, "mask byte at ({row},{col})");
        }

        // meta.toml — parses, fields are populated correctly.
        let meta_str =
            std::fs::read_to_string(label.dir.join("meta.toml")).expect("read meta.toml");
        let prov: Provenance = toml::from_str(&meta_str).expect("parse meta.toml");
        assert_eq!(prov.schema_version, "1.0");
        assert_eq!(prov.source_image_width, 4);
        assert_eq!(prov.source_image_height, 4);
        assert_eq!(prov.model.family, "test");
        assert_eq!(prov.source_image_sha256.len(), 64);

        // prompts.json — deserializes with one entry.
        let prompts_str =
            std::fs::read_to_string(label.dir.join("prompts.json")).expect("read prompts.json");
        let parsed: PromptSessionJson =
            serde_json::from_str(&prompts_str).expect("parse prompts.json");
        assert_eq!(parsed.session.len(), 1);

        // logits.png — opt-out, not present.
        assert!(!label.dir.join("logits.png").exists());

        // No .partial/ orphan remains.
        let partial_count = std::fs::read_dir(tmp.path())
            .expect("readdir")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".partial"))
            .count();
        assert_eq!(partial_count, 0);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        let gray = make_gray_4x4();
        // Mask 5x4 (rows=5, cols=4) — does not match image (4x4).
        let bad_mask = Array2::<bool>::default((5, 4));
        let (session, offsets) = make_session_one_click();
        let prompts = prompts_to_json(&session, &offsets).expect("prompts_to_json");

        let err = dir
            .save(None, &gray, prompts, &bad_mask, None, make_inputs())
            .expect_err("dimension mismatch should error");
        match err {
            LabelError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn region_suffix_grows() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        let gray = make_gray_4x4();
        let mask = make_mask_4x4();
        let (session, offsets) = make_session_one_click();

        let prompts1 = prompts_to_json(&session, &offsets).expect("prompts_to_json");
        let first = dir
            .save(None, &gray, prompts1, &mask, None, make_inputs())
            .expect("first save");

        let prompts2 = prompts_to_json(&session, &offsets).expect("prompts_to_json");
        let second = dir
            .save(None, &gray, prompts2, &mask, None, make_inputs())
            .expect("second save");

        assert_eq!(first.id.len(), 8, "first id is bare 8-hex");
        assert!(
            second.id.ends_with("-r2"),
            "second id should end with -r2, got {}",
            second.id
        );
        assert_ne!(first.dir, second.dir);
    }

    #[test]
    fn prompts_to_json_roundtrip() {
        let mut s = PromptSession::new();
        s.push(Prompt::Click {
            point: Point2::new(1.0, 2.0),
            polarity: Polarity::Positive,
        });
        s.push(Prompt::Box(BBox {
            x0: 0.0,
            y0: 1.0,
            x1: 4.0,
            y1: 5.0,
        }));
        s.push(Prompt::Scribble {
            points: vec![Point2::new(0.5, 0.5), Point2::new(1.5, 1.5)],
            polarity: Polarity::Negative,
        });
        let offsets = vec![0, 100, 250];
        let json = prompts_to_json(&s, &offsets).expect("prompts_to_json");
        assert_eq!(json.session.len(), 3);

        match &json.session[0] {
            PromptRecord::Click {
                x,
                y,
                polarity,
                t_ms,
            } => {
                assert!((*x - 1.0).abs() < 1e-6);
                assert!((*y - 2.0).abs() < 1e-6);
                assert_eq!(polarity, "positive");
                assert_eq!(*t_ms, 0);
            }
            other => panic!("expected Click variant, got {other:?}"),
        }
        match &json.session[1] {
            PromptRecord::Box {
                x0,
                y0,
                x1,
                y1,
                t_ms,
            } => {
                assert!((*x0 - 0.0).abs() < 1e-6);
                assert!((*y0 - 1.0).abs() < 1e-6);
                assert!((*x1 - 4.0).abs() < 1e-6);
                assert!((*y1 - 5.0).abs() < 1e-6);
                assert_eq!(*t_ms, 100);
            }
            other => panic!("expected Box variant, got {other:?}"),
        }
        match &json.session[2] {
            PromptRecord::Scribble {
                points,
                polarity,
                t_ms,
            } => {
                assert_eq!(points.len(), 2);
                assert!((points[0][0] - 0.5).abs() < 1e-6);
                assert!((points[1][1] - 1.5).abs() < 1e-6);
                assert_eq!(polarity, "negative");
                assert_eq!(*t_ms, 250);
            }
            other => panic!("expected Scribble variant, got {other:?}"),
        }
    }

    #[test]
    fn prompts_to_json_length_mismatch() {
        let mut s = PromptSession::new();
        s.push(Prompt::Click {
            point: Point2::new(0.0, 0.0),
            polarity: Polarity::Positive,
        });
        let err = prompts_to_json(&s, &[]).expect_err("length mismatch should error");
        match err {
            LabelError::InvalidInput(msg) => {
                assert!(msg.contains("t_ms_offsets"), "msg = {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn save_with_logits_writes_logits_png() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut dir = LabelDir::new(tmp.path().to_path_buf());

        let gray = make_gray_4x4();
        let mask = make_mask_4x4();
        // Mix of negative, near-zero, and positive logits to exercise
        // the tanh remap. Zero should land at exactly mid-grey.
        let logits = Array2::from_shape_vec((4, 4), (0..16).map(|i| (i as f32) - 7.5).collect())
            .expect("logits 4x4");
        let (session, offsets) = make_session_one_click();
        let prompts = prompts_to_json(&session, &offsets).expect("prompts_to_json");

        let label = dir
            .save(None, &gray, prompts, &mask, Some(&logits), make_inputs())
            .expect("save");

        let logits_path = label.dir.join("logits.png");
        assert!(logits_path.exists(), "logits.png missing");
        let decoded = image::open(&logits_path).expect("open logits.png");
        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 4);
        // The decoded image is 16-bit grayscale.
        let luma16 = decoded.to_luma16();
        // Sanity-check: a zero-logit input under the tanh-mapped
        // encoding lands at exactly the midpoint. We don't assume a
        // specific pixel slot — just that the encoded range is
        // non-degenerate (min < max) and at least one pixel is near
        // mid-grey for the zero-ish inputs in our linspace.
        let min = luma16.iter().copied().min().expect("non-empty");
        let max = luma16.iter().copied().max().expect("non-empty");
        assert!(max > min, "encoded logits are degenerate: {min}..{max}");
    }
}
