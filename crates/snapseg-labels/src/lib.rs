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

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
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
    /// `LabelDir::save`'s job on first use (added in M2-T03).
    pub fn new(root: PathBuf) -> Self {
        Self { root }
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
