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

pub mod coco;
pub use coco::ConversionReport;
mod prompts;
mod save;
mod types;

pub use prompts::prompts_to_json;
pub use save::ProvenanceInputs;
pub use types::{
    Label, LabelDir, LabelError, LabelQuality, ModelProvenance, OperatorNote, PolygonJson,
    PromptRecord, PromptSessionJson, Provenance, RuntimeProvenance, SCHEMA_VERSION,
};
