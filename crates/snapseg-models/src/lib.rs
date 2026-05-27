//! Per-family adapters implementing
//! [`snapseg_core::InteractiveSegmenter`]. Each module knows how to feed
//! one model family (RITM, FocalClick, MobileSAM, ...) — preprocessing,
//! click-map encoding, output decoding.
//!
//! Adapters share a [`snapseg_runtime::Backend`] and a small set of
//! preprocessing helpers from `snapseg_runtime::preprocess`.

/// FocalClick (Chen 2022) — click-driven with local-crop refinement.
/// Skeleton; full implementation lands per `snapseg-add-model`.
pub mod focalclick;
/// MobileSAM (Zhang 2023) — distilled SAM with click + box prompts.
/// Production adapter; the worked example for other families.
pub mod mobile_sam;
/// RITM (Sofiiuk 2021) — click-driven HRNet-based interactive
/// segmenter. Skeleton; full implementation lands per `snapseg-add-model`.
pub mod ritm;

use snapseg_core::SegError;

/// Dispatch helper: given a `family` string from a registry entry,
/// returns the human-readable label every adapter exposes via
/// [`InteractiveSegmenter::name`](snapseg_core::InteractiveSegmenter::name).
pub fn family_label(family: &str) -> Result<&'static str, SegError> {
    Ok(match family {
        "ritm" => "RITM",
        "focalclick" => "FocalClick",
        "mobile_sam" => "MobileSAM",
        "efficient_sam" => "EfficientSAM",
        "sam2_tiny" => "SAM2-Tiny",
        other => {
            return Err(SegError::InvalidInput(format!(
                "unknown model family `{other}`"
            )));
        }
    })
}
