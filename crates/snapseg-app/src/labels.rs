//! App-side glue between the in-memory [`SnapsegApp`] state and the
//! on-disk format owned by the `snapseg-labels` crate.
//!
//! The split keeps `SnapsegApp` free of label-format details: the UI
//! holds inference state (mask, logits, timing), this module turns
//! that state into a [`snapseg_labels::ProvenanceInputs`] and calls
//! [`snapseg_labels::LabelDir::save`]. The orchestration is the only
//! responsibility of this module — file I/O and atomicity live in the
//! labels crate.

use snapseg_labels::{
    Label, LabelError, ModelProvenance, OperatorNote, ProvenanceInputs, RuntimeProvenance,
    prompts_to_json,
};

use crate::app::SnapsegApp;

/// Build the caller-side portion of [`snapseg_labels::Provenance`] from
/// the app's current state. Fields the caller cannot know
/// (`source_image_sha256`, `source_image_width`, `source_image_height`,
/// `source_image_path`, `schema_version`, `created_at`) are filled in by
/// [`snapseg_labels::LabelDir::save`].
///
/// Caller is responsible for ensuring the app actually has a segmenter
/// loaded — this function falls back to empty strings rather than
/// erroring, on the assumption that the UI gates the Save button on the
/// same precondition.
pub(crate) fn build_provenance_inputs(app: &SnapsegApp) -> ProvenanceInputs {
    let registry_name = app.segmenter_registry_name.clone().unwrap_or_default();
    let family = app.segmenter_family.clone().unwrap_or_default();
    ProvenanceInputs {
        model: ModelProvenance {
            registry_name,
            family,
            encoder_sha256: app.encoder_sha256.clone(),
            decoder_sha256: app.decoder_sha256.clone(),
            model_sha256: None,
        },
        runtime: RuntimeProvenance {
            // TODO(M5): wire from Backend::execution_provider when the
            // inference path is threaded; today everything is CPU.
            execution_provider: "CPU".to_string(),
            encoder_ms: app.last_encoder_ms,
            decoder_ms: app.last_inference_ms.unwrap_or(0),
            snapseg_commit: option_env!("SNAPSEG_COMMIT_SHA")
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string()),
        },
        operator: OperatorNote {
            note: app.pending_note.clone(),
            quality: app.pending_quality,
        },
    }
}

/// Persist the latest segmentation as a label.
///
/// Returns the new [`Label`] (with `Label::id` and `Label::dir`) on
/// success. The caller (the side panel's "Save label" button) should
/// gate the action on `app.image.is_some() && app.last_mask.is_some()
/// && app.segmenter_family.is_some()` — the missing-state checks below
/// exist for defence in depth, not as the primary UX.
///
/// # Errors
///
/// - Any [`LabelError`] from [`snapseg_labels::LabelDir::save`] (I/O,
///   serialization, image encoding, dimension mismatch).
/// - [`LabelError::InvalidInput`] with `"no image"`, `"no mask"`, or
///   `"no segmenter"` if the app isn't in a savable state.
pub(crate) fn save_current_label(app: &mut SnapsegApp) -> Result<Label, LabelError> {
    let image = app
        .image
        .as_ref()
        .ok_or_else(|| LabelError::InvalidInput("no image loaded".into()))?;
    let mask = app
        .last_mask
        .as_ref()
        .ok_or_else(|| LabelError::InvalidInput("no mask available".into()))?;
    if app.segmenter_family.is_none() {
        return Err(LabelError::InvalidInput("no segmenter loaded".into()));
    }

    let prompts = prompts_to_json(&app.session, &app.prompt_t_ms)?;
    let inputs = build_provenance_inputs(app);
    let logits = app.last_logits.as_ref();

    app.label_dir.save(
        Some(image.path.as_path()),
        &image.gray,
        prompts,
        mask,
        logits,
        inputs,
    )
}
