//! RITM adapter (Reviving Iterative Training with Mask guidance,
//! Sofiiuk et al. 2021). HRNet backbone, click-driven; one network call
//! per user interaction.

use snapseg_core::{
    Capabilities, GrayImage, InteractiveSegmenter, PromptSession, SegError, SegmentationResult,
};

/// Skeleton adapter. The real implementation will own a
/// [`snapseg_runtime::Backend`], cache the source image, encode each click
/// as a Gaussian disk into a 2-channel click-map (pos / neg), and run the
/// network on `[image_rgb, click_pos, click_neg]`.
pub struct RitmSegmenter {
    name: String,
    input_size: (u32, u32),
    image: Option<GrayImage>,
}

impl RitmSegmenter {
    /// Construct a placeholder adapter that declares its
    /// [`Capabilities`] honestly but errors on `segment()`. Useful for
    /// the model picker UI before the real implementation lands.
    pub fn new(name: impl Into<String>, input_size: (u32, u32)) -> Self {
        Self {
            name: name.into(),
            input_size,
            image: None,
        }
    }
}

impl InteractiveSegmenter for RitmSegmenter {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            positive_clicks: true,
            negative_clicks: true,
            bbox: false,
            scribble: true,
            mask_input: true,
            recommended_input_size: self.input_size,
        }
    }

    fn set_image(&mut self, image: &GrayImage) -> Result<(), SegError> {
        self.image = Some(image.clone());
        Ok(())
    }

    fn segment(&mut self, _session: &PromptSession) -> Result<SegmentationResult, SegError> {
        Err(SegError::Backend(
            "RITM adapter not yet wired to ort".into(),
        ))
    }
}
