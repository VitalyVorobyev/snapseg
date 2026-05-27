//! MobileSAM adapter (Zhang et al. 2023). Distilled SAM with a tiny image
//! encoder (~10MB) plus the original SAM mask decoder. Click + box prompts;
//! image embedding is computed once per image, then each prompt is a cheap
//! decoder pass.

use snapseg_core::{
    Capabilities, GrayImage, InteractiveSegmenter, PromptSession, SegError, SegmentationResult,
};

pub struct MobileSamSegmenter {
    name: String,
    input_size: (u32, u32),
    image: Option<GrayImage>,
}

impl MobileSamSegmenter {
    pub fn new(name: impl Into<String>, input_size: (u32, u32)) -> Self {
        Self {
            name: name.into(),
            input_size,
            image: None,
        }
    }
}

impl InteractiveSegmenter for MobileSamSegmenter {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            positive_clicks: true,
            negative_clicks: true,
            bbox: true,
            scribble: false,
            mask_input: true,
            recommended_input_size: self.input_size,
        }
    }

    fn set_image(&mut self, image: &GrayImage) -> Result<(), SegError> {
        self.image = Some(image.clone());
        Ok(())
    }

    fn segment(&mut self, _session: &PromptSession) -> Result<SegmentationResult, SegError> {
        Err(SegError::Backend("MobileSAM adapter not yet wired to ort".into()))
    }
}
