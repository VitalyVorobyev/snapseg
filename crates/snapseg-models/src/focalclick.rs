//! FocalClick adapter (Chen et al., CVPR 2022). Click-driven like RITM but
//! supports a local-refinement pass on a small crop around the latest
//! click — much faster subsequent interactions on CPU.

use snapseg_core::{
    Capabilities, GrayImage, InteractiveSegmenter, PromptSession, SegError, SegmentationResult,
};

pub struct FocalClickSegmenter {
    name: String,
    input_size: (u32, u32),
    image: Option<GrayImage>,
}

impl FocalClickSegmenter {
    pub fn new(name: impl Into<String>, input_size: (u32, u32)) -> Self {
        Self {
            name: name.into(),
            input_size,
            image: None,
        }
    }
}

impl InteractiveSegmenter for FocalClickSegmenter {
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
        Err(SegError::Backend("FocalClick adapter not yet wired to ort".into()))
    }
}
