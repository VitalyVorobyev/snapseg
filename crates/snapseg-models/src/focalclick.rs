//! FocalClick adapter (Chen et al., CVPR 2022). Click-driven like RITM,
//! but with a two-stage architecture: a coarse global pass that
//! produces an image-wide mask, followed by an optional local
//! refinement pass on a small crop around the latest click. The local
//! pass is what makes subsequent interactions cheap on CPU.
//!
//! ## ONNX I/O contract
//!
//! Built against the upstream `XavierCHEN34/ClickSEG` FocalClick ONNX
//! export convention. Two operating modes share the same global
//! network signature; the local refinement is an optional second file.
//!
//! Global pass (used for the first click and whenever the user moves
//! the focus region):
//! ```text
//! in   `image`       : float32 [1, 3, H, W]   ImageNet-normalised RGB
//! in   `click_map`   : float32 [1, 2, H, W]   channel 0 = positive Gaussian disks,
//!                                              channel 1 = negative Gaussian disks
//! in   `prev_mask`   : float32 [1, 1, H, W]   previous global logits
//!                                              (zeros on first call)
//! out  `instances`   : float32 [1, 1, H, W]   coarse logits; threshold at 0
//! ```
//!
//! Local refinement pass (optional; consumes a crop around the latest
//! click and writes a refined logit patch back into the global mask):
//! ```text
//! in   `image_focus`     : float32 [1, 3, h, w]   cropped + resized
//! in   `click_map_focus` : float32 [1, 2, h, w]   clicks intersecting the crop
//! in   `prev_mask_focus` : float32 [1, 1, h, w]   coarse logits cropped
//! out  `instances_focus` : float32 [1, 1, h, w]   refined logits in the crop
//! ```
//!
//! H and W default to the registry-declared `input_size` (1024×1024 in
//! the bundled entry); h and w are the refinement crop size (commonly
//! 256). Community exports sometimes substitute `images` / `points`
//! (with a `[1, N, 3]` (y, x, polarity) layout) or `pred` / `pred_local`
//! for the output names; the real adapter must resolve those by
//! name-substring matching when it is wired up.
//!
//! Status: **stub — body unimplemented**. `segment()` returns
//! `SegError::Backend("not yet wired")`. The structural surface
//! (constructor, capabilities, set_image) is final so M6's body swap
//! does not become an API rewrite.

use std::collections::HashMap;
use std::path::PathBuf;

use snapseg_core::{
    Capabilities, GrayImage, InteractiveSegmenter, PromptSession, SegError, SegmentationResult,
};
use snapseg_runtime::RuntimeConfig;

/// FocalClick `InteractiveSegmenter`. Stub; full wiring lands when the
/// adapter is needed (model-adapter-integrator task).
pub struct FocalClickSegmenter {
    name: String,
    input_size: (u32, u32),
    image: Option<GrayImage>,
}

impl FocalClickSegmenter {
    /// Construct a placeholder adapter from a registry-resolved part
    /// map. `input_size` is the square edge length the global network
    /// was exported for (FocalClick is typically 1024). The stub
    /// ignores `parts` and `config` because no `Backend` is loaded yet.
    ///
    /// # Errors
    ///
    /// Currently infallible — always returns `Ok(Self { ... })`. When
    /// the real implementation lands, it will return
    /// [`SegError::InvalidInput`] if the expected `"global"` part (and
    /// optionally `"local"`) is missing from `parts`, and
    /// [`SegError::Backend`] if the underlying ort session fails to
    /// load.
    pub fn from_parts(
        name: String,
        _parts: &HashMap<String, PathBuf>,
        input_size: u32,
        _config: &RuntimeConfig,
    ) -> Result<Self, SegError> {
        Ok(Self {
            name,
            input_size: (input_size, input_size),
            image: None,
        })
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
        Err(SegError::Backend(
            "FocalClick adapter not yet wired to ort".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_constructs_stub_with_advertised_capabilities() {
        let parts: HashMap<String, PathBuf> = HashMap::new();
        let cfg = RuntimeConfig::default();
        let seg = FocalClickSegmenter::from_parts("focalclick-test".into(), &parts, 1024, &cfg)
            .expect("stub from_parts is infallible");
        let caps = seg.capabilities();
        assert!(caps.positive_clicks);
        assert!(caps.negative_clicks);
        assert!(!caps.bbox);
        assert!(caps.scribble);
        assert!(caps.mask_input);
        assert_eq!(caps.recommended_input_size, (1024, 1024));
    }
}
