//! RITM adapter (Reviving Iterative Training with Mask guidance,
//! Sofiiuk et al. 2021). HRNet-backboned, click-driven interactive
//! segmenter: each user click triggers exactly one network call that
//! consumes the full image, the accumulated click history rasterised
//! into a 2-channel positive/negative Gaussian disk map, and the
//! previous-iteration mask as a refinement prior.
//!
//! ## ONNX I/O contract
//!
//! Built against the `saic-vul/ritm_interactive_segmentation` upstream
//! ONNX export convention. Single network file; encoder + decoder are
//! fused into one graph.
//!
//! ```text
//! in   `image`       : float32 [1, 3, H, W]   ImageNet-normalised RGB
//! in   `click_map`   : float32 [1, 2, H, W]   channel 0 = positive Gaussian disks,
//!                                              channel 1 = negative Gaussian disks
//!                                              (σ = 5 px on the resized canvas)
//! in   `prev_mask`   : float32 [1, 1, H, W]   previous logits (zeros on first call)
//! out  `instances`   : float32 [1, 1, H, W]   logits; threshold at 0 for the mask
//! ```
//!
//! H and W default to the registry-declared `input_size` (1024×1024 in
//! the bundled entry). Community RITM exports occasionally substitute
//! `images` / `points` (with a `[1, N, 3]` (y, x, polarity) layout
//! instead of a rasterised click-map) or `pred` for the output name;
//! the real adapter must resolve those by name-substring matching when
//! it is wired up.
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

/// Skeleton adapter. The real implementation will own a
/// [`snapseg_runtime::Backend`], cache the source image, encode each
/// click as a Gaussian disk into a 2-channel click-map (pos / neg),
/// and run the network on `[image, click_map, prev_mask]`.
pub struct RitmSegmenter {
    name: String,
    input_size: (u32, u32),
    image: Option<GrayImage>,
}

impl RitmSegmenter {
    /// Construct a placeholder adapter from a registry-resolved part
    /// map. `input_size` is the square edge length the network was
    /// exported for (RITM is typically 1024). The stub ignores `parts`
    /// and `config` because no `Backend` is loaded yet.
    ///
    /// # Errors
    ///
    /// Currently infallible — always returns `Ok(Self { ... })`. When
    /// the real implementation lands, it will return
    /// [`SegError::InvalidInput`] if the expected `"model"` part is
    /// missing from `parts`, and [`SegError::Backend`] if the
    /// underlying ort session fails to load.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_constructs_stub_with_advertised_capabilities() {
        let parts: HashMap<String, PathBuf> = HashMap::new();
        let cfg = RuntimeConfig::default();
        let seg = RitmSegmenter::from_parts("ritm-test".into(), &parts, 1024, &cfg)
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
