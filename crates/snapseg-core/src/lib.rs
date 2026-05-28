//! Core contracts for `snapseg`: the data types every adapter, the runtime,
//! and the UI agree on. No ONNX, no GUI, no I/O — just shapes and traits.

use std::time::Duration;

use ndarray::{Array2, Array3};
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 2D point in image coordinates. Held as `f32` so subpixel-refined
/// boundaries and floating point click positions share one type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned bounding box prompt, in image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

/// Polarity of a click or stroke: does it indicate object (foreground) or
/// background?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Polarity {
    Positive,
    Negative,
}

/// A single user interaction. Adapters consume an ordered list of these via
/// [`PromptSession`] — order matters for models that condition on click
/// history (RITM, FocalClick).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Prompt {
    Click {
        point: Point2,
        polarity: Polarity,
    },
    Box(BBox),
    Scribble {
        points: Vec<Point2>,
        polarity: Polarity,
    },
}

/// Accumulated user input for one segmentation pass. `prev_logits` lets
/// mask-conditioned models (e.g. SAM-family second pass, FocalClick local
/// refinement) reuse the previous output instead of starting from scratch.
#[derive(Debug, Default, Clone)]
pub struct PromptSession {
    pub prompts: Vec<Prompt>,
    pub prev_logits: Option<Array2<f32>>,
}

impl PromptSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, prompt: Prompt) {
        self.prompts.push(prompt);
    }

    pub fn clear(&mut self) {
        self.prompts.clear();
        self.prev_logits = None;
    }

    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

/// What kinds of input a given segmenter actually supports. The UI reads
/// this to enable/disable the tool palette per-model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    pub positive_clicks: bool,
    pub negative_clicks: bool,
    pub bbox: bool,
    pub scribble: bool,
    /// Whether the model accepts a prior mask / previous logits as input.
    pub mask_input: bool,
    /// Native input resolution the model was trained for. Adapters
    /// resize/letterbox to this size, then upscale the output mask.
    pub recommended_input_size: (u32, u32),
}

impl Capabilities {
    /// Convenience: does this segmenter accept *any* form of click input?
    pub fn supports_clicks(&self) -> bool {
        self.positive_clicks || self.negative_clicks
    }
}

/// A single grayscale image, stored as `u8`. snapseg is grayscale-only
/// by design (see `CLAUDE.md` "Image colour"): the operator-facing
/// industrial pipeline is monochrome, and the workspace contract is
/// this type. RGB-pretrained adapters (MobileSAM, RITM, FocalClick)
/// replicate the channel internally via
/// `snapseg-runtime::preprocess::gray_to_rgb_chw{,_byte}` at the model
/// boundary — they don't see `GrayImage`'s shape change.
#[derive(Debug, Clone)]
pub struct GrayImage {
    pub width: u32,
    pub height: u32,
    pub data: Array2<u8>,
}

impl GrayImage {
    pub fn from_array(data: Array2<u8>) -> Self {
        let (h, w) = data.dim();
        Self {
            width: w as u32,
            height: h as u32,
            data,
        }
    }
}

/// One of the K masks predicted by a multi-mask model (canonical SAM /
/// MobileSAM predict three candidates per click). Each carries the
/// model's self-reported IoU estimate so a downstream consumer (the UI,
/// active-learning, label-export) can rank or cycle through them.
///
/// Single-mask families (RITM, FocalClick) populate this with one
/// element; the contract is "always non-empty when `mask` is".
#[derive(Debug, Clone)]
pub struct MaskCandidate {
    /// Binary foreground mask at the source image resolution.
    pub mask: Array2<bool>,
    /// Raw logits at the source image resolution (pre-threshold).
    pub logits: Array2<f32>,
    /// Model-reported IoU prediction in `[0, 1]`. Single-mask families
    /// report `1.0` (the model has no alternative to compare against).
    pub iou: f32,
}

/// Output of one segmentation pass. The boolean mask is the headline
/// artifact; `logits` is exposed for QA, active-learning sample selection,
/// and as `prev_logits` for the next iteration.
///
/// Multi-mask families (SAM / MobileSAM) populate `candidates` with all
/// K predictions; `mask` and `logits` mirror the selected candidate
/// (argmax-IoU by default). Single-mask families populate `candidates`
/// with one entry so the UI can iterate uniformly.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Selected mask. For multi-mask models this is the argmax-IoU
    /// candidate; for single-mask models it's the sole prediction.
    pub mask: Array2<bool>,
    /// Selected logits, paired with `mask`.
    pub logits: Array2<f32>,
    /// Wall-clock duration of the inference call (encoder + decoder
    /// time, but excluding pre-/post-processing on the caller side).
    pub inference_time: Duration,
    /// All K candidate masks. Always non-empty when `mask` is set;
    /// `candidates[0]` is not necessarily the selected one — see
    /// [`SegmentationResult::selected_index`].
    pub candidates: Vec<MaskCandidate>,
}

impl SegmentationResult {
    /// Index of the selected candidate inside `candidates`, by IoU
    /// argmax. Returns `0` for single-candidate results.
    ///
    /// The selection is recomputed on the fly rather than cached, so
    /// callers that mutate `candidates` see a consistent answer.
    pub fn selected_index(&self) -> usize {
        let mut best = 0usize;
        let mut best_iou = f32::NEG_INFINITY;
        for (i, c) in self.candidates.iter().enumerate() {
            if c.iou > best_iou {
                best_iou = c.iou;
                best = i;
            }
        }
        best
    }
}

/// Trait every model adapter implements. Split into a one-time `set_image`
/// (where SAM-family models compute their expensive image embedding) and a
/// per-interaction `segment` (cheap when the embedding is cached).
pub trait InteractiveSegmenter: Send {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;
    fn set_image(&mut self, image: &GrayImage) -> Result<(), SegError>;
    fn segment(&mut self, session: &PromptSession) -> Result<SegmentationResult, SegError>;

    /// Drop any per-prompt-session cached state (e.g. SAM's previous
    /// low-res logits, RITM's previous-iteration mask) without
    /// invalidating the encoder embedding or the image preprocessing.
    /// Called when the UI clears the prompt session and the operator
    /// starts segmenting a different object on the same image — without
    /// this hook the next click would condition on the previous
    /// object's mask, biasing the prediction.
    ///
    /// Default impl is a no-op so non-iterative adapters don't have to
    /// override.
    fn reset_prompt_state(&mut self) {}
}

/// Unified error type. Adapters and runtime can wrap their backend errors
/// here; downstream code matches on the variants it cares about.
#[derive(Debug, Error)]
pub enum SegError {
    #[error("no image has been set; call set_image() first")]
    NoImage,
    #[error("model does not support this prompt: {0}")]
    UnsupportedPrompt(&'static str),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Helper: convert a `[H, W, 3]` `Array3<f32>` (CHW-ready) into the
/// `[1, 3, H, W]` shape ONNX models expect. Lives here so adapters and the
/// runtime crate agree on layout, even though only the runtime crate touches
/// real ort tensors.
pub fn to_nchw(chw_last: &Array3<f32>) -> Array3<f32> {
    chw_last.clone().permuted_axes([2, 0, 1])
}
