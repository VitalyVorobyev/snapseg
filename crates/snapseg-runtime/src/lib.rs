//! ONNX Runtime wrapper.
//!
//! The public API is defined here; the actual `ort` session is wired once
//! the first real model adapter lands (see crate-level TODO and the
//! disabled `ort` dependency in `Cargo.toml`).

use std::path::PathBuf;

use thiserror::Error;

/// Execution providers we'll try, in priority order. CPU is the always-on
/// fallback. The runtime tries each in turn and uses the first that
/// initializes successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    TensorRt,
    Cuda,
    CoreMl,
    DirectMl,
    Cpu,
}

impl ExecutionProvider {
    pub const fn name(self) -> &'static str {
        match self {
            Self::TensorRt => "TensorRT",
            Self::Cuda => "CUDA",
            Self::CoreMl => "CoreML",
            Self::DirectMl => "DirectML",
            Self::Cpu => "CPU",
        }
    }
}

/// Runtime configuration: which EPs to try, in what order, and how to
/// configure the underlying session.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Ordered list of execution providers to attempt. The first one that
    /// initializes is used. CPU is implicitly appended if not present.
    pub ep_priority: Vec<ExecutionProvider>,
    /// Number of intra-op threads. `None` lets onnxruntime pick.
    pub intra_threads: Option<usize>,
    /// Number of inter-op threads.
    pub inter_threads: Option<usize>,
    /// Optimization level: 0 = none, 1 = basic, 2 = extended, 3 = all.
    pub optimization_level: u8,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            ep_priority: vec![
                ExecutionProvider::TensorRt,
                ExecutionProvider::Cuda,
                ExecutionProvider::CoreMl,
                ExecutionProvider::DirectMl,
                ExecutionProvider::Cpu,
            ],
            intra_threads: None,
            inter_threads: None,
            optimization_level: 3,
        }
    }
}

/// Handle to a loaded ONNX session. The session-loading implementation
/// arrives with the first real adapter; this struct exists so adapter
/// crates can compile against the API today.
pub struct Backend {
    model_path: PathBuf,
    chosen_ep: ExecutionProvider,
    // session: ort::Session,  // wired in the next iteration
}

impl Backend {
    /// Construct a backend. Currently a stub: stores the configuration but
    /// does not actually load the model. Real session creation is added
    /// alongside the first model adapter.
    pub fn load(model_path: PathBuf, _config: &RuntimeConfig) -> Result<Self, BackendError> {
        if !model_path.exists() {
            return Err(BackendError::ModelMissing(model_path));
        }
        Ok(Self {
            model_path,
            chosen_ep: ExecutionProvider::Cpu,
        })
    }

    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    pub fn execution_provider(&self) -> ExecutionProvider {
        self.chosen_ep
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("model file not found: {0}")]
    ModelMissing(PathBuf),
    #[error("failed to initialize any execution provider")]
    NoEpAvailable,
    #[error("runtime error: {0}")]
    Other(String),
}

pub mod preprocess;
