//! ONNX Runtime wrapper.
//!
//! Built on `ort` 2.x with the `load-dynamic` feature: nothing is linked
//! against onnxruntime at compile time. The shared library is discovered
//! at runtime via `ORT_DYLIB_PATH` or the system's default search path
//! (`brew install onnxruntime` on macOS lands it in
//! `/opt/homebrew/lib/libonnxruntime.dylib`).

use std::path::{Path, PathBuf};

use ort::execution_providers::CPUExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
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

#[derive(Debug, Clone, Copy)]
pub enum OptLevel {
    Disable,
    Basic,
    Extended,
    All,
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
    pub optimization_level: OptLevel,
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
            optimization_level: OptLevel::All,
        }
    }
}

/// Handle to a loaded ONNX session.
pub struct Backend {
    model_path: PathBuf,
    chosen_ep: ExecutionProvider,
    session: Session,
}

impl Backend {
    /// Load a model file into a session. Tries the EPs from
    /// `config.ep_priority` in order; only those compiled in via their
    /// feature flag are present in the dispatch list.
    pub fn load(model_path: PathBuf, config: &RuntimeConfig) -> Result<Self, BackendError> {
        if !model_path.exists() {
            return Err(BackendError::ModelMissing(model_path));
        }

        let opt = match config.optimization_level {
            OptLevel::Disable => GraphOptimizationLevel::Disable,
            OptLevel::Basic => GraphOptimizationLevel::Level1,
            OptLevel::Extended => GraphOptimizationLevel::Level2,
            OptLevel::All => GraphOptimizationLevel::Level3,
        };

        let mut builder = Session::builder()
            .map_err(|e| BackendError::Other(format!("session builder: {e}")))?
            .with_optimization_level(opt)
            .map_err(|e| BackendError::Other(format!("opt level: {e}")))?;

        if let Some(n) = config.intra_threads {
            builder = builder
                .with_intra_threads(n)
                .map_err(|e| BackendError::Other(format!("intra threads: {e}")))?;
        }
        if let Some(n) = config.inter_threads {
            builder = builder
                .with_inter_threads(n)
                .map_err(|e| BackendError::Other(format!("inter threads: {e}")))?;
        }

        let mut ep_dispatch = Vec::new();
        #[allow(unused_mut)]
        let mut chosen = ExecutionProvider::Cpu;
        for ep in &config.ep_priority {
            match ep {
                #[cfg(feature = "tensorrt")]
                ExecutionProvider::TensorRt => {
                    ep_dispatch.push(
                        ort::execution_providers::TensorRTExecutionProvider::default().build(),
                    );
                    chosen = *ep;
                }
                #[cfg(feature = "cuda")]
                ExecutionProvider::Cuda => {
                    ep_dispatch.push(
                        ort::execution_providers::CUDAExecutionProvider::default().build(),
                    );
                    chosen = *ep;
                }
                #[cfg(feature = "coreml")]
                ExecutionProvider::CoreMl => {
                    ep_dispatch.push(
                        ort::execution_providers::CoreMLExecutionProvider::default().build(),
                    );
                    chosen = *ep;
                }
                #[cfg(feature = "directml")]
                ExecutionProvider::DirectMl => {
                    ep_dispatch.push(
                        ort::execution_providers::DirectMLExecutionProvider::default().build(),
                    );
                    chosen = *ep;
                }
                ExecutionProvider::Cpu => {
                    ep_dispatch.push(CPUExecutionProvider::default().build());
                }
                _ => {
                    // Provider not built in via its feature flag — skip.
                }
            }
        }

        builder = builder
            .with_execution_providers(ep_dispatch)
            .map_err(|e| BackendError::Other(format!("execution providers: {e}")))?;

        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| BackendError::LoadFailed(format!("{}: {e}", model_path.display())))?;

        tracing::info!(
            path = %model_path.display(),
            ep = chosen.name(),
            "ort session loaded"
        );

        Ok(Self {
            model_path,
            chosen_ep: chosen,
            session,
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn execution_provider(&self) -> ExecutionProvider {
        self.chosen_ep
    }

    /// Mutable access to the underlying session — adapters use this to
    /// build input value maps and call `session.run()`.
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("model file not found: {0}")]
    ModelMissing(PathBuf),
    #[error("failed to load model: {0}")]
    LoadFailed(String),
    #[error("runtime error: {0}")]
    Other(String),
}

pub mod preprocess;
