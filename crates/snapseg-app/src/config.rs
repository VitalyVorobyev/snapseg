//! Project-local operator configuration.
//!
//! On startup, snapseg looks for `./snapseg.toml` in the current
//! working directory. If present, its `default_model` is auto-loaded
//! (no file pickers), `label_dir` overrides the built-in default, and
//! `refine_edges` sets the subpixel-refinement toggle's starting state.
//!
//! The config file is operator-private (gitignored). A `snapseg.toml.example`
//! at the repo root documents the schema; copy and edit.
//!
//! Reload at runtime via the "Reload config" side-panel button. Reload
//! never re-runs encoder inference unless the model entry changed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

/// Where the config lives. Resolved relative to the process cwd.
pub const DEFAULT_CONFIG_NAME: &str = "snapseg.toml";

/// Top-level on-disk shape of `snapseg.toml`. Every field is optional;
/// an empty / absent file is a no-op (the app uses its built-in defaults).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    /// Override the label save directory. Relative paths are resolved
    /// from cwd.
    #[serde(default)]
    pub label_dir: Option<PathBuf>,

    /// Initial state of the "Refine subpixel edges" toggle. Defaults
    /// to `false` (matching the in-code default) when absent.
    #[serde(default)]
    pub refine_edges: Option<bool>,

    /// Auto-load this model on startup. When absent, the user must
    /// click "Load MobileSAM…" as before.
    #[serde(default)]
    pub default_model: Option<ModelConfig>,

    /// Absolute path to the `libonnxruntime` shared library `ort` should
    /// `dlopen` at runtime. When set, snapseg exports it as
    /// `ORT_DYLIB_PATH` before constructing any session — useful when
    /// the system-installed onnxruntime is ABI-incompatible with the
    /// pinned `ort` crate (e.g. Homebrew's onnxruntime is newer than
    /// what `ort 2.0.0-rc.10` was built against). When absent, `ort`
    /// uses its default search path.
    #[serde(default)]
    pub onnxruntime_path: Option<PathBuf>,
}

/// One model entry: same shape as a registry entry plus a per-part
/// `local_path` map (the local-paths-only equivalent of `models.toml`'s
/// download-able entries).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Display name (also used as the registry name in label provenance).
    pub name: String,
    /// Family slug — selects the adapter (`"mobile_sam"`, `"ritm"`, ...).
    pub family: String,
    /// Square input edge length the model was exported for.
    pub input_size: u32,
    /// `part_name → on-disk ONNX path`. For MobileSAM the keys are
    /// `"encoder"` and `"decoder"`; for single-network families (`ritm`,
    /// `focalclick`) the single key is `"model"`.
    pub parts: HashMap<String, PathBuf>,
}

/// Errors that can occur when reading or parsing `snapseg.toml`.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The config file was not found at the given path.
    #[error("config file not found at {0}")]
    NotFound(PathBuf),
    /// An I/O error occurred reading the config file.
    #[error("I/O error reading config: {0}")]
    Io(#[from] std::io::Error),
    /// The TOML content failed to parse against the [`AppConfig`] schema.
    #[error("TOML parse error in config: {0}")]
    Toml(#[from] toml::de::Error),
}

impl AppConfig {
    /// Load `./snapseg.toml` from cwd. Returns `Ok(None)` when the file
    /// is absent (a no-op startup); `Err` only on read or parse failure.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Io`] on any I/O failure other than
    /// file-not-found, or [`ConfigError::Toml`] on a parse error.
    pub fn load_default() -> Result<Option<Self>, ConfigError> {
        let path = PathBuf::from(DEFAULT_CONFIG_NAME);
        match Self::load_from(&path) {
            Ok(cfg) => Ok(Some(cfg)),
            Err(ConfigError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Load from an explicit path.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::NotFound`] when the file does not exist,
    /// [`ConfigError::Io`] on other read failures, and [`ConfigError::Toml`]
    /// on parse errors.
    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConfigError::NotFound(path.to_owned())
            } else {
                ConfigError::Io(e)
            }
        })?;
        let cfg: Self = toml::from_str(&text)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(contents.as_bytes()).expect("write");
        f
    }

    #[test]
    fn minimal_config_parses() {
        let f = write_tmp(
            r#"
            label_dir = "./labels"
            refine_edges = true

            [default_model]
            name = "mobile-sam"
            family = "mobile_sam"
            input_size = 1024

            [default_model.parts]
            encoder = "/tmp/e.onnx"
            decoder = "/tmp/d.onnx"
        "#,
        );
        let cfg = AppConfig::load_from(f.path()).expect("parse");
        assert_eq!(
            cfg.label_dir.as_ref().map(|p| p.to_str().unwrap()),
            Some("./labels")
        );
        assert_eq!(cfg.refine_edges, Some(true));
        let model = cfg.default_model.expect("default_model");
        assert_eq!(model.name, "mobile-sam");
        assert_eq!(model.family, "mobile_sam");
        assert_eq!(model.parts.len(), 2);
    }

    #[test]
    fn onnxruntime_path_parses() {
        let f = write_tmp(r#"onnxruntime_path = "/opt/ort/libonnxruntime.dylib""#);
        let cfg = AppConfig::load_from(f.path()).expect("parse");
        assert_eq!(
            cfg.onnxruntime_path.as_ref().and_then(|p| p.to_str()),
            Some("/opt/ort/libonnxruntime.dylib"),
        );
    }

    #[test]
    fn unknown_field_is_rejected() {
        let f = write_tmp(
            r#"
            label_dir = "./labels"
            mystery   = "no"
        "#,
        );
        assert!(AppConfig::load_from(f.path()).is_err());
    }
}
