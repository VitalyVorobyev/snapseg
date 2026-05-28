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

/// Encoder canvas the model was exported for.
///
/// The TOML side accepts two shapes via serde's untagged enum:
///
/// ```toml
/// # Canonical, square SAM-style canvas.
/// input_size = 1024
///
/// # Non-square canvas, ordered `[H, W]` (rows, columns) — matches
/// # `orig_im_size` and `models.toml`'s registry array convention.
/// input_size = [682, 1024]
/// ```
///
/// Use [`InputSize::as_hw`] to read it back as a `(height, width)` tuple
/// regardless of which shape the operator wrote.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(untagged)]
pub enum InputSize {
    /// Scalar form `input_size = 1024` — interpreted as a square `(N, N)`
    /// canvas. Back-compatible with pre-2026-05 configs.
    Square(u32),
    /// Array form `input_size = [682, 1024]` — explicit `[H, W]`.
    HW([u32; 2]),
}

impl InputSize {
    /// Return the canvas as `(height, width)` regardless of which TOML
    /// shape was written.
    pub fn as_hw(&self) -> (u32, u32) {
        match self {
            InputSize::Square(n) => (*n, *n),
            InputSize::HW([h, w]) => (*h, *w),
        }
    }
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
    /// Encoder canvas (`[H, W]` or a scalar interpreted as a square).
    /// See [`InputSize`] for the on-disk shapes accepted.
    pub input_size: InputSize,
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

/// A config plus the file it was loaded from. The source path lets
/// callers resolve config-relative paths (model parts, onnxruntime_path)
/// against the directory the config file lives in — more intuitive than
/// resolving against cwd, and robust to `cargo run -p` setting cwd to
/// the package directory rather than the workspace root.
#[derive(Debug, Clone)]
pub struct LoadedConfig {
    /// Parsed config.
    pub config: AppConfig,
    /// Absolute path the config was loaded from.
    pub source: PathBuf,
}

impl LoadedConfig {
    /// Directory containing the config file. Used as the base for
    /// resolving any relative paths inside the config.
    pub fn dir(&self) -> &Path {
        self.source.parent().unwrap_or(Path::new("."))
    }

    /// Resolve a path against [`LoadedConfig::dir`] if relative; return
    /// the path as-is if already absolute.
    pub fn resolve(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.dir().join(p)
        }
    }
}

impl AppConfig {
    /// Find `snapseg.toml` by walking up from cwd, then load it.
    /// Returns `Ok(None)` when no config exists anywhere up the chain;
    /// `Err` only on read or parse failure.
    ///
    /// Walking up handles `cargo run -p <pkg>` setting cwd to the
    /// package directory: from `crates/snapseg-app/` we walk to
    /// `crates/`, then to the workspace root where `snapseg.toml`
    /// lives.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Io`] on any I/O failure other than
    /// file-not-found, or [`ConfigError::Toml`] on a parse error.
    pub fn load_default() -> Result<Option<LoadedConfig>, ConfigError> {
        let Some(path) = find_config_upward()? else {
            return Ok(None);
        };
        let config = Self::load_from(&path)?;
        Ok(Some(LoadedConfig {
            config,
            source: path,
        }))
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

/// Walk up from cwd looking for `snapseg.toml`. Returns the first hit
/// or `None` if none exists between cwd and the filesystem root.
fn find_config_upward() -> Result<Option<PathBuf>, ConfigError> {
    let mut dir = std::env::current_dir()?;
    loop {
        let candidate = dir.join(DEFAULT_CONFIG_NAME);
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
        if !dir.pop() {
            return Ok(None);
        }
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
        assert_eq!(model.input_size.as_hw(), (1024, 1024));
        assert_eq!(model.parts.len(), 2);
    }

    #[test]
    fn non_square_input_size_parses() {
        let f = write_tmp(
            r#"
            [default_model]
            name = "mobile-sam"
            family = "mobile_sam"
            input_size = [682, 1024]

            [default_model.parts]
            encoder = "/tmp/e.onnx"
            decoder = "/tmp/d.onnx"
        "#,
        );
        let cfg = AppConfig::load_from(f.path()).expect("parse");
        let model = cfg.default_model.expect("default_model");
        // [H, W] — height first to match orig_im_size / registry.
        assert_eq!(model.input_size.as_hw(), (682, 1024));
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
