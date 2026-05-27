//! Model registry: TOML schema, cache resolution, sha256-verified
//! downloads. The app's model picker is built on this.
//!
//! Entries are multi-part because SAM-family models ship as two ONNX
//! files (image encoder + mask decoder); single-network models like RITM
//! simply declare one part. Adapters look up the part they need by name
//! (e.g. `"encoder"`, `"decoder"`, or `"model"`).

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// One downloadable ONNX file. A model entry has one or more of these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPart {
    /// Logical name an adapter looks up. Conventions:
    ///   - `"model"`     for single-network models (RITM, FocalClick, ...)
    ///   - `"encoder"`/`"decoder"` for SAM-family two-stage models.
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub size_mb: u32,
}

/// One entry in `models.toml`. Adapters dispatch on `family` and pick the
/// `ModelPart` they need by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    /// Family selector — adapters dispatch on this string.
    /// Known values: "ritm", "focalclick", "mobile_sam", "efficient_sam",
    /// "sam2_tiny".
    pub family: String,
    /// `[height, width]` the model was exported for. Single-network models
    /// take this as their resize target; SAM-family models use it as the
    /// encoder input shape.
    pub input_size: [u32; 2],
    pub license: String,
    #[serde(default)]
    pub notes: String,
    /// One or more ONNX files that make up this model.
    #[serde(default)]
    pub parts: Vec<ModelPart>,
}

impl ModelEntry {
    pub fn part(&self, name: &str) -> Option<&ModelPart> {
        self.parts.iter().find(|p| p.name == name)
    }
}

/// Top-level TOML file structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(rename = "model", default)]
    pub models: Vec<ModelEntry>,
}

impl Registry {
    pub fn from_toml_file(path: &Path) -> Result<Self, RegistryError> {
        let text = fs::read_to_string(path)?;
        let reg: Registry = toml::from_str(&text)?;
        Ok(reg)
    }

    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }
}

/// Where to keep downloaded model files. Uses platform-appropriate cache
/// dir; falls back to `./models/` if `directories` can't resolve one.
pub fn default_cache_dir() -> PathBuf {
    ProjectDirs::from("dev", "vitavision", "snapseg")
        .map(|p| p.cache_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("./models"))
}

/// Resolve every part of a model entry to a local path. Returns a map keyed
/// by part name (`"encoder"`, `"decoder"`, `"model"`, ...).
pub fn resolve(entry: &ModelEntry, cache_dir: &Path) -> Result<HashMap<String, PathBuf>, RegistryError> {
    let mut out = HashMap::with_capacity(entry.parts.len());
    fs::create_dir_all(cache_dir)?;
    for part in &entry.parts {
        let dest = cache_dir.join(format!("{}--{}.onnx", entry.name, part.name));
        if !(dest.exists() && verify_sha256(&dest, &part.sha256)?) {
            #[cfg(feature = "download")]
            {
                download(&part.url, &dest)?;
                if !verify_sha256(&dest, &part.sha256)? {
                    return Err(RegistryError::Sha256Mismatch {
                        name: format!("{}/{}", entry.name, part.name),
                    });
                }
            }
            #[cfg(not(feature = "download"))]
            {
                return Err(RegistryError::Missing {
                    name: format!("{}/{}", entry.name, part.name),
                    path: dest,
                });
            }
        }
        out.insert(part.name.clone(), dest);
    }
    Ok(out)
}

/// SHA-256 of a file as lowercase hex.
pub fn file_sha256(path: &Path) -> Result<String, RegistryError> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path)?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

fn verify_sha256(path: &Path, expected_hex: &str) -> Result<bool, RegistryError> {
    // Empty sha (or all zeros) means "trust whatever is cached" — useful
    // during early bring-up before a real hash is pinned.
    if expected_hex.is_empty() || expected_hex.chars().all(|c| c == '0') {
        return Ok(true);
    }
    let got = file_sha256(path)?;
    Ok(got.eq_ignore_ascii_case(expected_hex))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

#[cfg(feature = "download")]
fn download(url: &str, dest: &Path) -> Result<(), RegistryError> {
    tracing::info!(url = %url, dest = %dest.display(), "downloading model");
    let resp = reqwest::blocking::get(url)
        .map_err(|e| RegistryError::Download(e.to_string()))?
        .error_for_status()
        .map_err(|e| RegistryError::Download(e.to_string()))?;
    let bytes = resp
        .bytes()
        .map_err(|e| RegistryError::Download(e.to_string()))?;
    fs::write(dest, &bytes)?;
    Ok(())
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("model `{name}` not present at {path:?} and downloads are disabled")]
    Missing { name: String, path: PathBuf },
    #[error("sha256 mismatch for model `{name}`")]
    Sha256Mismatch { name: String },
    #[error("download failed: {0}")]
    Download(String),
}
