//! Model registry: TOML schema, cache resolution, sha256-verified
//! downloads. The app's model picker is built on this.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// One entry in `models.toml`. The fields mirror what an adapter needs to
/// load a model: where to get it, how big to expect, what input shape it
/// was exported with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    /// Family selector — adapters dispatch on this string.
    /// Known values: "ritm", "focalclick", "mobile_sam", "efficient_sam",
    /// "sam2_tiny".
    pub family: String,
    /// HTTP(S) URL to fetch the ONNX file from when missing.
    pub url: String,
    /// Hex-encoded SHA-256 of the ONNX file.
    pub sha256: String,
    pub size_mb: u32,
    /// `[height, width]` the model expects.
    pub input_size: [u32; 2],
    pub license: String,
    #[serde(default)]
    pub notes: String,
}

/// Top-level TOML file structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(rename = "model", default)]
    pub models: Vec<ModelEntry>,
}

impl Registry {
    /// Parse a TOML file into a registry.
    pub fn from_toml_file(path: &Path) -> Result<Self, RegistryError> {
        let text = fs::read_to_string(path)?;
        let reg: Registry = toml::from_str(&text)?;
        Ok(reg)
    }

    /// Look up an entry by `name`.
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

/// Resolve an entry to an on-disk path: returns the cached path if present
/// and sha256-valid, otherwise downloads (when the `download` feature is
/// enabled) and verifies.
pub fn resolve(entry: &ModelEntry, cache_dir: &Path) -> Result<PathBuf, RegistryError> {
    let dest = cache_dir.join(format!("{}.onnx", entry.name));
    if dest.exists() && verify_sha256(&dest, &entry.sha256)? {
        return Ok(dest);
    }
    #[cfg(feature = "download")]
    {
        fs::create_dir_all(cache_dir)?;
        download(&entry.url, &dest)?;
        if !verify_sha256(&dest, &entry.sha256)? {
            return Err(RegistryError::Sha256Mismatch {
                name: entry.name.clone(),
            });
        }
        return Ok(dest);
    }
    #[cfg(not(feature = "download"))]
    {
        Err(RegistryError::Missing {
            name: entry.name.clone(),
            path: dest,
        })
    }
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
