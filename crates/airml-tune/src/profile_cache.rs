//! On-disk JSON cache for measured per-provider latency profiles.
//!
//! Entries are keyed by `(model_sha256, ort_version, macos_version)` and
//! store P50 latency in milliseconds for each tested provider configuration.
//!
//! # Usage
//!
//! ```rust,no_run
//! use airml_tune::profile_cache::{CacheKey, MeasuredProfile, ProfileCache};
//! use std::path::PathBuf;
//!
//! let cache = ProfileCache::new(PathBuf::from("/tmp/airml-cache.json"));
//! let key = CacheKey {
//!     model_sha256: "abc123".to_string(),
//!     ort_version: "2.0.0-rc.11".to_string(),
//!     macos_version: "14.4".to_string(),
//! };
//! if let Some(profile) = cache.get(&key) {
//!     println!("cached P50: {} ms", profile.p50_ms);
//! }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Cache lookup key.
///
/// The triple `(model_sha256, ort_version, macos_version)` uniquely identifies
/// a benchmark context so that cached results are not applied after OS or
/// runtime upgrades.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// SHA-256 hex digest of the ONNX model file.
    pub model_sha256: String,
    /// ORT version string (e.g. `"2.0.0-rc.11"`).
    pub ort_version: String,
    /// macOS version string (e.g. `"14.4.1"`).
    pub macos_version: String,
}

/// Measured latency profile for a specific model + provider combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredProfile {
    /// Median (P50) inference latency in milliseconds.
    pub p50_ms: f64,
    /// Provider label (e.g. `"CoreMLAneOnly"`).
    pub provider: String,
    /// ISO-8601 timestamp of when this measurement was recorded.
    pub recorded_at: String,
}

/// Serialization envelope for the on-disk JSON file.
#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheFile {
    entries: HashMap<String, MeasuredProfile>,
}

/// JSON-on-disk cache keyed by [`CacheKey`].
///
/// The cache is loaded lazily on first access. If the backing file does not
/// exist, the cache is treated as empty — no error is returned.
#[derive(Debug, Clone)]
pub struct ProfileCache {
    /// Path to the JSON cache file.
    path: PathBuf,
}

impl ProfileCache {
    /// Create a cache backed by the given file path.
    ///
    /// The file does not need to exist; it will be created on the first
    /// [`Self::insert`] call.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Look up a measured profile by cache key.
    ///
    /// Returns `None` if the key is not present or the cache file cannot be
    /// read / parsed.
    pub fn get(&self, key: &CacheKey) -> Option<MeasuredProfile> {
        let file = self.load().ok()?;
        let json_key = Self::encode_key(key);
        file.entries.get(&json_key).cloned()
    }

    /// Insert or overwrite a measured profile.
    ///
    /// Persists the updated cache to disk immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn insert(&self, key: CacheKey, profile: MeasuredProfile) -> Result<()> {
        let mut file = self.load().unwrap_or_default();
        let json_key = Self::encode_key(&key);
        file.entries.insert(json_key, profile);
        self.save(&file)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Load the cache from disk. Returns an empty cache if the file is absent.
    fn load(&self) -> Result<CacheFile> {
        if !self.path.exists() {
            return Ok(CacheFile::default());
        }
        let contents = std::fs::read_to_string(&self.path)?;
        let file: CacheFile = serde_json::from_str(&contents)?;
        Ok(file)
    }

    /// Persist the cache to disk, creating parent directories if needed.
    fn save(&self, file: &CacheFile) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(file)?;
        std::fs::write(&self.path, contents)?;
        Ok(())
    }

    /// Encode a [`CacheKey`] to a stable string for use as a JSON map key.
    fn encode_key(key: &CacheKey) -> String {
        format!(
            "{}::{}::{}",
            key.model_sha256, key.ort_version, key.macos_version
        )
    }
}
