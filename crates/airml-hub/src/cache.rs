//! Content-addressed on-disk model cache.
//!
//! Files are stored under `<root>/<sha256[..2]>/<sha256>` following the
//! git-style fanout layout to keep directory entry counts manageable.

use std::fs;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::error::{HubError, Result};

/// Summary statistics for a [`ModelCache`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStat {
    /// Number of distinct model blobs stored on disk.
    pub num_models: usize,
    /// Total size of all cached blobs in bytes.
    pub total_bytes: u64,
}

/// Content-addressed disk cache for ONNX model blobs.
///
/// Each blob is keyed by its lowercase hex-encoded SHA-256 digest and stored
/// under `<root>/<first-2-hex-chars>/<full-64-char-digest>`.
pub struct ModelCache {
    root: PathBuf,
}

impl ModelCache {
    /// Return the platform default cache root: `{cache_dir}/airml/models`.
    ///
    /// Falls back to `.cache/airml/models` if the OS cache directory cannot be
    /// determined.
    pub fn default_root() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("airml/models")
    }

    /// Create a cache rooted at `root`.  The directory is created lazily on
    /// first write; no I/O is performed here.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Validate that `sha256` is exactly 64 lowercase hex characters.
    fn validate_sha256(sha256: &str) -> Result<()> {
        if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(HubError::CacheError(format!(
                "invalid sha256 key (must be 64 hex chars): {sha256:?}"
            )));
        }
        Ok(())
    }

    /// Return the filesystem path where a blob with digest `sha256` is stored.
    ///
    /// Returns [`HubError::CacheError`] if `sha256` is not a valid 64-character
    /// lowercase hex string.
    pub fn path_for_sha256(&self, sha256: &str) -> Result<PathBuf> {
        Self::validate_sha256(sha256)?;
        let fanout = &sha256[..2];
        Ok(self.root.join(fanout).join(sha256))
    }

    /// Return `true` if a blob with digest `sha256` is present in the cache.
    pub fn contains(&self, sha256: &str) -> bool {
        match self.path_for_sha256(sha256) {
            Ok(p) => p.exists(),
            Err(_) => false,
        }
    }

    /// Atomically store `bytes` in the cache under `sha256`.
    ///
    /// The hash of `bytes` is verified *before* the file is renamed into place.
    /// If the digest does not match, the temporary file is deleted and
    /// [`HubError::Sha256Mismatch`] is returned.
    ///
    /// Returns the path to the stored blob.
    pub fn store_atomic(&self, sha256: &str, bytes: &[u8]) -> Result<PathBuf> {
        // Verify first — never persist corrupt data.
        let actual = hex::encode(Sha256::digest(bytes));
        if actual != sha256 {
            return Err(HubError::Sha256Mismatch {
                expected: sha256.to_owned(),
                actual,
            });
        }

        let dest = self.path_for_sha256(sha256)?;

        // Short-circuit if already cached (idempotent).
        if dest.exists() {
            return Ok(dest);
        }

        // Ensure parent directory exists.
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let tmp = dest.with_extension("tmp");

        // Write to temp file.
        fs::write(&tmp, bytes).map_err(|e| {
            HubError::IoError(std::io::Error::new(
                e.kind(),
                format!("writing tmp blob {}: {e}", tmp.display()),
            ))
        })?;

        // Atomic rename.
        if let Err(e) = fs::rename(&tmp, &dest) {
            // Best-effort cleanup.
            let _ = fs::remove_file(&tmp);
            return Err(HubError::IoError(std::io::Error::new(
                e.kind(),
                format!("renaming blob into cache: {e}"),
            )));
        }

        Ok(dest)
    }

    /// Load the cached blob for `sha256` into memory.
    ///
    /// Returns [`HubError::CacheError`] if the blob is not present.
    pub fn load(&self, sha256: &str) -> Result<Vec<u8>> {
        let path = self.path_for_sha256(sha256)?;
        if !path.exists() {
            return Err(HubError::CacheError(format!(
                "blob not found in cache: {sha256}"
            )));
        }
        Ok(fs::read(&path)?)
    }

    /// Return aggregate statistics for the cache.
    ///
    /// Silently skips entries that cannot be read (e.g. due to permissions).
    pub fn stat(&self) -> CacheStat {
        let mut num_models = 0usize;
        let mut total_bytes = 0u64;

        let Ok(fanout_entries) = fs::read_dir(&self.root) else {
            return CacheStat { num_models, total_bytes };
        };

        for fanout in fanout_entries.flatten() {
            let Ok(blobs) = fs::read_dir(fanout.path()) else {
                continue;
            };
            for blob in blobs.flatten() {
                if let Ok(meta) = blob.metadata() {
                    if meta.is_file() {
                        num_models += 1;
                        total_bytes += meta.len();
                    }
                }
            }
        }

        CacheStat { num_models, total_bytes }
    }

    /// Remove the cached blob for `sha256`.
    ///
    /// Returns [`HubError::CacheError`] if the blob is not present.
    pub fn evict(&self, sha256: &str) -> Result<()> {
        let path = self.path_for_sha256(sha256)?;
        if !path.exists() {
            return Err(HubError::CacheError(format!(
                "cannot evict: blob not found in cache: {sha256}"
            )));
        }
        fs::remove_file(&path)?;
        Ok(())
    }
}
