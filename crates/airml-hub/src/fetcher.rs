//! Model download orchestration.
//!
//! [`Fetcher`] is the main entry point for resolving a [`ModelUri`] to a local
//! file path, downloading and caching the blob as needed.

use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::cache::ModelCache;
use crate::error::{HubError, Result};
use crate::registry;
use crate::uri::ModelUri;

/// HTTP timeout for all model downloads.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

/// Orchestrates model resolution, downloading, and caching.
pub struct Fetcher {
    cache: ModelCache,
    ua: String,
}

impl Fetcher {
    /// Create a [`Fetcher`] using the default cache root and User-Agent
    /// `"airml/0.2"`.
    pub fn new() -> Self {
        Self {
            cache: ModelCache::new(ModelCache::default_root()),
            ua: "airml/0.2".to_owned(),
        }
    }

    /// Create a [`Fetcher`] with a custom [`ModelCache`] (useful for tests).
    pub fn with_cache(cache: ModelCache) -> Self {
        Self {
            cache,
            ua: "airml/0.2".to_owned(),
        }
    }

    /// Resolve a [`ModelUri`] to an on-disk path, downloading if necessary.
    ///
    /// | Variant | Behaviour |
    /// |---|---|
    /// | `LocalPath` | Verified to exist; returned as-is. |
    /// | `Registry(id)` | Looked up; downloaded from HF if not cached; sha256 verified. |
    /// | `HuggingFace { repo, file }` | URL constructed; downloaded; stored under content hash. |
    /// | `Url(u)` | Downloaded; stored under content hash; returned. |
    #[tracing::instrument(skip(self), fields(uri = ?uri))]
    pub fn resolve_to_path(&self, uri: &ModelUri) -> Result<PathBuf> {
        match uri {
            ModelUri::LocalPath(p) => {
                if !p.exists() {
                    return Err(HubError::IoError(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("local path does not exist: {}", p.display()),
                    )));
                }
                Ok(p.clone())
            }

            ModelUri::Registry(id) => {
                let entry = registry::lookup(id)
                    .ok_or_else(|| HubError::UnknownModel(id.clone()))?;

                // If we have a real (non-placeholder) sha256 and it's cached, return it.
                let is_placeholder = entry.sha256 == "PLACEHOLDER_TO_VERIFY";

                if !is_placeholder && self.cache.contains(entry.sha256) {
                    tracing::debug!(sha = %&entry.sha256[..8], "cache hit: {}", id);
                    return self.cache.path_for_sha256(entry.sha256);
                }

                // Download from HuggingFace.
                let url = hf_url(entry.hf_repo, entry.file);
                let size_mb = entry.size_bytes / 1_000_000;
                eprintln!("downloading {url} ({size_mb} MB)...");

                let bytes = self.http_get(&url)?;

                let actual_sha = hex::encode(Sha256::digest(&bytes));

                // Verify sha256 when we have a real expected value.
                if !is_placeholder && actual_sha != entry.sha256 {
                    return Err(HubError::Sha256Mismatch {
                        expected: entry.sha256.to_owned(),
                        actual: actual_sha,
                    });
                }

                let path = self.cache.store_atomic(&actual_sha, &bytes)?;
                let sha_short = &actual_sha[..8];
                tracing::info!(bytes = bytes.len(), sha = %sha_short, "downloaded model");
                eprintln!("✓ cached {} ({})", id, sha_short);
                Ok(path)
            }

            ModelUri::HuggingFace { repo, file } => {
                let url = hf_url(repo, file);
                self.download_and_store(&url, &url)
            }

            ModelUri::Url(u) => self.download_and_store(u, u),
        }
    }

    /// Download `url`, hash the result, store in cache, return the path.
    fn download_and_store(&self, url: &str, label: &str) -> Result<PathBuf> {
        // Derive a short display name from the URL for progress output.
        let short = url.rsplit('/').next().unwrap_or(url);
        eprintln!("downloading {short}...");

        let bytes = self.http_get(url)?;
        let sha = hex::encode(Sha256::digest(&bytes));

        let path = self.cache.store_atomic(&sha, &bytes)?;
        let sha_short = &sha[..8];
        tracing::info!(bytes = bytes.len(), sha = %sha_short, "downloaded model");
        eprintln!("✓ cached {} ({})", label, sha_short);
        Ok(path)
    }

    /// Perform a blocking HTTP GET with a 5-minute timeout.
    ///
    /// Returns the full response body as bytes.  Non-2xx responses are mapped
    /// to [`HubError::HttpError`].
    pub(crate) fn http_get(&self, url: &str) -> Result<Vec<u8>> {
        let response = ureq::AgentBuilder::new()
            .timeout(DOWNLOAD_TIMEOUT)
            .user_agent(&self.ua)
            .build()
            .get(url)
            .call()
            .map_err(|e| HubError::HttpError(format!("{url}: {e}")))?;

        let status = response.status();
        if !(200..300).contains(&status) {
            return Err(HubError::HttpError(format!("{url}: HTTP {status}")));
        }

        let mut buf = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| HubError::HttpError(format!("{url}: failed reading body: {e}")))?;

        Ok(buf)
    }
}

impl Default for Fetcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a HuggingFace `resolve/main` URL.
fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}
