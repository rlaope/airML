//! Model URI parsing.
//!
//! Supports four URI forms:
//!
//! | Input | Variant |
//! |---|---|
//! | `hf://owner/repo` | `HuggingFace { repo: "owner/repo", file: "model.onnx" }` |
//! | `hf://owner/repo/path/to/file.onnx` | `HuggingFace { repo, file }` |
//! | `https://…` / `http://…` | `Url(…)` |
//! | known registry id | `Registry(id)` |
//! | anything else | `LocalPath(PathBuf)` |

use std::path::PathBuf;

use crate::error::Result;
use crate::registry;

/// A resolved model source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelUri {
    /// A short id matching an entry in the built-in [`registry`].
    Registry(String),
    /// A HuggingFace repository + file path.
    HuggingFace {
        /// Repository slug, e.g. `"Xenova/clip-vit-base-patch32"`.
        repo: String,
        /// Path within the repo, e.g. `"onnx/model.onnx"`.
        file: String,
    },
    /// An arbitrary HTTP/HTTPS URL.
    Url(String),
    /// A local filesystem path.
    LocalPath(PathBuf),
}

impl ModelUri {
    /// Parse a raw URI string into a [`ModelUri`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HubError::MalformedUri`] if the string starts
    /// with `hf://` but contains no repository segment.
    pub fn parse(s: &str) -> Result<Self> {
        // hf:// scheme
        if let Some(rest) = s.strip_prefix("hf://") {
            return Self::parse_hf(rest, s);
        }

        // HTTP(S) URL
        if s.starts_with("https://") || s.starts_with("http://") {
            return Ok(Self::Url(s.to_owned()));
        }

        // Known registry id
        if registry::lookup(s).is_some() {
            return Ok(Self::Registry(s.to_owned()));
        }

        // Fall back to local path
        Ok(Self::LocalPath(PathBuf::from(s)))
    }

    /// Parse the portion of a `hf://` URI that follows the scheme.
    ///
    /// `hf://<owner>/<repo>` → repo = `"<owner>/<repo>"`, file = `"model.onnx"`
    /// `hf://<owner>/<repo>/<path>` → repo = `"<owner>/<repo>"`, file = `"<path>"`
    fn parse_hf(rest: &str, original: &str) -> Result<Self> {
        // `rest` is everything after "hf://"
        // The first two slash-delimited segments form the repo slug.
        let parts: Vec<&str> = rest.splitn(3, '/').collect();
        match parts.as_slice() {
            // hf://owner  or  hf://  — missing repo component
            [_] | [] => Err(crate::error::HubError::MalformedUri(format!(
                "hf:// URI requires at least owner/repo: {original:?}"
            ))),
            // hf://owner/repo  — default file
            [owner, repo] => Ok(Self::HuggingFace {
                repo: format!("{owner}/{repo}"),
                file: "model.onnx".to_owned(),
            }),
            // hf://owner/repo/sub/dir/file.ext
            [owner, repo, file_path] => Ok(Self::HuggingFace {
                repo: format!("{owner}/{repo}"),
                file: file_path.to_string(),
            }),
            _ => unreachable!("splitn(3) produces at most 3 elements"),
        }
    }
}
