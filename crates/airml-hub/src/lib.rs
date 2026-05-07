//! `airml-hub` — content-addressed model download and caching for airML.
//!
//! # Overview
//!
//! This crate provides "zero-friction model acquisition":
//!
//! ```text
//! airml run  -m hf://Xenova/clip-vit-base-patch32
//! airml pull bge-small-en
//! ```
//!
//! Models are downloaded once, verified against their SHA-256 digest, and
//! stored in a content-addressed on-disk cache
//! (`~/Library/Caches/airml/models` on macOS, `~/.cache/airml/models` on
//! Linux).  Subsequent invocations return the cached path instantly.
//!
//! # Quick start
//!
//! ```no_run
//! use airml_hub::{Fetcher, ModelUri};
//!
//! let fetcher = Fetcher::new();
//! let uri = ModelUri::parse("bge-small-en").unwrap();
//! let path = fetcher.resolve_to_path(&uri).unwrap();
//! println!("model at {}", path.display());
//! ```

pub mod cache;
pub mod error;
pub mod fetcher;
pub mod registry;
pub mod uri;

pub use cache::{CacheStat, ModelCache};
pub use error::{HubError, Result};
pub use fetcher::Fetcher;
pub use registry::{ModelEntry, REGISTRY};
pub use uri::ModelUri;
