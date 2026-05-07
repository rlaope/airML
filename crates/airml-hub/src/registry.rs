//! Built-in curated model registry.
//!
//! Maps short human-friendly IDs (e.g. `"bge-small-en"`) to their upstream
//! HuggingFace coordinates and expected SHA-256 digests.
//!
// TODO: replace placeholder sha256 values by running scripts/refresh-registry.sh
//       which downloads and hashes each entry.

/// A single entry in the built-in model registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    /// Short identifier used on the command line, e.g. `"bge-small-en"`.
    pub id: &'static str,
    /// HuggingFace repository slug, e.g. `"BAAI/bge-small-en-v1.5"`.
    pub hf_repo: &'static str,
    /// Path within the repository, e.g. `"onnx/model.onnx"`.
    pub file: &'static str,
    /// Expected lowercase hex SHA-256 of the downloaded blob.
    ///
    /// Set to `"PLACEHOLDER_TO_VERIFY"` until verified by the refresh script.
    pub sha256: &'static str,
    /// Approximate download size in bytes.
    pub size_bytes: u64,
    /// Human-readable description shown in `airml list`.
    pub description: &'static str,
}

/// All models shipped with this version of airML.
pub const REGISTRY: &[ModelEntry] = &[
    ModelEntry {
        id: "bge-small-en",
        hf_repo: "BAAI/bge-small-en-v1.5",
        file: "onnx/model.onnx",
        sha256: "PLACEHOLDER_TO_VERIFY",
        size_bytes: 133_000_000,
        description: "BGE Small English v1.5 — compact general-purpose text embeddings (~133 MB)",
    },
    ModelEntry {
        id: "all-minilm-l6-v2",
        hf_repo: "sentence-transformers/all-MiniLM-L6-v2",
        file: "onnx/model.onnx",
        sha256: "PLACEHOLDER_TO_VERIFY",
        size_bytes: 90_000_000,
        description: "all-MiniLM-L6-v2 — fast sentence embeddings (~90 MB)",
    },
    ModelEntry {
        id: "clip-vit-b32",
        hf_repo: "Xenova/clip-vit-base-patch32",
        file: "onnx/model.onnx",
        sha256: "PLACEHOLDER_TO_VERIFY",
        size_bytes: 605_000_000,
        description: "CLIP ViT-B/32 — joint image + text embeddings (~605 MB)",
    },
    ModelEntry {
        id: "mobilenetv3-small",
        hf_repo: "onnx/models",
        file: "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        sha256: "PLACEHOLDER_TO_VERIFY",
        size_bytes: 14_000_000,
        description: "MobileNetV2 image classification (~14 MB)",
    },
    ModelEntry {
        id: "whisper-tiny-encoder",
        hf_repo: "Xenova/whisper-tiny",
        file: "onnx/encoder_model.onnx",
        sha256: "PLACEHOLDER_TO_VERIFY",
        size_bytes: 80_000_000,
        description: "Whisper Tiny encoder — speech feature extraction (~80 MB)",
    },
];

/// Look up a registry entry by its short `id`.
///
/// Returns `None` if no entry matches.
pub fn lookup(id: &str) -> Option<&'static ModelEntry> {
    REGISTRY.iter().find(|e| e.id == id)
}

/// Return a slice of all built-in registry entries.
pub fn all() -> &'static [ModelEntry] {
    REGISTRY
}
