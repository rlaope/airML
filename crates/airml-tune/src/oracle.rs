//! Backend recommendation oracle.
//!
//! Classifies an ONNX model from its `ModelMetadata` and recommends the
//! optimal CoreML compute units or CPU fallback without executing the model.
//!
//! Two classification paths are available:
//!
//! - [`BackendOracle::profile_from_metadata`] — fast heuristic from input
//!   tensor names/shapes; no file I/O required.
//! - [`BackendOracle::profile_from_path`] — reads the ONNX graph and counts
//!   op types; more accurate, requires the model file.

use std::path::Path;

use airml_core::ModelMetadata;

use crate::graph_parser::{self, GraphParseError};

/// Classification of the dominant operation type in a model.
///
/// Used together with [`ModelFamily`] to select the best compute units.
/// New variants will be added in v0.3 once graph-op-histogram parsing lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpClass {
    /// Dominated by convolution ops (e.g. MobileNet, ResNet).
    ConvHeavy,
    /// Dominated by GEMM / matrix-multiply ops (e.g. CLIP projection heads).
    GemmHeavy,
    /// Dominated by attention / transformer ops (e.g. BERT, all-MiniLM).
    AttentionHeavy,
    /// Contains significant control-flow (e.g. autoregressive LMs with KV cache).
    ControlFlow,
    /// Mix of multiple op classes (e.g. CLIP dual encoder).
    Mixed,
}

/// High-level model family inferred from input/output tensor names and shapes.
///
/// v0.3 will parse the actual ONNX graph for a finer-grained classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFamily {
    /// Image-classification / detection models with 4-D NCHW inputs.
    Vision,
    /// Text encoder models (BERT-style) with `input_ids` / `attention_mask`.
    TextEncoder,
    /// Dual-encoder models combining image and text inputs (e.g. CLIP).
    ImageTextDual,
    /// Autoregressive language models with KV-cache inputs.
    LanguageModel,
    /// Family could not be determined from available metadata.
    Unknown,
}

/// Derived properties of a model used to drive the recommendation.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Number of input tensors.
    pub num_inputs: usize,
    /// Number of output tensors.
    pub num_outputs: usize,
    /// Rough parameter count estimate when available.
    ///
    /// Currently always `None`; v0.3 will compute this from weight tensors.
    pub total_params_estimate: Option<u64>,
    /// `true` if any input dimension is `<= 0` (dynamic / symbolic shape).
    pub has_dynamic_shapes: bool,
    /// Dominant operation class inferred from model family.
    pub dominant_op_class: OpClass,
    /// Model family inferred from input names and shapes.
    pub model_family: ModelFamily,
}

/// Recommended backend configuration for a model.
///
/// Map this to actual providers via [`crate::dispatch::recommendation_to_providers`]
/// when the `coreml` feature is enabled.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendRecommendation {
    /// Run on CPU only (always available, no CoreML).
    CpuOnly,
    /// Use CoreML with all compute units (CPU + GPU + ANE).
    CoreMLAll,
    /// Use CoreML with CPU + ANE only — best for conv/attention-heavy workloads.
    CoreMLAneOnly,
    /// Use CoreML with CPU + GPU only — better for control-flow heavy models.
    CoreMLGpuOnly,
    /// CPU only with an explanatory reason string.
    CpuOnlyWithReason(String),
}

/// Stateless oracle that maps model metadata to backend recommendations.
///
/// The oracle applies a fast, deterministic rule set that requires no
/// benchmark runs. For latency-critical production use, pair it with
/// [`crate::profile_cache::ProfileCache`] to override with measured data.
#[derive(Debug, Default, Clone)]
pub struct BackendOracle;

impl BackendOracle {
    /// Create a new oracle instance.
    pub fn new() -> Self {
        Self
    }

    /// Derive a [`ModelProfile`] from ONNX session metadata.
    ///
    /// Heuristics applied (in order):
    ///
    /// 1. **`model_family`** — inferred from input names and shapes:
    ///    - `pixel_values` + `input_ids` present → [`ModelFamily::ImageTextDual`]
    ///    - any input name contains `past_key_values` → [`ModelFamily::LanguageModel`]
    ///    - `input_ids` or `attention_mask` present → [`ModelFamily::TextEncoder`]
    ///    - 4-D NCHW input with 3 channels → [`ModelFamily::Vision`]
    ///    - otherwise → [`ModelFamily::Unknown`]
    ///
    /// 2. **`dominant_op_class`** — currently derived from `model_family`; v0.3
    ///    will replace this with an ONNX graph op-histogram pass.
    ///
    /// 3. **`has_dynamic_shapes`** — `true` if any input dimension is `<= 0`.
    pub fn profile_from_metadata(&self, meta: &ModelMetadata) -> ModelProfile {
        let model_family = Self::infer_family(meta);
        let dominant_op_class = Self::infer_op_class(model_family);
        let has_dynamic_shapes = meta
            .inputs
            .iter()
            .any(|t| t.shape.iter().any(|&d| d <= 0));

        ModelProfile {
            num_inputs: meta.inputs.len(),
            num_outputs: meta.outputs.len(),
            // TODO(v0.3): compute from weight initializer tensors in the ONNX graph.
            total_params_estimate: None,
            has_dynamic_shapes,
            dominant_op_class,
            model_family,
        }
    }

    /// Recommend the optimal backend given a pre-computed [`ModelProfile`].
    ///
    /// Decision table:
    ///
    /// | Family | Op class | Dynamic shapes | Recommendation |
    /// |---|---|---|---|
    /// | Vision | ConvHeavy | any | CoreMLAneOnly |
    /// | TextEncoder | AttentionHeavy | false | CoreMLAneOnly |
    /// | TextEncoder | AttentionHeavy | true | CoreMLAll |
    /// | ImageTextDual | Mixed | any | CoreMLAll |
    /// | LanguageModel | ControlFlow | any | CoreMLGpuOnly |
    /// | Unknown | any | true | CpuOnlyWithReason |
    /// | Unknown | any | false | CoreMLAll |
    pub fn recommend(&self, profile: &ModelProfile) -> BackendRecommendation {
        match (profile.model_family, profile.dominant_op_class) {
            (ModelFamily::Vision, OpClass::ConvHeavy) => BackendRecommendation::CoreMLAneOnly,

            (ModelFamily::TextEncoder, OpClass::AttentionHeavy) => {
                if profile.has_dynamic_shapes {
                    BackendRecommendation::CoreMLAll
                } else {
                    BackendRecommendation::CoreMLAneOnly
                }
            }

            (ModelFamily::ImageTextDual, OpClass::Mixed) => BackendRecommendation::CoreMLAll,

            (ModelFamily::LanguageModel, OpClass::ControlFlow) => {
                BackendRecommendation::CoreMLGpuOnly
            }

            (ModelFamily::Unknown, _) => {
                if profile.has_dynamic_shapes {
                    BackendRecommendation::CpuOnlyWithReason(
                        "unknown family with dynamic shapes; falling back to CPU".to_string(),
                    )
                } else {
                    BackendRecommendation::CoreMLAll
                }
            }

            // Catch-all for any future family/op-class combinations introduced in v0.3.
            _ => BackendRecommendation::CoreMLAll,
        }
    }

    /// Convenience method: profile then recommend in one call.
    ///
    /// Equivalent to `oracle.recommend(&oracle.profile_from_metadata(meta))`.
    #[tracing::instrument(skip(meta))]
    pub fn recommend_for_metadata(&self, meta: &ModelMetadata) -> BackendRecommendation {
        let profile = self.profile_from_metadata(meta);
        let recommendation = self.recommend(&profile);
        tracing::info!(?recommendation, "backend dispatch");
        recommendation
    }

    /// Derive a [`ModelProfile`] from the ONNX file at `path`.
    ///
    /// This path parses the actual graph op histogram for an accurate
    /// `dominant_op_class`, unlike [`Self::profile_from_metadata`] which
    /// infers the op class from input tensor names alone.
    ///
    /// `model_family` and `has_dynamic_shapes` still require loaded session
    /// metadata and are set to `Unknown` / `false` respectively here; callers
    /// that have session metadata available should prefer combining both:
    ///
    /// ```rust,no_run
    /// # use airml_tune::oracle::BackendOracle;
    /// # use airml_core::ModelMetadata;
    /// # use std::path::Path;
    /// # let oracle = BackendOracle::new();
    /// # let meta: &ModelMetadata = todo!();
    /// # let path: &Path = todo!();
    /// let mut profile = oracle.profile_from_metadata(meta);
    /// if let Ok(hist) = airml_tune::histogram_from_path(path) {
    ///     profile.dominant_op_class = hist.dominant_class();
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`GraphParseError`] if the file cannot be read or is not a
    /// valid ONNX protobuf.
    pub fn profile_from_path(&self, path: &Path) -> Result<ModelProfile, GraphParseError> {
        let hist = graph_parser::histogram_from_path(path)?;
        let dominant_op_class = hist.dominant_class();
        Ok(ModelProfile {
            num_inputs: 0,
            num_outputs: 0,
            total_params_estimate: None,
            has_dynamic_shapes: false,
            dominant_op_class,
            model_family: ModelFamily::Unknown,
        })
    }

    /// Convenience method: parse graph, profile, then recommend in one call.
    ///
    /// When session metadata is also available, prefer calling
    /// [`Self::profile_from_metadata`], patching `dominant_op_class` from the
    /// histogram, and then calling [`Self::recommend`].
    ///
    /// # Errors
    ///
    /// Returns [`GraphParseError`] if the file cannot be read or parsed.
    #[tracing::instrument(skip(self))]
    pub fn recommend_for_path(
        &self,
        path: &Path,
    ) -> Result<BackendRecommendation, GraphParseError> {
        let profile = self.profile_from_path(path)?;
        let recommendation = self.recommend(&profile);
        tracing::info!(?recommendation, "backend dispatch");
        Ok(recommendation)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn infer_family(meta: &ModelMetadata) -> ModelFamily {
        let names: Vec<&str> = meta.inputs.iter().map(|t| t.name.as_str()).collect();

        let has_pixel_values = names.contains(&"pixel_values");
        let has_input_ids = names.contains(&"input_ids");
        let has_past_kv = names
            .iter()
            .any(|n| n.contains("past_key_values") || n.contains("past_key") || n.starts_with("past_"));
        let has_attention_mask = names.contains(&"attention_mask");

        // Priority order: most specific first.
        if has_pixel_values && has_input_ids {
            return ModelFamily::ImageTextDual;
        }

        if has_past_kv && has_input_ids {
            return ModelFamily::LanguageModel;
        }

        if has_input_ids || has_attention_mask {
            return ModelFamily::TextEncoder;
        }

        // 4-D NCHW input with 3 channels in dim-1 → vision
        if meta.inputs.iter().any(|t| Self::is_nchw_rgb(&t.shape)) {
            return ModelFamily::Vision;
        }

        ModelFamily::Unknown
    }

    /// Returns `true` for shapes like `[N, 3, H, W]` where N/H/W may be dynamic.
    fn is_nchw_rgb(shape: &[i64]) -> bool {
        shape.len() == 4 && shape[1] == 3
    }

    /// Infer dominant op class from model family.
    ///
    /// TODO(v0.3): replace with an ONNX graph op-histogram pass that counts
    /// Conv, MatMul, Attention, and control-flow nodes directly.
    fn infer_op_class(family: ModelFamily) -> OpClass {
        match family {
            ModelFamily::Vision => OpClass::ConvHeavy,
            ModelFamily::TextEncoder => OpClass::AttentionHeavy,
            ModelFamily::ImageTextDual => OpClass::Mixed,
            ModelFamily::LanguageModel => OpClass::ControlFlow,
            ModelFamily::Unknown => OpClass::Mixed,
        }
    }
}
