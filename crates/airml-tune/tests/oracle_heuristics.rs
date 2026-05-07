//! Integration tests for [`BackendOracle`] heuristics.
//!
//! All tests use synthetic [`ModelMetadata`] instances; no real ONNX models
//! are loaded. The goal is to verify that each model family maps to the
//! expected [`BackendRecommendation`].

use airml_core::ModelMetadata;
use airml_core::TensorInfo;
use airml_tune::oracle::{BackendOracle, BackendRecommendation, ModelFamily, OpClass};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn make_metadata(inputs: Vec<TensorInfo>, outputs: Vec<TensorInfo>) -> ModelMetadata {
    ModelMetadata {
        name: None,
        description: None,
        version: None,
        producer: None,
        inputs,
        outputs,
    }
}

fn tensor(name: &str, shape: Vec<i64>) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        shape,
        dtype: "Float32".to_string(),
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

/// MobileNetV3-Small: single 4-D NCHW image input with 3 channels.
/// Expected: Vision / ConvHeavy → CoreMLAneOnly.
#[test]
fn test_mobilenet_vision_recommends_ane_only() {
    let meta = make_metadata(
        vec![tensor("input", vec![1, 3, 224, 224])],
        vec![tensor("output", vec![1, 1000])],
    );
    let oracle = BackendOracle::new();
    let profile = oracle.profile_from_metadata(&meta);

    assert_eq!(profile.model_family, ModelFamily::Vision);
    assert_eq!(profile.dominant_op_class, OpClass::ConvHeavy);
    assert!(!profile.has_dynamic_shapes);

    let rec = oracle.recommend(&profile);
    assert_eq!(rec, BackendRecommendation::CoreMLAneOnly);
}

/// all-MiniLM-L6-v2 (static shapes): `input_ids` + `attention_mask`.
/// Expected: TextEncoder / AttentionHeavy / static → CoreMLAneOnly.
#[test]
fn test_minilm_static_recommends_ane_only() {
    let meta = make_metadata(
        vec![
            tensor("input_ids", vec![1, 128]),
            tensor("attention_mask", vec![1, 128]),
        ],
        vec![tensor("last_hidden_state", vec![1, 128, 384])],
    );
    let oracle = BackendOracle::new();
    let profile = oracle.profile_from_metadata(&meta);

    assert_eq!(profile.model_family, ModelFamily::TextEncoder);
    assert_eq!(profile.dominant_op_class, OpClass::AttentionHeavy);
    assert!(!profile.has_dynamic_shapes);

    let rec = oracle.recommend(&profile);
    assert_eq!(rec, BackendRecommendation::CoreMLAneOnly);
}

/// all-MiniLM-L6-v2 with dynamic sequence length (`-1`).
/// Expected: TextEncoder / AttentionHeavy / dynamic → CoreMLAll.
#[test]
fn test_minilm_dynamic_recommends_coreml_all() {
    let meta = make_metadata(
        vec![
            tensor("input_ids", vec![1, -1]),
            tensor("attention_mask", vec![1, -1]),
        ],
        vec![tensor("last_hidden_state", vec![1, -1, 384])],
    );
    let oracle = BackendOracle::new();
    let profile = oracle.profile_from_metadata(&meta);

    assert_eq!(profile.model_family, ModelFamily::TextEncoder);
    assert!(profile.has_dynamic_shapes);

    let rec = oracle.recommend(&profile);
    assert_eq!(rec, BackendRecommendation::CoreMLAll);
}

/// CLIP-ViT-B/32: `pixel_values` + `input_ids` → ImageTextDual.
/// Expected: Mixed → CoreMLAll.
#[test]
fn test_clip_image_text_dual_recommends_coreml_all() {
    let meta = make_metadata(
        vec![
            tensor("pixel_values", vec![1, 3, 224, 224]),
            tensor("input_ids", vec![1, 77]),
            tensor("attention_mask", vec![1, 77]),
        ],
        vec![
            tensor("image_embeds", vec![1, 512]),
            tensor("text_embeds", vec![1, 512]),
        ],
    );
    let oracle = BackendOracle::new();
    let profile = oracle.profile_from_metadata(&meta);

    assert_eq!(profile.model_family, ModelFamily::ImageTextDual);
    assert_eq!(profile.dominant_op_class, OpClass::Mixed);

    let rec = oracle.recommend(&profile);
    assert_eq!(rec, BackendRecommendation::CoreMLAll);
}

/// Llama-3.2-1B: `input_ids` + `past_key_values` inputs → LanguageModel.
/// Expected: ControlFlow → CoreMLGpuOnly.
#[test]
fn test_llama_language_model_recommends_gpu_only() {
    let meta = make_metadata(
        vec![
            tensor("input_ids", vec![1, -1]),
            tensor("past_key_values.0.key", vec![1, 32, -1, 128]),
            tensor("past_key_values.0.value", vec![1, 32, -1, 128]),
            tensor("attention_mask", vec![1, -1]),
        ],
        vec![tensor("logits", vec![1, -1, 32000])],
    );
    let oracle = BackendOracle::new();
    let profile = oracle.profile_from_metadata(&meta);

    assert_eq!(profile.model_family, ModelFamily::LanguageModel);
    assert_eq!(profile.dominant_op_class, OpClass::ControlFlow);

    let rec = oracle.recommend(&profile);
    assert_eq!(rec, BackendRecommendation::CoreMLGpuOnly);
}

/// Unknown family with dynamic shapes → CPU fallback with reason.
#[test]
fn test_unknown_dynamic_recommends_cpu_with_reason() {
    let meta = make_metadata(
        vec![tensor("custom_input", vec![1, -1, 256])],
        vec![tensor("custom_output", vec![1, -1])],
    );
    let oracle = BackendOracle::new();
    let profile = oracle.profile_from_metadata(&meta);

    assert_eq!(profile.model_family, ModelFamily::Unknown);
    assert!(profile.has_dynamic_shapes);

    let rec = oracle.recommend(&profile);
    assert!(matches!(rec, BackendRecommendation::CpuOnlyWithReason(_)));
    if let BackendRecommendation::CpuOnlyWithReason(reason) = rec {
        assert!(reason.contains("unknown family"));
    }
}

/// Unknown family with static shapes → CoreMLAll (optimistic default).
#[test]
fn test_unknown_static_recommends_coreml_all() {
    let meta = make_metadata(
        vec![tensor("custom_input", vec![1, 256, 256])],
        vec![tensor("custom_output", vec![1, 128])],
    );
    let oracle = BackendOracle::new();
    let profile = oracle.profile_from_metadata(&meta);

    assert_eq!(profile.model_family, ModelFamily::Unknown);
    assert!(!profile.has_dynamic_shapes);

    let rec = oracle.recommend(&profile);
    assert_eq!(rec, BackendRecommendation::CoreMLAll);
}

/// Convenience method `recommend_for_metadata` produces the same result
/// as the two-step `profile_from_metadata` + `recommend` for a vision model.
#[test]
fn test_recommend_for_metadata_convenience() {
    let meta = make_metadata(
        vec![tensor("input", vec![1, 3, 224, 224])],
        vec![tensor("output", vec![1, 1000])],
    );
    let oracle = BackendOracle::new();

    let two_step = oracle.recommend(&oracle.profile_from_metadata(&meta));
    let one_step = oracle.recommend_for_metadata(&meta);

    assert_eq!(two_step, one_step);
}

/// `BackendOracle::default()` works and behaves identically to `::new()`.
#[test]
#[allow(clippy::default_constructed_unit_structs)]
fn test_oracle_default_impl() {
    let meta = make_metadata(
        vec![tensor("input", vec![1, 3, 224, 224])],
        vec![tensor("output", vec![1, 1000])],
    );
    let a = BackendOracle::new().recommend_for_metadata(&meta);
    let b = BackendOracle::default().recommend_for_metadata(&meta);
    assert_eq!(a, b);
}
