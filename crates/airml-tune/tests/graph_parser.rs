//! Integration tests for the ONNX graph parser.
//!
//! These tests verify end-to-end behaviour of [`airml_tune::graph_parser`]
//! without requiring real ONNX model files — minimal ONNX protobuf bytes are
//! constructed inline.

use airml_tune::graph_parser::{histogram_from_bytes, GraphParseError, OpHistogram};
use airml_tune::OpClass;

// ---------------------------------------------------------------------------
// Minimal protobuf builder helpers
// ---------------------------------------------------------------------------

fn encode_varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        } else {
            out.push(b | 0x80);
        }
    }
    out
}

fn encode_len_delimited(field: u32, data: &[u8]) -> Vec<u8> {
    let tag = (field << 3) | 2;
    let mut out = encode_varint(tag as u64);
    out.extend(encode_varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

fn node_proto(op_type: &str) -> Vec<u8> {
    // NodeProto.op_type = field 4, wire type 2 (string)
    encode_len_delimited(4, op_type.as_bytes())
}

fn graph_proto(nodes: &[&str]) -> Vec<u8> {
    // GraphProto.node = field 1, repeated
    let mut out = Vec::new();
    for op in nodes {
        out.extend(encode_len_delimited(1, &node_proto(op)));
    }
    out
}

fn model_proto(nodes: &[&str]) -> Vec<u8> {
    // ModelProto.graph = field 7
    encode_len_delimited(7, &graph_proto(nodes))
}

// ---------------------------------------------------------------------------
// Test 1: malformed / garbage input returns Decode error, never panics
// ---------------------------------------------------------------------------

#[test]
fn test_empty_bytes_errors() {
    // Completely empty → no graph field found → Decode error
    let result = histogram_from_bytes(&[]);
    assert!(
        matches!(result, Err(GraphParseError::Decode(_))),
        "expected Decode error on empty input, got: {:?}",
        result
    );
}

#[test]
fn test_garbage_bytes_errors_not_panics() {
    // 10 zero bytes → not a valid ONNX model; must return Err, never panic
    let result = histogram_from_bytes(&[0u8; 10]);
    assert!(result.is_err(), "expected error on all-zero input, got Ok");
}

#[test]
fn test_truncated_varint_returns_decode_error() {
    // A single 0x80 byte is a truncated multi-byte varint — should not panic
    let result = histogram_from_bytes(&[0x80]);
    assert!(result.is_err(), "expected error on truncated varint");
}

// ---------------------------------------------------------------------------
// Test 2: minimal valid ONNX bytes parse correctly
// ---------------------------------------------------------------------------

#[test]
fn test_real_minimal_onnx_parses() {
    // Construct a tiny model with 3 Conv nodes and 2 Relu nodes.
    let bytes = model_proto(&["Conv", "Relu", "Conv", "Relu", "Conv"]);
    let hist = histogram_from_bytes(&bytes).expect("minimal ONNX should parse");

    assert_eq!(hist.total, 5);
    assert_eq!(hist.counts.get("Conv").copied().unwrap_or(0), 3);
    assert_eq!(hist.counts.get("Relu").copied().unwrap_or(0), 2);
    // 3/5 = 0.60 > 0.30 → ConvHeavy
    assert_eq!(hist.dominant_class(), OpClass::ConvHeavy);
}

// ---------------------------------------------------------------------------
// Test 3: OpHistogram::dominant_class threshold cases
// ---------------------------------------------------------------------------

fn histogram_from_pairs(pairs: &[(&str, usize)]) -> OpHistogram {
    let mut h = OpHistogram::default();
    for (op, count) in pairs {
        for _ in 0..*count {
            h.counts.entry(op.to_string()).and_modify(|n| *n += 1).or_insert(1);
            h.total += 1;
        }
    }
    h
}

#[test]
fn test_dominant_class_thresholds() {
    // ConvHeavy: Conv fraction 35/100 = 0.35 > 0.30
    let h = histogram_from_pairs(&[("Conv", 35), ("Relu", 65)]);
    assert_eq!(h.dominant_class(), OpClass::ConvHeavy, "ConvHeavy threshold");

    // GemmHeavy: MatMul fraction 45/100 = 0.45 > 0.40
    let h = histogram_from_pairs(&[("MatMul", 45), ("Add", 55)]);
    assert_eq!(h.dominant_class(), OpClass::GemmHeavy, "GemmHeavy threshold");

    // AttentionHeavy: Attention fraction 25/100 = 0.25 > 0.20
    let h = histogram_from_pairs(&[("Attention", 25), ("Add", 75)]);
    assert_eq!(h.dominant_class(), OpClass::AttentionHeavy, "AttentionHeavy threshold");

    // ControlFlow: Loop fraction 6/100 = 0.06 > 0.05
    let h = histogram_from_pairs(&[("Loop", 6), ("Add", 94)]);
    assert_eq!(h.dominant_class(), OpClass::ControlFlow, "ControlFlow threshold");

    // Mixed: all fractions below their thresholds
    let h = histogram_from_pairs(&[("Conv", 10), ("MatMul", 15), ("Add", 75)]);
    assert_eq!(h.dominant_class(), OpClass::Mixed, "Mixed fallback");
}

#[test]
fn test_dominant_class_boundary_not_exceeded() {
    // Exactly at threshold boundary → should NOT trigger
    // Conv: exactly 30/100 = 0.30 (not strictly > 0.30)
    let h = histogram_from_pairs(&[("Conv", 30), ("Add", 70)]);
    assert_ne!(
        h.dominant_class(),
        OpClass::ConvHeavy,
        "exact threshold boundary should not trigger ConvHeavy"
    );
}

#[test]
fn test_empty_graph_returns_empty_graph_error() {
    let bytes = model_proto(&[]);
    let result = histogram_from_bytes(&bytes);
    assert!(
        matches!(result, Err(GraphParseError::EmptyGraph)),
        "expected EmptyGraph, got: {:?}",
        result
    );
}

#[test]
fn test_layernorm_counts_toward_attention_class() {
    // LayerNormalization counts toward AttentionHeavy
    let h = histogram_from_pairs(&[("LayerNormalization", 22), ("Add", 78)]);
    assert_eq!(h.dominant_class(), OpClass::AttentionHeavy);
}

#[test]
fn test_op_names_are_case_sensitive() {
    // "conv" (lowercase) should NOT count toward ConvHeavy
    let h = histogram_from_pairs(&[("conv", 50), ("Add", 50)]);
    assert_ne!(
        h.dominant_class(),
        OpClass::ConvHeavy,
        "op matching must be case-sensitive per ONNX spec"
    );
}

/// Real-world fixture test — only runs when AIRML_TEST_ONNX_PATH is set.
#[test]
#[ignore = "requires AIRML_TEST_ONNX_PATH env var pointing to a real .onnx file"]
fn test_real_model_parses_and_classifies() {
    let path = std::env::var("AIRML_TEST_ONNX_PATH")
        .expect("set AIRML_TEST_ONNX_PATH to a valid .onnx file path");
    let hist = airml_tune::histogram_from_path(std::path::Path::new(&path))
        .expect("real model should parse without error");
    assert!(hist.total > 0, "real model must have at least one node");
    println!("total={} class={:?}", hist.total, hist.dominant_class());
}
