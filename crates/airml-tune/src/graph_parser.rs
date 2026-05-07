//! ONNX graph parser — extracts an op-type histogram from a model file.
//!
//! # Approach
//!
//! Rather than pulling in `prost` + `onnx-pb` (which pins prost 0.6 and requires
//! `protoc` at build time), this module implements a minimal protobuf wire-format
//! reader sufficient to walk the ONNX `ModelProto` structure and collect
//! `NodeProto.op_type` strings.
//!
//! ## ONNX protobuf field map (relevant subset)
//!
//! ```text
//! ModelProto  { graph: GraphProto = field 7 }
//! GraphProto  { node: repeated NodeProto = field 1 }
//! NodeProto   { op_type: string = field 4 }
//! ```
//!
//! All three are length-delimited (wire type 2).
//!
//! # Memory limits
//!
//! The current implementation reads the entire file into memory with
//! `std::fs::read`. This is fine for models up to ~2 GB.
//!
//! TODO: streaming parse for models larger than 2 GB.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use thiserror::Error;

use crate::oracle::OpClass;

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

/// Errors that can arise during ONNX graph parsing.
#[derive(Debug, Error)]
pub enum GraphParseError {
    /// I/O error reading the model file.
    #[error("I/O error reading ONNX model: {0}")]
    Io(#[from] io::Error),

    /// Protobuf wire-format decode error or unexpected structure.
    #[error("ONNX decode error: {0}")]
    Decode(String),

    /// The model graph contains no nodes.
    #[error("ONNX graph is empty (no nodes found)")]
    EmptyGraph,
}

// ---------------------------------------------------------------------------
// OpHistogram
// ---------------------------------------------------------------------------

/// Count of each `op_type` string found in the ONNX graph.
///
/// Construct via [`histogram_from_bytes`] or [`histogram_from_path`], then
/// call [`OpHistogram::dominant_class`] to obtain an [`OpClass`].
#[derive(Debug, Clone, Default)]
pub struct OpHistogram {
    /// Raw per-op counts. Keys are ONNX op_type strings (case-sensitive).
    pub counts: HashMap<String, usize>,
    /// Total number of nodes counted.
    pub total: usize,
}

impl OpHistogram {
    /// Increment the count for `op_type` by one.
    fn record(&mut self, op_type: String) {
        *self.counts.entry(op_type).or_insert(0) += 1;
        self.total += 1;
    }

    /// Determine the dominant [`OpClass`] using fractional thresholds.
    ///
    /// Thresholds (applied in priority order):
    ///
    /// | Class | Ops counted | Fraction threshold |
    /// |---|---|---|
    /// | [`OpClass::ConvHeavy`] | Conv, ConvTranspose, DepthwiseConv | > 0.30 |
    /// | [`OpClass::GemmHeavy`] | MatMul, Gemm | > 0.40 |
    /// | [`OpClass::AttentionHeavy`] | Attention, MultiHeadAttention, LayerNormalization | > 0.20 |
    /// | [`OpClass::ControlFlow`] | If, Loop, Scan, Where | > 0.05 |
    /// | [`OpClass::Mixed`] | (fallback) | — |
    ///
    /// If `total == 0`, returns [`OpClass::Mixed`].
    pub fn dominant_class(&self) -> OpClass {
        if self.total == 0 {
            return OpClass::Mixed;
        }

        let total = self.total as f64;

        let conv_count = self.sum_ops(&["Conv", "ConvTranspose", "DepthwiseConv"]);
        let gemm_count = self.sum_ops(&["MatMul", "Gemm"]);
        let attn_count =
            self.sum_ops(&["Attention", "MultiHeadAttention", "LayerNormalization"]);
        let ctrl_count = self.sum_ops(&["If", "Loop", "Scan", "Where"]);

        // Priority order: conv, gemm, attention, control-flow.
        if conv_count as f64 / total > 0.30 {
            return OpClass::ConvHeavy;
        }
        if gemm_count as f64 / total > 0.40 {
            return OpClass::GemmHeavy;
        }
        if attn_count as f64 / total > 0.20 {
            return OpClass::AttentionHeavy;
        }
        if ctrl_count as f64 / total > 0.05 {
            return OpClass::ControlFlow;
        }

        OpClass::Mixed
    }

    /// Sum counts for the given op-type names.
    fn sum_ops(&self, names: &[&str]) -> usize {
        names.iter().map(|n| self.counts.get(*n).copied().unwrap_or(0)).sum()
    }
}

// ---------------------------------------------------------------------------
// Public parse API
// ---------------------------------------------------------------------------

/// Build an [`OpHistogram`] by parsing ONNX protobuf bytes.
///
/// Returns [`GraphParseError::Decode`] on malformed input — never panics.
pub fn histogram_from_bytes(bytes: &[u8]) -> Result<OpHistogram, GraphParseError> {
    let model = decode_model(bytes)?;
    if model.total == 0 {
        return Err(GraphParseError::EmptyGraph);
    }
    Ok(model)
}

/// Build an [`OpHistogram`] from an ONNX file on disk.
///
/// TODO: streaming parse for models larger than 2 GB.
pub fn histogram_from_path(path: &Path) -> Result<OpHistogram, GraphParseError> {
    let bytes = std::fs::read(path).map_err(GraphParseError::Io)?;
    histogram_from_bytes(&bytes)
}

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format decoder
// ---------------------------------------------------------------------------
//
// Wire types:
//   0 = varint
//   1 = 64-bit
//   2 = length-delimited
//   5 = 32-bit
//
// Field tag = (field_number << 3) | wire_type
//
// ONNX field numbers used here:
//   ModelProto.graph  = field 7, wire type 2
//   GraphProto.node   = field 1, wire type 2
//   NodeProto.op_type = field 4, wire type 2

/// Decode a varint from `buf` starting at `*pos`. Returns the varint value and
/// advances `*pos`. Returns `Err` on truncated input.
fn decode_varint(buf: &[u8], pos: &mut usize) -> Result<u64, GraphParseError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        if *pos >= buf.len() {
            return Err(GraphParseError::Decode(
                "truncated varint".to_string(),
            ));
        }
        let b = buf[*pos];
        *pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(GraphParseError::Decode(
                "varint overflow (>64 bits)".to_string(),
            ));
        }
    }
}

/// Skip `n` bytes of `buf` from `*pos`, advancing it. Returns `Err` on
/// truncated input.
fn skip_bytes(buf: &[u8], pos: &mut usize, n: usize) -> Result<(), GraphParseError> {
    if *pos + n > buf.len() {
        return Err(GraphParseError::Decode(format!(
            "unexpected end of buffer: need {n} bytes at offset {pos}",
            n = n,
            pos = *pos,
        )));
    }
    *pos += n;
    Ok(())
}

/// Read a length-delimited field (wire type 2) and return its byte slice.
/// The length varint is consumed from `*pos`.
fn read_len_delimited<'a>(buf: &'a [u8], pos: &mut usize) -> Result<&'a [u8], GraphParseError> {
    let len = decode_varint(buf, pos)? as usize;
    if *pos + len > buf.len() {
        return Err(GraphParseError::Decode(format!(
            "length-delimited field claims {} bytes but only {} remain",
            len,
            buf.len() - *pos,
        )));
    }
    let slice = &buf[*pos..*pos + len];
    *pos += len;
    Ok(slice)
}

/// Skip one unknown field value given its wire type. `*pos` is advanced past
/// the field's data.
fn skip_field(buf: &[u8], pos: &mut usize, wire_type: u64) -> Result<(), GraphParseError> {
    match wire_type {
        0 => {
            decode_varint(buf, pos)?;
        }
        1 => skip_bytes(buf, pos, 8)?,
        2 => {
            let len = decode_varint(buf, pos)? as usize;
            skip_bytes(buf, pos, len)?;
        }
        5 => skip_bytes(buf, pos, 4)?,
        _ => {
            return Err(GraphParseError::Decode(format!(
                "unknown protobuf wire type {wire_type} at offset {pos}",
                wire_type = wire_type,
                pos = *pos,
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ONNX-specific decoders
// ---------------------------------------------------------------------------

/// Parse a `NodeProto` byte slice and return its `op_type` string (field 4).
///
/// All other fields are skipped. Returns `Ok(None)` if field 4 is absent.
fn decode_node_op_type(buf: &[u8]) -> Result<Option<String>, GraphParseError> {
    let mut pos = 0;
    let mut op_type: Option<String> = None;
    while pos < buf.len() {
        let tag = decode_varint(buf, &mut pos)?;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;

        if field_number == 4 && wire_type == 2 {
            // op_type is a string (UTF-8 bytes).
            let bytes = read_len_delimited(buf, &mut pos)?;
            let s = std::str::from_utf8(bytes).map_err(|e| {
                GraphParseError::Decode(format!("op_type is not valid UTF-8: {e}"))
            })?;
            op_type = Some(s.to_string());
        } else {
            skip_field(buf, &mut pos, wire_type)?;
        }
    }
    Ok(op_type)
}

/// Parse a `GraphProto` byte slice and accumulate op counts into `hist`.
///
/// Only field 1 (`node`, repeated `NodeProto`) is processed; all others skipped.
fn decode_graph(buf: &[u8], hist: &mut OpHistogram) -> Result<(), GraphParseError> {
    let mut pos = 0;
    while pos < buf.len() {
        let tag = decode_varint(buf, &mut pos)?;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;

        if field_number == 1 && wire_type == 2 {
            // node: NodeProto
            let node_bytes = read_len_delimited(buf, &mut pos)?;
            if let Some(op) = decode_node_op_type(node_bytes)? {
                if !op.is_empty() {
                    hist.record(op);
                }
            }
        } else {
            skip_field(buf, &mut pos, wire_type)?;
        }
    }
    Ok(())
}

/// Parse a `ModelProto` byte slice and return an [`OpHistogram`].
///
/// Only field 7 (`graph`, `GraphProto`) is processed; all others skipped.
fn decode_model(buf: &[u8]) -> Result<OpHistogram, GraphParseError> {
    let mut pos = 0;
    let mut hist = OpHistogram::default();
    let mut found_graph = false;

    while pos < buf.len() {
        let tag = decode_varint(buf, &mut pos)?;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;

        if field_number == 7 && wire_type == 2 {
            // graph: GraphProto
            let graph_bytes = read_len_delimited(buf, &mut pos)?;
            decode_graph(graph_bytes, &mut hist)?;
            found_graph = true;
        } else {
            skip_field(buf, &mut pos, wire_type)?;
        }
    }

    if !found_graph {
        // Not a valid ONNX model — no graph field found.
        return Err(GraphParseError::Decode(
            "no graph field (field 7) found in ModelProto — is this an ONNX file?".to_string(),
        ));
    }

    Ok(hist)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Protobuf helpers for building minimal ONNX bytes in tests
    // -----------------------------------------------------------------------

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

    fn encode_string_field(field: u32, s: &str) -> Vec<u8> {
        encode_len_delimited(field, s.as_bytes())
    }

    /// Build a minimal NodeProto with the given op_type (field 4).
    fn node_proto(op_type: &str) -> Vec<u8> {
        encode_string_field(4, op_type)
    }

    /// Build a GraphProto (field 7 of ModelProto) containing the given nodes.
    fn graph_proto(nodes: &[&str]) -> Vec<u8> {
        let mut graph_bytes = Vec::new();
        for op in nodes {
            graph_bytes.extend(encode_len_delimited(1, &node_proto(op)));
        }
        graph_bytes
    }

    /// Wrap a graph payload into a minimal ModelProto byte string.
    fn model_proto(nodes: &[&str]) -> Vec<u8> {
        let g = graph_proto(nodes);
        encode_len_delimited(7, &g)
    }

    // -----------------------------------------------------------------------
    // Unit tests for OpHistogram::dominant_class thresholds
    // -----------------------------------------------------------------------

    #[test]
    fn test_dominant_class_empty_is_mixed() {
        let h = OpHistogram::default();
        assert_eq!(h.dominant_class(), OpClass::Mixed);
    }

    #[test]
    fn test_dominant_class_conv_heavy_threshold() {
        // 31 Conv out of 100 → > 0.30 → ConvHeavy
        let mut h = OpHistogram::default();
        for _ in 0..31 {
            h.record("Conv".to_string());
        }
        for _ in 0..69 {
            h.record("Relu".to_string());
        }
        assert_eq!(h.dominant_class(), OpClass::ConvHeavy);
    }

    #[test]
    fn test_dominant_class_conv_below_threshold_is_not_conv() {
        // Exactly 30% Conv — threshold is *strictly greater than* 0.30
        let mut h = OpHistogram::default();
        for _ in 0..30 {
            h.record("Conv".to_string());
        }
        for _ in 0..70 {
            h.record("Relu".to_string());
        }
        // 30/100 = 0.30, not > 0.30 → should NOT be ConvHeavy
        assert_ne!(h.dominant_class(), OpClass::ConvHeavy);
    }

    #[test]
    fn test_dominant_class_gemm_heavy_threshold() {
        // 41 MatMul out of 100 → > 0.40 → GemmHeavy
        let mut h = OpHistogram::default();
        for _ in 0..41 {
            h.record("MatMul".to_string());
        }
        for _ in 0..59 {
            h.record("Add".to_string());
        }
        assert_eq!(h.dominant_class(), OpClass::GemmHeavy);
    }

    #[test]
    fn test_dominant_class_attention_heavy_threshold() {
        // 21 Attention out of 100 → > 0.20 → AttentionHeavy
        let mut h = OpHistogram::default();
        for _ in 0..21 {
            h.record("Attention".to_string());
        }
        for _ in 0..79 {
            h.record("LayerNorm".to_string());
        }
        assert_eq!(h.dominant_class(), OpClass::AttentionHeavy);
    }

    #[test]
    fn test_dominant_class_control_flow_threshold() {
        // 6 Loop out of 100 → > 0.05 → ControlFlow
        let mut h = OpHistogram::default();
        for _ in 0..6 {
            h.record("Loop".to_string());
        }
        for _ in 0..94 {
            h.record("Add".to_string());
        }
        assert_eq!(h.dominant_class(), OpClass::ControlFlow);
    }

    #[test]
    fn test_dominant_class_mixed_fallback() {
        // Low fractions for all classes → Mixed
        let mut h = OpHistogram::default();
        for _ in 0..5 {
            h.record("Conv".to_string());
        }
        for _ in 0..5 {
            h.record("MatMul".to_string());
        }
        for _ in 0..90 {
            h.record("Relu".to_string());
        }
        assert_eq!(h.dominant_class(), OpClass::Mixed);
    }

    #[test]
    fn test_dominant_class_conv_wins_over_gemm_by_priority() {
        // Both Conv and MatMul above threshold — Conv is checked first
        let mut h = OpHistogram::default();
        // 35 Conv (> 0.30) and 45 MatMul (> 0.40) out of... wait, 35+45=80
        // total=80, conv=35/80=0.4375 > 0.30, gemm=45/80=0.5625 > 0.40
        // Conv check fires first in our priority order.
        for _ in 0..35 {
            h.record("Conv".to_string());
        }
        for _ in 0..45 {
            h.record("MatMul".to_string());
        }
        assert_eq!(h.dominant_class(), OpClass::ConvHeavy);
    }

    // -----------------------------------------------------------------------
    // Unit tests for the wire-format decoder
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_bytes_returns_decode_error() {
        // Completely empty bytes → no graph field found
        let result = histogram_from_bytes(&[]);
        assert!(
            matches!(result, Err(GraphParseError::Decode(_))),
            "expected Decode error, got: {result:?}",
            result = result
        );
    }

    #[test]
    fn test_garbage_bytes_returns_decode_error_not_panic() {
        // 10 garbage bytes should not panic
        let result = histogram_from_bytes(&[0u8; 10]);
        // Could be Decode or EmptyGraph — must NOT be Ok
        assert!(result.is_err(), "expected error on garbage input");
    }

    #[test]
    fn test_minimal_onnx_model_parses() {
        let bytes = model_proto(&["Conv", "Relu", "Conv", "GlobalAveragePool"]);
        let hist = histogram_from_bytes(&bytes).expect("should parse minimal ONNX");
        assert_eq!(hist.total, 4);
        assert_eq!(hist.counts["Conv"], 2);
        assert_eq!(hist.counts["Relu"], 1);
        assert_eq!(hist.counts["GlobalAveragePool"], 1);
        // 2/4 = 0.50 > 0.30 → ConvHeavy
        assert_eq!(hist.dominant_class(), OpClass::ConvHeavy);
    }

    #[test]
    fn test_conv_heavy_model_classified_correctly() {
        // 35 Conv, 10 Relu, 5 BN = 50 total → Conv fraction 0.70 > 0.30
        let mut ops: Vec<&str> = vec!["Conv"; 35];
        ops.extend(vec!["Relu"; 10]);
        ops.extend(vec!["BatchNormalization"; 5]);
        let bytes = model_proto(&ops);
        let hist = histogram_from_bytes(&bytes).expect("parse");
        assert_eq!(hist.dominant_class(), OpClass::ConvHeavy);
    }

    #[test]
    fn test_attention_model_classified_correctly() {
        // 25 Attention, 75 Add
        let mut ops: Vec<&str> = vec!["Attention"; 25];
        ops.extend(vec!["Add"; 75]);
        let bytes = model_proto(&ops);
        let hist = histogram_from_bytes(&bytes).expect("parse");
        assert_eq!(hist.dominant_class(), OpClass::AttentionHeavy);
    }

    #[test]
    fn test_empty_graph_returns_empty_graph_error() {
        // Graph with zero nodes → EmptyGraph
        let bytes = model_proto(&[]);
        let result = histogram_from_bytes(&bytes);
        assert!(
            matches!(result, Err(GraphParseError::EmptyGraph)),
            "expected EmptyGraph, got: {result:?}",
            result = result
        );
    }

    /// Real-world fixture test — skipped unless AIRML_TEST_ONNX_PATH env var is set.
    #[test]
    #[ignore = "requires AIRML_TEST_ONNX_PATH env var pointing to a real .onnx file"]
    fn test_real_model_from_env() {
        let path = std::env::var("AIRML_TEST_ONNX_PATH")
            .expect("AIRML_TEST_ONNX_PATH must be set to run this test");
        let hist = histogram_from_path(Path::new(&path)).expect("should parse real model");
        println!("total ops: {}", hist.total);
        println!("dominant class: {:?}", hist.dominant_class());
        println!("counts: {:?}", hist.counts);
        assert!(hist.total > 0);
    }
}
