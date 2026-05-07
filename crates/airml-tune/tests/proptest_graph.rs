//! Property-based tests for [`airml_tune::graph_parser`] and
//! [`airml_tune::OpHistogram::dominant_class`].

use airml_tune::graph_parser::{histogram_from_bytes, OpHistogram};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf builder (duplicated from graph_parser unit tests so
// this integration test file is self-contained)
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
    encode_len_delimited(4, op_type.as_bytes())
}

fn graph_proto(nodes: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for op in nodes {
        out.extend(encode_len_delimited(1, &node_proto(op)));
    }
    out
}

fn model_proto_from_strs(nodes: &[&str]) -> Vec<u8> {
    encode_len_delimited(7, &graph_proto(nodes))
}

// ---------------------------------------------------------------------------
// Property 1: arbitrary byte slices never panic
// ---------------------------------------------------------------------------

proptest! {
    /// For any byte slice up to 16 KiB, `histogram_from_bytes` returns a
    /// `Result` and never panics. Most random inputs will return `Err` —
    /// that is expected and correct.
    #[test]
    fn arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..=16384)) {
        let result = histogram_from_bytes(&bytes);
        // We only assert no panic; both Ok and Err are acceptable.
        let _ = result;
    }
}

// ---------------------------------------------------------------------------
// Property 2: dominant_class is commutative (order-independent)
// ---------------------------------------------------------------------------

/// A single (op_name, count) entry for building an OpHistogram.
#[derive(Debug, Clone)]
struct OpEntry {
    name: &'static str,
    count: usize,
}

/// Strategy that produces a list of (op_name, count) pairs from a fixed set
/// of ONNX-relevant op names. Total node count is kept bounded.
fn op_entries_strategy() -> impl Strategy<Value = Vec<OpEntry>> {
    const OPS: &[&str] = &[
        "Conv",
        "ConvTranspose",
        "MatMul",
        "Gemm",
        "Attention",
        "MultiHeadAttention",
        "LayerNormalization",
        "If",
        "Loop",
        "Scan",
        "Where",
        "Relu",
        "Add",
        "Mul",
        "BatchNormalization",
        "GlobalAveragePool",
    ];
    prop::collection::vec(
        (0usize..OPS.len(), 0usize..=20usize),
        1..=8,
    )
    .prop_map(|pairs| {
        pairs
            .into_iter()
            .map(|(idx, count)| OpEntry { name: OPS[idx % OPS.len()], count })
            .collect()
    })
}

/// Build an [`OpHistogram`] from an ordered list of `(op_name, count)` pairs.
fn build_histogram(entries: &[OpEntry]) -> OpHistogram {
    let mut h = OpHistogram::default();
    for entry in entries {
        for _ in 0..entry.count {
            h.counts
                .entry(entry.name.to_string())
                .and_modify(|n| *n += 1)
                .or_insert(1);
            h.total += 1;
        }
    }
    h
}

proptest! {
    /// `dominant_class` is commutative: the same set of (op, count) pairs
    /// produces the same `OpClass` regardless of the order they are inserted.
    ///
    /// We verify this by comparing a forward-order histogram with a
    /// reversed-order histogram built from the same entries.
    #[test]
    fn dominant_class_is_order_independent(entries in op_entries_strategy()) {
        let forward = build_histogram(&entries);

        let mut reversed = entries.clone();
        reversed.reverse();
        let backward = build_histogram(&reversed);

        prop_assert_eq!(
            forward.dominant_class(),
            backward.dominant_class(),
            "dominant_class should be the same regardless of insertion order; forward={:?} backward={:?}",
            forward,
            backward
        );
    }

    /// Well-formed minimal ONNX bytes built from a known op list parse
    /// successfully and produce a histogram whose total equals the node count.
    #[test]
    fn well_formed_onnx_parses_with_correct_total(entries in op_entries_strategy()) {
        // Flatten entries into a flat op-name list for model_proto_from_strs
        let ops: Vec<&str> = entries
            .iter()
            .flat_map(|e| std::iter::repeat(e.name).take(e.count))
            .collect();

        if ops.is_empty() {
            // model_proto with zero nodes returns EmptyGraph — acceptable
            return Ok(());
        }

        let bytes = model_proto_from_strs(&ops);
        let result = histogram_from_bytes(&bytes);
        prop_assert!(result.is_ok(), "well-formed ONNX bytes should parse: {result:?}");

        let hist = result.unwrap();
        prop_assert_eq!(
            hist.total,
            ops.len(),
            "histogram total should equal node count"
        );
    }
}
