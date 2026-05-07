//! Integration tests for airml-tune graph parsing against the
//! synthetic-identity.onnx fixture.  No ORT required.

use std::path::PathBuf;

use airml_tune::{histogram_from_path, OpHistogram};

/// Path to the synthetic-identity.onnx fixture bundled with the repo.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/synthetic-identity.onnx")
}

#[test]
fn tune_histogram_from_identity_model_has_ops() {
    let path = fixture_path();
    assert!(path.exists(), "synthetic-identity.onnx fixture missing");

    let hist: OpHistogram = histogram_from_path(&path)
        .expect("histogram_from_path must succeed on the synthetic identity model");

    // The synthetic-identity.onnx has exactly one Identity op.
    assert!(hist.total >= 1, "expected at least one op in the Identity model");
    assert!(
        hist.counts.contains_key("Identity"),
        "expected 'Identity' op in counts, got: {:?}",
        hist.counts
    );
}

#[test]
fn tune_dominant_class_does_not_panic_on_identity_model() {
    let path = fixture_path();
    let hist = histogram_from_path(&path)
        .expect("histogram_from_path must succeed on the synthetic identity model");

    // dominant_class() must not panic regardless of the result.
    // A single Identity op does not meet any class threshold → Mixed.
    let class = hist.dominant_class();
    assert_eq!(
        class,
        airml_tune::OpClass::Mixed,
        "a single Identity op should fall back to Mixed"
    );
}

#[test]
fn tune_histogram_from_bytes_matches_from_path() {
    let path = fixture_path();
    let bytes = std::fs::read(&path).expect("reading fixture");

    let from_path = histogram_from_path(&path).expect("from_path");
    let from_bytes = airml_tune::histogram_from_bytes(&bytes).expect("from_bytes");

    assert_eq!(
        from_path.total, from_bytes.total,
        "histogram_from_path and histogram_from_bytes must agree on total"
    );
    assert_eq!(
        from_path.counts, from_bytes.counts,
        "histogram_from_path and histogram_from_bytes must agree on counts"
    );
}
