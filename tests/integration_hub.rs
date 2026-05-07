//! Integration tests for the airml-hub ModelCache using the synthetic-identity
//! ONNX fixture.  These tests do NOT require ORT — they only exercise
//! filesystem-level cache operations.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Compute the lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Path to the synthetic-identity.onnx fixture bundled with the repo.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/synthetic-identity.onnx")
}

#[test]
fn hub_cache_store_and_load_roundtrip() {
    let fixture = fixture_path();
    assert!(fixture.exists(), "synthetic-identity.onnx fixture missing");

    let bytes = std::fs::read(&fixture).expect("reading fixture");
    let digest = sha256_hex(&bytes);

    // Use a tempdir so the test is hermetic and leaves no artefacts.
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = airml_hub::ModelCache::new(tmp.path().to_path_buf());

    // Blob must not be present before storing.
    assert!(!cache.contains(&digest), "cache should be empty initially");

    // Store atomically.
    let stored_path = cache.store_atomic(&digest, &bytes).expect("store_atomic");
    assert!(stored_path.exists(), "stored blob path must exist on disk");

    // Contains must now return true.
    assert!(cache.contains(&digest), "cache must report blob present after store");

    // Load must return identical bytes.
    let loaded = cache.load(&digest).expect("load");
    assert_eq!(loaded, bytes, "loaded bytes must match original");
}

#[test]
fn hub_cache_sha256_mismatch_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = airml_hub::ModelCache::new(tmp.path().to_path_buf());

    let bytes = b"some model bytes";
    // Deliberately wrong digest — all zeros (64 hex chars).
    let wrong_digest = "0".repeat(64);

    let result = cache.store_atomic(&wrong_digest, bytes);
    assert!(
        result.is_err(),
        "store_atomic must reject a mismatched sha256"
    );
}

#[test]
fn hub_cache_stat_reflects_stored_blob() {
    let fixture = fixture_path();
    let bytes = std::fs::read(&fixture).expect("reading fixture");
    let digest = sha256_hex(&bytes);

    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = airml_hub::ModelCache::new(tmp.path().to_path_buf());

    let stat_before = cache.stat();
    assert_eq!(stat_before.num_models, 0);
    assert_eq!(stat_before.total_bytes, 0);

    cache.store_atomic(&digest, &bytes).expect("store");

    let stat_after = cache.stat();
    assert_eq!(stat_after.num_models, 1);
    assert_eq!(stat_after.total_bytes, bytes.len() as u64);
}

#[test]
fn hub_cache_store_is_idempotent() {
    let fixture = fixture_path();
    let bytes = std::fs::read(&fixture).expect("reading fixture");
    let digest = sha256_hex(&bytes);

    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = airml_hub::ModelCache::new(tmp.path().to_path_buf());

    // Store twice — must succeed both times.
    let p1 = cache.store_atomic(&digest, &bytes).expect("first store");
    let p2 = cache.store_atomic(&digest, &bytes).expect("second store");
    assert_eq!(p1, p2, "idempotent store must return same path");
    assert_eq!(cache.stat().num_models, 1, "must not duplicate the entry");
}
