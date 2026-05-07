//! Integration test: write → contains → load → stat → evict roundtrip.

use airml_hub::cache::{CacheStat, ModelCache};
use airml_hub::error::HubError;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// Compute the hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn make_cache() -> (TempDir, ModelCache) {
    let dir = TempDir::new().unwrap();
    let cache = ModelCache::new(dir.path().to_path_buf());
    (dir, cache)
}

#[test]
fn store_then_contains_returns_true() {
    let (_dir, cache) = make_cache();
    let data = b"hello airml hub";
    let sha = sha256_hex(data);

    cache.store_atomic(&sha, data).unwrap();
    assert!(cache.contains(&sha));
}

#[test]
fn load_returns_identical_bytes() {
    let (_dir, cache) = make_cache();
    let data = b"the quick brown fox";
    let sha = sha256_hex(data);

    cache.store_atomic(&sha, data).unwrap();
    let loaded = cache.load(&sha).unwrap();
    assert_eq!(loaded, data);
}

#[test]
fn stat_reports_one_model_after_one_store() {
    let (_dir, cache) = make_cache();
    let data = b"stat test payload";
    let sha = sha256_hex(data);

    cache.store_atomic(&sha, data).unwrap();

    let stat = cache.stat();
    assert_eq!(stat.num_models, 1);
    assert_eq!(stat.total_bytes, data.len() as u64);
}

#[test]
fn stat_empty_cache_returns_zeros() {
    let (_dir, cache) = make_cache();
    assert_eq!(cache.stat(), CacheStat { num_models: 0, total_bytes: 0 });
}

#[test]
fn wrong_sha256_is_rejected_before_store() {
    let (_dir, cache) = make_cache();
    let data = b"real data";
    // Compute sha of different data so it mismatches.
    let wrong_sha = sha256_hex(b"other data");

    let result = cache.store_atomic(&wrong_sha, data);
    assert!(
        matches!(result, Err(HubError::Sha256Mismatch { .. })),
        "expected Sha256Mismatch, got {result:?}"
    );
    // Blob must NOT be present after rejection.
    assert!(!cache.contains(&wrong_sha));
}

#[test]
fn store_is_idempotent() {
    let (_dir, cache) = make_cache();
    let data = b"idempotent write";
    let sha = sha256_hex(data);

    let p1 = cache.store_atomic(&sha, data).unwrap();
    let p2 = cache.store_atomic(&sha, data).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(cache.stat().num_models, 1);
}

#[test]
fn evict_removes_blob() {
    let (_dir, cache) = make_cache();
    let data = b"evict me";
    let sha = sha256_hex(data);

    cache.store_atomic(&sha, data).unwrap();
    assert!(cache.contains(&sha));

    cache.evict(&sha).unwrap();
    assert!(!cache.contains(&sha));
    assert_eq!(cache.stat().num_models, 0);
}

#[test]
fn evict_nonexistent_is_error() {
    let (_dir, cache) = make_cache();
    let sha = sha256_hex(b"ghost");
    let result = cache.evict(&sha);
    assert!(
        matches!(result, Err(HubError::CacheError(_))),
        "expected CacheError, got {result:?}"
    );
}

#[test]
fn path_for_sha256_uses_fanout() {
    let (_dir, cache) = make_cache();
    let sha = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";
    let path = cache.path_for_sha256(sha).unwrap();
    // First two chars form the fanout directory.
    let components: Vec<_> = path.components().collect();
    let last = components.last().unwrap().as_os_str().to_str().unwrap();
    let second_last = components[components.len() - 2]
        .as_os_str()
        .to_str()
        .unwrap();
    assert_eq!(last, sha);
    assert_eq!(second_last, &sha[..2]);
}

#[test]
fn invalid_sha256_key_returns_cache_error() {
    let (_dir, cache) = make_cache();
    // Too short.
    assert!(matches!(
        cache.path_for_sha256("abc"),
        Err(HubError::CacheError(_))
    ));
    // Contains non-hex chars.
    let bad = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
    assert!(matches!(
        cache.path_for_sha256(bad),
        Err(HubError::CacheError(_))
    ));
}
