//! Property-based tests for [`airml_hub::cache::ModelCache`].

use airml_hub::cache::ModelCache;
use airml_hub::error::HubError;
use proptest::prelude::*;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Build a fresh temporary cache for each test case.
fn make_cache() -> (TempDir, ModelCache) {
    let dir = TempDir::new().expect("tempdir");
    let cache = ModelCache::new(dir.path().to_path_buf());
    (dir, cache)
}

proptest! {
    /// For any byte slice up to 4096 bytes, storing under its actual SHA-256
    /// and then loading returns the identical bytes.
    #[test]
    fn store_then_load_roundtrip(bytes in prop::collection::vec(any::<u8>(), 0..=4096)) {
        let (_dir, cache) = make_cache();
        let sha = sha256_hex(&bytes);

        let store_result = cache.store_atomic(&sha, &bytes);
        prop_assert!(store_result.is_ok(), "store_atomic failed: {store_result:?}");

        let load_result = cache.load(&sha);
        prop_assert!(load_result.is_ok(), "load failed after successful store: {load_result:?}");
        prop_assert_eq!(load_result.unwrap(), bytes);
    }

    /// Storing under the correct SHA and loading under a different valid-format
    /// SHA returns either an error or different bytes — never the wrong content.
    #[test]
    fn load_wrong_sha_never_returns_stored_content(
        bytes in prop::collection::vec(any::<u8>(), 1..=1024),
        // Generate a second 32-byte value for the "other" SHA key
        other_seed in prop::collection::vec(any::<u8>(), 1..=64),
    ) {
        let (_dir, cache) = make_cache();
        let correct_sha = sha256_hex(&bytes);

        // Build a different SHA from the seed; ensure it differs from the correct one
        let other_sha_raw = sha256_hex(&other_seed);
        // If they happen to collide, skip — this is extremely rare but possible
        prop_assume!(other_sha_raw != correct_sha);

        // Store under the correct SHA
        let store_result = cache.store_atomic(&correct_sha, &bytes);
        prop_assert!(store_result.is_ok(), "store_atomic failed: {store_result:?}");

        // Loading under the other (absent) SHA must be an error — never Ok with our bytes
        let load_result = cache.load(&other_sha_raw);
        match load_result {
            Err(_) => { /* expected: blob not present */ }
            Ok(loaded) => {
                prop_assert_ne!(
                    loaded, bytes,
                    "loading wrong SHA returned the content stored under a different key"
                );
            }
        }
    }

    /// Storing with a mismatched SHA always returns Sha256Mismatch and the
    /// blob is never persisted.
    #[test]
    fn wrong_sha_rejected_and_not_cached(
        bytes in prop::collection::vec(any::<u8>(), 1..=512),
        wrong_seed in prop::collection::vec(any::<u8>(), 1..=64),
    ) {
        let (_dir, cache) = make_cache();
        let actual_sha = sha256_hex(&bytes);
        let wrong_sha = sha256_hex(&wrong_seed);
        prop_assume!(wrong_sha != actual_sha);

        let result = cache.store_atomic(&wrong_sha, &bytes);
        prop_assert!(
            matches!(result, Err(HubError::Sha256Mismatch { .. })),
            "expected Sha256Mismatch, got: {result:?}"
        );
        prop_assert!(!cache.contains(&wrong_sha), "rejected blob must not be cached");
    }
}
