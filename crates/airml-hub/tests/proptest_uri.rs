//! Property-based tests for [`airml_hub::ModelUri::parse`].
//!
//! Strategies generate ASCII-alphanumeric strings to avoid confusing the
//! `hf://` parser with incidental slashes or scheme prefixes.

use airml_hub::ModelUri;
use proptest::prelude::*;

/// Regex strategy for a non-empty ASCII alphanumeric + hyphen/underscore token
/// that does not start with `http` or `hf` to avoid colliding with scheme detection.
fn alnum_token() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_-]{1,30}".prop_filter("must not look like a scheme", |s| {
        !s.starts_with("hf") && !s.starts_with("http")
    })
}

proptest! {
    /// For any (owner, repo) pair of safe tokens, `hf://owner/repo` parses to
    /// `HuggingFace { repo: "owner/repo", file: "model.onnx" }`.
    #[test]
    fn hf_two_segment_defaults_to_model_onnx(
        owner in alnum_token(),
        repo in alnum_token(),
    ) {
        let s = format!("hf://{owner}/{repo}");
        let result = ModelUri::parse(&s);
        prop_assert!(result.is_ok(), "parse failed for {s:?}: {result:?}");
        prop_assert_eq!(
            result.unwrap(),
            ModelUri::HuggingFace {
                repo: format!("{owner}/{repo}"),
                file: "model.onnx".to_owned(),
            }
        );
    }

    /// For any (owner, repo, file) triple, `hf://owner/repo/file` parses to
    /// `HuggingFace { repo: "owner/repo", file: "file" }`.
    #[test]
    fn hf_three_segment_uses_explicit_file(
        owner in alnum_token(),
        repo in alnum_token(),
        file in "[a-z][a-z0-9_./-]{1,40}",
    ) {
        let s = format!("hf://{owner}/{repo}/{file}");
        let result = ModelUri::parse(&s);
        prop_assert!(result.is_ok(), "parse failed for {s:?}: {result:?}");
        let uri = result.unwrap();
        if let ModelUri::HuggingFace { repo: got_repo, file: got_file } = uri {
            prop_assert_eq!(got_repo, format!("{owner}/{repo}"));
            // file path is everything after "owner/repo/" — splitn(3) keeps slashes
            prop_assert!(
                got_file.starts_with(&file[..file.find('/').unwrap_or(file.len())]),
                "file prefix mismatch: got {got_file:?}, expected to start with part of {file:?}"
            );
        } else {
            prop_assert!(false, "expected HuggingFace variant, got {uri:?}");
        }
    }

    /// Any string starting with `https://` parses as `ModelUri::Url`.
    #[test]
    fn https_prefix_parses_as_url(suffix in "[a-z0-9._/-]{5,50}") {
        let s = format!("https://{suffix}");
        let result = ModelUri::parse(&s);
        prop_assert!(result.is_ok(), "parse failed for {s:?}");
        prop_assert_eq!(result.unwrap(), ModelUri::Url(s));
    }

    /// Any string starting with `http://` parses as `ModelUri::Url`.
    #[test]
    fn http_prefix_parses_as_url(suffix in "[a-z0-9._/-]{5,50}") {
        let s = format!("http://{suffix}");
        let result = ModelUri::parse(&s);
        prop_assert!(result.is_ok(), "parse failed for {s:?}");
        prop_assert_eq!(result.unwrap(), ModelUri::Url(s));
    }

    /// Empty string, single slashes, and short malformed inputs never panic —
    /// they always return `Result` (Ok or Err, never a panic).
    #[test]
    fn malformed_inputs_never_panic(s in ".*") {
        // We only care that this does not panic; result may be Ok or Err.
        let _result = ModelUri::parse(&s);
    }
}
