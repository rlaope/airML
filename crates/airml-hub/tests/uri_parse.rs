//! Tests for [`airml_hub::ModelUri::parse`].

use airml_hub::ModelUri;
use std::path::PathBuf;

// --- hf:// scheme ---

#[test]
fn hf_repo_only_defaults_to_model_onnx() {
    let uri = ModelUri::parse("hf://Xenova/clip-vit-base-patch32").unwrap();
    assert_eq!(
        uri,
        ModelUri::HuggingFace {
            repo: "Xenova/clip-vit-base-patch32".to_owned(),
            file: "model.onnx".to_owned(),
        }
    );
}

#[test]
fn hf_repo_with_single_file_segment() {
    let uri = ModelUri::parse("hf://BAAI/bge-small-en-v1.5/onnx/model.onnx").unwrap();
    assert_eq!(
        uri,
        ModelUri::HuggingFace {
            repo: "BAAI/bge-small-en-v1.5".to_owned(),
            file: "onnx/model.onnx".to_owned(),
        }
    );
}

#[test]
fn hf_repo_with_deep_file_path() {
    // hf://owner/repo/sub/dir/model.onnx — file contains slashes
    let uri =
        ModelUri::parse("hf://owner/repo/sub/dir/model.onnx").unwrap();
    assert_eq!(
        uri,
        ModelUri::HuggingFace {
            repo: "owner/repo".to_owned(),
            file: "sub/dir/model.onnx".to_owned(),
        }
    );
}

#[test]
fn hf_missing_repo_segment_is_error() {
    // hf://owner_only  — only one slash segment, no repo
    let result = ModelUri::parse("hf://orphan-owner");
    assert!(result.is_err(), "expected MalformedUri, got {result:?}");
}

// --- https / http ---

#[test]
fn https_url_variant() {
    let s = "https://example.com/model.onnx";
    let uri = ModelUri::parse(s).unwrap();
    assert_eq!(uri, ModelUri::Url(s.to_owned()));
}

#[test]
fn http_url_variant() {
    let s = "http://localhost:8080/model.onnx";
    let uri = ModelUri::parse(s).unwrap();
    assert_eq!(uri, ModelUri::Url(s.to_owned()));
}

// --- built-in registry ids ---

#[test]
fn known_registry_id_bge() {
    let uri = ModelUri::parse("bge-small-en").unwrap();
    assert_eq!(uri, ModelUri::Registry("bge-small-en".to_owned()));
}

#[test]
fn known_registry_id_clip() {
    let uri = ModelUri::parse("clip-vit-b32").unwrap();
    assert_eq!(uri, ModelUri::Registry("clip-vit-b32".to_owned()));
}

#[test]
fn known_registry_id_whisper() {
    let uri = ModelUri::parse("whisper-tiny-encoder").unwrap();
    assert_eq!(uri, ModelUri::Registry("whisper-tiny-encoder".to_owned()));
}

// --- local path fallback ---

#[test]
fn unknown_bare_token_becomes_local_path() {
    let uri = ModelUri::parse("not-a-registry-id").unwrap();
    assert_eq!(uri, ModelUri::LocalPath(PathBuf::from("not-a-registry-id")));
}

#[test]
fn absolute_path_becomes_local_path() {
    let uri = ModelUri::parse("/tmp/my-model.onnx").unwrap();
    assert_eq!(uri, ModelUri::LocalPath(PathBuf::from("/tmp/my-model.onnx")));
}
