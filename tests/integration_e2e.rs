//! End-to-end integration tests that actually exercise ORT.
//!
//! These tests require ONNX Runtime to be loadable via `ORT_DYLIB_PATH`.
//! When ORT is unavailable they skip with a clear stderr message.

use std::path::PathBuf;

fn ort_available() -> bool {
    use std::env;
    if let Ok(p) = env::var("ORT_DYLIB_PATH") {
        return std::path::Path::new(&p).exists();
    }
    // Try the airml-installed location too.
    if let Some(home) = dirs::home_dir() {
        let p = home.join(".airml/onnxruntime/v1.20.0/lib/libonnxruntime.dylib");
        if p.exists() {
            // SAFETY: Setting an env var before any ORT initialisation is
            // safe here: integration test binaries are single-threaded at
            // this point and no ORT globals have been touched yet.
            unsafe {
                env::set_var("ORT_DYLIB_PATH", p.display().to_string());
            }
            return true;
        }
    }
    false
}

fn skip_if_no_ort(name: &str) -> bool {
    if ort_available() {
        return false;
    }
    eprintln!(
        "[skip] {name}: ORT_DYLIB_PATH not set and no airml-installed runtime found. \
         Run `airml install-runtime` or set ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib"
    );
    true
}

#[test]
fn e2e_identity_model_roundtrip() {
    if skip_if_no_ort("e2e_identity_model_roundtrip") {
        return;
    }
    use airml_core::{
        ndarray::{ArrayD, IxDyn},
        InferenceEngine,
    };

    let model_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/synthetic-identity.onnx");
    assert!(model_path.exists(), "synthetic-identity.onnx fixture missing");

    let mut engine =
        InferenceEngine::from_file(&model_path).expect("loading synthetic identity model");

    let input =
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let outputs = engine.run(input.clone()).expect("inference");
    assert_eq!(outputs.len(), 1, "expected 1 output");
    let y = &outputs[0];
    assert_eq!(y.shape(), &[1, 4], "output shape");

    let y_vec: Vec<f32> = y.iter().copied().collect();
    let expected = [1.0_f32, 2.0, 3.0, 4.0];
    for (got, want) in y_vec.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "identity mismatch: got {got}, want {want}"
        );
    }
}

#[test]
fn e2e_metadata_inspection() {
    if skip_if_no_ort("e2e_metadata_inspection") {
        return;
    }
    use airml_core::InferenceEngine;

    let model_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/synthetic-identity.onnx");
    let engine = InferenceEngine::from_file(&model_path).expect("loading");

    let meta = engine.metadata();
    assert_eq!(meta.inputs.len(), 1);
    assert_eq!(meta.outputs.len(), 1);
    assert_eq!(meta.inputs[0].name, "x");
    assert_eq!(meta.outputs[0].name, "y");
    // Shape is reported as i64; [1, 4] expected for synthetic-identity.
    assert_eq!(meta.inputs[0].shape, vec![1_i64, 4]);
}

#[test]
fn e2e_repeat_invocation_consistent() {
    if skip_if_no_ort("e2e_repeat_invocation_consistent") {
        return;
    }
    use airml_core::{
        ndarray::{ArrayD, IxDyn},
        InferenceEngine,
    };

    let model_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/synthetic-identity.onnx");
    let mut engine = InferenceEngine::from_file(&model_path).expect("loading");
    let input =
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![5.5_f32, -1.5, 0.0, 999.999]).unwrap();

    let r1 = engine.run(input.clone()).unwrap()[0].clone();
    let r2 = engine.run(input.clone()).unwrap()[0].clone();
    assert_eq!(
        r1.iter().copied().collect::<Vec<_>>(),
        r2.iter().copied().collect::<Vec<_>>()
    );
}
