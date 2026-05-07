//! Example: Image classification with an ONNX model.
//!
//! Usage:
//!   cargo run --example 01_image_classify --features=coreml -- model.onnx cat.jpg
//!
//! The model must accept a float32 NCHW tensor of shape [1, 3, 224, 224] and
//! return a 1D or 2D float32 logits/scores tensor.

use anyhow::{Context, Result};
use image::imageops::FilterType;
use ndarray::{ArrayD, IxDyn};

use airml_core::InferenceEngine;
use airml_providers::auto_select_providers;
use airml_core::SessionConfig;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.onnx> <image.jpg|png>", args[0]);
        std::process::exit(2);
    }

    let model_path = &args[1];
    let image_path = &args[2];

    // Load model with the best available execution provider.
    let config = SessionConfig::new().with_providers(auto_select_providers());
    let mut engine = InferenceEngine::from_file_with_config(model_path, config)
        .with_context(|| format!("Failed to load model from '{model_path}'"))?;

    // ---- Image preprocessing ------------------------------------------------
    // Load and resize to 224x224.
    let img = image::open(image_path)
        .with_context(|| format!("Failed to open image '{image_path}'"))?;
    let img = img.resize_exact(224, 224, FilterType::Triangle).to_rgb8();

    // ImageNet normalisation constants (RGB order).
    let mean = [0.485_f32, 0.456, 0.406];
    let std  = [0.229_f32, 0.224, 0.225];

    // Build NCHW tensor: [1, 3, 224, 224].
    // enumerate_pixels() yields (x_col, y_row, pixel); NCHW offset = c*H*W + y*W + x.
    let mut data = vec![0.0_f32; 3 * 224 * 224];
    for (x, y, pixel) in img.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        let [r, g, b] = pixel.0;
        let channels = [(r, 0usize), (g, 1), (b, 2)];
        for (raw, c) in channels {
            data[c * 224 * 224 + y * 224 + x] =
                (raw as f32 / 255.0 - mean[c]) / std[c];
        }
    }

    let input = ArrayD::from_shape_vec(IxDyn(&[1, 3, 224, 224]), data)
        .context("Failed to create input tensor")?;

    // ---- Inference ----------------------------------------------------------
    let outputs = engine.run(input).context("Inference failed")?;
    let logits = outputs.first().context("Model returned no outputs")?;

    // Flatten to 1-D slice of class scores.
    let scores: Vec<f32> = logits.iter().copied().collect();

    // Softmax.
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

    // Top-5 by probability.
    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top-5 predictions:");
    for (idx, prob) in indexed.iter().take(5) {
        println!("  idx={idx:<6}  prob={prob:.4}");
    }

    Ok(())
}
