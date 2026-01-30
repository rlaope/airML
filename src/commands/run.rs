//! Run command implementation
//!
//! Executes model inference on input data.

use std::fs;
use std::path::Path;

use airml_core::{ndarray, InferenceEngine, SessionConfig};
use airml_preprocess::ImagePreprocessor;
use airml_providers::auto_select_providers;
use anyhow::{Context, Result};

use crate::cli::RunArgs;

/// Execute the run command
pub fn execute(args: &RunArgs, verbose: bool) -> Result<()> {
    if verbose {
        println!("Loading model: {}", args.model.display());
    }

    // Configure session with providers
    let providers = select_providers(&args.provider)?;
    let config = SessionConfig::new().with_providers(providers);

    // Load model
    let engine = InferenceEngine::from_file_with_config(&args.model, config)
        .context("Failed to load model")?;

    if verbose {
        let meta = engine.metadata();
        if let Some(name) = &meta.name {
            println!("Model: {}", name);
        }
        println!("Inputs: {:?}", engine.inputs());
        println!("Outputs: {:?}", engine.outputs());
    }

    // Create preprocessor
    let preprocessor = create_preprocessor(&args.preprocess, &engine)?;

    // Load and preprocess input
    if verbose {
        println!("Processing input: {}", args.input.display());
    }

    let input_tensor = preprocessor
        .load_and_process(&args.input)
        .context("Failed to preprocess input")?;

    // Run inference
    if verbose {
        println!("Running inference...");
    }

    let mut engine = engine;
    let outputs = engine
        .run(input_tensor.into_dyn())
        .context("Inference failed")?;

    // Process outputs
    if args.raw {
        print_raw_output(&outputs);
    } else {
        print_classification_output(&outputs, &args.labels, args.top_k)?;
    }

    Ok(())
}

fn select_providers(provider_name: &str) -> Result<Vec<airml_providers::ExecutionProviderDispatch>> {
    match provider_name {
        "auto" => Ok(auto_select_providers()),
        "cpu" => Ok(vec![airml_providers::CpuProvider::default().into_dispatch()]),
        #[cfg(feature = "coreml")]
        "coreml" | "neural-engine" => {
            Ok(vec![airml_providers::CoreMLProvider::default().into_dispatch()])
        }
        _ => {
            println!("Warning: Unknown provider '{}', using auto-selection", provider_name);
            Ok(auto_select_providers())
        }
    }
}

fn create_preprocessor(preset: &str, engine: &InferenceEngine) -> Result<ImagePreprocessor> {
    let input_info = engine.inputs().first();

    let (width, height) = if let Some(info) = input_info {
        // Try to get dimensions from model input shape
        if info.shape.len() >= 4 {
            let h = info.shape[2];
            let w = info.shape[3];
            if h > 0 && w > 0 {
                (w as u32, h as u32)
            } else {
                (224, 224) // Default
            }
        } else {
            (224, 224)
        }
    } else {
        (224, 224)
    };

    let preprocessor = match preset {
        "imagenet" => ImagePreprocessor::imagenet(),
        "clip" => ImagePreprocessor::clip(),
        "yolo" => ImagePreprocessor::yolo(width),
        "none" => ImagePreprocessor::custom(width, height, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        _ => {
            println!("Warning: Unknown preset '{}', using ImageNet", preset);
            ImagePreprocessor::imagenet()
        }
    };

    // Override dimensions if model specifies them
    Ok(ImagePreprocessor {
        width,
        height,
        ..preprocessor
    })
}

fn print_raw_output(outputs: &[ndarray::ArrayD<f32>]) {
    for (i, output) in outputs.iter().enumerate() {
        println!("Output {}: shape={:?}", i, output.shape());
        println!("{:?}", output);
    }
}

fn print_classification_output(
    outputs: &[ndarray::ArrayD<f32>],
    labels_path: &Option<std::path::PathBuf>,
    top_k: usize,
) -> Result<()> {
    let output = outputs.first().context("No output from model")?;
    let flat: Vec<f32> = output.iter().copied().collect();

    // Apply softmax
    let softmax = softmax(&flat);

    // Get top-k indices
    let mut indexed: Vec<(usize, f32)> = softmax.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Load labels if provided
    let labels = if let Some(path) = labels_path {
        load_labels(path)?
    } else {
        Vec::new()
    };

    println!("\nTop {} predictions:", top_k);
    println!("{:-<50}", "");

    for (idx, prob) in indexed.iter().take(top_k) {
        let label = labels.get(*idx).map(|s| s.as_str()).unwrap_or("Unknown");
        let bar_len = (prob * 40.0) as usize;
        let bar: String = "=".repeat(bar_len);

        println!(
            "{:4} {:>6.2}% {} {}",
            idx,
            prob * 100.0,
            bar,
            label
        );
    }

    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

fn load_labels<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let content = fs::read_to_string(path).context("Failed to read labels file")?;
    Ok(content.lines().map(|s| s.to_string()).collect())
}
