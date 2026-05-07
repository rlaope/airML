//! Embed command implementation
//!
//! Generates text embeddings using ONNX models.

use airml_core::{ndarray::ArrayD, InferenceEngine, SessionConfig};
use airml_preprocess::TextPreprocessor;
use airml_providers::auto_select_providers;
use anyhow::{Context, Result};

use crate::cli::EmbedArgs;

/// Execute the embed command
pub fn execute(args: &EmbedArgs, verbose: bool) -> Result<()> {
    if verbose {
        println!("Loading tokenizer: {}", args.tokenizer.display());
    }

    // Load tokenizer
    let preprocessor = TextPreprocessor::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
        .with_max_length(args.max_length);

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
        println!("Model inputs: {:?}", engine.inputs());
        println!("Model outputs: {:?}", engine.outputs());
    }

    // Tokenize text
    if verbose {
        println!("Tokenizing text...");
    }

    let tokenized = preprocessor
        .encode(&args.text)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize: {}", e))?;

    let (input_ids, attention_mask) = tokenized.to_array();

    if verbose {
        println!("Token count: {}", tokenized.input_ids.len());
    }

    // Run inference
    if verbose {
        println!("Running inference...");
    }

    let mut engine = engine;

    // Most embedding models expect input_ids and attention_mask
    // We need to handle this based on model inputs
    let model_inputs = engine.inputs();

    let t0 = std::time::Instant::now();
    let outputs = if model_inputs.len() >= 2 {
        // Model expects multiple inputs (input_ids, attention_mask)
        engine
            .run_multiple(vec![
                input_ids.into_dyn().mapv(|x| x as f32),
                attention_mask.into_dyn().mapv(|x| x as f32),
            ])
            .context("Inference failed")?
    } else {
        // Model expects single input
        engine
            .run(input_ids.into_dyn().mapv(|x| x as f32))
            .context("Inference failed")?
    };
    let elapsed = t0.elapsed();

    if let Some(c) = crate::metrics::INFERENCE_COUNT.get() {
        c.inc();
    }
    if let Some(h) = crate::metrics::INFERENCE_LATENCY.get() {
        h.observe(elapsed.as_secs_f64());
    }

    // Get embeddings from output
    let embeddings = extract_embeddings(&outputs)?;

    // Optionally normalize
    let embeddings = if args.normalize {
        l2_normalize(&embeddings)
    } else {
        embeddings
    };

    // Output results
    match args.output.as_str() {
        "json" => print_json(&embeddings, &args.text),
        "raw" => print_raw(&embeddings),
        _ => print_json(&embeddings, &args.text),
    }

    Ok(())
}

fn select_providers(provider_name: &str) -> Result<Vec<airml_providers::ExecutionProviderDispatch>> {
    match provider_name {
        "auto" => Ok(auto_select_providers()),
        "cpu" => Ok(vec![airml_providers::CpuProvider::default().into_dispatch()]),
        #[cfg(feature = "coreml")]
        "coreml" => Ok(vec![airml_providers::CoreMLProvider::default().into_dispatch()]),
        #[cfg(feature = "coreml")]
        "neural-engine" => Ok(vec![
            airml_providers::CoreMLProvider::default()
                .neural_engine_only()
                .into_dispatch(),
        ]),
        _ => {
            println!("Warning: Unknown provider '{}', using auto-selection", provider_name);
            Ok(auto_select_providers())
        }
    }
}

fn extract_embeddings(outputs: &[ArrayD<f32>]) -> Result<Vec<f32>> {
    let output = outputs.first().context("No output from model")?;

    // Handle different output shapes:
    // - [batch, seq_len, hidden] -> take [CLS] token or mean pooling
    // - [batch, hidden] -> direct embedding
    let shape = output.shape();

    let embeddings: Vec<f32> = match shape.len() {
        2 => {
            // [batch, hidden] - direct embedding
            output.iter().copied().collect()
        }
        3 => {
            // [batch, seq_len, hidden] - use mean pooling
            let hidden_size = shape[2];
            let seq_len = shape[1];

            // Mean pooling across sequence dimension
            let mut pooled = vec![0.0f32; hidden_size];
            for i in 0..seq_len {
                for j in 0..hidden_size {
                    pooled[j] += output[[0, i, j]];
                }
            }
            for v in &mut pooled {
                *v /= seq_len as f32;
            }
            pooled
        }
        _ => {
            // Flatten whatever we get
            output.iter().copied().collect()
        }
    };

    Ok(embeddings)
}

fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

fn print_json(embeddings: &[f32], text: &str) {
    println!("{{");
    println!("  \"text\": {:?},", text);
    println!("  \"dimension\": {},", embeddings.len());
    println!("  \"embedding\": [");

    let chunk_size = 8;
    for (i, chunk) in embeddings.chunks(chunk_size).enumerate() {
        let values: Vec<String> = chunk.iter().map(|v| format!("{:.6}", v)).collect();
        let is_last = (i + 1) * chunk_size >= embeddings.len();
        println!(
            "    {}{}",
            values.join(", "),
            if is_last { "" } else { "," }
        );
    }

    println!("  ]");
    println!("}}");
}

fn print_raw(embeddings: &[f32]) {
    for v in embeddings {
        println!("{:.6}", v);
    }
}
