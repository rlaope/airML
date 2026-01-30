//! Info command implementation
//!
//! Displays information about an ONNX model.

use airml_core::InferenceEngine;
use anyhow::{Context, Result};

use crate::cli::InfoArgs;

/// Execute the info command
pub fn execute(args: &InfoArgs) -> Result<()> {
    let engine = InferenceEngine::from_file(&args.model)
        .context("Failed to load model")?;

    let meta = engine.metadata();

    println!("Model Information");
    println!("{:=<60}", "");

    // Basic info
    if let Some(name) = &meta.name {
        println!("Name:        {}", name);
    }
    if let Some(desc) = &meta.description {
        println!("Description: {}", desc);
    }
    if let Some(version) = &meta.version {
        println!("Version:     {}", version);
    }
    if let Some(producer) = &meta.producer {
        println!("Producer:    {}", producer);
    }

    // File info
    let file_size = std::fs::metadata(&args.model)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("File Size:   {}", format_size(file_size));

    println!();

    // Inputs
    println!("Inputs ({}):", engine.inputs().len());
    println!("{:-<60}", "");
    for input in engine.inputs() {
        let shape_str = format_shape(&input.shape);
        println!("  {} : {} [{}]", input.name, shape_str, input.dtype);
    }

    println!();

    // Outputs
    println!("Outputs ({}):", engine.outputs().len());
    println!("{:-<60}", "");
    for output in engine.outputs() {
        let shape_str = format_shape(&output.shape);
        println!("  {} : {} [{}]", output.name, shape_str, output.dtype);
    }

    if args.verbose {
        println!();
        println!("Detailed Type Information:");
        println!("{:-<60}", "");
        for input in engine.inputs() {
            println!("  Input '{}': {}", input.name, input.dtype);
        }
        for output in engine.outputs() {
            println!("  Output '{}': {}", output.name, output.dtype);
        }
    }

    Ok(())
}

fn format_shape(shape: &[i64]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }

    let dims: Vec<String> = shape
        .iter()
        .map(|&d| {
            if d < 0 {
                "?".to_string()
            } else {
                d.to_string()
            }
        })
        .collect();

    format!("[{}]", dims.join(", "))
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
