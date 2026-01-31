//! CLI interface definition
//!
//! Defines the command-line interface using clap.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// airML - Lightweight ML Runtime
///
/// Run ML models without Python. Fast, portable, and efficient.
#[derive(Parser, Debug)]
#[command(name = "airml")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run inference on an input
    Run(RunArgs),

    /// Display model information
    Info(InfoArgs),

    /// Benchmark model inference
    Bench(BenchArgs),

    /// Display system information
    System,

    /// Generate text embeddings
    #[cfg(feature = "nlp")]
    Embed(EmbedArgs),
}

/// Arguments for the `run` command
#[derive(Parser, Debug)]
pub struct RunArgs {
    /// Path to the ONNX model file
    #[arg(short, long)]
    pub model: PathBuf,

    /// Path to the input file (image)
    #[arg(short, long)]
    pub input: PathBuf,

    /// Path to labels file (one label per line)
    #[arg(short, long)]
    pub labels: Option<PathBuf>,

    /// Number of top predictions to show
    #[arg(short = 'k', long, default_value = "5")]
    pub top_k: usize,

    /// Execution provider to use (cpu, coreml, neural-engine)
    #[arg(short, long, default_value = "auto")]
    pub provider: String,

    /// Image preprocessing preset (imagenet, clip, yolo, none)
    #[arg(long, default_value = "imagenet")]
    pub preprocess: String,

    /// Output raw tensor values
    #[arg(long)]
    pub raw: bool,
}

/// Arguments for the `info` command
#[derive(Parser, Debug)]
pub struct InfoArgs {
    /// Path to the ONNX model file
    #[arg(short, long)]
    pub model: PathBuf,

    /// Show detailed information
    #[arg(short, long)]
    pub verbose: bool,
}

/// Arguments for the `bench` command
#[derive(Parser, Debug)]
pub struct BenchArgs {
    /// Path to the ONNX model file
    #[arg(short, long)]
    pub model: PathBuf,

    /// Number of inference iterations
    #[arg(short = 'n', long, default_value = "100")]
    pub iterations: usize,

    /// Number of warmup iterations
    #[arg(short, long, default_value = "10")]
    pub warmup: usize,

    /// Execution provider to use
    #[arg(short, long, default_value = "auto")]
    pub provider: String,

    /// Input shape for random data (e.g., "1,3,224,224")
    #[arg(long)]
    pub shape: Option<String>,
}

/// Arguments for the `embed` command
#[cfg(feature = "nlp")]
#[derive(Parser, Debug)]
pub struct EmbedArgs {
    /// Path to the ONNX embedding model file
    #[arg(short, long)]
    pub model: PathBuf,

    /// Path to the tokenizer.json file
    #[arg(short, long)]
    pub tokenizer: PathBuf,

    /// Text to embed
    #[arg(long)]
    pub text: String,

    /// Maximum sequence length
    #[arg(long, default_value = "512")]
    pub max_length: usize,

    /// Execution provider to use (cpu, coreml, neural-engine)
    #[arg(short, long, default_value = "auto")]
    pub provider: String,

    /// Output format (json, raw)
    #[arg(long, default_value = "json")]
    pub output: String,

    /// Normalize output embeddings (L2 normalization)
    #[arg(long)]
    pub normalize: bool,
}
