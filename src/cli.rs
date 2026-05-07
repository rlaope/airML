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

    /// Log format (text or json)
    #[arg(long, global = true, default_value = "text", value_parser = ["text", "json"])]
    pub log_format: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, global = true, default_value = "info")]
    pub log_level: String,
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

    /// Auto-download ONNX Runtime dylib to ~/.airml/onnxruntime/
    InstallRuntime(InstallRuntimeArgs),

    /// Download a model into the local cache
    Pull(PullArgs),

    /// Generate tokens from a language model (stub — KV-cache pending)
    Generate(GenerateArgs),

    /// Serve an OpenAI-compatible embeddings HTTP API
    Serve(ServeArgs),
}

/// Arguments for the `run` command
#[derive(Parser, Debug)]
pub struct RunArgs {
    /// Path to the ONNX model file, or a URI (hf://owner/repo, registry ID)
    #[arg(short, long)]
    pub model: String,

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

/// Arguments for the `install-runtime` command
#[derive(Parser, Debug)]
pub struct InstallRuntimeArgs {
    /// ONNX Runtime version to download
    #[arg(long, default_value = "1.20.0")]
    pub version: String,

    /// Re-download even if already present
    #[arg(long)]
    pub force: bool,

    /// Target platform (auto-detected if omitted): macos-arm64, macos-x86_64, linux-arm64, linux-x86_64
    #[arg(long)]
    pub platform: Option<String>,

    /// Keep the downloaded .tgz archive after extraction (default: delete it)
    #[arg(long)]
    pub keep_archive: bool,
}

/// Arguments for the `pull` command
#[derive(Parser, Debug)]
pub struct PullArgs {
    /// Registry ID (e.g. bge-small-en) or URI (e.g. hf://Xenova/clip-vit-base-patch32)
    #[arg(required_unless_present = "list")]
    pub model: Option<String>,

    /// List available registry models instead of pulling
    #[arg(long)]
    pub list: bool,

    /// Override the default cache directory
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

/// Arguments for the `generate` command
#[derive(Parser, Debug)]
pub struct GenerateArgs {
    /// Path to the model file or URI
    #[arg(short, long)]
    pub model: String,

    /// Input prompt text
    #[arg(long)]
    pub prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "64")]
    pub max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value = "0.8")]
    pub temperature: f32,

    /// Top-k sampling cutoff
    #[arg(long, default_value = "40")]
    pub top_k: usize,
}

/// Arguments for the `serve` command
#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// Address to bind the HTTP server on
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,

    /// Default model when the request body omits "model"
    #[arg(long)]
    pub default_model: Option<String>,

    /// Bearer token required on /v1/* routes (disabled if not set)
    #[arg(long)]
    pub auth_token: Option<String>,

    /// Maximum allowed request body size in bytes
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    pub max_request_bytes: usize,

    /// Override the default Hub cache directory
    #[arg(long)]
    pub cache_dir: Option<std::path::PathBuf>,
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
