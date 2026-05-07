//! airML - Lightweight ML Runtime
//!
//! A fast, portable ML inference runtime that runs without Python.
//!
//! # Features
//!
//! - Single binary deployment (~50MB)
//! - Fast cold start (0.01-0.05s)
//! - Apple Silicon/Metal acceleration
//! - ONNX model support

mod cli;
mod commands;
pub mod metrics;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Commands};

fn init_tracing(format: &str, level: &str) {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));
    let builder = tracing_subscriber::fmt().with_env_filter(env_filter);
    if format == "json" {
        builder.json().init();
    } else {
        builder.init();
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    init_tracing(&cli.log_format, &cli.log_level);

    if let Err(e) = metrics::init() {
        tracing::warn!(error = %e, "failed to initialize Prometheus metrics; continuing without metrics");
    }

    match &cli.command {
        Commands::Run(args) => commands::run(args, cli.verbose),
        Commands::Info(args) => commands::info(args),
        Commands::Bench(args) => commands::bench(args),
        Commands::System => commands::system(),
        #[cfg(feature = "nlp")]
        Commands::Embed(args) => commands::embed(args, cli.verbose),
        Commands::InstallRuntime(args) => commands::install_runtime(args),
        Commands::Pull(args) => commands::pull(args),
        Commands::Generate(args) => commands::generate(args),
        Commands::Serve(args) => commands::serve(args),
    }
}
