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

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Run(args) => commands::run(args, cli.verbose),
        Commands::Info(args) => commands::info(args),
        Commands::Bench(args) => commands::bench(args),
        Commands::System => commands::system(),
    }
}
