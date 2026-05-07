//! Pull command implementation.
//!
//! Downloads a model into the local content-addressed cache, or lists the
//! built-in registry when `--list` is passed.

use airml_hub::{registry, Fetcher, ModelCache, ModelUri};
use anyhow::{Context, Result};

use crate::cli::PullArgs;

/// Execute the pull command.
pub fn execute(args: &PullArgs) -> Result<()> {
    if args.list {
        print_registry();
        return Ok(());
    }

    let model_str = args
        .model
        .as_deref()
        .context("A model ID or URI is required (or pass --list)")?;

    let uri = ModelUri::parse(model_str)
        .with_context(|| format!("Failed to parse model URI: {model_str}"))?;

    let fetcher = if let Some(cache_dir) = &args.cache_dir {
        Fetcher::with_cache(ModelCache::new(cache_dir.clone()))
    } else {
        Fetcher::new()
    };

    let path = fetcher
        .resolve_to_path(&uri)
        .with_context(|| format!("Failed to resolve model: {model_str}"))?;

    println!("{}", path.display());

    Ok(())
}

/// Print the registry table.
fn print_registry() {
    let entries = registry::all();
    println!("{:<24} {:<36} {:>9}  Description", "ID", "HF Repo", "Size");
    println!("{}", "-".repeat(100));
    for e in entries {
        let size_mb = e.size_bytes / 1_000_000;
        println!("{:<24} {:<36} {:>6} MB  {}", e.id, e.hf_repo, size_mb, e.description);
    }
}
