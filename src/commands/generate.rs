//! Generate command implementation — STUB.
//!
//! Autoregressive token generation is not yet implemented. The engine is
//! missing IoBinding support required for efficient KV-cache management.
//!
//! # Planned v0.3 API surface
//!
//! ```rust,ignore
//! use airml_core::{InferenceEngine, IoBinding, SessionConfig};
//! use airml_providers::auto_select_providers;
//!
//! // 1. Load model with optimised providers.
//! let config = SessionConfig::new().with_providers(auto_select_providers());
//! let mut engine = InferenceEngine::from_file_with_config(&model_path, config)?;
//!
//! // 2. Tokenise prompt → input_ids tensor (shape [1, seq_len]).
//!
//! // 3. Allocate KV-cache buffers for each decoder layer via IoBinding.
//! let mut binding = IoBinding::new(&engine)?;
//! // binding.bind_kv_cache(...);
//!
//! // 4. Autoregressive loop:
//! //    - run engine with IoBinding
//! //    - sample next token (temperature / top-k)
//! //    - append to sequence; shift KV-cache
//! //    - stop at EOS or max_tokens
//!
//! // 5. Decode token ids → UTF-8 text and print.
//! ```

use anyhow::Result;

use crate::cli::GenerateArgs;

/// Execute the generate command (stub).
///
/// Prints the prompt back with a notice and exits 0. No tokens are generated.
pub fn execute(args: &GenerateArgs) -> Result<()> {
    println!("airml generate");
    println!("{}", "=".repeat(60));
    println!();
    println!("  model : {}", args.model);
    println!("  prompt: {}", args.prompt);
    println!("  max-tokens  : {}", args.max_tokens);
    println!("  temperature : {}", args.temperature);
    println!("  top-k       : {}", args.top_k);
    println!();
    println!("[STUB: airml-core IoBinding refactor pending — see ROADMAP.md sprint 4]");
    println!();
    println!(
        "Token generation is not yet implemented. The inference engine requires \
IoBinding support for efficient KV-cache management, which is scheduled for \
the v0.3 sprint. Once that lands this command will be a drop-in replacement \
with the same flags."
    );
    println!();
    println!("Prompt (echoed):");
    println!("{}", args.prompt);

    Ok(())
}
