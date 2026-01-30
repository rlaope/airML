//! System command implementation
//!
//! Displays system information and available execution providers.

use airml_providers::system_info;
use anyhow::Result;

/// Execute the system command
pub fn execute() -> Result<()> {
    let info = system_info();

    println!("System Information");
    println!("{:=<60}", "");

    println!("Operating System: {}", info.os);
    println!("Architecture:     {}", info.arch);
    println!("Apple Silicon:    {}", if info.is_apple_silicon { "Yes" } else { "No" });

    println!();
    println!("Available Execution Providers:");
    println!("{:-<60}", "");

    for provider in &info.available_providers {
        let status = match provider.as_str() {
            "cpu" => "Ready",
            "coreml" => {
                if info.is_apple_silicon {
                    "Ready (Apple Silicon detected)"
                } else {
                    "Available (x86 mode)"
                }
            }
            _ => "Unknown",
        };
        println!("  {:<15} {}", provider, status);
    }

    println!();
    println!("Build Configuration:");
    println!("{:-<60}", "");

    #[cfg(feature = "cpu")]
    println!("  CPU support:    Enabled");
    #[cfg(not(feature = "cpu"))]
    println!("  CPU support:    Disabled");

    #[cfg(feature = "coreml")]
    println!("  CoreML support: Enabled");
    #[cfg(not(feature = "coreml"))]
    println!("  CoreML support: Disabled");

    #[cfg(feature = "nlp")]
    println!("  NLP support:    Enabled");
    #[cfg(not(feature = "nlp"))]
    println!("  NLP support:    Disabled");

    println!();
    println!("Version: {}", env!("CARGO_PKG_VERSION"));

    Ok(())
}
