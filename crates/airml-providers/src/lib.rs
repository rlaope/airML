//! airML Providers - Execution Provider module
//!
//! Provides execution provider configuration and auto-selection.

mod cpu;

#[cfg(feature = "coreml")]
mod coreml;

pub use cpu::CpuProvider;
pub use ort::execution_providers::ExecutionProviderDispatch;

#[cfg(feature = "coreml")]
pub use coreml::{CoreMLConfig, CoreMLProvider};

/// Available execution providers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// CPU execution (always available)
    Cpu,
    /// CoreML execution (macOS only)
    #[cfg(feature = "coreml")]
    CoreML,
}

/// Check if running on Apple Silicon
pub fn is_apple_silicon() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        true
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        false
    }
}

/// Auto-select the best available execution provider
pub fn auto_select_providers() -> Vec<ExecutionProviderDispatch> {
    let mut providers = Vec::new();

    #[cfg(feature = "coreml")]
    {
        if is_apple_silicon() {
            providers.push(CoreMLProvider::default().into_dispatch());
        }
    }

    // CPU is always available as fallback
    providers.push(CpuProvider::default().into_dispatch());

    providers
}

/// Get system information
pub fn system_info() -> SystemInfo {
    SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        is_apple_silicon: is_apple_silicon(),
        available_providers: available_providers(),
    }
}

/// List available providers
pub fn available_providers() -> Vec<String> {
    #[allow(unused_mut)]
    let mut providers = vec!["cpu".to_string()];

    #[cfg(feature = "coreml")]
    {
        if cfg!(target_os = "macos") {
            providers.push("coreml".to_string());
        }
    }

    providers
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// CPU architecture
    pub arch: String,
    /// Whether running on Apple Silicon
    pub is_apple_silicon: bool,
    /// Available execution providers
    pub available_providers: Vec<String>,
}
