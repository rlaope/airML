//! airML Providers - Execution Provider module
//!
//! Provides execution provider configuration and auto-selection.

mod cpu;

#[cfg(feature = "coreml")]
mod coreml;

pub use cpu::CpuProvider;
pub use ort::execution_providers::ExecutionProviderDispatch;

#[cfg(feature = "coreml")]
pub use coreml::{ComputeUnits, CoreMLConfig, CoreMLModelFormat, CoreMLProvider};

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
#[allow(clippy::vec_init_then_push)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_apple_silicon_returns_bool() {
        let result = is_apple_silicon();
        // On aarch64 macOS it must be true; everywhere else false.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(result);
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        assert!(!result);
    }

    #[test]
    fn test_system_info_populates_fields() {
        let info = system_info();
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        assert!(info.available_providers.contains(&"cpu".to_string()));
    }

    #[test]
    fn test_available_providers_always_contains_cpu() {
        let providers = available_providers();
        assert!(providers.contains(&"cpu".to_string()));
    }

    #[test]
    fn test_auto_select_providers_returns_at_least_one() {
        let providers = auto_select_providers();
        assert!(!providers.is_empty());
    }

    #[test]
    fn test_provider_enum_equality() {
        assert_eq!(Provider::Cpu, Provider::Cpu);
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn test_compute_units_default_is_all() {
        let units = ComputeUnits::default();
        assert!(matches!(units, ComputeUnits::All));
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn test_coreml_provider_builder_chain() {
        // Verify the builder chain doesn't panic and produces a dispatchable provider.
        let _dispatch = CoreMLProvider::new()
            .with_compute_units(ComputeUnits::CpuOnly)
            .with_subgraphs(true)
            .with_static_shapes(false)
            .into_dispatch();
        // If we got here without panicking, the chain is valid.
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn test_coreml_config_default_no_cache_dir() {
        let config = CoreMLConfig::default();
        assert!(config.cache_dir.is_none());
        assert!(matches!(config.compute_units, ComputeUnits::All));
        assert!(!config.enable_subgraphs);
        assert!(!config.require_static_shapes);
        assert!(config.model_format.is_none());
    }
}
