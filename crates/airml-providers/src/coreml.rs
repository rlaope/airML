//! CoreML Execution Provider
//!
//! Execution provider for Apple CoreML/Metal acceleration on macOS.

use ort::execution_providers::{CoreMLExecutionProvider, ExecutionProviderDispatch};

/// Compute units for CoreML execution
#[derive(Debug, Clone, Copy, Default)]
pub enum ComputeUnits {
    /// Use all available compute units (CPU, GPU, Neural Engine)
    #[default]
    All,
    /// Use CPU and GPU only
    CpuAndGpu,
    /// Use CPU only
    CpuOnly,
}

/// CoreML execution provider configuration
#[derive(Debug, Clone)]
pub struct CoreMLConfig {
    /// Which compute units to use
    pub compute_units: ComputeUnits,
    /// Require static input shapes
    pub require_static_shapes: bool,
    /// Enable model caching
    pub enable_cache: bool,
}

impl Default for CoreMLConfig {
    fn default() -> Self {
        Self {
            compute_units: ComputeUnits::All,
            require_static_shapes: false,
            enable_cache: true,
        }
    }
}

/// CoreML execution provider
#[derive(Debug, Clone, Default)]
pub struct CoreMLProvider {
    config: CoreMLConfig,
}

impl CoreMLProvider {
    /// Create a new CoreML provider with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration
    pub fn with_config(config: CoreMLConfig) -> Self {
        Self { config }
    }

    /// Set compute units
    pub fn with_compute_units(mut self, units: ComputeUnits) -> Self {
        self.config.compute_units = units;
        self
    }

    /// Set neural engine only mode
    pub fn neural_engine_only(self) -> Self {
        self.with_compute_units(ComputeUnits::All)
    }

    /// Set GPU only mode (no ANE)
    pub fn gpu_only(self) -> Self {
        self.with_compute_units(ComputeUnits::CpuAndGpu)
    }

    /// Convert to ORT execution provider dispatch
    pub fn into_dispatch(self) -> ExecutionProviderDispatch {
        let mut provider = CoreMLExecutionProvider::default();

        match self.config.compute_units {
            ComputeUnits::All => {
                // Default - uses all available compute units
            }
            ComputeUnits::CpuAndGpu => {
                provider = provider.with_ane_only();
            }
            ComputeUnits::CpuOnly => {
                provider = provider.with_cpu_only();
            }
        }

        if self.config.require_static_shapes {
            provider = provider.with_subgraphs();
        }

        provider.build().into()
    }
}
