//! CoreML Execution Provider
//!
//! Execution provider for Apple CoreML/Metal acceleration on macOS.
//! Supports CPU, GPU, and Neural Engine (ANE) on Apple Silicon.

use ort::ep::coreml::{ComputeUnits as OrtComputeUnits, CoreML, ModelFormat};
use ort::execution_providers::ExecutionProviderDispatch;

/// Compute units for CoreML execution
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ComputeUnits {
    /// Use all available compute units (CPU, GPU, Neural Engine)
    #[default]
    All,
    /// Use CPU and Neural Engine (ANE) - optimal for most models on Apple Silicon
    CpuAndNeuralEngine,
    /// Use CPU and GPU only (no ANE)
    CpuAndGpu,
    /// Use CPU only
    CpuOnly,
}

impl ComputeUnits {
    /// Convert to ort's ComputeUnits enum
    fn to_ort(self) -> OrtComputeUnits {
        match self {
            ComputeUnits::All => OrtComputeUnits::All,
            ComputeUnits::CpuAndNeuralEngine => OrtComputeUnits::CPUAndNeuralEngine,
            ComputeUnits::CpuAndGpu => OrtComputeUnits::CPUAndGPU,
            ComputeUnits::CpuOnly => OrtComputeUnits::CPUOnly,
        }
    }
}

/// CoreML execution provider configuration
#[derive(Debug, Clone)]
pub struct CoreMLConfig {
    /// Which compute units to use
    pub compute_units: ComputeUnits,
    /// Enable subgraph execution (for models with control flow)
    pub enable_subgraphs: bool,
    /// Require static input shapes
    pub require_static_shapes: bool,
    /// Model format (NeuralNetwork or MLProgram)
    pub model_format: Option<CoreMLModelFormat>,
    /// Cache directory for compiled models
    pub cache_dir: Option<String>,
}

/// CoreML model format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLModelFormat {
    /// NeuralNetwork format - better compatibility with older macOS/iOS
    NeuralNetwork,
    /// MLProgram format - supports more operators, potentially better performance
    MLProgram,
}

impl Default for CoreMLConfig {
    fn default() -> Self {
        Self {
            compute_units: ComputeUnits::All,
            enable_subgraphs: false,
            require_static_shapes: false,
            model_format: None,
            cache_dir: None,
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

    /// Set Neural Engine only mode (CPU + ANE, no GPU)
    /// This is optimal for most inference tasks on Apple Silicon
    pub fn neural_engine_only(self) -> Self {
        self.with_compute_units(ComputeUnits::CpuAndNeuralEngine)
    }

    /// Set GPU only mode (CPU + GPU, no ANE)
    pub fn gpu_only(self) -> Self {
        self.with_compute_units(ComputeUnits::CpuAndGpu)
    }

    /// Set CPU only mode
    pub fn cpu_only(self) -> Self {
        self.with_compute_units(ComputeUnits::CpuOnly)
    }

    /// Enable subgraph execution for models with control flow operators
    pub fn with_subgraphs(mut self, enable: bool) -> Self {
        self.config.enable_subgraphs = enable;
        self
    }

    /// Require static input shapes
    pub fn with_static_shapes(mut self, require: bool) -> Self {
        self.config.require_static_shapes = require;
        self
    }

    /// Set model format
    pub fn with_model_format(mut self, format: CoreMLModelFormat) -> Self {
        self.config.model_format = Some(format);
        self
    }

    /// Set cache directory for compiled models
    pub fn with_cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.cache_dir = Some(dir.into());
        self
    }

    /// Convert to ORT execution provider dispatch
    pub fn into_dispatch(self) -> ExecutionProviderDispatch {
        let mut provider = CoreML::default();

        // Set compute units
        provider = provider.with_compute_units(self.config.compute_units.to_ort());

        // Enable subgraphs if requested
        if self.config.enable_subgraphs {
            provider = provider.with_subgraphs(true);
        }

        // Require static shapes if requested
        if self.config.require_static_shapes {
            provider = provider.with_static_input_shapes(true);
        }

        // Set model format if specified
        if let Some(format) = self.config.model_format {
            let ort_format = match format {
                CoreMLModelFormat::NeuralNetwork => ModelFormat::NeuralNetwork,
                CoreMLModelFormat::MLProgram => ModelFormat::MLProgram,
            };
            provider = provider.with_model_format(ort_format);
        }

        // Set cache directory if specified
        if let Some(dir) = &self.config.cache_dir {
            provider = provider.with_model_cache_dir(dir);
        }

        provider.build()
    }
}
