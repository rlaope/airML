//! ONNX Session configuration and management
//!
//! Handles ONNX Runtime session creation with various execution providers.

use ort::execution_providers::ExecutionProviderDispatch;

/// Configuration for creating an ONNX Runtime session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Number of intra-op threads (0 = auto)
    pub intra_threads: usize,
    /// Number of inter-op threads (0 = auto)
    pub inter_threads: usize,
    /// Memory optimization level
    pub optimization_level: OptimizationLevel,
    /// Execution providers to use (in order of preference)
    pub providers: Vec<ExecutionProviderDispatch>,
}

/// ONNX Runtime graph optimization level
#[derive(Debug, Clone, Copy, Default)]
pub enum OptimizationLevel {
    /// No optimization
    Disabled,
    /// Basic optimizations
    Basic,
    /// Extended optimizations
    #[default]
    Extended,
    /// All optimizations enabled
    All,
}

impl From<OptimizationLevel> for ort::session::builder::GraphOptimizationLevel {
    fn from(level: OptimizationLevel) -> Self {
        match level {
            OptimizationLevel::Disabled => ort::session::builder::GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => ort::session::builder::GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => ort::session::builder::GraphOptimizationLevel::Level2,
            OptimizationLevel::All => ort::session::builder::GraphOptimizationLevel::Level3,
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            intra_threads: 0,
            inter_threads: 0,
            optimization_level: OptimizationLevel::Extended,
            providers: Vec::new(),
        }
    }
}

impl SessionConfig {
    /// Create a new session configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of intra-op threads
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    /// Set the number of inter-op threads
    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    /// Set the optimization level
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Set execution providers
    pub fn with_providers(mut self, providers: Vec<ExecutionProviderDispatch>) -> Self {
        self.providers = providers;
        self
    }

    /// Add an execution provider
    pub fn add_provider(mut self, provider: ExecutionProviderDispatch) -> Self {
        self.providers.push(provider);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::session::builder::GraphOptimizationLevel;

    #[test]
    fn test_session_config_default_values() {
        let config = SessionConfig::default();
        assert_eq!(config.intra_threads, 0);
        assert_eq!(config.inter_threads, 0);
        assert!(config.providers.is_empty());
        assert!(matches!(config.optimization_level, OptimizationLevel::Extended));
    }

    #[test]
    fn test_optimization_level_into_ort() {
        let disabled: GraphOptimizationLevel = OptimizationLevel::Disabled.into();
        let basic: GraphOptimizationLevel = OptimizationLevel::Basic.into();
        let extended: GraphOptimizationLevel = OptimizationLevel::Extended.into();
        let all: GraphOptimizationLevel = OptimizationLevel::All.into();

        assert!(matches!(disabled, GraphOptimizationLevel::Disable));
        assert!(matches!(basic, GraphOptimizationLevel::Level1));
        assert!(matches!(extended, GraphOptimizationLevel::Level2));
        assert!(matches!(all, GraphOptimizationLevel::Level3));
    }

    #[test]
    fn test_session_config_with_providers_replaces() {
        let config = SessionConfig::new().with_providers(vec![]);
        assert!(config.providers.is_empty());
    }

    #[test]
    fn test_session_config_chained_builder() {
        let config = SessionConfig::new()
            .with_intra_threads(4)
            .with_inter_threads(2)
            .with_optimization_level(OptimizationLevel::Basic);

        assert_eq!(config.intra_threads, 4);
        assert_eq!(config.inter_threads, 2);
        assert!(matches!(config.optimization_level, OptimizationLevel::Basic));
    }
}
