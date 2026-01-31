//! CPU Execution Provider
//!
//! Default execution provider using CPU for inference.

use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};

/// CPU execution provider configuration
#[derive(Debug, Clone, Default)]
pub struct CpuProvider {
    /// Use arena memory allocator
    pub use_arena: bool,
}

impl CpuProvider {
    /// Create a new CPU provider with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable arena memory allocator
    pub fn with_arena(mut self, use_arena: bool) -> Self {
        self.use_arena = use_arena;
        self
    }

    /// Convert to ORT execution provider dispatch
    pub fn into_dispatch(self) -> ExecutionProviderDispatch {
        CPUExecutionProvider::default().build()
    }
}
