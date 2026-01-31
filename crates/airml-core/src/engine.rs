//! Inference Engine
//!
//! The main interface for loading and running ONNX models.

use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{AirMLError, Result};
use crate::session::SessionConfig;

/// Metadata about a loaded model
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name (from ONNX metadata)
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model version
    pub version: Option<i64>,
    /// Producer name (framework that created the model)
    pub producer: Option<String>,
    /// Input tensor information
    pub inputs: Vec<TensorInfo>,
    /// Output tensor information
    pub outputs: Vec<TensorInfo>,
}

/// Information about a tensor (input or output)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Tensor shape (-1 for dynamic dimensions)
    pub shape: Vec<i64>,
    /// Element type as string
    pub dtype: String,
}

/// Main inference engine for running ONNX models
pub struct InferenceEngine {
    session: Session,
    metadata: ModelMetadata,
}

impl InferenceEngine {
    /// Load a model from a file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_file_with_config(path, SessionConfig::default())
    }

    /// Load a model from a file path with custom configuration
    pub fn from_file_with_config<P: AsRef<Path>>(path: P, config: SessionConfig) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(AirMLError::ModelNotFound(path.display().to_string()));
        }

        let session = Self::build_session_from_file(path, &config)?;
        let metadata = Self::extract_metadata(&session)?;

        Ok(Self { session, metadata })
    }

    /// Load a model from bytes (for embedded models via include_bytes!)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with_config(bytes, SessionConfig::default())
    }

    /// Load a model from bytes with custom configuration
    pub fn from_bytes_with_config(bytes: &[u8], config: SessionConfig) -> Result<Self> {
        let session = Self::build_session_from_bytes(bytes, &config)?;
        let metadata = Self::extract_metadata(&session)?;

        Ok(Self { session, metadata })
    }

    fn build_session_from_file(path: &Path, config: &SessionConfig) -> Result<Session> {
        let mut builder = Session::builder()
            .map_err(|e| AirMLError::OrtError(e.to_string()))?;

        builder = Self::apply_config(builder, config)?;

        builder
            .commit_from_file(path)
            .map_err(|e| AirMLError::ModelLoadError(e.to_string()))
    }

    fn build_session_from_bytes(bytes: &[u8], config: &SessionConfig) -> Result<Session> {
        let mut builder = Session::builder()
            .map_err(|e| AirMLError::OrtError(e.to_string()))?;

        builder = Self::apply_config(builder, config)?;

        builder
            .commit_from_memory(bytes)
            .map_err(|e| AirMLError::ModelLoadError(e.to_string()))
    }

    fn apply_config(
        mut builder: ort::session::builder::SessionBuilder,
        config: &SessionConfig,
    ) -> Result<ort::session::builder::SessionBuilder> {
        let opt_level: GraphOptimizationLevel = config.optimization_level.into();
        builder = builder
            .with_optimization_level(opt_level)
            .map_err(|e| AirMLError::OrtError(e.to_string()))?;

        if config.intra_threads > 0 {
            builder = builder
                .with_intra_threads(config.intra_threads)
                .map_err(|e| AirMLError::OrtError(e.to_string()))?;
        }

        if config.inter_threads > 0 {
            builder = builder
                .with_inter_threads(config.inter_threads)
                .map_err(|e| AirMLError::OrtError(e.to_string()))?;
        }

        if !config.providers.is_empty() {
            builder = builder
                .with_execution_providers(config.providers.clone())
                .map_err(|e| AirMLError::OrtError(e.to_string()))?;
        }

        Ok(builder)
    }

    fn extract_metadata(session: &Session) -> Result<ModelMetadata> {
        let inputs: Vec<TensorInfo> = session
            .inputs()
            .iter()
            .map(|input| {
                let dtype = input.dtype();
                let shape: Vec<i64> = dtype
                    .tensor_shape()
                    .map(|s| s.iter().copied().collect())
                    .unwrap_or_default();
                TensorInfo {
                    name: input.name().to_string(),
                    shape,
                    dtype: format!("{:?}", dtype),
                }
            })
            .collect();

        let outputs: Vec<TensorInfo> = session
            .outputs()
            .iter()
            .map(|output| {
                let dtype = output.dtype();
                let shape: Vec<i64> = dtype
                    .tensor_shape()
                    .map(|s| s.iter().copied().collect())
                    .unwrap_or_default();
                TensorInfo {
                    name: output.name().to_string(),
                    shape,
                    dtype: format!("{:?}", dtype),
                }
            })
            .collect();

        let session_meta = session
            .metadata()
            .map_err(|e| AirMLError::OrtError(e.to_string()))?;

        Ok(ModelMetadata {
            name: session_meta.name(),
            description: session_meta.description(),
            version: session_meta.version(),
            producer: session_meta.producer(),
            inputs,
            outputs,
        })
    }

    /// Run inference with a single f32 input tensor
    pub fn run(&mut self, input: ArrayD<f32>) -> Result<Vec<ArrayD<f32>>> {
        let input_name = self
            .metadata
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| AirMLError::ConfigError("Model has no inputs".to_string()))?;

        self.run_named(vec![(&input_name, input)])
    }

    /// Run inference with multiple input tensors (matched by order)
    pub fn run_multiple(&mut self, inputs: Vec<ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>> {
        if inputs.len() != self.metadata.inputs.len() {
            return Err(AirMLError::ConfigError(format!(
                "Expected {} inputs, got {}",
                self.metadata.inputs.len(),
                inputs.len()
            )));
        }

        // Collect input names first to avoid borrow conflict
        let input_names: Vec<String> = self
            .metadata
            .inputs
            .iter()
            .map(|info| info.name.clone())
            .collect();

        let named_inputs: Vec<(&str, ArrayD<f32>)> = input_names
            .iter()
            .zip(inputs)
            .map(|(name, arr)| (name.as_str(), arr))
            .collect();

        self.run_named(named_inputs)
    }

    /// Run inference with named inputs
    pub fn run_named(&mut self, inputs: Vec<(&str, ArrayD<f32>)>) -> Result<Vec<ArrayD<f32>>> {
        // Create input tensors
        let mut ort_inputs: Vec<(String, Tensor<f32>)> = Vec::new();
        for (name, arr) in inputs {
            let shape: Vec<usize> = arr.shape().to_vec();
            let data: Vec<f32> = arr.iter().copied().collect();
            let tensor = Tensor::from_array((shape, data))
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
            ort_inputs.push((name.to_string(), tensor));
        }

        // Build input references as (name, value) pairs
        let input_values: Vec<(std::borrow::Cow<'_, str>, ort::session::SessionInputValue<'_>)> =
            ort_inputs
                .iter()
                .map(|(name, tensor)| {
                    (
                        std::borrow::Cow::Borrowed(name.as_str()),
                        ort::session::SessionInputValue::from(tensor),
                    )
                })
                .collect();

        // Run inference
        let outputs = self
            .session
            .run(input_values)
            .map_err(|e| AirMLError::InferenceError(e.to_string()))?;

        // Extract output tensors
        let mut results = Vec::new();
        for value in outputs.values() {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e: ort::Error| AirMLError::InferenceError(e.to_string()))?;

            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let data_vec: Vec<f32> = data.to_vec();
            let array = ArrayD::from_shape_vec(IxDyn(&shape_vec), data_vec)
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;

            results.push(array);
        }

        Ok(results)
    }

    /// Get model metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get input tensor info
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.metadata.inputs
    }

    /// Get output tensor info
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.metadata.outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_builder() {
        let config = SessionConfig::new()
            .with_intra_threads(4)
            .with_inter_threads(2);

        assert_eq!(config.intra_threads, 4);
        assert_eq!(config.inter_threads, 2);
    }
}
