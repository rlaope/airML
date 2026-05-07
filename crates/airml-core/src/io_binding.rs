//! Pre-allocated input/output binding for low-allocation inference.
//!
//! [`BoundSession`] wraps an [`InferenceEngine`] together with an
//! [`ort::io_binding::IoBinding`] and a set of pre-allocated KV-cache buffers
//! so autoregressive decoder LLMs can call `step()` token-by-token without
//! re-allocating per-call tensors.
//!
//! Decoder LLMs exported via HuggingFace `optimum` follow this convention:
//!
//! - Inputs:
//!   - `input_ids` (i64, `[batch, seq_len]`)
//!   - `attention_mask` (i64, `[batch, total_len]`)
//!   - `past_key_values.{layer}.key` / `past_key_values.{layer}.value`
//!     (f32 or f16, `[batch, num_heads, past_len, head_dim]`)
//! - Outputs:
//!   - `logits` (f32, `[batch, seq_len, vocab_size]`)
//!   - `present.{layer}.key` / `present.{layer}.value`
//!
//! The first call uses `seq_len = prompt_length` and an empty past_kv.
//! Each subsequent call uses `seq_len = 1` and the previous step's
//! `present.*` as the new `past_key_values.*`.
//!
//! # KV cache shape
//!
//! [`KvShape`] describes the per-layer key/value tensor geometry. For Qwen2.5
//! and Llama-3.2 this is typically `[1, num_heads, past_len, head_dim]`.
//!
//! # Example (zero-copy intent, no model loaded)
//!
//! ```rust,no_run
//! # use airml_core::{InferenceEngine, BoundSession, KvDtype, KvShape};
//! # fn main() -> airml_core::Result<()> {
//! let engine = InferenceEngine::from_file("decoder.onnx")?;
//! let kv = KvShape { batch: 1, num_heads: 8, head_dim: 64 };
//! let mut bound = BoundSession::from_engine(engine, /*num_layers=*/ 24, kv, KvDtype::F32)?;
//! bound.reset_kv_cache();
//! for _ in 0..5 {
//!     let _logits = bound.step(/*token=*/ 42)?;
//! }
//! # Ok(())
//! # }
//! ```

use half::f16;
use ndarray::ArrayD;
use ort::io_binding::IoBinding;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::value::{DynValue, Tensor};

use crate::engine::{InferenceEngine, ModelMetadata, TensorInfo};
use crate::error::{AirMLError, Result};

/// Per-layer KV-cache geometry.
///
/// Each layer's key/value tensor has shape
/// `[batch, num_heads, past_len, head_dim]`.
#[derive(Debug, Clone, Copy)]
pub struct KvShape {
    /// Batch size (almost always 1 for chat).
    pub batch: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
}

/// KV cache element type.
///
/// Most HF optimum exports use f32; some quantised builds use f16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    /// 32-bit float keys/values.
    F32,
    /// 16-bit float keys/values (`half::f16`).
    F16,
}

/// Per-layer KV cache buffer.
///
/// Stored as a flat owned vector that we re-shape into a [`Tensor`] on each
/// `step()`. Re-binding the same vec each call is cheap; the heap allocation
/// happens only on `reset_kv_cache()` and on growth.
#[derive(Debug)]
struct KvBuffer {
    f32: Option<Vec<f32>>,
    f16: Option<Vec<f16>>,
    /// Current `past_len` represented in the buffer (0 == empty cache).
    past_len: usize,
}

impl KvBuffer {
    fn new_empty(dtype: KvDtype) -> Self {
        Self {
            f32: matches!(dtype, KvDtype::F32).then(Vec::new),
            f16: matches!(dtype, KvDtype::F16).then(Vec::new),
            past_len: 0,
        }
    }

    fn reset(&mut self) {
        self.past_len = 0;
        if let Some(v) = self.f32.as_mut() {
            v.clear();
        }
        if let Some(v) = self.f16.as_mut() {
            v.clear();
        }
    }

    fn is_empty(&self) -> bool {
        self.past_len == 0
    }
}

/// A wrapper around [`InferenceEngine`] that pre-allocates I/O buffers and
/// drives a decoder-only LLM step-by-step with KV-cache reuse.
///
/// `BoundSession` does **not** replace [`InferenceEngine::run_named`] — it is
/// a parallel API specialised for autoregressive token generation.
pub struct BoundSession {
    engine: InferenceEngine,
    binding: Option<IoBinding>,
    /// `past_key_values.{layer}.key` buffers.
    keys: Vec<KvBuffer>,
    /// `past_key_values.{layer}.value` buffers.
    values: Vec<KvBuffer>,
    layer_count: usize,
    kv_shape: KvShape,
    dtype: KvDtype,
    /// Number of `step()` invocations since the last `reset_kv_cache()`.
    /// Tracks the next `position_id` we would generate.
    position: usize,
    /// Cached layout discovered from model metadata (input/output names).
    layout: Layout,
}

#[derive(Debug, Clone)]
struct Layout {
    has_input_ids: bool,
    has_attention_mask: bool,
    /// `past_key_values.{i}.key` input names (length = layer_count if present).
    past_key_inputs: Vec<String>,
    past_value_inputs: Vec<String>,
    /// `present.{i}.key` output names.
    present_key_outputs: Vec<String>,
    present_value_outputs: Vec<String>,
    /// `logits` output name (or first f32 output, fallback).
    logits_output: Option<String>,
}

impl Layout {
    fn discover(metadata: &ModelMetadata, expected_layers: usize) -> Self {
        let mut past_key_inputs = vec![String::new(); expected_layers];
        let mut past_value_inputs = vec![String::new(); expected_layers];
        let mut present_key_outputs = vec![String::new(); expected_layers];
        let mut present_value_outputs = vec![String::new(); expected_layers];
        let mut has_input_ids = false;
        let mut has_attention_mask = false;
        let mut logits_output = None;

        for input in &metadata.inputs {
            if input.name == "input_ids" {
                has_input_ids = true;
            } else if input.name == "attention_mask" {
                has_attention_mask = true;
            } else if let Some((idx, kind)) = parse_kv_name(&input.name, "past_key_values") {
                if idx < expected_layers {
                    match kind {
                        KvKind::Key => past_key_inputs[idx] = input.name.clone(),
                        KvKind::Value => past_value_inputs[idx] = input.name.clone(),
                    }
                }
            }
        }

        for output in &metadata.outputs {
            if output.name == "logits" {
                logits_output = Some(output.name.clone());
            } else if let Some((idx, kind)) = parse_kv_name(&output.name, "present") {
                if idx < expected_layers {
                    match kind {
                        KvKind::Key => present_key_outputs[idx] = output.name.clone(),
                        KvKind::Value => present_value_outputs[idx] = output.name.clone(),
                    }
                }
            }
        }

        // Fallback: if we did not find a literal "logits" output, use the first
        // output that isn't a present_kv tensor.
        if logits_output.is_none() {
            for output in &metadata.outputs {
                if parse_kv_name(&output.name, "present").is_none() {
                    logits_output = Some(output.name.clone());
                    break;
                }
            }
        }

        Self {
            has_input_ids,
            has_attention_mask,
            past_key_inputs,
            past_value_inputs,
            present_key_outputs,
            present_value_outputs,
            logits_output,
        }
    }

    /// Number of fully-resolved past_key_values layer pairs.
    fn resolved_layers(&self) -> usize {
        self.past_key_inputs
            .iter()
            .zip(&self.past_value_inputs)
            .take_while(|(k, v)| !k.is_empty() && !v.is_empty())
            .count()
    }

    fn looks_like_decoder_llm(&self) -> bool {
        self.has_input_ids && self.resolved_layers() > 0
    }
}

#[derive(Debug, Clone, Copy)]
enum KvKind {
    Key,
    Value,
}

/// Parse `past_key_values.7.key` or `present.7.value` into `(7, Key|Value)`.
fn parse_kv_name(name: &str, prefix: &str) -> Option<(usize, KvKind)> {
    let rest = name.strip_prefix(prefix)?.strip_prefix('.')?;
    let mut parts = rest.splitn(2, '.');
    let idx: usize = parts.next()?.parse().ok()?;
    let kind = match parts.next()? {
        "key" => KvKind::Key,
        "value" => KvKind::Value,
        _ => return None,
    };
    Some((idx, kind))
}

impl BoundSession {
    /// Construct a [`BoundSession`] from a loaded [`InferenceEngine`].
    ///
    /// `layer_count` is the number of decoder layers (i.e. how many
    /// `past_key_values.{i}.key` / `value` inputs the model accepts).
    /// `kv_shape` describes the per-layer KV geometry.
    /// `dtype` selects the cache element type.
    ///
    /// This does **not** load a model — pass an existing [`InferenceEngine`].
    pub fn from_engine(
        engine: InferenceEngine,
        layer_count: usize,
        kv_shape: KvShape,
        dtype: KvDtype,
    ) -> Result<Self> {
        let layout = Layout::discover(engine.metadata(), layer_count);

        let keys = (0..layer_count).map(|_| KvBuffer::new_empty(dtype)).collect();
        let values = (0..layer_count).map(|_| KvBuffer::new_empty(dtype)).collect();

        let binding = engine
            .session
            .create_binding()
            .map_err(|e| AirMLError::OrtError(e.to_string()))?;

        Ok(Self {
            engine,
            binding: Some(binding),
            keys,
            values,
            layer_count,
            kv_shape,
            dtype,
            position: 0,
            layout,
        })
    }

    /// Number of decoder layers configured at construction time.
    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Configured KV element dtype.
    pub fn dtype(&self) -> KvDtype {
        self.dtype
    }

    /// Configured KV geometry.
    pub fn kv_shape(&self) -> KvShape {
        self.kv_shape
    }

    /// Logical position counter — number of tokens fed since the last reset.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether the wrapped model looks like a decoder-only LLM (has
    /// `input_ids` and at least one `past_key_values.*` input).
    pub fn looks_like_decoder_llm(&self) -> bool {
        self.layout.looks_like_decoder_llm()
    }

    /// Inspect the underlying model metadata (delegates to engine).
    pub fn metadata(&self) -> &ModelMetadata {
        self.engine.metadata()
    }

    /// Inspect the input tensor info.
    pub fn inputs(&self) -> &[TensorInfo] {
        self.engine.inputs()
    }

    /// Inspect the output tensor info.
    pub fn outputs(&self) -> &[TensorInfo] {
        self.engine.outputs()
    }

    /// Reset all KV-cache buffers to empty (`past_len = 0`).
    ///
    /// This must be called between independent prompts.
    pub fn reset_kv_cache(&mut self) {
        for buf in self.keys.iter_mut() {
            buf.reset();
        }
        for buf in self.values.iter_mut() {
            buf.reset();
        }
        self.position = 0;
        // Drop and recreate the binding so any previously-bound output tensors
        // are released; a stale binding would otherwise still reference the
        // old past_kv ptrs after we reallocate the buffers.
        if let Ok(new_binding) = self.engine.session.create_binding() {
            self.binding = Some(new_binding);
        }
    }

    /// Feed a single token through the decoder and return the logits for the
    /// **last** sequence position.
    ///
    /// On the first call after `reset_kv_cache()`, prefer [`Self::prefill`] to
    /// feed the entire prompt in one shot. `step()` always passes
    /// `seq_len = 1`.
    pub fn step(&mut self, token: i64) -> Result<Vec<f32>> {
        self.run_step(&[token])
    }

    /// Feed multiple prompt tokens in a single forward pass (prefill).
    ///
    /// Returns the logits for the **last** prompt position only.
    pub fn prefill(&mut self, tokens: &[i64]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(AirMLError::ConfigError(
                "prefill requires at least one token".into(),
            ));
        }
        self.run_step(tokens)
    }

    fn run_step(&mut self, tokens: &[i64]) -> Result<Vec<f32>> {
        if !self.layout.looks_like_decoder_llm() {
            return Err(AirMLError::ConfigError(
                "model does not appear to be a decoder LLM (no input_ids/past_key_values \
                 inputs found). Use `airml run` for non-LLM models."
                    .into(),
            ));
        }
        let resolved = self.layout.resolved_layers();
        if resolved < self.layer_count {
            return Err(AirMLError::ConfigError(format!(
                "configured layer_count={} but model only exposes {} past_key_values pairs",
                self.layer_count, resolved
            )));
        }

        let seq_len = tokens.len();
        let total_len = self.position + seq_len;

        // We rebuild a fresh IoBinding each step. This is O(num_layers)
        // pointer registrations, but avoids any unsafe lifetime juggling
        // and still spares the SessionInputValue allocation that
        // `run_named` performs.
        let mut binding = self
            .engine
            .session
            .create_binding()
            .map_err(|e| AirMLError::OrtError(e.to_string()))?;

        // input_ids: i64 [1, seq_len]
        let input_ids = Tensor::<i64>::from_array(([1usize, seq_len], tokens.to_vec()))
            .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
        binding
            .bind_input("input_ids", &input_ids)
            .map_err(|e| AirMLError::InferenceError(e.to_string()))?;

        // attention_mask: i64 [1, total_len] — all ones.
        let attn_mask_storage;
        if self.layout.has_attention_mask {
            attn_mask_storage = Tensor::<i64>::from_array((
                [1usize, total_len],
                vec![1i64; total_len],
            ))
            .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
            binding
                .bind_input("attention_mask", &attn_mask_storage)
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
        }

        // Bind past_key_values.{i}.key / value for every layer.
        // Hold the temporary tensors in a Vec so they live long enough.
        let mut held: Vec<KvTensorHolder> =
            Vec::with_capacity(self.layer_count * 2);

        for layer in 0..self.layer_count {
            let key_name = self.layout.past_key_inputs[layer].clone();
            let value_name = self.layout.past_value_inputs[layer].clone();
            let key_tensor = self.build_kv_tensor(layer, KvKind::Key)?;
            let value_tensor = self.build_kv_tensor(layer, KvKind::Value)?;

            match &key_tensor {
                KvTensorHolder::F32(t) => binding
                    .bind_input(key_name.clone(), t)
                    .map_err(|e| AirMLError::InferenceError(e.to_string()))?,
                KvTensorHolder::F16(t) => binding
                    .bind_input(key_name.clone(), t)
                    .map_err(|e| AirMLError::InferenceError(e.to_string()))?,
            }
            match &value_tensor {
                KvTensorHolder::F32(t) => binding
                    .bind_input(value_name.clone(), t)
                    .map_err(|e| AirMLError::InferenceError(e.to_string()))?,
                KvTensorHolder::F16(t) => binding
                    .bind_input(value_name.clone(), t)
                    .map_err(|e| AirMLError::InferenceError(e.to_string()))?,
            }

            held.push(key_tensor);
            held.push(value_tensor);
        }

        // Bind all outputs to the device — let ORT allocate the buffers.
        // We then extract logits + present_kv once run_binding returns.
        let cpu_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(|e| AirMLError::OrtError(e.to_string()))?;

        if let Some(name) = &self.layout.logits_output {
            binding
                .bind_output_to_device(name.clone(), &cpu_info)
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
        }
        for layer in 0..self.layer_count {
            binding
                .bind_output_to_device(self.layout.present_key_outputs[layer].clone(), &cpu_info)
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
            binding
                .bind_output_to_device(
                    self.layout.present_value_outputs[layer].clone(),
                    &cpu_info,
                )
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
        }

        let outputs = self
            .engine
            .session
            .run_binding(&binding)
            .map_err(|e| AirMLError::InferenceError(e.to_string()))?;

        // Extract logits.
        let logits_name = self
            .layout
            .logits_output
            .as_ref()
            .ok_or_else(|| AirMLError::InferenceError("model has no logits output".into()))?;
        let logits_value: &DynValue = outputs
            .get(logits_name)
            .ok_or_else(|| AirMLError::InferenceError("logits output missing".into()))?;
        let (shape, data) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
        let last_logits = extract_last_position_logits(shape, data)?;

        // Extract present_kv → swap into self.keys/self.values.
        let mut new_keys: Vec<KvBuffer> = (0..self.layer_count)
            .map(|_| KvBuffer::new_empty(self.dtype))
            .collect();
        let mut new_values: Vec<KvBuffer> = (0..self.layer_count)
            .map(|_| KvBuffer::new_empty(self.dtype))
            .collect();

        for layer in 0..self.layer_count {
            let key_out = &self.layout.present_key_outputs[layer];
            let value_out = &self.layout.present_value_outputs[layer];
            let key_value = outputs.get(key_out).ok_or_else(|| {
                AirMLError::InferenceError(format!("present output {} missing", key_out))
            })?;
            let value_value = outputs.get(value_out).ok_or_else(|| {
                AirMLError::InferenceError(format!("present output {} missing", value_out))
            })?;
            new_keys[layer] = extract_kv_buffer(self.dtype, key_value, total_len)?;
            new_values[layer] = extract_kv_buffer(self.dtype, value_value, total_len)?;
        }

        // Drop outputs / binding before mutating self.
        drop(outputs);
        drop(binding);
        drop(held);

        self.keys = new_keys;
        self.values = new_values;
        self.position = total_len;

        Ok(last_logits)
    }

    fn build_kv_tensor(&self, layer: usize, kind: KvKind) -> Result<KvTensorHolder> {
        let buf = match kind {
            KvKind::Key => &self.keys[layer],
            KvKind::Value => &self.values[layer],
        };
        let shape = [
            self.kv_shape.batch,
            self.kv_shape.num_heads,
            buf.past_len,
            self.kv_shape.head_dim,
        ];
        let elements = self.kv_shape.batch
            * self.kv_shape.num_heads
            * buf.past_len
            * self.kv_shape.head_dim;

        match self.dtype {
            KvDtype::F32 => {
                let data: Vec<f32> = if buf.is_empty() {
                    vec![0.0_f32; elements]
                } else {
                    buf.f32.as_ref().expect("dtype invariant").clone()
                };
                let tensor = Tensor::<f32>::from_array((shape, data))
                    .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
                Ok(KvTensorHolder::F32(tensor))
            }
            KvDtype::F16 => {
                let data: Vec<f16> = if buf.is_empty() {
                    vec![f16::ZERO; elements]
                } else {
                    buf.f16.as_ref().expect("dtype invariant").clone()
                };
                let tensor = Tensor::<f16>::from_array((shape, data))
                    .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
                Ok(KvTensorHolder::F16(tensor))
            }
        }
    }

}

fn extract_kv_buffer(dtype: KvDtype, value: &DynValue, expected_past_len: usize) -> Result<KvBuffer> {
    match dtype {
        KvDtype::F32 => {
            let (_shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
            Ok(KvBuffer {
                f32: Some(data.to_vec()),
                f16: None,
                past_len: expected_past_len,
            })
        }
        KvDtype::F16 => {
            let (_shape, data) = value
                .try_extract_tensor::<f16>()
                .map_err(|e| AirMLError::InferenceError(e.to_string()))?;
            Ok(KvBuffer {
                f32: None,
                f16: Some(data.to_vec()),
                past_len: expected_past_len,
            })
        }
    }
}

/// Owns a Tensor of the configured KV dtype, keeping it alive for the
/// duration of a `bind_input` call.
enum KvTensorHolder {
    F32(Tensor<f32>),
    F16(Tensor<f16>),
}

/// Given a `[batch, seq_len, vocab]` logits tensor, return the row for the
/// final sequence position.
fn extract_last_position_logits(shape: &[i64], data: &[f32]) -> Result<Vec<f32>> {
    if shape.len() != 3 {
        return Err(AirMLError::InferenceError(format!(
            "expected 3D logits tensor, got shape {:?}",
            shape
        )));
    }
    let seq_len = shape[1] as usize;
    let vocab = shape[2] as usize;
    if seq_len == 0 || vocab == 0 {
        return Err(AirMLError::InferenceError(
            "logits tensor has zero dimension".into(),
        ));
    }
    let total = data.len();
    if total < seq_len * vocab {
        return Err(AirMLError::InferenceError(
            "logits tensor smaller than declared shape".into(),
        ));
    }
    // Take the last sequence position from the first batch element.
    let start = (seq_len - 1) * vocab;
    Ok(data[start..start + vocab].to_vec())
}

/// Convenience wrapper for shape inference: count layers from output names
/// matching `present.{N}.key` (used by the `airml generate` CLI).
pub fn infer_layer_count_from_metadata(metadata: &ModelMetadata) -> usize {
    let mut max_idx: Option<usize> = None;
    for output in &metadata.outputs {
        if let Some((idx, KvKind::Key)) = parse_kv_name(&output.name, "present") {
            max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
        }
    }
    max_idx.map_or(0, |m| m + 1)
}

/// Helper: convert a flat logits slice to ndarray, preserving shape.
///
/// Useful for callers that want to apply temperature/top-k via ndarray.
pub fn logits_to_array(logits: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(ndarray::IxDyn(&[logits.len()]), logits.to_vec())
        .expect("flat logits always reshape to 1D")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_metadata() -> ModelMetadata {
        ModelMetadata {
            name: None,
            description: None,
            version: None,
            producer: None,
            inputs: vec![
                TensorInfo {
                    name: "input_ids".into(),
                    shape: vec![-1, -1],
                    dtype: "Int64".into(),
                },
                TensorInfo {
                    name: "attention_mask".into(),
                    shape: vec![-1, -1],
                    dtype: "Int64".into(),
                },
                TensorInfo {
                    name: "past_key_values.0.key".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "past_key_values.0.value".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "past_key_values.1.key".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "past_key_values.1.value".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
            ],
            outputs: vec![
                TensorInfo {
                    name: "logits".into(),
                    shape: vec![-1, -1, 32000],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "present.0.key".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "present.0.value".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "present.1.key".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "present.1.value".into(),
                    shape: vec![-1, 8, -1, 64],
                    dtype: "Float32".into(),
                },
            ],
        }
    }

    #[test]
    fn parse_kv_name_handles_well_formed_inputs() {
        assert!(matches!(
            parse_kv_name("past_key_values.7.key", "past_key_values"),
            Some((7, KvKind::Key))
        ));
        assert!(matches!(
            parse_kv_name("present.0.value", "present"),
            Some((0, KvKind::Value))
        ));
        assert!(parse_kv_name("input_ids", "past_key_values").is_none());
        assert!(parse_kv_name("past_key_values.x.key", "past_key_values").is_none());
    }

    #[test]
    fn layout_discovers_decoder_llm_shape() {
        let layout = Layout::discover(&dummy_metadata(), 2);
        assert!(layout.has_input_ids);
        assert!(layout.has_attention_mask);
        assert_eq!(layout.resolved_layers(), 2);
        assert_eq!(layout.past_key_inputs[1], "past_key_values.1.key");
        assert_eq!(layout.present_value_outputs[0], "present.0.value");
        assert_eq!(layout.logits_output.as_deref(), Some("logits"));
        assert!(layout.looks_like_decoder_llm());
    }

    #[test]
    fn layout_handles_non_llm_model() {
        let metadata = ModelMetadata {
            name: None,
            description: None,
            version: None,
            producer: None,
            inputs: vec![TensorInfo {
                name: "image".into(),
                shape: vec![1, 3, 224, 224],
                dtype: "Float32".into(),
            }],
            outputs: vec![TensorInfo {
                name: "scores".into(),
                shape: vec![1, 1000],
                dtype: "Float32".into(),
            }],
        };
        let layout = Layout::discover(&metadata, 0);
        assert!(!layout.has_input_ids);
        assert_eq!(layout.resolved_layers(), 0);
        assert!(!layout.looks_like_decoder_llm());
    }

    #[test]
    fn kv_buffer_reset_zeros_past_len() {
        let mut buf = KvBuffer {
            f32: Some(vec![1.0, 2.0, 3.0, 4.0]),
            f16: None,
            past_len: 2,
        };
        buf.reset();
        assert_eq!(buf.past_len, 0);
        assert!(buf.is_empty());
        assert!(buf.f32.as_ref().unwrap().is_empty());
    }

    #[test]
    fn infer_layer_count_counts_present_keys() {
        let metadata = dummy_metadata();
        assert_eq!(infer_layer_count_from_metadata(&metadata), 2);
    }

    #[test]
    fn extract_last_position_logits_returns_final_row() {
        // Shape [1, 3, 4] — three tokens, vocab 4. Last row should be [9, 10, 11, 12].
        let shape = [1i64, 3, 4];
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let last = extract_last_position_logits(&shape, &data).unwrap();
        assert_eq!(last, vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn extract_last_position_logits_rejects_bad_shape() {
        let shape = [1i64, 3];
        let data = vec![0.0_f32; 12];
        assert!(extract_last_position_logits(&shape, &data).is_err());
    }

    #[test]
    fn logits_to_array_round_trip() {
        let arr = logits_to_array(&[1.0, 2.0, 3.0]);
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr[ndarray::IxDyn(&[2])], 3.0);
    }
}
