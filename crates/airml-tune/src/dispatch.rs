//! Thin glue layer: maps [`BackendRecommendation`] to ORT execution providers.
//!
//! Only compiled when the `coreml` feature is enabled.

use airml_providers::{
    ComputeUnits, CoreMLProvider, CpuProvider, ExecutionProviderDispatch,
};

use crate::oracle::BackendRecommendation;

/// Convert a [`BackendRecommendation`] to an ordered list of ORT execution providers.
///
/// The returned list always ends with [`CpuProvider`] as a fallback so that
/// inference never fails due to a missing accelerator.
///
/// # Example
///
/// ```rust,ignore
/// use airml_tune::dispatch::recommendation_to_providers;
/// use airml_tune::oracle::{BackendOracle, BackendRecommendation};
///
/// let oracle = BackendOracle::new();
/// let rec = oracle.recommend_for_metadata(&metadata);
/// let providers = recommendation_to_providers(&rec);
/// ```
pub fn recommendation_to_providers(rec: &BackendRecommendation) -> Vec<ExecutionProviderDispatch> {
    let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();

    match rec {
        BackendRecommendation::CoreMLAll => {
            providers.push(
                CoreMLProvider::new()
                    .with_compute_units(ComputeUnits::All)
                    .into_dispatch(),
            );
        }
        BackendRecommendation::CoreMLAneOnly => {
            providers.push(
                CoreMLProvider::new()
                    .with_compute_units(ComputeUnits::CpuAndNeuralEngine)
                    .into_dispatch(),
            );
        }
        BackendRecommendation::CoreMLGpuOnly => {
            providers.push(
                CoreMLProvider::new()
                    .with_compute_units(ComputeUnits::CpuAndGpu)
                    .into_dispatch(),
            );
        }
        BackendRecommendation::CpuOnly | BackendRecommendation::CpuOnlyWithReason(_) => {
            // No CoreML provider; CPU fallback below covers this case.
        }
    }

    // CPU is always appended as the final fallback.
    providers.push(CpuProvider::default().into_dispatch());

    providers
}
