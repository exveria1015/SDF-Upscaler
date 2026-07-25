use serde::Serialize;

use crate::Plane;

/// Explicit validation boundary for a Rust route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityBoundary {
    /// Quantized alpha bytes match the pinned Python parity corpus.
    PythonReferenceQuantizedParity,
    ExperimentalNotPythonParity,
}

/// Diagnostics for an Alpha-SDF invocation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AlphaDiagnostics {
    pub implementation: &'static str,
    pub quality_boundary: QualityBoundary,
    pub input_size: [usize; 2],
    pub output_size: [usize; 2],
    pub constant_alpha_fast_path: bool,
    pub broad_soft_lanczos_fallback: bool,
    pub endpoint_fraction: f64,
    pub segment_count: usize,
    pub roi_fraction: f64,
    pub guard_accept_fraction: f64,
    pub changed_from_lanczos_fraction: f64,
}

/// Alpha image plus diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaOutput {
    pub alpha: Plane<f32>,
    pub diagnostics: AlphaDiagnostics,
}
