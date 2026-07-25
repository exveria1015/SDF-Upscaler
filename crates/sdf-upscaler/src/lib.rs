//! Pure-Rust SDF upscaling kernels.
//!
//! Alpha-SDF includes a canonical x4 route with quantized parity against the
//! pinned Python reference corpus. Color-/Adaptive-SDF remain Python-only
//! until their later parity milestones pass.

mod alpha;
mod compat;
mod diagnostics;
mod error;
mod image;
mod options;

pub use alpha::{quantize_alpha_u8, upscale_alpha_sdf_x4, upscale_alpha_sdf_x4_experimental};
pub use diagnostics::{AlphaDiagnostics, AlphaOutput, QualityBoundary};
pub use error::UpscaleError;
pub use image::{AlphaInput, Plane, PlaneRef};
pub use options::AlphaOptions;

/// Rust crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pipeline identifier written by the Rust CLI.
pub const ALPHA_EXPERIMENTAL_PIPELINE_ID: &str = "rust_alpha_sdf_roi_x4_experimental_v1";

/// Canonical Alpha-SDF x4 pipeline identifier.
pub const ALPHA_CANONICAL_PIPELINE_ID: &str = "rust_alpha_sdf_main_x4_python_quantized_parity_v1";
