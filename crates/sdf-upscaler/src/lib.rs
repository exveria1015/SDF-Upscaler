//! Pure-Rust SDF upscaling kernels.
//!
//! Alpha-SDF and Color-SDF include canonical x4 routes with quantized parity
//! against the pinned Python reference corpus. Adaptive-SDF remains
//! Python-only until its later parity milestone passes.

mod alpha;
mod color;
mod compat;
mod diagnostics;
mod error;
mod image;
mod options;

pub use alpha::{quantize_alpha_u8, upscale_alpha_sdf_x4, upscale_alpha_sdf_x4_experimental};
pub use color::upscale_color_sdf_x4;
pub use diagnostics::{
    AlphaDiagnostics, AlphaOutput, ColorDiagnostics, ColorOutput, QualityBoundary,
};
pub use error::UpscaleError;
pub use image::{AlphaInput, Plane, PlaneRef, Rgba8, Rgba8Ref};
pub use options::{AlphaOptions, ColorOptions};

/// Rust crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pipeline identifier written by the Rust CLI.
pub const ALPHA_EXPERIMENTAL_PIPELINE_ID: &str = "rust_alpha_sdf_roi_x4_experimental_v1";

/// Canonical Alpha-SDF x4 pipeline identifier.
pub const ALPHA_CANONICAL_PIPELINE_ID: &str = "rust_alpha_sdf_main_x4_python_quantized_parity_v1";

/// Canonical Color-SDF x4 pipeline identifier.
pub const COLOR_CANONICAL_PIPELINE_ID: &str =
    "rust_color_sdf_sal_classic_x4_python_quantized_parity_v1";
