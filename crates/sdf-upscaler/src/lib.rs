//! Pure-Rust SDF upscaling kernels.
//!
//! Alpha-SDF, Color-SDF, and Adaptive-SDF include canonical x4 routes with
//! quantized parity against the pinned Python reference corpus.

mod adaptive;
mod alpha;
mod color;
mod compat;
mod diagnostics;
mod error;
mod image;
mod options;
mod structure;

pub use adaptive::upscale_adaptive_sdf_x4;
pub use alpha::{quantize_alpha_u8, upscale_alpha_sdf_x4, upscale_alpha_sdf_x4_experimental};
pub use color::upscale_color_sdf_x4;
pub use diagnostics::{
    AdaptiveDiagnostics, AdaptiveOutput, AlphaDiagnostics, AlphaOutput, ColorDiagnostics,
    ColorOutput, QualityBoundary, StructureFusionDiagnostics,
};
pub use error::UpscaleError;
pub use image::{AlphaInput, Plane, PlaneRef, Rgba8, Rgba8Ref};
pub use options::{AdaptiveOptions, AlphaOptions, ColorOptions};

/// Rust crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pipeline identifier written by the Rust CLI.
pub const ALPHA_EXPERIMENTAL_PIPELINE_ID: &str = "rust_alpha_sdf_roi_x4_experimental_v1";

/// Canonical Alpha-SDF x4 pipeline identifier.
pub const ALPHA_CANONICAL_PIPELINE_ID: &str = "rust_alpha_sdf_main_x4_python_quantized_parity_v1";

/// Canonical Color-SDF x4 pipeline identifier.
pub const COLOR_CANONICAL_PIPELINE_ID: &str =
    "rust_color_sdf_sal_classic_x4_python_quantized_parity_v1";

/// Canonical Adaptive-SDF x4 pipeline identifier.
pub const ADAPTIVE_CANONICAL_PIPELINE_ID: &str =
    "rust_adaptive_color_sdf_x4_python_quantized_parity_v1";
