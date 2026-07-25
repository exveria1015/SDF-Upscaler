//! Pure-Rust SDF upscaling kernels.
//!
//! The first Rust milestone intentionally exposes only the experimental
//! Alpha-SDF ROI route. The canonical Python Alpha-/Color-/Adaptive-SDF APIs
//! remain authoritative until cross-language parity gates pass.

mod alpha;
mod compat;
mod diagnostics;
mod error;
mod image;
mod options;

pub use alpha::{quantize_alpha_u8, upscale_alpha_sdf_x4_experimental};
pub use diagnostics::{AlphaDiagnostics, AlphaOutput, QualityBoundary};
pub use error::UpscaleError;
pub use image::{AlphaInput, Plane, PlaneRef};
pub use options::AlphaOptions;

/// Rust crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pipeline identifier written by the Rust CLI.
pub const ALPHA_EXPERIMENTAL_PIPELINE_ID: &str = "rust_alpha_sdf_roi_x4_experimental_v1";
