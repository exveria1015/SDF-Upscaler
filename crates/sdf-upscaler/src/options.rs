use crate::UpscaleError;

/// Options for the first Rust-native Alpha-SDF ROI milestone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaOptions {
    pub roi_radius_lr: usize,
    pub effect_band_hr: f32,
    pub residual_cap: f32,
    pub reprojection_guard: bool,
}

impl Default for AlphaOptions {
    fn default() -> Self {
        Self {
            roi_radius_lr: 2,
            effect_band_hr: 4.0,
            residual_cap: 0.20,
            reprojection_guard: true,
        }
    }
}

impl AlphaOptions {
    pub(crate) fn validate(self) -> Result<(), UpscaleError> {
        if self.roi_radius_lr > 64 {
            return Err(UpscaleError::InvalidOptions(
                "ROI radius must be at most 64 LR pixels",
            ));
        }
        if !self.effect_band_hr.is_finite() || self.effect_band_hr <= 0.0 {
            return Err(UpscaleError::InvalidOptions(
                "effect band must be positive and finite",
            ));
        }
        if !self.residual_cap.is_finite() || self.residual_cap <= 0.0 || self.residual_cap > 0.5 {
            return Err(UpscaleError::InvalidOptions(
                "residual cap must be in (0, 0.5]",
            ));
        }
        Ok(())
    }
}
