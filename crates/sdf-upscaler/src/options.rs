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

/// Options for the canonical x4 Color-SDF (SAL-Classic RGB) route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorOptions {
    pub level: f32,
    pub nearest_segments: usize,
    pub support_radius_normal: f32,
    pub support_radius_tangent: f32,
    pub lanczos_lobes: usize,
    pub steer_kernel: bool,
    pub edge_band_hr: f32,
    pub condition_limit: f32,
    pub condition_soft_start: f32,
    pub arc_radius: f32,
    pub contour_gate_enabled: bool,
    pub alpha_sample_floor: f32,
    pub alpha_mass_floor: f32,
    pub local_range_epsilon: f32,
    pub max_residual: f32,
    pub strength: f32,
    pub hard_edge_gate: bool,
    pub line_art_bypass_enabled: bool,
    pub line_art_radius_limit: f32,
    pub line_art_min_component: usize,
    pub line_art_neighborhood: usize,
    pub line_art_peak_min: f32,
    pub line_art_max_foreground_area: Option<usize>,
    pub line_art_max_canvas_area: Option<usize>,
    pub line_art_ratio_start: f32,
    pub line_art_ratio_stop: f32,
}

impl Default for ColorOptions {
    fn default() -> Self {
        Self {
            level: 0.5,
            nearest_segments: 12,
            support_radius_normal: 3.0,
            support_radius_tangent: 3.0,
            lanczos_lobes: 3,
            steer_kernel: false,
            edge_band_hr: 3.5,
            condition_limit: 8.0,
            condition_soft_start: 2.0,
            arc_radius: 6.0,
            contour_gate_enabled: true,
            alpha_sample_floor: 1.0 / 255.0,
            alpha_mass_floor: 0.03,
            local_range_epsilon: 0.01,
            max_residual: 0.20,
            strength: 1.0,
            hard_edge_gate: true,
            line_art_bypass_enabled: true,
            line_art_radius_limit: 0.75,
            line_art_min_component: 3,
            line_art_neighborhood: 3,
            line_art_peak_min: 0.8,
            line_art_max_foreground_area: Some(4096),
            line_art_max_canvas_area: Some(16384),
            line_art_ratio_start: 0.05,
            line_art_ratio_stop: 0.15,
        }
    }
}

impl ColorOptions {
    pub(crate) fn validate(self) -> Result<(), UpscaleError> {
        if !self.level.is_finite() || !(0.0..1.0).contains(&self.level) {
            return Err(UpscaleError::InvalidOptions(
                "level must be strictly between zero and one",
            ));
        }
        if self.nearest_segments == 0 || self.lanczos_lobes == 0 {
            return Err(UpscaleError::InvalidOptions(
                "segment count and Lanczos lobes must be positive",
            ));
        }
        if !self.support_radius_normal.is_finite()
            || self.support_radius_normal <= 0.0
            || !self.support_radius_tangent.is_finite()
            || self.support_radius_tangent <= 0.0
        {
            return Err(UpscaleError::InvalidOptions(
                "support radii must be positive and finite",
            ));
        }
        if !self.edge_band_hr.is_finite()
            || self.edge_band_hr <= 0.0
            || !self.arc_radius.is_finite()
            || self.arc_radius <= 0.0
        {
            return Err(UpscaleError::InvalidOptions(
                "edge band and arc radius must be positive and finite",
            ));
        }
        if !self.condition_soft_start.is_finite()
            || !self.condition_limit.is_finite()
            || self.condition_soft_start < 1.0
            || self.condition_soft_start >= self.condition_limit
        {
            return Err(UpscaleError::InvalidOptions(
                "condition thresholds require 1 <= soft start < limit",
            ));
        }
        if !self.alpha_sample_floor.is_finite()
            || !(0.0..1.0).contains(&self.alpha_sample_floor)
            || !self.alpha_mass_floor.is_finite()
            || self.alpha_mass_floor <= 0.0
        {
            return Err(UpscaleError::InvalidOptions(
                "alpha support thresholds are invalid",
            ));
        }
        if !self.local_range_epsilon.is_finite()
            || self.local_range_epsilon < 0.0
            || !self.max_residual.is_finite()
            || self.max_residual < 0.0
            || !self.strength.is_finite()
            || !(0.0..=1.0).contains(&self.strength)
        {
            return Err(UpscaleError::InvalidOptions(
                "range, residual, or strength option is invalid",
            ));
        }
        if !self.line_art_radius_limit.is_finite()
            || self.line_art_radius_limit <= 0.0
            || self.line_art_min_component == 0
            || self.line_art_neighborhood == 0
            || self.line_art_neighborhood.is_multiple_of(2)
            || !self.line_art_peak_min.is_finite()
            || !(0.0..=1.0).contains(&self.line_art_peak_min)
            || !self.line_art_ratio_start.is_finite()
            || !self.line_art_ratio_stop.is_finite()
            || self.line_art_ratio_start < 0.0
            || self.line_art_ratio_start >= self.line_art_ratio_stop
        {
            return Err(UpscaleError::InvalidOptions(
                "line-art bypass options are invalid",
            ));
        }
        Ok(())
    }
}
