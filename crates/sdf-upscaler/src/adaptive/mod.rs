use crate::{
    AdaptiveDiagnostics, AdaptiveOptions, AdaptiveOutput, Plane, PlaneRef, Rgba8, Rgba8Ref,
    UpscaleError,
    alpha::{canonical::canonical_stages, quantize_alpha_u8},
    color::{linear_to_srgb, quantize_u8, segment_guided_rgb, srgb_to_linear},
    compat::{
        Resample, distance_transform_edt, distance_transform_edt_with_indices, resize_f32,
        resize_rgb_u8, uniform_filter_nearest,
    },
    structure::{edge_local_structure_fusion_x4, extract_structure_field},
};

const SCALE: usize = 4;

/// Upscale straight-sRGB RGBA8 with canonical x4 Adaptive-SDF.
///
/// When `supplied_alpha` is absent the canonical Alpha-SDF plane is used.
/// A supplied plane must have the x4 dimensions; finite values are clipped to
/// `[0, 1]` exactly as in the Python public API and are never reconstructed.
pub fn upscale_adaptive_sdf_x4(
    input: Rgba8Ref<'_>,
    supplied_alpha: Option<PlaneRef<'_, f32>>,
    options: &AdaptiveOptions,
) -> Result<AdaptiveOutput, UpscaleError> {
    options.validate()?;
    let width = input.width();
    let height = input.height();
    if width < 3 || height < 3 {
        return Err(UpscaleError::InvalidOptions(
            "Adaptive-SDF input must be at least 3x3",
        ));
    }
    let output_width = width
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_height = height
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let pixels = width
        .checked_mul(height)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_pixels = output_width
        .checked_mul(output_height)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let input_rgb_bytes = pixels
        .checked_mul(3)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let input_rgba_bytes = pixels
        .checked_mul(4)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_rgba_bytes = output_pixels
        .checked_mul(4)
        .ok_or(UpscaleError::DimensionOverflow)?;

    let mut rgba = Vec::with_capacity(input_rgba_bytes);
    let mut rgb = Vec::with_capacity(input_rgb_bytes);
    let mut alpha_lr = Vec::with_capacity(pixels);
    for y in 0..height {
        for pixel in input.row(y).chunks_exact(4) {
            rgba.extend_from_slice(pixel);
            rgb.extend_from_slice(&pixel[..3]);
            alpha_lr.push(f32::from(pixel[3]) / 255.0);
        }
    }
    let alpha_stages = canonical_stages(&alpha_lr, width, height)?;
    let alpha_hr = if let Some(alpha) = supplied_alpha {
        if alpha.width() != output_width || alpha.height() != output_height {
            return Err(UpscaleError::InvalidOptions(
                "supplied alpha must match the x4 output dimensions",
            ));
        }
        let mut packed = Vec::with_capacity(output_pixels);
        for y in 0..output_height {
            for &value in alpha.row(y) {
                if !value.is_finite() {
                    return Err(UpscaleError::InvalidAlpha {
                        index: packed.len(),
                    });
                }
                packed.push(value.clamp(0.0, 1.0));
            }
        }
        packed
    } else {
        alpha_stages.final_alpha.clone()
    };

    let (base_u8, sal_diagnostics) = segment_guided_rgb(
        &rgb,
        &alpha_lr,
        &alpha_hr,
        width,
        height,
        &alpha_stages.graph,
        &options.sal,
    )?;
    let base_linear = rgb_u8_to_linear(&base_u8);

    let bounded = bounded_visible_rgb(
        &rgba,
        width,
        height,
        options.hidden_rgb_radius,
        options.hidden_rgb_feather,
    );
    let bounded_base_u8 = resize_rgb_u8(
        &bounded,
        width,
        height,
        output_width,
        output_height,
        Resample::Lanczos3,
    );
    let bounded_base_linear = rgb_u8_to_linear(&bounded_base_u8);
    let structure_field = extract_structure_field(&bounded, width, height)?;
    let (structure_u8, structure_diagnostics) = edge_local_structure_fusion_x4(
        &bounded,
        &bounded_base_u8,
        width,
        height,
        &structure_field,
        options.structure_max_residual,
    )?;
    let structure_linear = rgb_u8_to_linear(&structure_u8);
    let structure_residual = structure_linear
        .iter()
        .zip(&bounded_base_linear)
        .map(|(structure, base)| {
            [
                structure[0] - base[0],
                structure[1] - base[1],
                structure[2] - base[2],
            ]
        })
        .collect::<Vec<_>>();

    let opaque = alpha_hr
        .iter()
        .map(|&value| value >= options.opaque_threshold)
        .collect::<Vec<_>>();
    let interior_distance = distance_transform_edt(&opaque, output_width, output_height);
    let interior_gate = interior_distance
        .into_iter()
        .map(|distance| {
            smoothstep(
                (distance - options.interior_start_hr)
                    / (options.interior_stop_hr - options.interior_start_hr),
            )
        })
        .collect::<Vec<_>>();
    let mut candidate = (0..output_pixels)
        .map(|index| {
            let weight = options.structure_strength * interior_gate[index];
            [
                (base_linear[index][0] + weight * structure_residual[index][0]).clamp(0.0, 1.0),
                (base_linear[index][1] + weight * structure_residual[index][1]).clamp(0.0, 1.0),
                (base_linear[index][2] + weight * structure_residual[index][2]).clamp(0.0, 1.0),
            ]
        })
        .collect::<Vec<_>>();
    let median = median_rgb_nearest(&candidate, output_width, output_height);
    let mut median_luma = vec![0.0_f32; output_pixels];
    let mut median_active = vec![0_u8; output_pixels];
    for index in 0..output_pixels {
        let mut residual = [0.0_f32; 3];
        for channel in 0..3 {
            residual[channel] = (median[index][channel] - candidate[index][channel])
                .clamp(-options.median_max_residual, options.median_max_residual);
            if opaque[index] {
                candidate[index][channel] =
                    (candidate[index][channel] + residual[channel]).clamp(0.0, 1.0);
            }
        }
        median_luma[index] =
            residual[0].abs() * 0.2126 + residual[1].abs() * 0.7152 + residual[2].abs() * 0.0722;
        median_active[index] = u8::from(opaque[index] && median_luma[index] > 1e-8);
    }

    let lr_linear = rgb_u8_to_linear(&bounded);
    let lr_premultiplied = lr_linear
        .iter()
        .zip(&alpha_lr)
        .map(|(pixel, &alpha)| [pixel[0] * alpha, pixel[1] * alpha, pixel[2] * alpha])
        .collect::<Vec<_>>();
    let base_reprojected = downsample_premultiplied(
        &base_linear,
        &alpha_hr,
        output_width,
        output_height,
        width,
        height,
    );
    let candidate_before = downsample_premultiplied(
        &candidate,
        &alpha_hr,
        output_width,
        output_height,
        width,
        height,
    );
    let base_error_map = mean_absolute_rgb(&base_reprojected, &lr_premultiplied);
    let candidate_before_error = mean_absolute_error(&candidate_before, &lr_premultiplied);

    let (local_min_lr, local_max_lr) = local_rgb_range(&lr_linear, width, height);
    let local_min_hr = resize_rgb_f32(
        &local_min_lr,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let local_max_hr = resize_rgb_f32(
        &local_max_lr,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let mut backprojected = candidate;
    for _ in 0..options.backprojection_iterations {
        let reprojected = downsample_premultiplied(
            &backprojected,
            &alpha_hr,
            output_width,
            output_height,
            width,
            height,
        );
        let error_lr = reprojected
            .iter()
            .zip(&lr_premultiplied)
            .map(|(actual, target)| {
                [
                    target[0] - actual[0],
                    target[1] - actual[1],
                    target[2] - actual[2],
                ]
            })
            .collect::<Vec<_>>();
        let error_hr = resize_rgb_f32(
            &error_lr,
            width,
            height,
            output_width,
            output_height,
            Resample::Lanczos3,
        );
        for index in 0..output_pixels {
            if opaque[index] {
                let safe_alpha = alpha_hr[index].max(1e-6);
                for channel in 0..3 {
                    backprojected[index][channel] +=
                        options.backprojection_strength * error_hr[index][channel] / safe_alpha;
                    backprojected[index][channel] = backprojected[index][channel].clamp(
                        local_min_hr[index][channel] - options.local_range_epsilon,
                        local_max_hr[index][channel] + options.local_range_epsilon,
                    );
                }
            }
            for value in &mut backprojected[index] {
                *value = value.clamp(0.0, 1.0);
            }
        }
    }

    let candidate_after = downsample_premultiplied(
        &backprojected,
        &alpha_hr,
        output_width,
        output_height,
        width,
        height,
    );
    let candidate_error_map = mean_absolute_rgb(&candidate_after, &lr_premultiplied);
    let base_error_smooth =
        uniform_filter_nearest(&base_error_map, width, height, options.reprojection_window);
    let candidate_error_smooth = uniform_filter_nearest(
        &candidate_error_map,
        width,
        height,
        options.reprojection_window,
    );
    let transition = options.reprojection_transition.max(1e-8);
    let reprojection_gate_lr = base_error_smooth
        .iter()
        .zip(&candidate_error_smooth)
        .map(|(&base, &candidate)| smoothstep((base - candidate) / transition))
        .collect::<Vec<_>>();
    let reprojection_gate_hr = resize_f32(
        &reprojection_gate_lr,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let mut guarded = base_linear.clone();
    for index in 0..output_pixels {
        if opaque[index] {
            for channel in 0..3 {
                guarded[index][channel] = (base_linear[index][channel]
                    + reprojection_gate_hr[index]
                        * (backprojected[index][channel] - base_linear[index][channel]))
                    .clamp(0.0, 1.0);
            }
        }
    }
    let guarded_reprojected = downsample_premultiplied(
        &guarded,
        &alpha_hr,
        output_width,
        output_height,
        width,
        height,
    );
    let base_error = mean_absolute_error(&base_reprojected, &lr_premultiplied);
    let candidate_after_error = mean_absolute_error(&candidate_after, &lr_premultiplied);
    let guarded_error = mean_absolute_error(&guarded_reprojected, &lr_premultiplied);
    let global_fallback = f64::from(guarded_error) > f64::from(base_error) + 1e-9;
    let mut final_linear = if global_fallback {
        base_linear
    } else {
        guarded
    };
    let final_error = if global_fallback {
        base_error
    } else {
        guarded_error
    };
    for index in 0..output_pixels {
        if alpha_hr[index] <= 1e-6 {
            final_linear[index] = [0.0; 3];
        }
    }
    let final_rgb = linear_to_rgb_u8(&final_linear);
    let alpha_u8 = quantize_alpha_u8(&alpha_hr);
    let mut output_rgba = Vec::with_capacity(output_rgba_bytes);
    for (pixel, &alpha) in final_rgb.chunks_exact(3).zip(&alpha_u8) {
        output_rgba.extend_from_slice(pixel);
        output_rgba.push(alpha);
    }
    Ok(AdaptiveOutput {
        image: Rgba8::new(output_width, output_height, output_rgba)?,
        diagnostics: AdaptiveDiagnostics {
            implementation: crate::ADAPTIVE_CANONICAL_PIPELINE_ID,
            sal: sal_diagnostics,
            structure: structure_diagnostics,
            structure_interior_gate_hr: Plane::new(output_width, output_height, interior_gate)?,
            median_active_mask_hr: Plane::new(output_width, output_height, median_active)?,
            median_residual_luma_hr: Plane::new(output_width, output_height, median_luma)?,
            reprojection_gate_lr: Plane::new(width, height, reprojection_gate_lr)?,
            reprojection_gate_hr: Plane::new(output_width, output_height, reprojection_gate_hr)?,
            lr_error_base: base_error,
            lr_error_candidate_before: candidate_before_error,
            lr_error_candidate_after: candidate_after_error,
            lr_error_final: final_error,
            global_reprojection_fallback: global_fallback,
        },
    })
}

fn bounded_visible_rgb(
    rgba: &[u8],
    width: usize,
    height: usize,
    radius: f32,
    feather: f32,
) -> Vec<u8> {
    let pixels = width * height;
    let known = rgba
        .chunks_exact(4)
        .map(|pixel| pixel[3] > 0)
        .collect::<Vec<_>>();
    let mut output = vec![0_u8; pixels * 3];
    for (index, pixel) in rgba.chunks_exact(4).enumerate() {
        if known[index] {
            output[index * 3..index * 3 + 3].copy_from_slice(&pixel[..3]);
        }
    }
    if radius == 0.0 || !known.iter().any(|&value| value) || known.iter().all(|&value| value) {
        return output;
    }
    let unknown = known.iter().map(|&value| !value).collect::<Vec<_>>();
    let (distance, nearest_y, nearest_x) =
        distance_transform_edt_with_indices(&unknown, width, height);
    for index in 0..pixels {
        if known[index] {
            continue;
        }
        let weight = if feather == 0.0 {
            f32::from(distance[index] <= radius)
        } else {
            smoothstep((radius - distance[index]) / feather)
        };
        let nearest = nearest_y[index] * width + nearest_x[index];
        for channel in 0..3 {
            output[index * 3 + channel] = (f32::from(rgba[nearest * 4 + channel]) * weight)
                .round_ties_even()
                .clamp(0.0, 255.0) as u8;
        }
    }
    output
}

fn rgb_u8_to_linear(input: &[u8]) -> Vec<[f32; 3]> {
    input
        .chunks_exact(3)
        .map(|pixel| {
            [
                srgb_to_linear(f32::from(pixel[0]) / 255.0),
                srgb_to_linear(f32::from(pixel[1]) / 255.0),
                srgb_to_linear(f32::from(pixel[2]) / 255.0),
            ]
        })
        .collect()
}

fn linear_to_rgb_u8(input: &[[f32; 3]]) -> Vec<u8> {
    input
        .iter()
        .flat_map(|pixel| {
            pixel.map(|value| quantize_u8(linear_to_srgb(value.clamp(0.0, 1.0)).clamp(0.0, 1.0)))
        })
        .collect()
}

fn resize_rgb_f32(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    output_width: usize,
    output_height: usize,
    filter: Resample,
) -> Vec<[f32; 3]> {
    let mut output = vec![[0.0_f32; 3]; output_width * output_height];
    for channel in 0..3 {
        let plane = input.iter().map(|pixel| pixel[channel]).collect::<Vec<_>>();
        let resized = resize_f32(&plane, width, height, output_width, output_height, filter);
        for (index, value) in resized.into_iter().enumerate() {
            output[index][channel] = value;
        }
    }
    output
}

fn downsample_premultiplied(
    straight: &[[f32; 3]],
    alpha: &[f32],
    width: usize,
    height: usize,
    output_width: usize,
    output_height: usize,
) -> Vec<[f32; 3]> {
    let premultiplied = straight
        .iter()
        .zip(alpha)
        .map(|(pixel, &alpha)| [pixel[0] * alpha, pixel[1] * alpha, pixel[2] * alpha])
        .collect::<Vec<_>>();
    resize_rgb_f32(
        &premultiplied,
        width,
        height,
        output_width,
        output_height,
        Resample::Lanczos3,
    )
}

fn local_rgb_range(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut minimum = vec![[f32::INFINITY; 3]; width * height];
    let mut maximum = vec![[f32::NEG_INFINITY; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    for channel in 0..3 {
                        let value = input[yy * width + xx][channel];
                        minimum[index][channel] = minimum[index][channel].min(value);
                        maximum[index][channel] = maximum[index][channel].max(value);
                    }
                }
            }
        }
    }
    (minimum, maximum)
}

fn median_rgb_nearest(input: &[[f32; 3]], width: usize, height: usize) -> Vec<[f32; 3]> {
    let mut output = vec![[0.0_f32; 3]; input.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let mut values = [0.0_f32; 9];
                let mut offset = 0;
                for yy in [y.saturating_sub(1), y, (y + 1).min(height - 1)] {
                    for xx in [x.saturating_sub(1), x, (x + 1).min(width - 1)] {
                        values[offset] = input[yy * width + xx][channel];
                        offset += 1;
                    }
                }
                values.sort_unstable_by(f32::total_cmp);
                output[y * width + x][channel] = values[4];
            }
        }
    }
    output
}

fn mean_absolute_rgb(left: &[[f32; 3]], right: &[[f32; 3]]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            ((left[0] - right[0]).abs() + (left[1] - right[1]).abs() + (left[2] - right[2]).abs())
                / 3.0
        })
        .collect()
}

fn mean_absolute_error(left: &[[f32; 3]], right: &[[f32; 3]]) -> f32 {
    let sum = left
        .iter()
        .zip(right)
        .map(|(left, right)| {
            (left[0] - right[0]).abs() + (left[1] - right[1]).abs() + (left[2] - right[2]).abs()
        })
        .sum::<f32>();
    sum / (left.len() * 3) as f32
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED_WIDTH: usize = 22;
    const FIXED_HEIGHT: usize = 18;

    #[test]
    fn bounded_visible_rgb_matches_python() {
        let input = include_bytes!("../../../../fixtures/parity/adaptive/fixed/input_rgba_u8.raw");
        assert_eq!(
            bounded_visible_rgb(input, FIXED_WIDTH, FIXED_HEIGHT, 16.0, 8.0),
            include_bytes!("../../../../fixtures/parity/adaptive/fixed/bounded_visible_rgb_u8.raw")
        );
    }

    #[test]
    fn fixed_adaptive_rgba_matches_python() {
        let input = include_bytes!("../../../../fixtures/parity/adaptive/fixed/input_rgba_u8.raw");
        let output = upscale_adaptive_sdf_x4(
            Rgba8Ref::packed(FIXED_WIDTH, FIXED_HEIGHT, input).unwrap(),
            None,
            &AdaptiveOptions::default(),
        )
        .unwrap();
        let expected =
            include_bytes!("../../../../fixtures/parity/adaptive/fixed/final_rgba_u8.raw");
        let differences = output
            .image
            .as_slice()
            .iter()
            .zip(expected)
            .enumerate()
            .filter(|(_, (actual, expected))| actual != expected)
            .map(|(index, (&actual, &expected))| (index, actual, expected))
            .collect::<Vec<_>>();
        assert!(
            differences.is_empty(),
            "{} RGBA mismatches; first {:?}",
            differences.len(),
            differences.first()
        );
        assert!(!output.diagnostics.global_reprojection_fallback);
        assert!(output.diagnostics.lr_error_final <= output.diagnostics.lr_error_base + 1e-9);
        for (actual, expected) in [
            (output.diagnostics.lr_error_base, 0.012_311_483_f32),
            (
                output.diagnostics.lr_error_candidate_before,
                0.012_756_122_f32,
            ),
            (
                output.diagnostics.lr_error_candidate_after,
                0.010_597_710_5_f32,
            ),
            (output.diagnostics.lr_error_final, 0.010_599_285_f32),
        ] {
            assert!(
                (actual - expected).abs() <= 2e-7,
                "diagnostic scalar {actual} != {expected}"
            );
        }
    }

    #[test]
    fn compact_adaptive_corpus_matches_python() {
        let root = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/sdf-upscaler")
            .unwrap();
        for (name, width, height) in [
            ("hidden_rgb_a", 9, 7),
            ("hidden_rgb_b", 9, 7),
            ("opaque", 7, 5),
            ("partial_alpha", 9, 7),
            ("transparent", 7, 5),
            ("line_art", 11, 9),
            ("odd_dimensions", 7, 5),
        ] {
            let input = std::fs::read(format!(
                "{root}/fixtures/parity/adaptive/cases/{name}/input_rgba_u8.raw"
            ))
            .unwrap();
            let expected = std::fs::read(format!(
                "{root}/fixtures/parity/adaptive/cases/{name}/output_rgba_u8.raw"
            ))
            .unwrap();
            let output = upscale_adaptive_sdf_x4(
                Rgba8Ref::packed(width, height, &input).unwrap(),
                None,
                &AdaptiveOptions::default(),
            )
            .unwrap();
            assert_eq!(output.image.as_slice(), expected, "case {name}");
        }
    }

    #[test]
    fn supplied_alpha_is_preserved() {
        let root = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/sdf-upscaler")
            .unwrap();
        let case = format!("{root}/fixtures/parity/adaptive/cases/supplied_alpha");
        let input = std::fs::read(format!("{case}/input_rgba_u8.raw")).unwrap();
        let alpha = std::fs::read(format!("{case}/hr_alpha_f32.raw"))
            .unwrap()
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();
        let expected = std::fs::read(format!("{case}/output_rgba_u8.raw")).unwrap();
        let output = upscale_adaptive_sdf_x4(
            Rgba8Ref::packed(7, 5, &input).unwrap(),
            Some(PlaneRef::packed(28, 20, &alpha).unwrap()),
            &AdaptiveOptions::default(),
        )
        .unwrap();
        assert_eq!(output.image.as_slice(), expected);
        assert_eq!(
            output
                .image
                .as_slice()
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            quantize_alpha_u8(&alpha)
        );
    }
}
