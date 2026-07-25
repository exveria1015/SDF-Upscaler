use crate::{
    ColorDiagnostics, ColorOptions, ColorOutput, Plane, Rgba8, Rgba8Ref, UpscaleError,
    alpha::{
        canonical::{
            LineArtReliabilityOptions, SegmentGraph, canonical_stages,
            hard_edge_segment_reliability, hr_queries, smoothstep,
        },
        quantize_alpha_u8,
    },
    compat::{Resample, nearest_segments, resize_f32},
};

const SCALE: usize = 4;

/// Upscale straight-sRGB RGBA8 with the canonical x4 Color-SDF route.
pub fn upscale_color_sdf_x4(
    input: Rgba8Ref<'_>,
    options: &ColorOptions,
) -> Result<ColorOutput, UpscaleError> {
    options.validate()?;
    if options.steer_kernel {
        return Err(UpscaleError::InvalidOptions(
            "steered Color-SDF kernels are not available in the parity route",
        ));
    }

    let width = input.width();
    let height = input.height();
    let output_width = width
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_height = height
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_pixels = output_width
        .checked_mul(output_height)
        .ok_or(UpscaleError::DimensionOverflow)?;
    output_pixels
        .checked_mul(4)
        .ok_or(UpscaleError::DimensionOverflow)?;

    let mut rgb = Vec::with_capacity(
        width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(UpscaleError::DimensionOverflow)?,
    );
    let mut alpha = Vec::with_capacity(
        width
            .checked_mul(height)
            .ok_or(UpscaleError::DimensionOverflow)?,
    );
    for y in 0..height {
        for pixel in input.row(y).chunks_exact(4) {
            rgb.extend_from_slice(&pixel[..3]);
            alpha.push(f32::from(pixel[3]) / 255.0);
        }
    }

    let alpha_stages = canonical_stages(&alpha, width, height)?;
    let (rgb_hr, diagnostics) = segment_guided_rgb(
        &rgb,
        &alpha,
        &alpha_stages.final_alpha,
        width,
        height,
        &alpha_stages.graph,
        options,
    )?;
    let alpha_u8 = quantize_alpha_u8(&alpha_stages.final_alpha);
    let mut rgba = Vec::with_capacity(output_pixels * 4);
    for (rgb, &alpha) in rgb_hr.chunks_exact(3).zip(&alpha_u8) {
        rgba.extend_from_slice(rgb);
        rgba.push(alpha);
    }
    Ok(ColorOutput {
        image: Rgba8::new(output_width, output_height, rgba)?,
        diagnostics,
    })
}

fn segment_guided_rgb(
    rgb: &[u8],
    alpha_lr: &[f32],
    alpha_hr: &[f32],
    width: usize,
    height: usize,
    graph: &SegmentGraph,
    options: &ColorOptions,
) -> Result<(Vec<u8>, ColorDiagnostics), UpscaleError> {
    let output_width = width * SCALE;
    let output_height = height * SCALE;
    let output_pixels = output_width * output_height;
    let base_u8 = resize_premultiplied_rgb_lanczos(rgb, alpha_lr, width, height);
    let mut output_linear = base_u8
        .iter()
        .map(|&value| srgb_to_linear(f32::from(value) / 255.0))
        .collect::<Vec<_>>();
    let mut gate = vec![0.0_f32; output_pixels];
    let mut condition = vec![f32::INFINITY; output_pixels];
    let mut contour_support = vec![0.0_f32; output_pixels];
    let mut residual_luma = vec![0.0_f32; output_pixels];

    if !graph.segments.is_empty() {
        let hr_points = hr_queries(output_width, output_height);
        let (_, _, hr_segment_indices, _, hr_distance_lr) =
            nearest_segments(&graph.segments, &hr_points, options.nearest_segments);
        let hr_segment_indices = hr_segment_indices
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let segment_reliability = if options.hard_edge_gate {
            let line_art = options
                .line_art_bypass_enabled
                .then_some(LineArtReliabilityOptions {
                    medial_radius_limit: options.line_art_radius_limit,
                    min_component: options.line_art_min_component,
                    neighborhood: options.line_art_neighborhood,
                    peak_min: options.line_art_peak_min,
                    max_foreground_area: options.line_art_max_foreground_area,
                    max_canvas_area: options.line_art_max_canvas_area,
                    ratio_start: f64::from(options.line_art_ratio_start),
                    ratio_stop: f64::from(options.line_art_ratio_stop),
                });
            hard_edge_segment_reliability(alpha_lr, width, height, graph, line_art)
        } else {
            vec![1.0; graph.segments.len()]
        };
        let frame_reliability = hr_segment_indices
            .iter()
            .map(|&index| segment_reliability[index])
            .collect::<Vec<_>>();
        let partial_alpha_gate = alpha_hr
            .iter()
            .map(|&alpha| (4.0 * alpha * (1.0 - alpha)).clamp(0.0, 1.0).sqrt())
            .collect::<Vec<_>>();
        let active = (0..output_pixels)
            .filter(|&index| {
                hr_distance_lr[index] * SCALE as f32 <= options.edge_band_hr
                    && alpha_hr[index] > 1e-6
                    && alpha_hr[index] < 1.0 - 1e-6
                    && partial_alpha_gate[index] > 1e-4
                    && frame_reliability[index] > 1e-4
            })
            .collect::<Vec<_>>();

        if !active.is_empty() {
            let lr_points = (0..height)
                .flat_map(|y| (0..width).map(move |x| [y as f32, x as f32]))
                .collect::<Vec<_>>();
            let (_, _, lr_segment_indices, _, _) =
                nearest_segments(&graph.segments, &lr_points, options.nearest_segments);
            let lr_segment_indices = lr_segment_indices
                .into_iter()
                .map(|value| value as usize)
                .collect::<Vec<_>>();
            let rgb_linear = rgb
                .iter()
                .map(|&value| srgb_to_linear(f32::from(value) / 255.0))
                .collect::<Vec<_>>();
            let support = options
                .support_radius_normal
                .max(options.support_radius_tangent)
                .ceil() as isize;

            for &target in &active {
                let point = hr_points[target];
                let target_segment = hr_segment_indices[target];
                let center_y = point[0].floor() as isize;
                let center_x = point[1].floor() as isize;
                let mut samples = Vec::with_capacity(((support * 2 + 1).pow(2)) as usize);
                let mut signed_mass = 0.0_f32;
                let mut absolute_mass = 0.0_f32;
                let mut raw_alpha_mass = 0.0_f32;
                let mut numerator = [0.0_f32; 3];
                let target_contour = graph.contour_ids[target_segment];
                let target_arc = graph.arc_midpoints[target_segment];
                for offset_y in -support..=support {
                    for offset_x in -support..=support {
                        let sample_y = center_y + offset_y;
                        let sample_x = center_x + offset_x;
                        if sample_y < 0
                            || sample_y >= height as isize
                            || sample_x < 0
                            || sample_x >= width as isize
                        {
                            continue;
                        }
                        let sample_y = sample_y as usize;
                        let sample_x = sample_x as usize;
                        let sample_index = sample_y * width + sample_x;
                        let delta_y = sample_y as f32 - point[0];
                        let delta_x = sample_x as f32 - point[1];
                        let normal_coordinate = -delta_y;
                        let tangent_coordinate = delta_x;
                        let raw_weight = lanczos_weight(
                            options.lanczos_lobes as f32 * normal_coordinate
                                / options.support_radius_normal,
                            options.lanczos_lobes,
                        ) * lanczos_weight(
                            options.lanczos_lobes as f32 * tangent_coordinate
                                / options.support_radius_tangent,
                            options.lanczos_lobes,
                        );
                        let sample_alpha = alpha_lr[sample_index];
                        let visible = sample_alpha >= options.alpha_sample_floor;
                        let sample_segment = lr_segment_indices[sample_index];
                        let mut contour_weight = 1.0_f32;
                        if options.contour_gate_enabled {
                            let sample_contour = graph.contour_ids[sample_segment];
                            if sample_contour != target_contour {
                                contour_weight = 0.0;
                            } else {
                                let mut arc_delta =
                                    (graph.arc_midpoints[sample_segment] - target_arc).abs();
                                if graph.contour_closed[target_contour as usize] != 0 {
                                    let contour_length =
                                        graph.contour_lengths[target_contour as usize];
                                    arc_delta =
                                        arc_delta.min((contour_length - arc_delta).max(0.0));
                                }
                                let normalized = arc_delta / options.arc_radius;
                                contour_weight = (-0.5 * normalized * normalized).exp();
                            }
                        }
                        let gated_weight = if visible {
                            raw_weight * contour_weight
                        } else {
                            0.0
                        };
                        let alpha_weight = gated_weight * sample_alpha;
                        signed_mass += alpha_weight;
                        absolute_mass += alpha_weight.abs();
                        if visible {
                            raw_alpha_mass += raw_weight.abs() * sample_alpha;
                        }
                        for channel in 0..3 {
                            numerator[channel] +=
                                alpha_weight * rgb_linear[sample_index * 3 + channel];
                        }
                        samples.push(Sample {
                            index: sample_index,
                            raw_weight,
                            contour_weight,
                            visible,
                        });
                    }
                }

                let sample_condition = absolute_mass / signed_mass.abs().max(1e-8);
                condition[target] = sample_condition;
                let sample_contour_support =
                    (absolute_mass / raw_alpha_mass.max(1e-8)).clamp(0.0, 1.0);
                contour_support[target] = sample_contour_support;
                let mut stable = signed_mass.abs() >= options.alpha_mass_floor
                    && absolute_mass >= options.alpha_mass_floor
                    && sample_condition <= options.condition_limit
                    && sample_contour_support > 1e-4;

                let mut candidate = [0.0_f32; 3];
                if stable {
                    for channel in 0..3 {
                        candidate[channel] = numerator[channel] / signed_mass;
                    }
                }
                let mut local_minimum = [f32::INFINITY; 3];
                let mut local_maximum = [f32::NEG_INFINITY; 3];
                for sample in &samples {
                    if sample.visible
                        && sample.contour_weight > 1e-5
                        && sample.raw_weight.abs() > 1e-8
                    {
                        for channel in 0..3 {
                            let value = rgb_linear[sample.index * 3 + channel];
                            local_minimum[channel] = local_minimum[channel].min(value);
                            local_maximum[channel] = local_maximum[channel].max(value);
                        }
                    }
                }
                stable &= local_minimum.iter().all(|value| value.is_finite())
                    && local_maximum.iter().all(|value| value.is_finite());
                if stable {
                    for channel in 0..3 {
                        candidate[channel] = candidate[channel].clamp(
                            local_minimum[channel] - options.local_range_epsilon,
                            local_maximum[channel] + options.local_range_epsilon,
                        );
                    }
                }

                let mut residual = [0.0_f32; 3];
                for channel in 0..3 {
                    residual[channel] = (candidate[channel] - output_linear[target * 3 + channel])
                        .clamp(-options.max_residual, options.max_residual);
                }
                let condition_gate = 1.0
                    - smoothstep(
                        (sample_condition - options.condition_soft_start)
                            / (options.condition_limit - options.condition_soft_start),
                    );
                let sample_gate = if stable {
                    condition_gate
                        * sample_contour_support
                        * partial_alpha_gate[target]
                        * frame_reliability[target]
                        * options.strength
                } else {
                    0.0
                };
                gate[target] = sample_gate;
                for channel in 0..3 {
                    output_linear[target * 3 + channel] = (output_linear[target * 3 + channel]
                        + residual[channel] * sample_gate)
                        .clamp(0.0, 1.0);
                }
                residual_luma[target] = residual[0].abs() * 0.2126
                    + residual[1].abs() * 0.7152
                    + residual[2].abs() * 0.0722;
            }
        }
    }

    for (index, &alpha) in alpha_hr.iter().enumerate() {
        if alpha <= 1e-6 {
            output_linear[index * 3..index * 3 + 3].fill(0.0);
        }
    }
    let output_u8 = output_linear
        .iter()
        .map(|&value| quantize_u8(linear_to_srgb(value.clamp(0.0, 1.0)).clamp(0.0, 1.0)))
        .collect::<Vec<_>>();
    let active_mask = gate
        .iter()
        .map(|&value| u8::from(value > 0.0))
        .collect::<Vec<_>>();
    let diagnostics = ColorDiagnostics {
        implementation: crate::COLOR_CANONICAL_PIPELINE_ID,
        gate_hr: Plane::new(output_width, output_height, gate)?,
        condition_hr: Plane::new(output_width, output_height, condition)?,
        contour_support_hr: Plane::new(output_width, output_height, contour_support)?,
        residual_luma_hr: Plane::new(output_width, output_height, residual_luma)?,
        active_mask_hr: Plane::new(output_width, output_height, active_mask)?,
    };
    Ok((output_u8, diagnostics))
}

#[derive(Clone, Copy)]
struct Sample {
    index: usize,
    raw_weight: f32,
    contour_weight: f32,
    visible: bool,
}

fn resize_premultiplied_rgb_lanczos(
    rgb: &[u8],
    alpha: &[f32],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let output_width = width * SCALE;
    let output_height = height * SCALE;
    let output_alpha = resize_f32(
        alpha,
        width,
        height,
        output_width,
        output_height,
        Resample::Lanczos3,
    )
    .into_iter()
    .map(|value| value.clamp(0.0, 1.0))
    .collect::<Vec<_>>();
    let mut premultiplied = vec![0.0_f32; width * height * 3];
    for index in 0..alpha.len() {
        for channel in 0..3 {
            premultiplied[index * 3 + channel] =
                srgb_to_linear(f32::from(rgb[index * 3 + channel]) / 255.0) * alpha[index];
        }
    }
    let mut straight = vec![0.0_f32; output_width * output_height * 3];
    for channel in 0..3 {
        let plane = (0..alpha.len())
            .map(|index| premultiplied[index * 3 + channel])
            .collect::<Vec<_>>();
        let resized = resize_f32(
            &plane,
            width,
            height,
            output_width,
            output_height,
            Resample::Lanczos3,
        );
        for index in 0..output_alpha.len() {
            if output_alpha[index] > 1e-6 {
                straight[index * 3 + channel] = resized[index] / output_alpha[index];
            }
        }
    }
    straight
        .into_iter()
        .map(|value| {
            let srgb = linear_to_srgb(value.clamp(0.0, 1.0)).clamp(0.0, 1.0);
            quantize_u8(srgb)
        })
        .collect()
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.max(0.0).powf(1.0 / 2.4) - 0.055
    }
}

fn quantize_u8(value: f32) -> u8 {
    (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8
}

fn lanczos_weight(value: f32, lobes: usize) -> f32 {
    if value.abs() >= lobes as f32 {
        0.0
    } else {
        sinc(value) * sinc(value / lobes as f32)
    }
}

fn sinc(value: f32) -> f32 {
    if value == 0.0 {
        1.0
    } else {
        let argument = std::f32::consts::PI * value;
        argument.sin() / argument
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED_WIDTH: usize = 22;
    const FIXED_HEIGHT: usize = 18;

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn assert_f32_close(name: &str, actual: &[f32], expected: &[u8], tolerance: f32) {
        let expected = f32s(expected);
        let mut maximum = 0.0_f32;
        let mut first = None;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                maximum = maximum.max((actual - expected).abs());
                first.get_or_insert((index, actual, expected));
            }
        }
        assert!(
            maximum <= tolerance,
            "{name} max error {maximum:e} exceeds {tolerance:e}; first {first:?}"
        );
    }

    #[test]
    fn fixed_premultiplied_fallback_matches_python() {
        let rgba = include_bytes!("../../../../fixtures/parity/color/fixed/input_rgba_u8.raw");
        let mut rgb = Vec::with_capacity(FIXED_WIDTH * FIXED_HEIGHT * 3);
        let mut alpha = Vec::with_capacity(FIXED_WIDTH * FIXED_HEIGHT);
        for pixel in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&pixel[..3]);
            alpha.push(f32::from(pixel[3]) / 255.0);
        }
        assert_eq!(
            resize_premultiplied_rgb_lanczos(&rgb, &alpha, FIXED_WIDTH, FIXED_HEIGHT),
            include_bytes!(
                "../../../../fixtures/parity/color/fixed/premultiplied_lanczos_rgb_u8.raw"
            )
        );
    }

    #[test]
    fn fixed_color_fixture_matches_python_quantized_output() {
        let input = include_bytes!("../../../../fixtures/parity/color/fixed/input_rgba_u8.raw");
        let output = upscale_color_sdf_x4(
            Rgba8Ref::packed(FIXED_WIDTH, FIXED_HEIGHT, input).unwrap(),
            &ColorOptions::default(),
        )
        .unwrap();
        assert_f32_close(
            "gate",
            output.diagnostics.gate_hr.as_slice(),
            include_bytes!("../../../../fixtures/parity/color/fixed/gate_hr_f32.raw"),
            1e-6,
        );
        assert_f32_close(
            "condition",
            output.diagnostics.condition_hr.as_slice(),
            include_bytes!("../../../../fixtures/parity/color/fixed/condition_hr_f32.raw"),
            1e-5,
        );
        assert_f32_close(
            "contour_support",
            output.diagnostics.contour_support_hr.as_slice(),
            include_bytes!("../../../../fixtures/parity/color/fixed/contour_support_hr_f32.raw"),
            1e-6,
        );
        assert_f32_close(
            "residual_luma",
            output.diagnostics.residual_luma_hr.as_slice(),
            include_bytes!("../../../../fixtures/parity/color/fixed/residual_luma_hr_f32.raw"),
            1e-6,
        );
        let expected = include_bytes!("../../../../fixtures/parity/color/fixed/final_rgba_u8.raw");
        let differences = output
            .image
            .as_slice()
            .iter()
            .zip(expected)
            .enumerate()
            .filter(|(_, (actual, expected))| actual != expected)
            .map(|(index, (&actual, &expected))| (index, actual, expected))
            .collect::<Vec<_>>();
        let per_channel = (0..4)
            .map(|channel| {
                differences
                    .iter()
                    .filter(|(index, _, _)| index % 4 == channel)
                    .count()
            })
            .collect::<Vec<_>>();
        assert!(
            differences.is_empty(),
            "{} RGBA byte mismatches ({per_channel:?}); first {:?}",
            differences.len(),
            differences.first()
        );
        assert_eq!(
            output
                .diagnostics
                .active_mask_hr
                .as_slice()
                .iter()
                .map(|&value| usize::from(value))
                .sum::<usize>(),
            666
        );
    }

    #[test]
    fn compact_color_corpus_matches_python() {
        for (name, width, height) in [
            ("opaque_flat", 7, 5),
            ("hard_chroma_split", 7, 5),
            ("partial_alpha_colored_edge", 9, 7),
            ("hidden_rgb_a", 9, 7),
            ("hidden_rgb_b", 9, 7),
            ("transparent_asset", 7, 5),
            ("line_art_color", 11, 9),
            ("odd_dimensions", 7, 5),
        ] {
            let input = std::fs::read(format!(
                "{}/fixtures/parity/color/cases/{name}/input_rgba_u8.raw",
                env!("CARGO_MANIFEST_DIR")
                    .strip_suffix("/crates/sdf-upscaler")
                    .unwrap()
            ))
            .unwrap();
            let expected = std::fs::read(format!(
                "{}/fixtures/parity/color/cases/{name}/output_rgba_u8.raw",
                env!("CARGO_MANIFEST_DIR")
                    .strip_suffix("/crates/sdf-upscaler")
                    .unwrap()
            ))
            .unwrap();
            let output = upscale_color_sdf_x4(
                Rgba8Ref::packed(width, height, &input).unwrap(),
                &ColorOptions::default(),
            )
            .unwrap();
            assert_eq!(output.image.as_slice(), expected, "case {name}");
        }
    }
}
