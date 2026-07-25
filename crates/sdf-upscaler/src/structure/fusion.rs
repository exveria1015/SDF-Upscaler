use crate::{
    Plane, StructureFusionDiagnostics, UpscaleError,
    color::{linear_to_srgb, quantize_u8, resize_premultiplied_rgb_lanczos, srgb_to_linear},
    compat::{Resample, distance_transform_edt_with_indices, resize_f32},
    structure::field::{EDGE_CONFIDENCE, NORMAL_X, NORMAL_Y, SIGNED_DISTANCE, StructureField},
};

const SCALE: usize = 4;
const PROPAGATION_RADIUS: f32 = 6.0;
const SIDE_OFFSET_LR: f32 = 1.25;
const EDGE_SOFTNESS_LR: f32 = 1.0;
const EDGE_BAND_START_LR: f32 = 0.25;
const EDGE_BAND_STOP_LR: f32 = 1.75;
const CONTRAST_START: f32 = 0.02;
const CONTRAST_STOP: f32 = 0.12;
const LOCAL_RANGE_EPSILON: f32 = 0.01;

pub(crate) fn edge_local_structure_fusion_x4(
    rgb: &[u8],
    hat: &[u8],
    width: usize,
    height: usize,
    field: &StructureField,
    max_residual: f32,
) -> Result<(Vec<u8>, StructureFusionDiagnostics), UpscaleError> {
    let output_width = width * SCALE;
    let output_height = height * SCALE;
    let pixels = width * height;
    let output_pixels = output_width * output_height;
    debug_assert_eq!(rgb.len(), pixels * 3);
    debug_assert_eq!(hat.len(), output_pixels * 3);
    debug_assert_eq!((field.width, field.height), (width, height));

    let empty_hr = vec![0.0_f32; output_pixels];
    let empty_lr = vec![0.0_f32; pixels];
    if !field.edge_mask.iter().any(|&value| value) || max_residual == 0.0 {
        let candidate = hat
            .chunks_exact(3)
            .map(|pixel| {
                [
                    f32::from(pixel[0]) / 255.0,
                    f32::from(pixel[1]) / 255.0,
                    f32::from(pixel[2]) / 255.0,
                ]
            })
            .collect();
        return Ok((
            hat.to_vec(),
            StructureFusionDiagnostics {
                structure_candidate_hr: Plane::new(output_width, output_height, candidate)?,
                structure_residual_hr: Plane::new(
                    output_width,
                    output_height,
                    vec![[0.0; 3]; output_pixels],
                )?,
                structure_gate_hr: Plane::new(output_width, output_height, empty_hr.clone())?,
                reprojection_gain_lr: Plane::new(width, height, empty_lr)?,
                reprojection_gate_hr: Plane::new(output_width, output_height, empty_hr.clone())?,
                applied_gate_hr: Plane::new(output_width, output_height, empty_hr)?,
                unconstrained_rgb_hr: Plane::new(
                    output_width,
                    output_height,
                    hat.chunks_exact(3)
                        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                        .collect(),
                )?,
            },
        ));
    }

    let linear_lr = rgb
        .iter()
        .map(|&value| srgb_to_linear(f32::from(value) / 255.0))
        .collect::<Vec<_>>();
    let non_edge = field
        .edge_mask
        .iter()
        .map(|&value| !value)
        .collect::<Vec<_>>();
    let (_, nearest_y, nearest_x) = distance_transform_edt_with_indices(&non_edge, width, height);
    let normal_x = field.channel(NORMAL_X);
    let normal_y = field.channel(NORMAL_Y);
    let mut plus_lr = vec![[0.0_f32; 3]; pixels];
    let mut minus_lr = vec![[0.0_f32; 3]; pixels];
    for index in 0..pixels {
        let nearest = nearest_y[index] * width + nearest_x[index];
        let edge_y = nearest_y[index] as f32;
        let edge_x = nearest_x[index] as f32;
        for channel in 0..3 {
            plus_lr[index][channel] = bilinear_sample(
                &linear_lr,
                width,
                height,
                channel,
                edge_y + normal_y[nearest] * SIDE_OFFSET_LR,
                edge_x + normal_x[nearest] * SIDE_OFFSET_LR,
            );
            minus_lr[index][channel] = bilinear_sample(
                &linear_lr,
                width,
                height,
                channel,
                edge_y - normal_y[nearest] * SIDE_OFFSET_LR,
                edge_x - normal_x[nearest] * SIDE_OFFSET_LR,
            );
        }
    }
    let plus_hr = resize_rgb_f32(
        &plus_lr,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let minus_hr = resize_rgb_f32(
        &minus_lr,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let signed_distance_lr = field
        .channel(SIGNED_DISTANCE)
        .into_iter()
        .map(|value| value * PROPAGATION_RADIUS)
        .collect::<Vec<_>>();
    let signed_distance_hr = resize_f32(
        &signed_distance_lr,
        width,
        height,
        output_width,
        output_height,
        Resample::Bicubic,
    );

    let mut candidate_linear = vec![[0.0_f32; 3]; output_pixels];
    for index in 0..output_pixels {
        let mix = smoothstep(0.5 + signed_distance_hr[index] / EDGE_SOFTNESS_LR);
        for channel in 0..3 {
            candidate_linear[index][channel] = minus_hr[index][channel]
                + (plus_hr[index][channel] - minus_hr[index][channel]) * mix;
        }
    }
    let (local_min_lr, local_max_lr) = local_rgb_range(&linear_lr, width, height);
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
    for index in 0..output_pixels {
        for channel in 0..3 {
            candidate_linear[index][channel] = candidate_linear[index][channel].clamp(
                local_min_hr[index][channel] - LOCAL_RANGE_EPSILON,
                local_max_hr[index][channel] + LOCAL_RANGE_EPSILON,
            );
        }
    }

    let alpha_opaque = vec![1.0_f32; pixels];
    let base_u8 = resize_premultiplied_rgb_lanczos(rgb, &alpha_opaque, width, height);
    let base_linear = base_u8
        .chunks_exact(3)
        .map(|pixel| {
            [
                srgb_to_linear(f32::from(pixel[0]) / 255.0),
                srgb_to_linear(f32::from(pixel[1]) / 255.0),
                srgb_to_linear(f32::from(pixel[2]) / 255.0),
            ]
        })
        .collect::<Vec<_>>();
    let hat_linear = hat
        .chunks_exact(3)
        .map(|pixel| {
            [
                srgb_to_linear(f32::from(pixel[0]) / 255.0),
                srgb_to_linear(f32::from(pixel[1]) / 255.0),
                srgb_to_linear(f32::from(pixel[2]) / 255.0),
            ]
        })
        .collect::<Vec<_>>();
    let residual = (0..output_pixels)
        .map(|index| {
            [
                (candidate_linear[index][0] - base_linear[index][0])
                    .clamp(-max_residual, max_residual),
                (candidate_linear[index][1] - base_linear[index][1])
                    .clamp(-max_residual, max_residual),
                (candidate_linear[index][2] - base_linear[index][2])
                    .clamp(-max_residual, max_residual),
            ]
        })
        .collect::<Vec<_>>();

    let edge_confidence_hr = resize_f32(
        &field.channel(EDGE_CONFIDENCE),
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let contrast_lr = plus_lr
        .iter()
        .zip(&minus_lr)
        .map(|(plus, minus)| {
            (0..3)
                .map(|channel| (plus[channel] - minus[channel]).abs())
                .fold(0.0_f32, f32::max)
        })
        .collect::<Vec<_>>();
    let contrast_gate_lr = contrast_lr
        .into_iter()
        .map(|value| smoothstep((value - CONTRAST_START) / (CONTRAST_STOP - CONTRAST_START)))
        .collect::<Vec<_>>();
    let contrast_gate_hr = resize_f32(
        &contrast_gate_lr,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let structure_gate = (0..output_pixels)
        .map(|index| {
            let band = 1.0
                - smoothstep(
                    (signed_distance_hr[index].abs() - EDGE_BAND_START_LR)
                        / (EDGE_BAND_STOP_LR - EDGE_BAND_START_LR),
                );
            (edge_confidence_hr[index] * band * contrast_gate_hr[index]).clamp(0.0, 1.0)
        })
        .collect::<Vec<_>>();
    let unconstrained_linear = (0..output_pixels)
        .map(|index| {
            let mut pixel = [0.0_f32; 3];
            for channel in 0..3 {
                pixel[channel] = (hat_linear[index][channel]
                    + structure_gate[index] * residual[index][channel])
                    .clamp(0.0, 1.0);
            }
            pixel
        })
        .collect::<Vec<_>>();
    let unconstrained_u8 = unconstrained_linear
        .iter()
        .map(|pixel| {
            [
                quantize_u8(linear_to_srgb(pixel[0]).clamp(0.0, 1.0)),
                quantize_u8(linear_to_srgb(pixel[1]).clamp(0.0, 1.0)),
                quantize_u8(linear_to_srgb(pixel[2]).clamp(0.0, 1.0)),
            ]
        })
        .collect::<Vec<_>>();

    // The canonical Adaptive-SDF route deliberately disables structure-local
    // reprojection. Its later premultiplied guard owns that decision.
    let reprojection_gain = vec![0.0_f32; pixels];
    let reprojection_gate = vec![1.0_f32; output_pixels];
    let applied_gate = structure_gate.clone();
    let output_u8 = unconstrained_u8
        .iter()
        .flat_map(|pixel| pixel.iter().copied())
        .collect::<Vec<_>>();
    let candidate_srgb = candidate_linear
        .iter()
        .map(|pixel| {
            [
                linear_to_srgb(pixel[0]).clamp(0.0, 1.0),
                linear_to_srgb(pixel[1]).clamp(0.0, 1.0),
                linear_to_srgb(pixel[2]).clamp(0.0, 1.0),
            ]
        })
        .collect::<Vec<_>>();
    Ok((
        output_u8,
        StructureFusionDiagnostics {
            structure_candidate_hr: Plane::new(output_width, output_height, candidate_srgb)?,
            structure_residual_hr: Plane::new(output_width, output_height, residual)?,
            structure_gate_hr: Plane::new(output_width, output_height, structure_gate)?,
            reprojection_gain_lr: Plane::new(width, height, reprojection_gain)?,
            reprojection_gate_hr: Plane::new(output_width, output_height, reprojection_gate)?,
            applied_gate_hr: Plane::new(output_width, output_height, applied_gate)?,
            unconstrained_rgb_hr: Plane::new(output_width, output_height, unconstrained_u8)?,
        },
    ))
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

fn local_rgb_range(input: &[f32], width: usize, height: usize) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut minimum = vec![[f32::INFINITY; 3]; width * height];
    let mut maximum = vec![[f32::NEG_INFINITY; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    for channel in 0..3 {
                        let value = input[(yy * width + xx) * 3 + channel];
                        minimum[index][channel] = minimum[index][channel].min(value);
                        maximum[index][channel] = maximum[index][channel].max(value);
                    }
                }
            }
        }
    }
    (minimum, maximum)
}

fn bilinear_sample(
    input: &[f32],
    width: usize,
    height: usize,
    channel: usize,
    y: f32,
    x: f32,
) -> f32 {
    let y = y.clamp(0.0, (height - 1) as f32);
    let x = x.clamp(0.0, (width - 1) as f32);
    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let y1 = (y0 + 1).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let wy = y - y0 as f32;
    let wx = x - x0 as f32;
    let top = input[(y0 * width + x0) * 3 + channel] * (1.0 - wx)
        + input[(y0 * width + x1) * 3 + channel] * wx;
    let bottom = input[(y1 * width + x0) * 3 + channel] * (1.0 - wx)
        + input[(y1 * width + x1) * 3 + channel] * wx;
    top * (1.0 - wy) + bottom * wy
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::extract_structure_field;

    fn expected_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect()
    }

    fn maximum_error(actual: &[f32], expected: &[u8]) -> f32 {
        actual
            .iter()
            .zip(expected_f32(expected))
            .map(|(&actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn fixed_structure_fusion_matches_python() {
        let bounded =
            include_bytes!("../../../../fixtures/parity/adaptive/fixed/bounded_visible_rgb_u8.raw");
        let hat =
            include_bytes!("../../../../fixtures/parity/adaptive/fixed/bounded_lanczos_rgb_u8.raw");
        let field = extract_structure_field(bounded, 22, 18).unwrap();
        let (output, diagnostics) =
            edge_local_structure_fusion_x4(bounded, hat, 22, 18, &field, 0.02).unwrap();
        assert_eq!(
            output,
            include_bytes!(
                "../../../../fixtures/parity/adaptive/fixed/structure_output_rgb_u8.raw"
            )
        );
        assert!(
            maximum_error(
                diagnostics.structure_gate_hr.as_slice(),
                include_bytes!(
                    "../../../../fixtures/parity/adaptive/fixed/structure_gate_hr_f32.raw"
                ),
            ) <= 2e-6
        );
        let residual = diagnostics
            .structure_residual_hr
            .as_slice()
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect::<Vec<_>>();
        let residual_error = maximum_error(
            &residual,
            include_bytes!(
                "../../../../fixtures/parity/adaptive/fixed/structure_residual_hr_f32.raw"
            ),
        );
        assert!(residual_error <= 2e-6, "residual error {residual_error}");
    }
}
