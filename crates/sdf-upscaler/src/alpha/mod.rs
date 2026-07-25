use crate::{
    AlphaDiagnostics, AlphaInput, AlphaOptions, AlphaOutput, Plane, QualityBoundary, UpscaleError,
};

mod canonical;

const SCALE: usize = 4;
const LEVEL: f32 = 0.5;
const GRID_CELL_SIZE: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
struct Point {
    y: f32,
    x: f32,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    start: Point,
    end: Point,
}

#[derive(Clone, Debug)]
struct AxisWeights {
    indices: [usize; 6],
    weights: [f32; 6],
}

#[derive(Debug)]
struct SegmentGrid {
    columns: usize,
    rows: usize,
    buckets: Vec<Vec<u32>>,
}

impl SegmentGrid {
    fn build(segments: &[Segment], width: usize, height: usize) -> Self {
        let columns = ((width as f32 / GRID_CELL_SIZE).ceil() as usize).max(1) + 1;
        let rows = ((height as f32 / GRID_CELL_SIZE).ceil() as usize).max(1) + 1;
        let mut buckets = vec![Vec::new(); columns * rows];
        for (index, segment) in segments.iter().enumerate() {
            let midpoint = Point {
                y: 0.5 * (segment.start.y + segment.end.y),
                x: 0.5 * (segment.start.x + segment.end.x),
            };
            let column = ((midpoint.x / GRID_CELL_SIZE).floor() as isize)
                .clamp(0, columns as isize - 1) as usize;
            let row = ((midpoint.y / GRID_CELL_SIZE).floor() as isize).clamp(0, rows as isize - 1)
                as usize;
            buckets[row * columns + column].push(index as u32);
        }
        Self {
            columns,
            rows,
            buckets,
        }
    }

    fn nearest(
        &self,
        segments: &[Segment],
        point: Point,
        maximum_distance: f32,
    ) -> Option<(Segment, f32)> {
        let center_column = (point.x / GRID_CELL_SIZE).floor() as isize;
        let center_row = (point.y / GRID_CELL_SIZE).floor() as isize;
        let cells = ((maximum_distance + 1.0) / GRID_CELL_SIZE).ceil() as isize;
        let mut best = None;
        let mut best_squared = maximum_distance * maximum_distance;
        for row in (center_row - cells)..=(center_row + cells) {
            if !(0..self.rows as isize).contains(&row) {
                continue;
            }
            for column in (center_column - cells)..=(center_column + cells) {
                if !(0..self.columns as isize).contains(&column) {
                    continue;
                }
                for &index in &self.buckets[row as usize * self.columns + column as usize] {
                    let segment = segments[index as usize];
                    let squared = point_segment_distance_squared(point, segment);
                    if squared < best_squared {
                        best_squared = squared;
                        best = Some(segment);
                    }
                }
            }
        }
        best.map(|segment| (segment, best_squared.sqrt()))
    }
}

/// Upscale alpha with the canonical x4 Alpha-SDF pipeline.
///
/// The quantized output is byte-exact against the pinned Python parity
/// corpus, including hard shapes, partial-alpha edges, line-art bypasses, and
/// an active SCS medial-lift case.
pub fn upscale_alpha_sdf_x4(input: AlphaInput<'_>) -> Result<AlphaOutput, UpscaleError> {
    let (alpha, width, height) = normalize_input(input)?;
    let output_width = width
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_height = height
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    output_width
        .checked_mul(output_height)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let stages = canonical::canonical_stages(&alpha, width, height)?;
    let endpoint_count = alpha
        .iter()
        .filter(|&&value| value <= 0.05 || value >= 0.95)
        .count();
    let endpoint_fraction = endpoint_count as f64 / alpha.len() as f64;
    let constant_value = alpha[0];
    let constant_alpha = (constant_value == 0.0 || constant_value == 1.0)
        && alpha.iter().all(|&value| value == constant_value);
    let changed = quantize_alpha_u8(&stages.final_alpha)
        .into_iter()
        .zip(quantize_alpha_u8(&stages.base))
        .filter(|(left, right)| left != right)
        .count();
    let reliable = stages
        .reliability
        .iter()
        .filter(|&&value| value > 0.0)
        .count();
    let diagnostics = AlphaDiagnostics {
        implementation: crate::ALPHA_CANONICAL_PIPELINE_ID,
        quality_boundary: QualityBoundary::PythonReferenceQuantizedParity,
        input_size: [width, height],
        output_size: [output_width, output_height],
        constant_alpha_fast_path: constant_alpha,
        broad_soft_lanczos_fallback: false,
        endpoint_fraction,
        segment_count: stages.graph.segments.len(),
        roi_fraction: reliable as f64 / stages.reliability.len() as f64,
        guard_accept_fraction: 1.0,
        changed_from_lanczos_fraction: changed as f64 / stages.final_alpha.len() as f64,
    };
    Ok(AlphaOutput {
        alpha: Plane::new(output_width, output_height, stages.final_alpha)?,
        diagnostics,
    })
}

/// Run the explicitly experimental pure-Rust Alpha-SDF ROI route.
///
/// This is not yet the canonical Python `resize_alpha_main_x4` algorithm. Its
/// quality status is returned in diagnostics so callers cannot silently treat
/// the milestone as parity-complete.
pub fn upscale_alpha_sdf_x4_experimental(
    input: AlphaInput<'_>,
    options: AlphaOptions,
) -> Result<AlphaOutput, UpscaleError> {
    options.validate()?;
    let (alpha, width, height) = normalize_input(input)?;
    let output_width = width
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_height = height
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    output_width
        .checked_mul(output_height)
        .ok_or(UpscaleError::DimensionOverflow)?;

    let endpoint_count = alpha
        .iter()
        .filter(|&&value| value <= 0.05 || value >= 0.95)
        .count();
    let endpoint_fraction = endpoint_count as f64 / alpha.len() as f64;
    let constant_value = alpha[0];
    let constant_alpha = (constant_value == 0.0 || constant_value == 1.0)
        && alpha.iter().all(|&value| value == constant_value);
    let broad_soft_fallback = !constant_alpha && endpoint_fraction < 0.30;
    let lanczos = if constant_alpha {
        vec![constant_value; output_width * output_height]
    } else {
        resize_lanczos_x4(&alpha, width, height)?
    };

    let segments = if constant_alpha || broad_soft_fallback || width < 2 || height < 2 {
        Vec::new()
    } else {
        marching_squares_segments(&alpha, width, height)
    };
    let edge_strength = local_alpha_range(&alpha, width, height);
    let roi = if segments.is_empty() {
        vec![false; width * height]
    } else {
        dilated_edge_roi(&edge_strength, width, height, options.roi_radius_lr)
    };
    let roi_count = roi.iter().filter(|&&value| value).count();
    let roi_fraction = roi_count as f64 / roi.len() as f64;
    let grid = SegmentGrid::build(&segments, width, height);
    let maximum_query_distance = options.roi_radius_lr as f32 + 2.0;
    let mut corrected = lanczos.clone();

    if !segments.is_empty() {
        for output_y in 0..output_height {
            let query_y = (output_y as f32 + 0.5) / SCALE as f32 - 0.5;
            let source_y = ((output_y as f32 + 0.5) / SCALE as f32)
                .floor()
                .clamp(0.0, height as f32 - 1.0) as usize;
            for output_x in 0..output_width {
                let source_x = ((output_x as f32 + 0.5) / SCALE as f32)
                    .floor()
                    .clamp(0.0, width as f32 - 1.0) as usize;
                let source_index = source_y * width + source_x;
                if !roi[source_index] {
                    continue;
                }
                let point = Point {
                    y: query_y,
                    x: (output_x as f32 + 0.5) / SCALE as f32 - 0.5,
                };
                let Some((segment, distance_lr)) =
                    grid.nearest(&segments, point, maximum_query_distance)
                else {
                    continue;
                };
                if distance_lr * SCALE as f32 > options.effect_band_hr {
                    continue;
                }
                let hard_edge_gate = smoothstep((edge_strength[source_index] - 0.05) / 0.45);
                if hard_edge_gate <= 0.0 {
                    continue;
                }
                let index = output_y * output_width + output_x;
                let base = corrected[index];
                let coverage = segment_coverage(segment, distance_lr, base >= LEVEL);
                let residual = (coverage - base).clamp(-options.residual_cap, options.residual_cap);
                let proposal = base + hard_edge_gate * residual;
                corrected[index] = topology_lock(base, proposal).clamp(0.0, 1.0);
            }
        }
    }

    let (accepted, guarded) = if options.reprojection_guard && !segments.is_empty() {
        apply_reprojection_guard(
            &mut corrected,
            &lanczos,
            &alpha,
            &roi,
            width,
            height,
            output_width,
        )
    } else {
        (roi_count as u64, roi_count as u64)
    };
    let guard_accept_fraction = if guarded == 0 {
        1.0
    } else {
        accepted as f64 / guarded as f64
    };
    let changed = quantize_alpha_u8(&corrected)
        .iter()
        .zip(quantize_alpha_u8(&lanczos))
        .filter(|(left, right)| left != &right)
        .count();
    let diagnostics = AlphaDiagnostics {
        implementation: crate::ALPHA_EXPERIMENTAL_PIPELINE_ID,
        quality_boundary: QualityBoundary::ExperimentalNotPythonParity,
        input_size: [width, height],
        output_size: [output_width, output_height],
        constant_alpha_fast_path: constant_alpha,
        broad_soft_lanczos_fallback: broad_soft_fallback,
        endpoint_fraction,
        segment_count: segments.len(),
        roi_fraction,
        guard_accept_fraction,
        changed_from_lanczos_fraction: changed as f64 / corrected.len() as f64,
    };
    Ok(AlphaOutput {
        alpha: Plane::new(output_width, output_height, corrected)?,
        diagnostics,
    })
}

/// Quantize normalized alpha with NumPy-compatible ties-to-even rounding.
pub fn quantize_alpha_u8(alpha: &[f32]) -> Vec<u8> {
    alpha
        .iter()
        .map(|&value| {
            (value.clamp(0.0, 1.0) * 255.0)
                .round_ties_even()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn normalize_input(input: AlphaInput<'_>) -> Result<(Vec<f32>, usize, usize), UpscaleError> {
    match input {
        AlphaInput::U8(plane) => {
            let mut output = Vec::with_capacity(
                plane
                    .width()
                    .checked_mul(plane.height())
                    .ok_or(UpscaleError::DimensionOverflow)?,
            );
            for y in 0..plane.height() {
                output.extend(plane.row(y).iter().map(|&value| f32::from(value) / 255.0));
            }
            Ok((output, plane.width(), plane.height()))
        }
        AlphaInput::F32(plane) => {
            let mut output = Vec::with_capacity(
                plane
                    .width()
                    .checked_mul(plane.height())
                    .ok_or(UpscaleError::DimensionOverflow)?,
            );
            for y in 0..plane.height() {
                for &value in plane.row(y) {
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        return Err(UpscaleError::InvalidAlpha {
                            index: output.len(),
                        });
                    }
                    output.push(value);
                }
            }
            Ok((output, plane.width(), plane.height()))
        }
    }
}

fn resize_lanczos_x4(alpha: &[f32], width: usize, height: usize) -> Result<Vec<f32>, UpscaleError> {
    let output_width = width
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_height = height
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let horizontal_weights = axis_weights(width, output_width);
    let vertical_weights = axis_weights(height, output_height);
    let mut horizontal = vec![
        0.0_f32;
        output_width
            .checked_mul(height)
            .ok_or(UpscaleError::DimensionOverflow)?
    ];
    for y in 0..height {
        let input_row = &alpha[y * width..(y + 1) * width];
        let row = &mut horizontal[y * output_width..(y + 1) * output_width];
        for (x, value) in row.iter_mut().enumerate() {
            let weights = &horizontal_weights[x];
            *value = weights
                .indices
                .iter()
                .zip(weights.weights.iter())
                .map(|(&index, &weight)| input_row[index] * weight)
                .sum::<f32>();
        }
    }
    let mut output = vec![
        0.0_f32;
        output_width
            .checked_mul(output_height)
            .ok_or(UpscaleError::DimensionOverflow)?
    ];
    for y in 0..output_height {
        let weights = &vertical_weights[y];
        let row = &mut output[y * output_width..(y + 1) * output_width];
        for (x, value) in row.iter_mut().enumerate() {
            *value = weights
                .indices
                .iter()
                .zip(weights.weights.iter())
                .map(|(&index, &weight)| horizontal[index * output_width + x] * weight)
                .sum::<f32>()
                .clamp(0.0, 1.0);
        }
    }
    Ok(output)
}

fn axis_weights(input_size: usize, output_size: usize) -> Vec<AxisWeights> {
    (0..output_size)
        .map(|output_index| {
            let coordinate = (output_index as f64 + 0.5) / SCALE as f64 - 0.5;
            let start = coordinate.floor() as isize - 2;
            let mut indices = [0_usize; 6];
            let mut weights = [0.0_f32; 6];
            let mut sum = 0.0_f64;
            for tap in 0..6 {
                let source = start + tap as isize;
                if !(0..input_size as isize).contains(&source) {
                    continue;
                }
                indices[tap] = source as usize;
                let weight = lanczos_weight(coordinate - source as f64);
                weights[tap] = weight as f32;
                sum += weight;
            }
            if sum != 0.0 {
                for weight in &mut weights {
                    *weight /= sum as f32;
                }
            }
            AxisWeights { indices, weights }
        })
        .collect()
}

fn lanczos_weight(value: f64) -> f64 {
    if value.abs() >= 3.0 {
        0.0
    } else {
        sinc(value) * sinc(value / 3.0)
    }
}

fn sinc(value: f64) -> f64 {
    if value.abs() < 1e-12 {
        1.0
    } else {
        let argument = std::f64::consts::PI * value;
        argument.sin() / argument
    }
}

fn marching_squares_segments(alpha: &[f32], width: usize, height: usize) -> Vec<Segment> {
    let mut segments = Vec::new();
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let top_left = alpha[y * width + x];
            let top_right = alpha[y * width + x + 1];
            let bottom_right = alpha[(y + 1) * width + x + 1];
            let bottom_left = alpha[(y + 1) * width + x];
            let case = u8::from(top_left >= LEVEL)
                | (u8::from(top_right >= LEVEL) << 1)
                | (u8::from(bottom_right >= LEVEL) << 2)
                | (u8::from(bottom_left >= LEVEL) << 3);
            if case == 0 || case == 15 {
                continue;
            }
            let corners = [top_left, top_right, bottom_right, bottom_left];
            let positions = [
                Point {
                    y: y as f32,
                    x: x as f32,
                },
                Point {
                    y: y as f32,
                    x: (x + 1) as f32,
                },
                Point {
                    y: (y + 1) as f32,
                    x: (x + 1) as f32,
                },
                Point {
                    y: (y + 1) as f32,
                    x: x as f32,
                },
            ];
            let edge_corners = [(0, 1), (1, 2), (2, 3), (3, 0)];
            let mut intersections: [Option<Point>; 4] = [None; 4];
            for (edge, &(first, second)) in edge_corners.iter().enumerate() {
                if (corners[first] >= LEVEL) != (corners[second] >= LEVEL) {
                    intersections[edge] = Some(interpolate_level(
                        positions[first],
                        positions[second],
                        corners[first],
                        corners[second],
                    ));
                }
            }
            let present = intersections
                .iter()
                .enumerate()
                .filter_map(|(edge, point)| point.map(|value| (edge, value)))
                .collect::<Vec<_>>();
            match present.len() {
                2 => segments.push(Segment {
                    start: present[0].1,
                    end: present[1].1,
                }),
                4 => {
                    let center_high =
                        0.25 * (top_left + top_right + bottom_right + bottom_left) >= LEVEL;
                    let pair_adjacent = match case {
                        5 => center_high,
                        10 => !center_high,
                        _ => true,
                    };
                    let pairs = if pair_adjacent {
                        [(0, 1), (2, 3)]
                    } else {
                        [(0, 3), (1, 2)]
                    };
                    for (first, second) in pairs {
                        segments.push(Segment {
                            start: intersections[first].expect("four-edge intersection"),
                            end: intersections[second].expect("four-edge intersection"),
                        });
                    }
                }
                _ => {}
            }
        }
    }
    segments
}

fn interpolate_level(start: Point, end: Point, start_value: f32, end_value: f32) -> Point {
    let denominator = end_value - start_value;
    let amount = if denominator.abs() < 1e-8 {
        0.5
    } else {
        ((LEVEL - start_value) / denominator).clamp(0.0, 1.0)
    };
    Point {
        y: start.y + amount * (end.y - start.y),
        x: start.x + amount * (end.x - start.x),
    }
}

fn local_alpha_range(alpha: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut minimum = 1.0_f32;
            let mut maximum = 0.0_f32;
            for sample_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for sample_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let sample = alpha[sample_y * width + sample_x];
                    minimum = minimum.min(sample);
                    maximum = maximum.max(sample);
                }
            }
            output[y * width + x] = maximum - minimum;
        }
    }
    output
}

fn dilated_edge_roi(
    edge_strength: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<bool> {
    let mut roi = vec![false; width * height];
    for y in 0..height {
        for x in 0..width {
            if edge_strength[y * width + x] <= 1.0 / 255.0 {
                continue;
            }
            for output_y in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                let start = output_y * width + x.saturating_sub(radius);
                let end = output_y * width + (x + radius).min(width - 1);
                roi[start..=end].fill(true);
            }
        }
    }
    roi
}

fn point_segment_distance_squared(point: Point, segment: Segment) -> f32 {
    let vector_y = segment.end.y - segment.start.y;
    let vector_x = segment.end.x - segment.start.x;
    let denominator = vector_y * vector_y + vector_x * vector_x;
    let projection = if denominator <= 1e-12 {
        0.0
    } else {
        ((point.y - segment.start.y) * vector_y + (point.x - segment.start.x) * vector_x)
            / denominator
    }
    .clamp(0.0, 1.0);
    let closest_y = segment.start.y + projection * vector_y;
    let closest_x = segment.start.x + projection * vector_x;
    let distance_y = point.y - closest_y;
    let distance_x = point.x - closest_x;
    distance_y * distance_y + distance_x * distance_x
}

fn segment_coverage(segment: Segment, distance_lr: f32, inside: bool) -> f32 {
    let tangent_y = segment.end.y - segment.start.y;
    let tangent_x = segment.end.x - segment.start.x;
    let length = (tangent_y * tangent_y + tangent_x * tangent_x)
        .sqrt()
        .max(1e-8);
    let normal_y = -tangent_x / length;
    let normal_x = tangent_y / length;
    let projected_width = (normal_y.abs() + normal_x.abs()) / SCALE as f32;
    let signed_distance = if inside { distance_lr } else { -distance_lr };
    (0.5 + signed_distance / projected_width.max(1e-6)).clamp(0.0, 1.0)
}

fn topology_lock(base: f32, candidate: f32) -> f32 {
    if base >= LEVEL {
        candidate.max(LEVEL)
    } else {
        candidate.min(f32::from_bits(LEVEL.to_bits() - 1))
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_reprojection_guard(
    corrected: &mut [f32],
    base: &[f32],
    alpha_lr: &[f32],
    roi: &[bool],
    width: usize,
    height: usize,
    output_width: usize,
) -> (u64, u64) {
    let mut accepted = 0_u64;
    let mut guarded = 0_u64;
    for source_y in 0..height {
        for source_x in 0..width {
            if !roi[source_y * width + source_x] {
                continue;
            }
            guarded += 1;
            let mut base_mean = 0.0_f32;
            let mut candidate_mean = 0.0_f32;
            for offset_y in 0..SCALE {
                for offset_x in 0..SCALE {
                    let index =
                        (source_y * SCALE + offset_y) * output_width + source_x * SCALE + offset_x;
                    base_mean += base[index];
                    candidate_mean += corrected[index];
                }
            }
            base_mean /= (SCALE * SCALE) as f32;
            candidate_mean /= (SCALE * SCALE) as f32;
            let target = alpha_lr[source_y * width + source_x];
            if (candidate_mean - target).abs() <= (base_mean - target).abs() + 1e-7 {
                accepted += 1;
            } else {
                for offset_y in 0..SCALE {
                    for offset_x in 0..SCALE {
                        let index = (source_y * SCALE + offset_y) * output_width
                            + source_x * SCALE
                            + offset_x;
                        corrected[index] = base[index];
                    }
                }
            }
        }
    }
    (accepted, guarded)
}

fn smoothstep(value: f32) -> f32 {
    let clipped = value.clamp(0.0, 1.0);
    clipped * clipped * (3.0 - 2.0 * clipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlaneRef;

    #[test]
    fn constants_are_exact() {
        for value in [0_u8, 255_u8] {
            let input = vec![value; 7 * 9];
            let output = upscale_alpha_sdf_x4_experimental(
                AlphaInput::U8(PlaneRef::packed(9, 7, &input).unwrap()),
                AlphaOptions::default(),
            )
            .unwrap();
            assert_eq!(output.alpha.width(), 36);
            assert_eq!(output.alpha.height(), 28);
            let expected = f32::from(value) / 255.0;
            assert!(
                output
                    .alpha
                    .as_slice()
                    .iter()
                    .all(|&sample| sample == expected)
            );
            assert!(output.diagnostics.constant_alpha_fast_path);
        }
    }

    #[test]
    fn u8_and_normalized_f32_inputs_match() {
        let input = (0_u8..63).collect::<Vec<_>>();
        let float = input
            .iter()
            .map(|&value| f32::from(value) / 255.0)
            .collect::<Vec<_>>();
        let from_u8 = upscale_alpha_sdf_x4_experimental(
            AlphaInput::U8(PlaneRef::packed(9, 7, &input).unwrap()),
            AlphaOptions::default(),
        )
        .unwrap();
        let from_f32 = upscale_alpha_sdf_x4_experimental(
            AlphaInput::F32(PlaneRef::packed(9, 7, &float).unwrap()),
            AlphaOptions::default(),
        )
        .unwrap();
        assert_eq!(from_u8.alpha, from_f32.alpha);
    }

    #[test]
    fn quantization_uses_ties_to_even() {
        let values = [0.5 / 255.0, 1.5 / 255.0, 2.5 / 255.0, 3.5 / 255.0];
        assert_eq!(quantize_alpha_u8(&values), vec![0, 2, 2, 4]);
    }

    #[test]
    fn topology_lock_does_not_cross_half() {
        assert!(topology_lock(0.49, 0.9) < 0.5);
        assert!(topology_lock(0.51, 0.1) >= 0.5);
    }

    #[test]
    fn invalid_float_alpha_is_rejected() {
        let input = [0.0_f32, f32::NAN];
        let error = upscale_alpha_sdf_x4_experimental(
            AlphaInput::F32(PlaneRef::packed(2, 1, &input).unwrap()),
            AlphaOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error, UpscaleError::InvalidAlpha { index: 1 });
    }
}
