use crate::{
    UpscaleError,
    compat::{
        Resample, Segment, convolve_square_nearest, distance_transform_edt, find_contours, label_8,
        maximum_filter_nearest, nearest_segments, resize_f32, skeletonize_zhang, sobel_nearest,
        uniform_filter_nearest,
    },
};

const SCALE: usize = 4;
const LEVEL: f32 = 0.5;
const NEAREST_SEGMENTS: usize = 12;

#[derive(Debug)]
#[allow(dead_code)] // Intermediate planes are retained for differential parity tests.
pub(super) struct CanonicalStages {
    pub base: Vec<f32>,
    pub candidate: Vec<f32>,
    pub reliability: Vec<f32>,
    pub analytic_lr: Vec<f32>,
    pub residual_lr: Vec<f32>,
    pub segment_coverage: Vec<f32>,
    pub ungated: Vec<f32>,
    pub combined_gate: Vec<f32>,
    pub protected: Vec<f32>,
    pub scs_lift: Vec<f32>,
    pub final_alpha: Vec<f32>,
    pub graph: SegmentGraph,
}

#[derive(Debug)]
#[allow(dead_code)] // Contour metadata is consumed by the following Color-SDF milestone.
pub(super) struct SegmentGraph {
    pub segments: Vec<Segment>,
    pub tangents: Vec<[f32; 2]>,
    pub contour_ids: Vec<i32>,
    pub arc_midpoints: Vec<f32>,
    pub contour_lengths: Vec<f32>,
    pub contour_closed: Vec<u8>,
}

pub(super) fn canonical_stages(
    alpha: &[f32],
    width: usize,
    height: usize,
) -> Result<CanonicalStages, UpscaleError> {
    let output_width = width
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let output_height = height
        .checked_mul(SCALE)
        .ok_or(UpscaleError::DimensionOverflow)?;
    output_width
        .checked_mul(output_height)
        .ok_or(UpscaleError::DimensionOverflow)?;

    let mut base = resize_f32(
        alpha,
        width,
        height,
        output_width,
        output_height,
        Resample::Lanczos3,
    );
    base.iter_mut()
        .for_each(|value| *value = value.clamp(0.0, 1.0));
    let graph = extract_segment_graph(alpha, width, height);
    if graph.segments.is_empty() {
        let zero = vec![0.0; base.len()];
        return Ok(CanonicalStages {
            candidate: base.clone(),
            reliability: zero.clone(),
            analytic_lr: alpha.to_vec(),
            residual_lr: zero.clone(),
            segment_coverage: base.clone(),
            ungated: base.clone(),
            combined_gate: vec![1.0; base.len()],
            protected: base.clone(),
            scs_lift: zero,
            final_alpha: base.clone(),
            base,
            graph,
        });
    }

    let sal = segment_adaptive_lanczos(alpha, width, height, &base, &graph);
    let ungated = bounded_residual(&base, &sal.candidate, &sal.reliability, None);
    let combined_gate = calibration_gate(
        alpha,
        width,
        height,
        &base,
        &ungated,
        output_width,
        output_height,
    );
    let protected = bounded_residual(
        &base,
        &sal.candidate,
        &sal.reliability,
        Some(&combined_gate),
    );
    let (final_alpha, scs_lift) =
        conditional_scs_lift(alpha, width, height, &base, &protected, &graph);
    Ok(CanonicalStages {
        base,
        candidate: sal.candidate,
        reliability: sal.reliability,
        analytic_lr: sal.analytic_lr,
        residual_lr: sal.residual_lr,
        segment_coverage: sal.segment_coverage,
        ungated,
        combined_gate,
        protected,
        scs_lift,
        final_alpha,
        graph,
    })
}

fn extract_segment_graph(alpha: &[f32], width: usize, height: usize) -> SegmentGraph {
    let (points, contour_point_counts) = find_contours(alpha, width, height);
    let mut segments = Vec::new();
    let mut tangents = Vec::new();
    let mut contour_ids = Vec::new();
    let mut arc_midpoints = Vec::new();
    let mut contour_lengths = Vec::new();
    let mut contour_closed = Vec::new();
    let mut offset = 0;
    for count in contour_point_counts {
        let count = count as usize;
        let contour = &points[offset..offset + count];
        offset += count;
        if contour.len() < 2 {
            continue;
        }
        let mut cumulative = 0.0_f64;
        let contour_id = contour_lengths.len() as i32;
        let mut total = 0.0_f32;
        for pair in contour.windows(2) {
            let start = pair[0];
            let end = pair[1];
            let vector_y = end[0] - start[0];
            let vector_x = end[1] - start[1];
            let length = (vector_y * vector_y + vector_x * vector_x).sqrt();
            if length <= 1e-8 {
                continue;
            }
            segments.push(Segment { start, end });
            tangents.push([vector_y / length, vector_x / length]);
            contour_ids.push(contour_id);
            arc_midpoints.push((cumulative + f64::from(0.5_f32 * length)) as f32);
            cumulative += f64::from(length);
            total += length;
        }
        if contour_ids.last().copied() != Some(contour_id) {
            continue;
        }
        contour_lengths.push(total);
        let delta_y = f64::from(contour[0][0]) - f64::from(contour[contour.len() - 1][0]);
        let delta_x = f64::from(contour[0][1]) - f64::from(contour[contour.len() - 1][1]);
        contour_closed.push(u8::from(
            (delta_y * delta_y + delta_x * delta_x).sqrt() <= 1e-5,
        ));
    }
    SegmentGraph {
        segments,
        tangents,
        contour_ids,
        arc_midpoints,
        contour_lengths,
        contour_closed,
    }
}

struct SalStages {
    candidate: Vec<f32>,
    reliability: Vec<f32>,
    analytic_lr: Vec<f32>,
    residual_lr: Vec<f32>,
    segment_coverage: Vec<f32>,
}

fn segment_adaptive_lanczos(
    alpha: &[f32],
    width: usize,
    height: usize,
    base: &[f32],
    graph: &SegmentGraph,
) -> SalStages {
    let output_width = width * SCALE;
    let output_height = height * SCALE;
    let hr_queries = hr_queries(output_width, output_height);
    let (_, _, hr_indices, _, hr_distances) =
        nearest_segments(&graph.segments, &hr_queries, NEAREST_SEGMENTS);
    let hr_indices = hr_indices
        .into_iter()
        .map(|value| value as usize)
        .collect::<Vec<_>>();
    let sign_hr = base
        .iter()
        .map(|&value| if value >= LEVEL { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let mut segment_coverage = vec![0.0; base.len()];
    for index in 0..base.len() {
        let tangent = graph.tangents[hr_indices[index]];
        let normal = [-tangent[1], tangent[0]];
        segment_coverage[index] =
            planar_coverage(sign_hr[index] * hr_distances[index], normal, 0.25);
    }

    let lr_queries = (0..height)
        .flat_map(|y| (0..width).map(move |x| [y as f32, x as f32]))
        .collect::<Vec<_>>();
    let (_, _, lr_indices, _, lr_distances) =
        nearest_segments(&graph.segments, &lr_queries, NEAREST_SEGMENTS);
    let lr_indices = lr_indices
        .into_iter()
        .map(|value| value as usize)
        .collect::<Vec<_>>();
    let mut analytic_lr = vec![0.0; alpha.len()];
    for index in 0..alpha.len() {
        let tangent = graph.tangents[lr_indices[index]];
        let normal = [-tangent[1], tangent[0]];
        let sign = if alpha[index] >= LEVEL { 1.0 } else { -1.0 };
        analytic_lr[index] = planar_coverage(sign * lr_distances[index], normal, 1.0);
    }
    let residual_lr = alpha
        .iter()
        .zip(&analytic_lr)
        .map(|(&observed, &analytic)| observed - analytic)
        .collect::<Vec<_>>();

    let segment_reliability = hard_edge_segment_reliability(alpha, width, height, graph);
    let mut candidate = base.to_vec();
    let mut reliability = vec![0.0_f32; base.len()];
    let support = 3_isize;
    for target in 0..base.len() {
        let distance_lr = hr_distances[target];
        let base_edge = 4.0 * base[target] * (1.0 - base[target]);
        let frame_reliability = segment_reliability[hr_indices[target]];
        if distance_lr * SCALE as f32 > 3.0 || base_edge <= 1e-4 || frame_reliability <= 1e-4 {
            continue;
        }
        let point = hr_queries[target];
        let center_y = point[0].floor() as isize;
        let center_x = point[1].floor() as isize;
        let mut signed_sum = 0.0_f32;
        let mut absolute_sum = 0.0_f32;
        let mut weighted_residual = 0.0_f32;
        let mut local_minimum = f32::INFINITY;
        let mut local_maximum = f32::NEG_INFINITY;
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
                let normal_coordinate = sample_y as f32 - point[0];
                let tangent_coordinate = sample_x as f32 - point[1];
                let weight = lanczos_weight(normal_coordinate) * lanczos_weight(tangent_coordinate);
                signed_sum += weight;
                absolute_sum += weight.abs();
                let residual = residual_lr[sample_y as usize * width + sample_x as usize];
                weighted_residual += weight * residual;
                if weight.abs() > 1e-8 {
                    local_minimum = local_minimum.min(residual);
                    local_maximum = local_maximum.max(residual);
                }
            }
        }
        let condition = absolute_sum / signed_sum.abs().max(1e-8);
        if signed_sum.abs() <= 1e-5 || condition > 8.0 {
            continue;
        }
        let reconstructed =
            (weighted_residual / signed_sum).clamp(local_minimum - 0.02, local_maximum + 0.02);
        candidate[target] = (segment_coverage[target] + reconstructed).clamp(0.0, 1.0);
        reliability[target] = frame_reliability;
    }
    SalStages {
        candidate,
        reliability,
        analytic_lr,
        residual_lr,
        segment_coverage,
    }
}

fn hard_edge_segment_reliability(
    alpha: &[f32],
    width: usize,
    height: usize,
    graph: &SegmentGraph,
) -> Vec<f32> {
    let gradient_x = sobel_nearest(alpha, width, height, 1)
        .into_iter()
        .map(|value| value / 8.0)
        .collect::<Vec<_>>();
    let gradient_y = sobel_nearest(alpha, width, height, 0)
        .into_iter()
        .map(|value| value / 8.0)
        .collect::<Vec<_>>();
    let gradient = gradient_x
        .iter()
        .zip(&gradient_y)
        .map(|(&x, &y)| x.hypot(y))
        .collect::<Vec<_>>();
    let background = alpha
        .iter()
        .map(|&value| f32::from(value <= 0.05))
        .collect::<Vec<_>>();
    let foreground = alpha
        .iter()
        .map(|&value| f32::from(value >= 0.95))
        .collect::<Vec<_>>();
    let kernel = vec![1.0_f32 / 25.0; 25];
    let local_background = convolve_square_nearest(&background, width, height, &kernel, 5);
    let local_foreground = convolve_square_nearest(&foreground, width, height, &kernel, 5);
    let mut reliability = graph
        .segments
        .iter()
        .map(|segment| {
            let midpoint = [
                (segment.start[0] + segment.end[0]) * 0.5,
                (segment.start[1] + segment.end[1]) * 0.5,
            ];
            let gradient_score = ((bilinear_sample(&gradient, width, height, midpoint) - 0.08)
                / 0.25)
                .clamp(0.0, 1.0);
            let binary_support = bilinear_sample(&local_background, width, height, midpoint)
                .min(bilinear_sample(&local_foreground, width, height, midpoint));
            let support_score = (binary_support / 0.12).clamp(0.0, 1.0);
            gradient_score * support_score
        })
        .collect::<Vec<_>>();
    let line_art_gate = persistent_centerline_support_gate(
        alpha,
        &vec![1.0; alpha.len()],
        width,
        height,
        CenterlineOptions {
            medial_radius_limit: 0.75,
            min_component: 3,
            neighborhood: 3,
            peak_min: 0.8,
            peak_stop: None,
            max_foreground_area: Some(4096),
            max_canvas_area: Some(16384),
            global_ratio: Some((0.05, 0.15)),
        },
    );
    for (value, segment) in reliability.iter_mut().zip(&graph.segments) {
        let midpoint = [
            (segment.start[0] + segment.end[0]) * 0.5,
            (segment.start[1] + segment.end[1]) * 0.5,
        ];
        *value = value.max(bilinear_sample(&line_art_gate, width, height, midpoint));
    }
    reliability
}

fn calibration_gate(
    alpha: &[f32],
    width: usize,
    height: usize,
    base: &[f32],
    candidate: &[f32],
    output_width: usize,
    output_height: usize,
) -> Vec<f32> {
    let intermediate_mass = alpha
        .iter()
        .map(|&value| 4.0 * value * (1.0 - value))
        .collect::<Vec<_>>();
    let broad_score = uniform_filter_nearest(&intermediate_mass, width, height, 5);
    let broad_gate = broad_score
        .iter()
        .map(|&score| 1.0 - smoothstep((score - 0.16) / (0.52 - 0.16)))
        .collect::<Vec<_>>();

    let base_reprojected = resize_f32(
        base,
        output_width,
        output_height,
        width,
        height,
        Resample::Lanczos3,
    );
    let candidate_reprojected = resize_f32(
        candidate,
        output_width,
        output_height,
        width,
        height,
        Resample::Lanczos3,
    );
    let base_absolute_error = base_reprojected
        .iter()
        .zip(alpha)
        .map(|(&value, &target)| (value - target).abs())
        .collect::<Vec<_>>();
    let candidate_absolute_error = candidate_reprojected
        .iter()
        .zip(alpha)
        .map(|(&value, &target)| (value - target).abs())
        .collect::<Vec<_>>();
    let base_error = uniform_filter_nearest(&base_absolute_error, width, height, 5);
    let candidate_error = uniform_filter_nearest(&candidate_absolute_error, width, height, 5);
    let reprojection_gate = base_error
        .iter()
        .zip(&candidate_error)
        .map(|(&base_value, &candidate_value)| smoothstep((base_value - candidate_value) / 0.001))
        .collect::<Vec<_>>();
    let support_score = uniform_filter_nearest(&reprojection_gate, width, height, 9);
    let support_gate = support_score
        .iter()
        .map(|&score| smoothstep((score - 0.15) / (0.50 - 0.15)))
        .collect::<Vec<_>>();
    let centerline_support_gate = persistent_centerline_support_gate(
        alpha,
        &reprojection_gate,
        width,
        height,
        CenterlineOptions {
            medial_radius_limit: 0.75,
            min_component: 3,
            neighborhood: 3,
            peak_min: 0.8,
            peak_stop: None,
            max_foreground_area: Some(4096),
            max_canvas_area: None,
            global_ratio: Some((0.10, 0.30)),
        },
    );

    let broad_hr = resize_f32(
        &broad_gate,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let reprojection_hr = resize_f32(
        &reprojection_gate,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let support_hr = resize_f32(
        &support_gate,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    let centerline_support_hr = resize_f32(
        &centerline_support_gate,
        width,
        height,
        output_width,
        output_height,
        Resample::Bilinear,
    );
    broad_hr
        .iter()
        .zip(&reprojection_hr)
        .zip(&support_hr)
        .zip(&centerline_support_hr)
        .map(|(((&broad, &reprojection), &support), &centerline)| {
            (broad * reprojection * support)
                .max(centerline)
                .clamp(0.0, 1.0)
        })
        .collect()
}

#[derive(Clone, Copy)]
struct CenterlineOptions {
    medial_radius_limit: f32,
    min_component: usize,
    neighborhood: usize,
    peak_min: f32,
    peak_stop: Option<f32>,
    max_foreground_area: Option<usize>,
    max_canvas_area: Option<usize>,
    global_ratio: Option<(f64, f64)>,
}

fn persistent_centerline_support_gate(
    alpha: &[f32],
    confidence: &[f32],
    width: usize,
    height: usize,
    options: CenterlineOptions,
) -> Vec<f32> {
    let mask = alpha
        .iter()
        .map(|&value| value >= LEVEL)
        .collect::<Vec<_>>();
    let foreground_area = mask.iter().filter(|&&value| value).count();
    let peak = alpha.iter().copied().fold(0.0_f32, f32::max);
    if foreground_area == 0
        || foreground_area == mask.len()
        || peak < options.peak_min
        || options.peak_stop.is_some_and(|stop| peak >= stop)
        || options
            .max_foreground_area
            .is_some_and(|maximum| foreground_area > maximum)
        || options
            .max_canvas_area
            .is_some_and(|maximum| alpha.len() > maximum)
    {
        return vec![0.0; alpha.len()];
    }
    let skeleton = skeletonize_zhang(&mask, width, height);
    let mut global_gate = 0.0_f32;
    if let Some((ratio_start, ratio_stop)) = options.global_ratio {
        let (labels, component_count) = label_8(&skeleton, width, height);
        let mut sizes = vec![0_usize; component_count + 1];
        for label in labels {
            sizes[label] += 1;
        }
        if sizes.iter().skip(1).copied().max().unwrap_or(0) >= options.min_component {
            let skeleton_area = skeleton.iter().filter(|&&value| value).count();
            let ratio = skeleton_area as f64 / (foreground_area as f64).max(1.0);
            global_gate = smoothstep(((ratio - ratio_start) / (ratio_stop - ratio_start)) as f32);
        }
    }
    let distance = distance_transform_edt(&mask, width, height);
    let thin = skeleton
        .iter()
        .zip(distance)
        .map(|(&on_skeleton, radius)| {
            on_skeleton && (radius - 0.5).max(0.5) <= options.medial_radius_limit
        })
        .collect::<Vec<_>>();
    let (labels, component_count) = label_8(&thin, width, height);
    if component_count == 0 {
        return vec![global_gate; alpha.len()];
    }
    let mut sizes = vec![0_usize; component_count + 1];
    for &label in &labels {
        sizes[label] += 1;
    }
    let persistent = labels
        .iter()
        .map(|&label| label != 0 && sizes[label] >= options.min_component)
        .collect::<Vec<_>>();
    let persistent_f32 = persistent
        .iter()
        .map(|&value| f32::from(value))
        .collect::<Vec<_>>();
    let neighborhood = maximum_filter_nearest(&persistent_f32, width, height, options.neighborhood);
    neighborhood
        .iter()
        .zip(confidence)
        .map(|(&near, &confidence)| (near * confidence).max(global_gate).clamp(0.0, 1.0))
        .collect()
}

fn conditional_scs_lift(
    alpha: &[f32],
    width: usize,
    height: usize,
    base: &[f32],
    protected: &[f32],
    graph: &SegmentGraph,
) -> (Vec<f32>, Vec<f32>) {
    let zero = || vec![0.0_f32; protected.len()];
    let mask = alpha
        .iter()
        .map(|&value| value >= LEVEL)
        .collect::<Vec<_>>();
    let foreground_area = mask.iter().filter(|&&value| value).count();
    if foreground_area == 0
        || foreground_area == mask.len()
        || foreground_area > 4096
        || !alpha.iter().any(|&value| value >= 0.95)
        || !alpha.iter().any(|&value| value <= 0.05)
    {
        return (protected.to_vec(), zero());
    }
    let skeleton = skeletonize_zhang(&mask, width, height);
    if !skeleton.iter().any(|&value| value) {
        return (protected.to_vec(), zero());
    }
    let radius_map = distance_transform_edt(&mask, width, height)
        .into_iter()
        .map(|distance| (distance - 0.5).max(0.5))
        .collect::<Vec<_>>();
    let mut thin = skeleton
        .iter()
        .zip(&radius_map)
        .map(|(&value, &radius)| value && radius <= 0.75)
        .collect::<Vec<_>>();
    let (labels, component_count) = label_8(&thin, width, height);
    if component_count == 0 {
        return (protected.to_vec(), zero());
    }
    let mut sizes = vec![0_usize; component_count + 1];
    for &label in &labels {
        sizes[label] += 1;
    }
    for (value, &label) in thin.iter_mut().zip(&labels) {
        *value &= label != 0 && sizes[label] >= 3;
    }
    let mut centers = Vec::new();
    let mut radii = Vec::new();
    for (index, &value) in thin.iter().enumerate() {
        if value {
            centers.push([(index / width) as f32, (index % width) as f32]);
            radii.push(radius_map[index]);
        }
    }
    if centers.is_empty() {
        return (protected.to_vec(), zero());
    }

    let segment_reference = segment_reference(base, width, height, graph);
    let mut active_centers = Vec::new();
    let mut active_radii = Vec::new();
    let mut activations = Vec::new();
    for (center, radius) in centers.into_iter().zip(radii) {
        let center_y = (((center[0] + 0.5) * SCALE as f32 - 0.5).round_ties_even() as isize)
            .clamp(0, (height * SCALE - 1) as isize) as usize;
        let center_x = (((center[1] + 0.5) * SCALE as f32 - 0.5).round_ties_even() as isize)
            .clamp(0, (width * SCALE - 1) as isize) as usize;
        let center_alpha = segment_reference[center_y * width * SCALE + center_x];
        let activation = ((1.0 - center_alpha) / 0.5).clamp(0.0, 1.0);
        if activation > 0.0 {
            active_centers.push(center);
            active_radii.push(radius);
            activations.push(activation);
        }
    }
    if active_centers.is_empty() {
        return (protected.to_vec(), zero());
    }

    let queries = hr_queries(width * SCALE, height * SCALE);
    let nearest_count = active_centers.len().min(16);
    let mut lift = vec![0.0_f32; protected.len()];
    for (index, &query) in queries.iter().enumerate() {
        let mut neighbors = active_centers
            .iter()
            .enumerate()
            .map(|(center_index, center)| {
                let delta_y = query[0] - center[0];
                let delta_x = query[1] - center[1];
                (
                    f64::from(delta_y) * f64::from(delta_y)
                        + f64::from(delta_x) * f64::from(delta_x),
                    center_index,
                )
            })
            .collect::<Vec<_>>();
        neighbors
            .sort_unstable_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        let mut maximum_lift = 0.0_f32;
        for &(_, center_index) in neighbors.iter().take(nearest_count) {
            let center = active_centers[center_index];
            let delta_y = query[0] - center[0];
            let delta_x = query[1] - center[1];
            let distance = (delta_y * delta_y + delta_x * delta_x).sqrt();
            let signed_distance_hr = (active_radii[center_index] - distance) * SCALE as f32;
            let coverage = (0.5 + signed_distance_hr).clamp(0.0, 1.0);
            maximum_lift = maximum_lift
                .max(activations[center_index] * (coverage - protected[index]).max(0.0));
        }
        lift[index] = if protected[index] >= LEVEL {
            maximum_lift.min(0.2)
        } else {
            0.0
        };
    }
    let below_half = f32::from_bits(LEVEL.to_bits() - 1);
    let output = protected
        .iter()
        .zip(&lift)
        .map(|(&value, &addition)| {
            let corrected = (value + addition).clamp(0.0, 1.0);
            if value >= LEVEL {
                corrected.max(LEVEL)
            } else {
                corrected.min(below_half)
            }
        })
        .collect();
    (output, lift)
}

fn segment_reference(base: &[f32], width: usize, height: usize, graph: &SegmentGraph) -> Vec<f32> {
    let queries = hr_queries(width * SCALE, height * SCALE);
    let (_, _, _, _, distances) = nearest_segments(&graph.segments, &queries, NEAREST_SEGMENTS);
    base.iter()
        .zip(distances)
        .map(|(&base, distance)| {
            let sign = if base >= LEVEL { 1.0 } else { -1.0 };
            (0.5 + sign * distance * SCALE as f32).clamp(0.0, 1.0)
        })
        .collect()
}

fn bounded_residual(
    base: &[f32],
    candidate: &[f32],
    reliability: &[f32],
    additional_gate: Option<&[f32]>,
) -> Vec<f32> {
    let below_half = f32::from_bits(LEVEL.to_bits() - 1);
    (0..base.len())
        .map(|index| {
            let gate = additional_gate.map_or(1.0, |values| values[index]);
            let residual = (candidate[index] - base[index]).clamp(-0.2, 0.2);
            let corrected = base[index] + reliability[index] * gate * residual;
            if base[index] >= LEVEL {
                corrected.max(LEVEL).clamp(0.0, 1.0)
            } else {
                corrected.min(below_half).clamp(0.0, 1.0)
            }
        })
        .collect()
}

fn hr_queries(width: usize, height: usize) -> Vec<[f32; 2]> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                [
                    (y as f32 + 0.5) / SCALE as f32 - 0.5,
                    (x as f32 + 0.5) / SCALE as f32 - 0.5,
                ]
            })
        })
        .collect()
}

fn planar_coverage(signed_distance_lr: f32, normal: [f32; 2], pixel_size_lr: f32) -> f32 {
    let projected_width = pixel_size_lr * (normal[0].abs() + normal[1].abs());
    (0.5 + signed_distance_lr / projected_width.max(1e-6)).clamp(0.0, 1.0)
}

fn bilinear_sample(plane: &[f32], width: usize, height: usize, point: [f32; 2]) -> f32 {
    let y = point[0].clamp(0.0, height as f32 - 1.0);
    let x = point[1].clamp(0.0, width as f32 - 1.0);
    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let y1 = (y0 + 1).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let fy = y - y0 as f32;
    let fx = x - x0 as f32;
    plane[y0 * width + x0] * (1.0 - fy) * (1.0 - fx)
        + plane[y0 * width + x1] * (1.0 - fy) * fx
        + plane[y1 * width + x0] * fy * (1.0 - fx)
        + plane[y1 * width + x1] * fy * fx
}

fn lanczos_weight(value: f32) -> f32 {
    if value.abs() >= 3.0 {
        0.0
    } else {
        sinc(value) * sinc(value / 3.0)
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

fn smoothstep(value: f32) -> f32 {
    let clipped = value.clamp(0.0, 1.0);
    clipped * clipped * (3.0 - 2.0 * clipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 22;
    const HEIGHT: usize = 18;

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|item| f32::from_le_bytes(item.try_into().unwrap()))
            .collect()
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn input() -> Vec<f32> {
        include_bytes!("../../../../fixtures/parity/alpha/input_alpha_u8.raw")
            .iter()
            .map(|&value| f32::from(value) / 255.0)
            .collect()
    }

    macro_rules! intermediate {
        ($path:literal) => {
            include_bytes!(concat!(
                "../../../../fixtures/parity/alpha_intermediates/",
                $path
            ))
        };
    }

    #[test]
    fn fixed_fixture_graph_and_early_geometry_match() {
        let stages = canonical_stages(&input(), WIDTH, HEIGHT).unwrap();
        let starts = stages
            .graph
            .segments
            .iter()
            .flat_map(|segment| segment.start)
            .collect::<Vec<_>>();
        let ends = stages
            .graph
            .segments
            .iter()
            .flat_map(|segment| segment.end)
            .collect::<Vec<_>>();
        assert_eq!(
            bytes(&starts),
            intermediate!("geometry/contour_starts_f32.raw")
        );
        assert_eq!(bytes(&ends), intermediate!("geometry/contour_ends_f32.raw"));
        assert_eq!(
            bytes(
                &stages
                    .graph
                    .tangents
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
            ),
            intermediate!("geometry/contour_tangents_f32.raw")
        );
        assert_eq!(
            bytes(&stages.analytic_lr),
            intermediate!("geometry/analytic_coverage_lr_f32.raw")
        );
        assert_eq!(
            bytes(&stages.residual_lr),
            intermediate!("geometry/residual_lr_f32.raw")
        );
        assert_eq!(
            bytes(&stages.segment_coverage),
            intermediate!("geometry/segment_coverage_hr_f32.raw")
        );
    }

    #[test]
    fn fixed_fixture_value_stages_report_exact_boundaries() {
        let stages = canonical_stages(&input(), WIDTH, HEIGHT).unwrap();
        for (name, actual, expected) in [
            (
                "base",
                stages.base.as_slice(),
                intermediate!("stages/base_lanczos_hr_f32.raw").as_slice(),
            ),
            (
                "candidate",
                stages.candidate.as_slice(),
                intermediate!("stages/candidate_hr_f32.raw").as_slice(),
            ),
            (
                "reliability",
                stages.reliability.as_slice(),
                intermediate!("stages/reliability_hr_f32.raw").as_slice(),
            ),
            (
                "ungated",
                stages.ungated.as_slice(),
                intermediate!("stages/ungated_hr_f32.raw").as_slice(),
            ),
            (
                "combined_gate",
                stages.combined_gate.as_slice(),
                intermediate!("calibration/combined_gate_hr_f32.raw").as_slice(),
            ),
            (
                "protected",
                stages.protected.as_slice(),
                intermediate!("stages/protected_hr_f32.raw").as_slice(),
            ),
        ] {
            let expected = f32s(expected);
            let mismatches = actual
                .iter()
                .zip(&expected)
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
            let maximum = actual
                .iter()
                .zip(&expected)
                .map(|(&left, &right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("{name}: mismatches={mismatches}, max={maximum:e}");
        }
        assert_eq!(
            bytes(&stages.base),
            intermediate!("stages/base_lanczos_hr_f32.raw")
        );
        assert_eq!(
            crate::alpha::quantize_alpha_u8(&stages.final_alpha),
            include_bytes!("../../../../fixtures/parity/alpha/expected_alpha_u8.raw")
        );
        assert_eq!(
            bytes(&stages.scs_lift),
            intermediate!("stages/scs_lift_hr_f32.raw")
        );
    }

    fn corpus_gap(input: &[u8], expected: &[u8], width: usize, height: usize) -> (usize, u8) {
        let alpha = input
            .iter()
            .map(|&value| f32::from(value) / 255.0)
            .collect::<Vec<_>>();
        let stages = canonical_stages(&alpha, width, height).unwrap();
        let actual = crate::alpha::quantize_alpha_u8(&stages.final_alpha);
        actual
            .iter()
            .zip(expected)
            .fold((0, 0), |(count, maximum), (&left, &right)| {
                (
                    count + usize::from(left != right),
                    maximum.max(left.abs_diff(right)),
                )
            })
    }

    #[test]
    fn corpus_reports_remaining_line_art_gaps() {
        macro_rules! case {
            ($name:literal, $width:literal, $height:literal) => {{
                let gap = corpus_gap(
                    include_bytes!(concat!(
                        "../../../../fixtures/parity/alpha_cases/",
                        $name,
                        "/input_alpha_u8.raw"
                    )),
                    include_bytes!(concat!(
                        "../../../../fixtures/parity/alpha_cases/",
                        $name,
                        "/expected_alpha_u8.raw"
                    )),
                    $width,
                    $height,
                );
                eprintln!("{}: changed={}, max={}", $name, gap.0, gap.1);
                gap
            }};
        }
        let gaps = [
            case!("zero", 9, 7),
            case!("one", 9, 7),
            case!("broad_soft", 11, 9),
            case!("rectangle", 13, 9),
            case!("ellipse", 15, 11),
            case!("line_horizontal", 13, 7),
            case!("line_vertical", 7, 13),
            case!("line_diagonal", 11, 11),
            case!("holes", 15, 13),
            case!("open_contour", 15, 13),
            case!("partial_alpha_edges", 13, 9),
            case!("odd_size_motif", 11, 9),
            case!("scs_lift", 9, 9),
        ];
        assert!(gaps.into_iter().all(|gap| gap == (0, 0)));
    }
}
