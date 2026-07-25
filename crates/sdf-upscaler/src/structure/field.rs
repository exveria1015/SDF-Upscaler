//! Packed-HWC eleven-channel RGB structure field.

use std::collections::VecDeque;

use crate::{
    UpscaleError,
    compat::{distance_transform_edt_with_indices, label_8, skeletonize_zhang},
};

pub(crate) const CHANNELS: usize = 11;
pub(crate) const EDGE_CONFIDENCE: usize = 0;
pub(crate) const SIGNED_DISTANCE: usize = 1;
pub(crate) const NORMAL_X: usize = 3;
pub(crate) const NORMAL_Y: usize = 4;

const SCALES: [f64; 3] = [0.8, 1.4, 2.2];
const LOW_THRESHOLD: f32 = 0.025;
const HIGH_THRESHOLD: f32 = 0.075;
const MIN_COMPONENT_SIZE: usize = 3;
const PROPAGATION_RADIUS: f32 = 6.0;
const CONTRAST_OFFSET: f32 = 1.25;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StructureField {
    pub width: usize,
    pub height: usize,
    /// Packed `[y * width * 11 + x * 11 + channel]` values.
    pub features: Vec<f32>,
    pub edge_mask: Vec<bool>,
    pub contour_labels: Vec<i32>,
}

impl StructureField {
    pub(crate) fn channel(&self, channel: usize) -> Vec<f32> {
        assert!(channel < CHANNELS);
        self.features
            .chunks_exact(CHANNELS)
            .map(|pixel| pixel[channel])
            .collect()
    }
}

pub(crate) fn extract_structure_field(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<StructureField, UpscaleError> {
    if width < 3 || height < 3 {
        return Err(UpscaleError::InvalidOptions(
            "structure RGB must be at least 3x3",
        ));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(UpscaleError::DimensionOverflow)?;
    let required = pixels
        .checked_mul(3)
        .ok_or(UpscaleError::DimensionOverflow)?;
    if rgb.len() != required {
        return Err(UpscaleError::BufferTooSmall {
            required,
            actual: rgb.len(),
        });
    }

    let mut red = vec![0.0_f32; pixels];
    let mut green = vec![0.0_f32; pixels];
    let mut blue = vec![0.0_f32; pixels];
    let mut luma = vec![0.0_f32; pixels];
    let mut opponent_a = vec![0.0_f32; pixels];
    let mut opponent_b = vec![0.0_f32; pixels];
    for index in 0..pixels {
        red[index] = srgb_to_linear(rgb[index * 3]);
        green[index] = srgb_to_linear(rgb[index * 3 + 1]);
        blue[index] = srgb_to_linear(rgb[index * 3 + 2]);
        luma[index] = 0.2126 * red[index] + 0.7152 * green[index] + 0.0722 * blue[index];
        opponent_a[index] = 0.5 * (red[index] - green[index]) + 0.5;
        opponent_b[index] = 0.5 * (blue[index] - 0.5 * (red[index] + green[index])) + 0.5;
    }

    let planes = [&luma, &red, &green, &blue];
    let mut edge_mask = vec![false; pixels];
    let mut confidence = vec![0.0_f32; pixels];
    let mut best_magnitude = vec![0.0_f32; pixels];
    let mut best_x = vec![0.0_f32; pixels];
    let mut best_y = vec![0.0_f32; pixels];
    for sigma in SCALES {
        let canny_bleed =
            gaussian_filter(&vec![1.0; pixels], width, height, sigma, Border::Constant);
        for plane in planes {
            let smoothed = gaussian_filter(plane, width, height, sigma, Border::Reflect);
            let gradient_x = sobel(&smoothed, width, height, Axis::X, Border::Reflect);
            let gradient_y = sobel(&smoothed, width, height, Axis::Y, Border::Reflect);
            let mut magnitude = vec![0.0_f32; pixels];
            for index in 0..pixels {
                let x = gradient_x[index] / 8.0;
                let y = gradient_y[index] / 8.0;
                magnitude[index] = x.hypot(y);
                if magnitude[index] > best_magnitude[index] {
                    best_magnitude[index] = magnitude[index];
                    best_x[index] = x;
                    best_y[index] = y;
                }
                confidence[index] = confidence[index].max(1.0 - (-magnitude[index] / 0.06).exp());
            }
            let candidate = canny(
                plane,
                &canny_bleed,
                width,
                height,
                sigma,
                LOW_THRESHOLD,
                HIGH_THRESHOLD,
            );
            for index in 0..pixels {
                edge_mask[index] |= candidate[index];
            }
        }
    }

    retain_components(&mut edge_mask, width, height, MIN_COMPONENT_SIZE);
    edge_mask = skeletonize_zhang(&edge_mask, width, height);
    let (labels, _) = label_8(&edge_mask, width, height);
    let contour_labels = labels
        .into_iter()
        .map(|value| value as i32)
        .collect::<Vec<_>>();
    if !edge_mask.iter().any(|&value| value) {
        return Ok(StructureField {
            width,
            height,
            features: vec![0.0; pixels * CHANNELS],
            edge_mask,
            contour_labels,
        });
    }

    let mut normal_x_raw = vec![0.0_f32; pixels];
    let mut normal_y_raw = vec![0.0_f32; pixels];
    for index in 0..pixels {
        let denominator = best_magnitude[index].max(1e-8);
        normal_x_raw[index] = best_x[index] / denominator;
        normal_y_raw[index] = best_y[index] / denominator;
    }

    let non_edge = edge_mask.iter().map(|&value| !value).collect::<Vec<_>>();
    let (distance, nearest_y, nearest_x) =
        distance_transform_edt_with_indices(&non_edge, width, height);
    let smooth_normal_x = gaussian_filter(&normal_x_raw, width, height, 1.0, Border::Reflect);
    let smooth_normal_y = gaussian_filter(&normal_y_raw, width, height, 1.0, Border::Reflect);
    let curvature_x = sobel(&smooth_normal_x, width, height, Axis::X, Border::Reflect);
    let curvature_y = sobel(&smooth_normal_y, width, height, Axis::Y, Border::Reflect);
    let curvature_raw = curvature_x
        .iter()
        .zip(&curvature_y)
        .map(|(&x, &y)| (x + y) / 8.0)
        .collect::<Vec<_>>();

    let junction = junction_mask(&edge_mask, width, height);
    let mut junction_confidence = gaussian_filter(&junction, width, height, 1.0, Border::Constant);
    let junction_peak = junction_confidence.iter().copied().fold(0.0, f32::max);
    if junction_peak > 0.0 {
        for value in &mut junction_confidence {
            *value /= junction_peak;
        }
    }

    let mut features = vec![0.0_f32; pixels * CHANNELS];
    for index in 0..pixels {
        let nearest_index = nearest_y[index] * width + nearest_x[index];
        let mut normal_x = normal_x_raw[nearest_index];
        let mut normal_y = normal_y_raw[nearest_index];
        let normal_length = normal_x.hypot(normal_y).max(1e-8);
        normal_x /= normal_length;
        normal_y /= normal_length;
        let support = (-distance[index] / PROPAGATION_RADIUS).exp();
        let delta_x = (index % width) as f32 - nearest_x[index] as f32;
        let delta_y = (index / width) as f32 - nearest_y[index] as f32;
        let base = index * CHANNELS;
        features[base] = (confidence[nearest_index] * support).clamp(0.0, 1.0);
        features[base + 1] =
            ((delta_x * normal_x + delta_y * normal_y) / PROPAGATION_RADIUS).clamp(-1.0, 1.0);
        features[base + 2] = (distance[index] / PROPAGATION_RADIUS).clamp(0.0, 1.0);
        features[base + 3] = normal_x;
        features[base + 4] = normal_y;
        features[base + 5] = -normal_y;
        features[base + 6] = normal_x;
        features[base + 7] = (curvature_raw[nearest_index] / 0.5).clamp(-1.0, 1.0);
        features[base + 8] = (junction_confidence[nearest_index] * support).clamp(0.0, 1.0);
        let edge_x = nearest_x[index] as f32;
        let edge_y = nearest_y[index] as f32;
        let plus_luma = bilinear_sample(
            &luma,
            width,
            height,
            edge_x + normal_x * CONTRAST_OFFSET,
            edge_y + normal_y * CONTRAST_OFFSET,
        );
        let minus_luma = bilinear_sample(
            &luma,
            width,
            height,
            edge_x - normal_x * CONTRAST_OFFSET,
            edge_y - normal_y * CONTRAST_OFFSET,
        );
        features[base + 9] = (plus_luma - minus_luma).clamp(-1.0, 1.0) * support;
        let plus_a = bilinear_sample(
            &opponent_a,
            width,
            height,
            edge_x + normal_x * CONTRAST_OFFSET,
            edge_y + normal_y * CONTRAST_OFFSET,
        );
        let minus_a = bilinear_sample(
            &opponent_a,
            width,
            height,
            edge_x - normal_x * CONTRAST_OFFSET,
            edge_y - normal_y * CONTRAST_OFFSET,
        );
        let plus_b = bilinear_sample(
            &opponent_b,
            width,
            height,
            edge_x + normal_x * CONTRAST_OFFSET,
            edge_y + normal_y * CONTRAST_OFFSET,
        );
        let minus_b = bilinear_sample(
            &opponent_b,
            width,
            height,
            edge_x - normal_x * CONTRAST_OFFSET,
            edge_y - normal_y * CONTRAST_OFFSET,
        );
        features[base + 10] = (plus_a - minus_a).hypot(plus_b - minus_b).clamp(0.0, 1.0) * support;
    }

    Ok(StructureField {
        width,
        height,
        features,
        edge_mask,
        contour_labels,
    })
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[derive(Clone, Copy)]
enum Border {
    Reflect,
    Constant,
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as isize;
    let sigma2 = sigma * sigma;
    let mut kernel = (-radius..=radius)
        .map(|offset| (-(offset * offset) as f64 / (2.0 * sigma2)).exp())
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f64>();
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn gaussian_filter(
    input: &[f32],
    width: usize,
    height: usize,
    sigma: f64,
    border: Border,
) -> Vec<f32> {
    let kernel = gaussian_kernel(sigma);
    let vertical = correlate_axis(input, width, height, &kernel, Axis::Y, border);
    correlate_axis(&vertical, width, height, &kernel, Axis::X, border)
}

fn correlate_axis(
    input: &[f32],
    width: usize,
    height: usize,
    kernel: &[f64],
    axis: Axis,
    border: Border,
) -> Vec<f32> {
    let radius = (kernel.len() / 2) as isize;
    let mut output = vec![0.0_f32; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f64;
            for (kernel_index, &weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                let (sample_x, sample_y) = match axis {
                    Axis::X => (x as isize + offset, y as isize),
                    Axis::Y => (x as isize, y as isize + offset),
                };
                if let Some(sample) = sample_index(sample_x, sample_y, width, height, border) {
                    sum += f64::from(input[sample]) * weight;
                }
            }
            output[y * width + x] = sum as f32;
        }
    }
    output
}

fn sample_index(x: isize, y: isize, width: usize, height: usize, border: Border) -> Option<usize> {
    match border {
        Border::Constant if x < 0 || y < 0 || x >= width as isize || y >= height as isize => None,
        Border::Constant => Some(y as usize * width + x as usize),
        Border::Reflect => {
            Some(reflect_half_sample(y, height) * width + reflect_half_sample(x, width))
        }
    }
}

fn reflect_half_sample(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * length as isize;
    let wrapped = index.rem_euclid(period);
    if wrapped >= length as isize {
        (period - 1 - wrapped) as usize
    } else {
        wrapped as usize
    }
}

fn sobel(input: &[f32], width: usize, height: usize, axis: Axis, border: Border) -> Vec<f32> {
    let derivative = [-1.0_f64, 0.0, 1.0];
    let smoothing = [1.0_f64, 2.0, 1.0];
    match axis {
        Axis::X => {
            let first = correlate_axis(input, width, height, &derivative, Axis::X, border);
            correlate_axis(&first, width, height, &smoothing, Axis::Y, border)
        }
        Axis::Y => {
            let first = correlate_axis(input, width, height, &derivative, Axis::Y, border);
            correlate_axis(&first, width, height, &smoothing, Axis::X, border)
        }
    }
}

fn canny(
    plane: &[f32],
    bleed: &[f32],
    width: usize,
    height: usize,
    sigma: f64,
    low: f32,
    high: f32,
) -> Vec<bool> {
    let mut smoothed = gaussian_filter(plane, width, height, sigma, Border::Constant);
    for (value, &bleed) in smoothed.iter_mut().zip(bleed) {
        *value /= bleed + f32::EPSILON;
    }
    let jsobel = sobel(&smoothed, width, height, Axis::X, Border::Reflect);
    let isobel = sobel(&smoothed, width, height, Axis::Y, Border::Reflect);
    let magnitude = isobel
        .iter()
        .zip(&jsobel)
        .map(|(&y, &x)| y.hypot(x))
        .collect::<Vec<_>>();
    let mut suppressed = vec![0.0_f32; plane.len()];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = y * width + x;
            let m = magnitude[index];
            if m < low {
                continue;
            }
            let iy = isobel[index];
            let jx = jsobel[index];
            let is_down = iy <= 0.0;
            let is_up = iy >= 0.0;
            let is_left = jx <= 0.0;
            let is_right = jx >= 0.0;
            let cond1 = (is_up && is_right) || (is_down && is_left);
            let cond2 = (is_down && is_right) || (is_up && is_left);
            if !cond1 && !cond2 {
                continue;
            }
            let abs_i = iy.abs();
            let abs_j = jx.abs();
            let (weight, first_a, first_b, second_a, second_b) = if cond1 {
                if abs_i > abs_j {
                    (
                        abs_j / abs_i,
                        magnitude[(y + 1) * width + x],
                        magnitude[(y + 1) * width + x + 1],
                        magnitude[(y - 1) * width + x],
                        magnitude[(y - 1) * width + x - 1],
                    )
                } else {
                    (
                        abs_i / abs_j,
                        magnitude[y * width + x + 1],
                        magnitude[(y + 1) * width + x + 1],
                        magnitude[y * width + x - 1],
                        magnitude[(y - 1) * width + x - 1],
                    )
                }
            } else if abs_i < abs_j {
                (
                    abs_i / abs_j,
                    magnitude[y * width + x + 1],
                    magnitude[(y - 1) * width + x + 1],
                    magnitude[y * width + x - 1],
                    magnitude[(y + 1) * width + x - 1],
                )
            } else {
                (
                    abs_j / abs_i,
                    magnitude[(y - 1) * width + x],
                    magnitude[(y - 1) * width + x + 1],
                    magnitude[(y + 1) * width + x],
                    magnitude[(y + 1) * width + x - 1],
                )
            };
            let plus = first_b * weight + first_a * (1.0 - weight);
            let minus = second_b * weight + second_a * (1.0 - weight);
            if plus <= m && minus <= m {
                suppressed[index] = m;
            }
        }
    }
    hysteresis(&suppressed, width, height, high)
}

fn hysteresis(suppressed: &[f32], width: usize, height: usize, high: f32) -> Vec<bool> {
    let mut output = vec![false; suppressed.len()];
    let mut queued = vec![false; suppressed.len()];
    let mut queue = VecDeque::new();
    for (index, &value) in suppressed.iter().enumerate() {
        if value >= high && value > 0.0 {
            output[index] = true;
            queued[index] = true;
            queue.push_back(index);
        }
    }
    while let Some(index) = queue.pop_front() {
        let x = index % width;
        let y = index / width;
        for neighbor_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                let neighbor = neighbor_y * width + neighbor_x;
                if !queued[neighbor] && suppressed[neighbor] > 0.0 {
                    queued[neighbor] = true;
                    output[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    output
}

fn retain_components(mask: &mut [bool], width: usize, height: usize, minimum: usize) {
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    let mut component = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        queue.push_back(start);
        component.clear();
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let x = index % width;
            let y = index / width;
            for neighbor_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbor = neighbor_y * width + neighbor_x;
                    if mask[neighbor] && !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        if component.len() < minimum {
            for &index in &component {
                mask[index] = false;
            }
        }
    }
}

fn junction_mask(mask: &[bool], width: usize, height: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; mask.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if !mask[index] {
                continue;
            }
            let mut neighbors = 0_u8;
            for neighbor_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if (neighbor_x != x || neighbor_y != y) && mask[neighbor_y * width + neighbor_x]
                    {
                        neighbors += 1;
                    }
                }
            }
            output[index] = f32::from(neighbors >= 3);
        }
    }
    output
}

fn bilinear_sample(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let top = plane[y0 * width + x0] * (1.0 - wx) + plane[y0 * width + x1] * wx;
    let bottom = plane[y1 * width + x0] * (1.0 - wx) + plane[y1 * width + x1] * wx;
    top * (1.0 - wy) + bottom * wy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn fixed_structure_field_matches_python() {
        let rgba = include_bytes!("../../../../fixtures/parity/adaptive/fixed/input_rgba_u8.raw");
        let bounded =
            include_bytes!("../../../../fixtures/parity/adaptive/fixed/bounded_visible_rgb_u8.raw");
        assert_eq!(rgba.len(), 18 * 22 * 4);
        let field = extract_structure_field(bounded, 22, 18).unwrap();
        let edge =
            include_bytes!("../../../../fixtures/parity/adaptive/fixed/structure_edge_mask_u8.raw");
        assert_eq!(
            field
                .edge_mask
                .iter()
                .map(|&value| u8::from(value))
                .collect::<Vec<_>>(),
            edge
        );
        let labels = include_bytes!(
            "../../../../fixtures/parity/adaptive/fixed/structure_contour_labels_i32.raw"
        )
        .chunks_exact(4)
        .map(|value| i32::from_le_bytes(value.try_into().unwrap()))
        .collect::<Vec<_>>();
        assert_eq!(field.contour_labels, labels);
        let expected = expected_f32(include_bytes!(
            "../../../../fixtures/parity/adaptive/fixed/structure_features_hwc_f32.raw"
        ));
        let maximum = field
            .features
            .iter()
            .zip(&expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum <= 2e-6, "maximum feature error {maximum}");
    }
}
