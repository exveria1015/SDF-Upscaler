//! Compatibility primitives frozen against the Python parity micro-fixtures.
//!
//! This module is deliberately private: it exists to make the later Alpha
//! port mechanically reproducible, not to establish a public image API.

#![allow(dead_code)]

use std::{
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Resample {
    Bilinear,
    Lanczos3,
}

pub(crate) fn resize_f32(
    input: &[f32],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
    filter: Resample,
) -> Vec<f32> {
    assert_eq!(input.len(), width * height);
    let horizontal = resize_axis(input, width, height, out_width, filter, true);
    resize_axis(&horizontal, out_width, height, out_height, filter, false)
}

fn resize_axis(
    input: &[f32],
    width: usize,
    height: usize,
    output_size: usize,
    filter: Resample,
    horizontal: bool,
) -> Vec<f32> {
    let source_size = if horizontal { width } else { height };
    let mut weights = Vec::with_capacity(output_size);
    for destination in 0..output_size {
        weights.push(pillow_weights(
            source_size,
            output_size,
            destination,
            filter,
        ));
    }
    let (out_width, out_height) = if horizontal {
        (output_size, height)
    } else {
        (width, output_size)
    };
    let mut output = vec![0.0; out_width * out_height];
    for y in 0..out_height {
        for x in 0..out_width {
            let destination = if horizontal { x } else { y };
            let mut value = 0.0_f64;
            for &(source, coefficient) in &weights[destination] {
                let sample = if horizontal {
                    input[y * width + source]
                } else {
                    input[source * width + x]
                };
                value += f64::from(sample) * coefficient;
            }
            output[y * out_width + x] = value as f32;
        }
    }
    output
}

fn pillow_weights(
    source_size: usize,
    output_size: usize,
    destination: usize,
    filter: Resample,
) -> Vec<(usize, f64)> {
    let scale = source_size as f64 / output_size as f64;
    let filter_scale = scale.max(1.0);
    let support = match filter {
        Resample::Bilinear => 1.0 * filter_scale,
        Resample::Lanczos3 => 3.0 * filter_scale,
    };
    let center = (destination as f64 + 0.5) * scale;
    let first = (center - support).floor() as isize;
    let last = (center + support).ceil() as isize;
    let mut unnormalized = Vec::new();
    for source in first..=last {
        if !(0..source_size as isize).contains(&source) {
            continue;
        }
        let position = (source as f64 + 0.5 - center) / filter_scale;
        let weight = match filter {
            Resample::Bilinear => (1.0 - position.abs()).max(0.0),
            Resample::Lanczos3 => lanczos3(position),
        };
        if weight != 0.0 {
            unnormalized.push((source as usize, weight));
        }
    }
    let total: f64 = unnormalized.iter().map(|(_, value)| value).sum();
    unnormalized
        .into_iter()
        .map(|(index, value)| (index, value / total))
        .collect()
}

fn lanczos3(value: f64) -> f64 {
    if value == 0.0 {
        1.0
    } else if value.abs() >= 3.0 {
        0.0
    } else {
        let pi_value = core::f64::consts::PI * value;
        (pi_value.sin() / pi_value) * ((pi_value / 3.0).sin() / (pi_value / 3.0))
    }
}

pub(crate) fn sobel_nearest(input: &[f32], width: usize, height: usize, axis: usize) -> Vec<f32> {
    assert_eq!(input.len(), width * height);
    let kernel = if axis == 1 {
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    } else {
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    };
    correlate_nearest(input, width, height, &kernel)
}

pub(crate) fn convolve_nearest(
    input: &[f32],
    width: usize,
    height: usize,
    kernel: &[[f32; 3]; 3],
) -> Vec<f32> {
    convolve_square_nearest(
        input,
        width,
        height,
        &kernel.iter().flatten().copied().collect::<Vec<_>>(),
        3,
    )
}

pub(crate) fn convolve_square_nearest(
    input: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    size: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), width * height);
    assert!(size % 2 == 1);
    assert_eq!(kernel.len(), size * size);
    let radius = (size / 2) as isize;
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f64;
            for kernel_y in 0..size {
                for kernel_x in 0..size {
                    let sample_y = (y as isize + kernel_y as isize - radius)
                        .clamp(0, height as isize - 1) as usize;
                    let sample_x = (x as isize + kernel_x as isize - radius)
                        .clamp(0, width as isize - 1) as usize;
                    let coefficient = kernel[(size - 1 - kernel_y) * size + (size - 1 - kernel_x)];
                    sum += f64::from(input[sample_y * width + sample_x]) * f64::from(coefficient);
                }
            }
            output[y * width + x] = sum as f32;
        }
    }
    output
}

fn correlate_nearest(
    input: &[f32],
    width: usize,
    height: usize,
    kernel: &[[f32; 3]; 3],
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f64;
            for (ky, kernel_row) in kernel.iter().enumerate() {
                for (kx, coefficient) in kernel_row.iter().enumerate() {
                    let sample_y =
                        (y as isize + ky as isize - 1).clamp(0, height as isize - 1) as usize;
                    let sample_x =
                        (x as isize + kx as isize - 1).clamp(0, width as isize - 1) as usize;
                    sum += f64::from(input[sample_y * width + sample_x]) * f64::from(*coefficient);
                }
            }
            output[y * width + x] = sum as f32;
        }
    }
    output
}

pub(crate) fn uniform_filter_nearest(
    input: &[f32],
    width: usize,
    height: usize,
    size: usize,
) -> Vec<f32> {
    assert!(size % 2 == 1);
    // scipy.ndimage.uniform_filter visits axes in array order: Y, then X.
    // Each uniform_filter1d pass accumulates in f64 and stores an f32
    // intermediate before the following axis.
    let vertical = uniform_filter_axis(input, width, height, size, false);
    uniform_filter_axis(&vertical, width, height, size, true)
}

fn uniform_filter_axis(
    input: &[f32],
    width: usize,
    height: usize,
    size: usize,
    horizontal: bool,
) -> Vec<f32> {
    let radius = (size / 2) as isize;
    let mut output = vec![0.0; input.len()];
    if horizontal {
        for y in 0..height {
            let mut sum = 0.0_f64;
            for offset in -radius..=radius {
                let x = offset.clamp(0, width as isize - 1) as usize;
                sum += f64::from(input[y * width + x]);
            }
            output[y * width] = (sum / size as f64) as f32;
            for x in 1..width {
                let remove = (x as isize - radius - 1).clamp(0, width as isize - 1) as usize;
                let add = (x as isize + radius).clamp(0, width as isize - 1) as usize;
                sum += f64::from(input[y * width + add]) - f64::from(input[y * width + remove]);
                output[y * width + x] = (sum / size as f64) as f32;
            }
        }
        return output;
    }
    for x in 0..width {
        let mut sum = 0.0_f64;
        for offset in -radius..=radius {
            let y = offset.clamp(0, height as isize - 1) as usize;
            sum += f64::from(input[y * width + x]);
        }
        output[x] = (sum / size as f64) as f32;
        for y in 1..height {
            let remove = (y as isize - radius - 1).clamp(0, height as isize - 1) as usize;
            let add = (y as isize + radius).clamp(0, height as isize - 1) as usize;
            sum += f64::from(input[add * width + x]) - f64::from(input[remove * width + x]);
            output[y * width + x] = (sum / size as f64) as f32;
        }
    }
    output
}

pub(crate) fn maximum_filter_nearest(
    input: &[f32],
    width: usize,
    height: usize,
    size: usize,
) -> Vec<f32> {
    assert!(size % 2 == 1);
    let radius = (size / 2) as isize;
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut maximum = f32::NEG_INFINITY;
            for ky in -radius..=radius {
                for kx in -radius..=radius {
                    let yy = (y as isize + ky).clamp(0, height as isize - 1) as usize;
                    let xx = (x as isize + kx).clamp(0, width as isize - 1) as usize;
                    maximum = maximum.max(input[yy * width + xx]);
                }
            }
            output[y * width + x] = maximum;
        }
    }
    output
}

pub(crate) fn find_contours(
    input: &[f32],
    width: usize,
    height: usize,
) -> (Vec<[f32; 2]>, Vec<i32>) {
    assert_eq!(input.len(), width * height);
    let mut segments = Vec::<(ContourPoint, ContourPoint)>::new();
    for y in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            // skimage converts the plane to f64 before marching squares. Its
            // bit order is UL, UR, LL, LR and equality is classified low.
            let ul = f64::from(input[y * width + x]);
            let ur = f64::from(input[y * width + x + 1]);
            let ll = f64::from(input[(y + 1) * width + x]);
            let lr = f64::from(input[(y + 1) * width + x + 1]);
            let mask = u8::from(ul > 0.5)
                | (u8::from(ur > 0.5) << 1)
                | (u8::from(ll > 0.5) << 2)
                | (u8::from(lr > 0.5) << 3);
            if mask == 0 || mask == 15 {
                continue;
            }
            let fraction = |from: f64, to: f64| {
                if to == from {
                    0.0
                } else {
                    (0.5 - from) / (to - from)
                }
            };
            let top = ContourPoint(y as f64, x as f64 + fraction(ul, ur));
            let bottom = ContourPoint(y as f64 + 1.0, x as f64 + fraction(ll, lr));
            let left = ContourPoint(y as f64 + fraction(ul, ll), x as f64);
            let right = ContourPoint(y as f64 + fraction(ur, lr), x as f64 + 1.0);
            // Segment directions keep low-valued samples on the left. Cases
            // 6/9 use fully_connected="high".
            let pairs: &[(ContourPoint, ContourPoint)] = match mask {
                1 => &[(top, left)],
                2 => &[(right, top)],
                3 => &[(right, left)],
                4 => &[(left, bottom)],
                5 => &[(top, bottom)],
                6 => &[(left, top), (right, bottom)],
                7 => &[(right, bottom)],
                8 => &[(bottom, right)],
                9 => &[(top, right), (bottom, left)],
                10 => &[(bottom, top)],
                11 => &[(bottom, left)],
                12 => &[(left, right)],
                13 => &[(top, right)],
                14 => &[(left, top)],
                _ => unreachable!(),
            };
            segments.extend_from_slice(pairs);
        }
    }
    stitch_contours(segments)
}

#[derive(Clone, Copy, Debug)]
struct ContourPoint(f64, f64);

impl PartialEq for ContourPoint {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits() && self.1.to_bits() == other.1.to_bits()
    }
}

impl Eq for ContourPoint {}

impl Hash for ContourPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
        self.1.to_bits().hash(state);
    }
}

fn stitch_contours(segments: Vec<(ContourPoint, ContourPoint)>) -> (Vec<[f32; 2]>, Vec<i32>) {
    let mut contours: Vec<Option<VecDeque<ContourPoint>>> = Vec::new();
    let mut starts: HashMap<ContourPoint, usize> = HashMap::new();
    let mut ends: HashMap<ContourPoint, usize> = HashMap::new();
    for (from, to) in segments {
        if from == to {
            continue;
        }
        let tail_id = starts.remove(&to);
        let head_id = ends.remove(&from);
        match (tail_id, head_id) {
            (Some(tail_id), Some(head_id)) if tail_id == head_id => {
                contours[head_id].as_mut().unwrap().push_back(to);
            }
            (Some(tail_id), Some(head_id)) if tail_id > head_id => {
                let tail = contours[tail_id].take().unwrap();
                let head = contours[head_id].as_mut().unwrap();
                head.extend(tail);
                starts.insert(*head.front().unwrap(), head_id);
                ends.insert(*head.back().unwrap(), head_id);
            }
            (Some(tail_id), Some(head_id)) => {
                let head = contours[head_id].take().unwrap();
                starts.remove(head.front().unwrap());
                let tail = contours[tail_id].as_mut().unwrap();
                for point in head.into_iter().rev() {
                    tail.push_front(point);
                }
                starts.insert(*tail.front().unwrap(), tail_id);
                ends.insert(*tail.back().unwrap(), tail_id);
            }
            (None, None) => {
                let id = contours.len();
                contours.push(Some(VecDeque::from([from, to])));
                starts.insert(from, id);
                ends.insert(to, id);
            }
            (Some(tail_id), None) => {
                let tail = contours[tail_id].as_mut().unwrap();
                tail.push_front(from);
                starts.insert(from, tail_id);
            }
            (None, Some(head_id)) => {
                let head = contours[head_id].as_mut().unwrap();
                head.push_back(to);
                ends.insert(to, head_id);
            }
        }
    }
    let mut points = Vec::new();
    let mut lengths = Vec::new();
    for contour in contours.into_iter().flatten() {
        lengths.push(contour.len() as i32);
        // positive_orientation="high" reverses every assembled contour.
        points.extend(
            contour
                .into_iter()
                .rev()
                .map(|point| [point.0 as f32, point.1 as f32]),
        );
    }
    (points, lengths)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Segment {
    pub start: [f32; 2],
    pub end: [f32; 2],
}

pub(crate) type SegmentQuery = (Vec<f32>, Vec<i64>, Vec<i64>, Vec<[f32; 2]>, Vec<f32>);

pub(crate) fn nearest_segments(
    segments: &[Segment],
    queries: &[[f32; 2]],
    k: usize,
) -> SegmentQuery {
    assert!(!segments.is_empty());
    assert!(k > 0);
    let mut candidate_distances = Vec::new();
    let mut candidate_indices = Vec::new();
    let mut exact_indices = Vec::new();
    let mut closest = Vec::new();
    let mut exact_distances = Vec::new();
    for &query in queries {
        let candidates = midpoint_neighbors(segments, query, k);
        for &(distance, index) in &candidates {
            candidate_distances.push(distance as f32);
            candidate_indices.push(index as i64);
        }
        let mut best = None;
        for &(_, index) in &candidates {
            let segment = segments[index];
            let dy = segment.end[0] - segment.start[0];
            let dx = segment.end[1] - segment.start[1];
            let denominator = dy * dy + dx * dx;
            let projection = ((query[0] - segment.start[0]) * dy
                + (query[1] - segment.start[1]) * dx)
                / denominator.max(1e-12);
            let projection = projection.clamp(0.0, 1.0);
            let point = [
                segment.start[0] + projection * dy,
                segment.start[1] + projection * dx,
            ];
            let offset_y = query[0] - point[0];
            let offset_x = query[1] - point[1];
            let squared_distance = offset_y * offset_y + offset_x * offset_x;
            if best
                .as_ref()
                .is_none_or(|(best_squared, _, _)| squared_distance < *best_squared)
            {
                best = Some((squared_distance, index, point));
            }
        }
        let (squared_distance, index, point) = best.unwrap();
        exact_indices.push(index as i64);
        closest.push(point);
        exact_distances.push(squared_distance.sqrt());
    }
    (
        candidate_distances,
        candidate_indices,
        exact_indices,
        closest,
        exact_distances,
    )
}

#[derive(Clone, Copy)]
struct HeapItem {
    priority: f64,
    index: usize,
}

fn heap_push(heap: &mut Vec<HeapItem>, item: HeapItem) {
    heap.push(item);
    let mut child = heap.len() - 1;
    while child > 0 {
        let parent = (child - 1) / 2;
        if heap[child].priority >= heap[parent].priority {
            break;
        }
        heap.swap(child, parent);
        child = parent;
    }
}

fn heap_pop(heap: &mut Vec<HeapItem>) -> HeapItem {
    let output = heap[0];
    let last = heap.pop().unwrap();
    if heap.is_empty() {
        return output;
    }
    heap[0] = last;
    let mut parent = 0;
    loop {
        let left = parent * 2 + 1;
        let right = left + 1;
        let left_smaller = left < heap.len() && heap[parent].priority > heap[left].priority;
        let right_smaller = right < heap.len() && heap[parent].priority > heap[right].priority;
        if !left_smaller && !right_smaller {
            break;
        }
        let child = if right < heap.len() && heap[left].priority > heap[right].priority {
            right
        } else {
            left
        };
        heap.swap(parent, child);
        parent = child;
    }
    output
}

fn midpoint_neighbors(segments: &[Segment], query: [f32; 2], k: usize) -> Vec<(f64, usize)> {
    let capacity = k.min(segments.len());
    let mut heap = Vec::with_capacity(capacity);
    let mut upper_bound = f64::INFINITY;
    for (index, segment) in segments.iter().enumerate() {
        // NumPy forms the midpoint in f32 before cKDTree stores it as f64.
        let midpoint_y = (segment.start[0] + segment.end[0]) * 0.5;
        let midpoint_x = (segment.start[1] + segment.end[1]) * 0.5;
        let dy = f64::from(midpoint_y) - f64::from(query[0]);
        let dx = f64::from(midpoint_x) - f64::from(query[1]);
        let distance_squared = dy * dy + dx * dx;
        if distance_squared < upper_bound {
            if heap.len() == capacity {
                heap_pop(&mut heap);
            }
            heap_push(
                &mut heap,
                HeapItem {
                    priority: -distance_squared,
                    index,
                },
            );
            if heap.len() == capacity {
                upper_bound = -heap[0].priority;
            }
        }
    }
    // cKDTree heap-sorts by popping farthest-first into the result backwards.
    let mut output = vec![(0.0, 0); heap.len()];
    for target in (0..output.len()).rev() {
        let item = heap_pop(&mut heap);
        output[target] = ((-item.priority).sqrt(), item.index);
    }
    output
}
