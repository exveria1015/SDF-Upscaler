use std::collections::VecDeque;

const ZHANG_LUT: [u8; 256] = [
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
];

pub(crate) fn skeletonize_zhang(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    assert_eq!(mask.len(), width * height);
    let padded_width = width + 2;
    let padded_height = height + 2;
    let mut skeleton = vec![0_u8; padded_width * padded_height];
    for y in 0..height {
        for x in 0..width {
            skeleton[(y + 1) * padded_width + x + 1] = u8::from(mask[y * width + x]);
        }
    }
    let mut cleaned = skeleton.clone();
    loop {
        let mut removed = false;
        for first_pass in [true, false] {
            for y in 1..padded_height - 1 {
                for x in 1..padded_width - 1 {
                    let index = y * padded_width + x;
                    if skeleton[index] == 0 {
                        continue;
                    }
                    let neighborhood = skeleton[index - padded_width - 1]
                        + 2 * skeleton[index - padded_width]
                        + 4 * skeleton[index - padded_width + 1]
                        + 8 * skeleton[index + 1]
                        + 16 * skeleton[index + padded_width + 1]
                        + 32 * skeleton[index + padded_width]
                        + 64 * skeleton[index + padded_width - 1]
                        + 128 * skeleton[index - 1];
                    let class = ZHANG_LUT[neighborhood as usize];
                    if class == 3 || (class == 1 && first_pass) || (class == 2 && !first_pass) {
                        cleaned[index] = 0;
                        removed = true;
                    }
                }
            }
            skeleton.copy_from_slice(&cleaned);
        }
        if !removed {
            break;
        }
    }
    let mut output = vec![false; mask.len()];
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = skeleton[(y + 1) * padded_width + x + 1] != 0;
        }
    }
    output
}

pub(crate) fn label_8(mask: &[bool], width: usize, height: usize) -> (Vec<usize>, usize) {
    assert_eq!(mask.len(), width * height);
    let mut labels = vec![0_usize; mask.len()];
    let mut count = 0;
    let mut queue = VecDeque::new();
    for start in 0..mask.len() {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        count += 1;
        labels[start] = count;
        queue.push_back(start);
        while let Some(index) = queue.pop_front() {
            let x = index % width;
            let y = index / width;
            for neighbor_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbor = neighbor_y * width + neighbor_x;
                    if mask[neighbor] && labels[neighbor] == 0 {
                        labels[neighbor] = count;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    (labels, count)
}

pub(crate) fn distance_transform_edt(mask: &[bool], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(mask.len(), width * height);
    let feature = mask.iter().map(|&value| !value).collect::<Vec<_>>();
    assert!(feature.iter().any(|&value| value));
    let mut vertical_distance = vec![usize::MAX; mask.len()];
    for x in 0..width {
        let mut last = None;
        for y in 0..height {
            let index = y * width + x;
            if feature[index] {
                last = Some(y);
            }
            if let Some(source_y) = last {
                vertical_distance[index] = (y - source_y) * (y - source_y);
            }
        }
        last = None;
        for y in (0..height).rev() {
            let index = y * width + x;
            if feature[index] {
                last = Some(y);
            }
            if let Some(source_y) = last {
                let candidate = (source_y - y) * (source_y - y);
                if candidate < vertical_distance[index] {
                    vertical_distance[index] = candidate;
                }
            }
        }
    }
    let mut output = vec![0.0_f32; mask.len()];
    for y in 0..height {
        let squared = edt_row(&vertical_distance[y * width..(y + 1) * width]);
        for x in 0..width {
            output[y * width + x] = (squared[x] as f64).sqrt() as f32;
        }
    }
    output
}

fn edt_row(vertical: &[usize]) -> Vec<usize> {
    let valid = vertical
        .iter()
        .enumerate()
        .filter_map(|(x, &value)| (value != usize::MAX).then_some(x))
        .collect::<Vec<_>>();
    assert!(!valid.is_empty());
    let mut sites = vec![0_usize; valid.len()];
    let mut boundaries = vec![0.0_f64; valid.len() + 1];
    let mut envelope = 0_usize;
    sites[0] = valid[0];
    boundaries[0] = f64::NEG_INFINITY;
    boundaries[1] = f64::INFINITY;
    for &candidate in valid.iter().skip(1) {
        let mut intersection;
        loop {
            let active = sites[envelope];
            intersection = ((vertical[candidate] + candidate * candidate) as f64
                - (vertical[active] + active * active) as f64)
                / (2.0 * (candidate - active) as f64);
            if intersection > boundaries[envelope] || envelope == 0 {
                break;
            }
            envelope -= 1;
        }
        envelope += 1;
        sites[envelope] = candidate;
        boundaries[envelope] = intersection;
        boundaries[envelope + 1] = f64::INFINITY;
    }
    let count = envelope + 1;
    let mut squared = vec![0_usize; vertical.len()];
    let mut active = 0_usize;
    for (x, value) in squared.iter_mut().enumerate() {
        while active + 1 < count && boundaries[active + 1] < x as f64 {
            active += 1;
        }
        let site = sites[active];
        let delta = x.abs_diff(site);
        *value = delta * delta + vertical[site];
    }
    squared
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morphology_handles_cross_and_components() {
        let mask = [
            false, true, false, false, //
            true, true, true, false, //
            false, true, false, false, //
            false, false, false, true,
        ];
        let skeleton = skeletonize_zhang(&mask, 4, 4);
        assert!(skeleton.iter().any(|&value| value));
        let (_, count) = label_8(&mask, 4, 4);
        assert_eq!(count, 2);
        let distance = distance_transform_edt(&mask, 4, 4);
        assert_eq!(distance[0], 0.0);
        assert_eq!(distance[5], 2.0_f32.sqrt());
    }
}
