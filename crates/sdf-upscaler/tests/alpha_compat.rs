#[path = "../src/compat/mod.rs"]
mod compat;

use compat::{
    Resample, Segment, convolve_nearest, convolve_square_nearest, find_contours,
    maximum_filter_nearest, nearest_segments, resize_f32, sobel_nearest, uniform_filter_nearest,
};

macro_rules! fixture {
    ($path:literal) => {
        include_bytes!(concat!(
            "../../../fixtures/parity/alpha_intermediates/",
            $path
        ))
    };
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|item| f32::from_le_bytes(item.try_into().unwrap()))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn pillow_f_mode_resampling_matches_frozen_fixture_bytes() {
    let source = f32s(fixture!("pillow/source_f32.raw"));
    let bilinear = resize_f32(&source, 3, 2, 7, 5, Resample::Bilinear);
    assert_eq!(
        f32_bytes(&bilinear),
        fixture!("pillow/bilinear_7x5_f32.raw")
    );
    let lanczos = resize_f32(&source, 3, 2, 7, 5, Resample::Lanczos3);
    assert_eq!(f32_bytes(&lanczos), fixture!("pillow/lanczos_7x5_f32.raw"));
}

#[test]
fn scipy_nearest_filters_match_frozen_fixture_bytes() {
    let source = f32s(fixture!("scipy/source_f32.raw"));
    assert_eq!(
        f32_bytes(&sobel_nearest(&source, 4, 3, 1)),
        fixture!("scipy/sobel_x_nearest_f32.raw")
    );
    assert_eq!(
        f32_bytes(&sobel_nearest(&source, 4, 3, 0)),
        fixture!("scipy/sobel_y_nearest_f32.raw")
    );
    let kernel = [
        [0.0, 1.0 / 6.0, 0.0],
        [1.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0],
        [0.0, 1.0 / 6.0, 0.0],
    ];
    assert_eq!(
        f32_bytes(&convolve_nearest(&source, 4, 3, &kernel)),
        fixture!("scipy/convolve_nearest_f32.raw")
    );
    assert_eq!(
        f32_bytes(&uniform_filter_nearest(&source, 4, 3, 3)),
        fixture!("scipy/uniform_3_nearest_f32.raw")
    );
}

#[test]
fn scipy_five_by_five_hard_edge_convolution_matches_stage_bytes() {
    let alpha = include_bytes!("../../../fixtures/parity/alpha/input_alpha_u8.raw")
        .iter()
        .map(|&value| f32::from(value) / 255.0)
        .collect::<Vec<_>>();
    let kernel = vec![1.0_f32 / 25.0; 25];
    let background = alpha
        .iter()
        .map(|&value| f32::from(value <= 0.05))
        .collect::<Vec<_>>();
    let foreground = alpha
        .iter()
        .map(|&value| f32::from(value >= 0.95))
        .collect::<Vec<_>>();
    assert_eq!(
        f32_bytes(&convolve_square_nearest(&background, 22, 18, &kernel, 5)),
        fixture!("filters/local_background_lr_f32.raw")
    );
    assert_eq!(
        f32_bytes(&convolve_square_nearest(&foreground, 22, 18, &kernel, 5)),
        fixture!("filters/local_foreground_lr_f32.raw")
    );
}

#[test]
fn skimage_micro_contours_match_frozen_fixture_bytes() {
    for (input, width, height, points, lengths) in [
        (
            fixture!("contours/ambiguous_input_f32.raw").as_slice(),
            2,
            2,
            fixture!("contours/ambiguous_points_f32.raw").as_slice(),
            fixture!("contours/ambiguous_lengths_i32.raw").as_slice(),
        ),
        (
            fixture!("contours/border_input_f32.raw").as_slice(),
            3,
            3,
            fixture!("contours/border_points_f32.raw").as_slice(),
            fixture!("contours/border_lengths_i32.raw").as_slice(),
        ),
    ] {
        let (actual_points, actual_lengths) = find_contours(&f32s(input), width, height);
        let flat: Vec<f32> = actual_points.into_iter().flatten().collect();
        assert_eq!(f32_bytes(&flat), points);
        assert_eq!(i32_bytes(&actual_lengths), lengths);
    }
}

#[test]
fn midpoint_k12_order_and_exact_segment_query_match_fixture_bytes() {
    let segments: Vec<_> = (0..12)
        .map(|index| Segment {
            start: [0.0, index as f32],
            end: [0.0, index as f32 + 1.0],
        })
        .collect();
    let values = f32s(fixture!("k12/query_points_f32.raw"));
    let queries: Vec<[f32; 2]> = values
        .chunks_exact(2)
        .map(|item| [item[0], item[1]])
        .collect();
    let (distances, tree, exact, closest, exact_distances) =
        nearest_segments(&segments, &queries, 12);
    assert_eq!(
        f32_bytes(&distances),
        fixture!("k12/tree_distances_f32.raw")
    );
    assert_eq!(i64_bytes(&tree), fixture!("k12/tree_indices_i64.raw"));
    assert_eq!(i64_bytes(&exact), fixture!("k12/exact_indices_i64.raw"));
    assert_eq!(
        f32_bytes(&closest.into_iter().flatten().collect::<Vec<_>>()),
        fixture!("k12/exact_closest_f32.raw")
    );
    assert_eq!(
        f32_bytes(&exact_distances),
        fixture!("k12/exact_distances_f32.raw")
    );
}

#[test]
fn general_filters_preserve_constants_and_dilate_maxima() {
    let constant = vec![0.375_f32; 5 * 4];
    assert_eq!(
        uniform_filter_nearest(&constant, 5, 4, 5),
        vec![0.375; 5 * 4]
    );
    let mut impulse = vec![0.0_f32; 5 * 5];
    impulse[2 * 5 + 2] = 1.0;
    let dilated = maximum_filter_nearest(&impulse, 5, 5, 3);
    assert_eq!(dilated.iter().filter(|&&value| value == 1.0).count(), 9);
}

#[test]
fn general_contours_close_a_filled_rectangle() {
    let input = [
        0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 1.0, 1.0, 0.0, //
        0.0, 1.0, 1.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let (points, lengths) = find_contours(&input, 5, 4);
    assert_eq!(lengths, vec![11]);
    assert_eq!(points.first(), points.last());
}

#[test]
fn general_segment_query_uses_exact_distance_after_midpoint_ranking() {
    let segments = [
        Segment {
            start: [0.0, 0.0],
            end: [0.0, 4.0],
        },
        Segment {
            start: [2.0, 2.0],
            end: [3.0, 2.0],
        },
        Segment {
            start: [5.0, 5.0],
            end: [6.0, 6.0],
        },
    ];
    let (_, _, indices, closest, distances) =
        nearest_segments(&segments, &[[1.0, 3.5]], segments.len());
    assert_eq!(indices, vec![0]);
    assert_eq!(closest, vec![[0.0, 3.5]]);
    assert_eq!(distances, vec![1.0]);
}
