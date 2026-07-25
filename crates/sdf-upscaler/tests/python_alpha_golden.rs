use sdf_upscaler::{
    AlphaInput, AlphaOptions, PlaneRef, QualityBoundary, quantize_alpha_u8,
    upscale_alpha_sdf_x4_experimental,
};

const WIDTH: usize = 22;
const HEIGHT: usize = 18;
const OUTPUT_WIDTH: usize = WIDTH * 4;
const OUTPUT_HEIGHT: usize = HEIGHT * 4;

#[test]
fn experimental_route_is_measured_against_python_golden() {
    let input = include_bytes!("../../../fixtures/parity/alpha/input_alpha_u8.raw");
    let expected = include_bytes!("../../../fixtures/parity/alpha/expected_alpha_u8.raw");
    let plane = PlaneRef::packed(WIDTH, HEIGHT, input).unwrap();
    let output =
        upscale_alpha_sdf_x4_experimental(AlphaInput::U8(plane), AlphaOptions::default()).unwrap();
    let actual = quantize_alpha_u8(output.alpha.as_slice());

    assert_eq!(output.alpha.width(), OUTPUT_WIDTH);
    assert_eq!(output.alpha.height(), OUTPUT_HEIGHT);
    assert_eq!(actual.len(), expected.len());
    assert_eq!(
        output.diagnostics.quality_boundary,
        QualityBoundary::ExperimentalNotPythonParity
    );

    let mut absolute_sum = 0_u64;
    let mut maximum = 0_u8;
    let mut changed = 0_usize;
    let mut topology_changes = 0_usize;
    for (&left, &right) in actual.iter().zip(expected.iter()) {
        let difference = left.abs_diff(right);
        absolute_sum += u64::from(difference);
        maximum = maximum.max(difference);
        changed += usize::from(difference != 0);
        topology_changes += usize::from((left >= 128) != (right >= 128));
    }
    let mae = absolute_sum as f64 / actual.len() as f64;
    let changed_fraction = changed as f64 / actual.len() as f64;
    eprintln!(
        "python golden gap: mae_u8={mae:.6}, max_abs_u8={maximum}, \
         changed_fraction={changed_fraction:.6}, topology_changes={topology_changes}"
    );

    // These are drift alarms for the experimental route, not parity claims.
    // The canonical gate remains exact byte equality and is deliberately not
    // asserted until Pillow/SciPy/skimage compatibility kernels are ported.
    assert!(mae <= 12.0);
    assert!(maximum <= 192);
    assert!(topology_changes <= actual.len() / 20);
}
