use bridge_kernels_reference::{
    residual_add_in_place, weighted_head_rms_norm_in_place, weighted_rms_norm_in_place,
    weighted_rms_norm_into, KernelError,
};

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 1.0e-6, "{actual} != {expected}");
}

#[test]
fn weighted_rms_norm_matches_the_scalar_definition_in_and_out_of_place() {
    let input = [3.0_f32, 4.0];
    let weight = [1.0_f32, 2.0];
    let epsilon = 1.0e-5_f32;
    let scale = 1.0 / ((25.0_f32 / 2.0) + epsilon).sqrt();
    let expected = [3.0 * scale, 8.0 * scale];

    let mut output = [f32::NAN; 2];
    weighted_rms_norm_into(&input, &weight, epsilon, &mut output).unwrap();
    close(output[0], expected[0]);
    close(output[1], expected[1]);

    let mut in_place = input;
    weighted_rms_norm_in_place(&mut in_place, &weight, epsilon).unwrap();
    assert_eq!(in_place.map(f32::to_bits), output.map(f32::to_bits));
}

#[test]
fn per_head_norm_restarts_reduction_for_every_head() {
    let mut values = [3.0_f32, 4.0, 0.0, 2.0];
    weighted_head_rms_norm_in_place(&mut values, &[1.0, 0.5], 2, 1.0e-5).unwrap();

    let scale0 = 1.0 / ((25.0_f32 / 2.0) + 1.0e-5).sqrt();
    let scale1 = 1.0 / ((4.0_f32 / 2.0) + 1.0e-5).sqrt();
    let expected = [3.0 * scale0, 2.0 * scale0, 0.0, scale1];
    for (actual, expected) in values.into_iter().zip(expected) {
        close(actual, expected);
    }
}

#[test]
fn residual_add_is_in_place_and_exact() {
    let mut values = [1.0_f32, -2.0, 3.5];
    residual_add_in_place(&mut values, &[2.0, 1.0, -0.5]).unwrap();
    assert_eq!(values, [3.0, -1.0, 3.0]);
}

#[test]
fn norm_and_residual_validation_are_atomic() {
    let sentinel = [f32::from_bits(0x7fc0_00a5), f32::from_bits(0x8000_0000)];
    let mut output = sentinel;
    assert!(matches!(
        weighted_rms_norm_into(&[1.0, 2.0], &[1.0], 1.0e-5, &mut output),
        Err(KernelError::DimensionMismatch {
            field: "RMSNorm weight",
            expected: 2,
            actual: 1,
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    assert!(matches!(
        weighted_rms_norm_into(&[1.0, f32::NAN], &[1.0, 1.0], 1.0e-5, &mut output),
        Err(KernelError::NonFiniteValue {
            field: "RMSNorm input",
            index: 1,
            ..
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    let mut values = [1.0_f32, 0.0];
    let before = values.map(f32::to_bits);
    assert!(weighted_rms_norm_in_place(&mut values, &[f32::MAX, f32::MAX], 1.0e-5).is_err());
    assert_eq!(values.map(f32::to_bits), before);

    let mut residual = [f32::MAX, 1.0];
    let before = residual.map(f32::to_bits);
    assert!(residual_add_in_place(&mut residual, &[f32::MAX, 2.0]).is_err());
    assert_eq!(residual.map(f32::to_bits), before);
}
