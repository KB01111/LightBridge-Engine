use bridge_kernels_reference::{apply_neox_yarn_rope_in_place, Hy3RopeParams, KernelError};
use bridge_model_hy3::Hy3Profile;

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 2.0e-6, "{actual} != {expected}");
}

fn plain_params() -> Hy3RopeParams {
    Hy3RopeParams {
        head_dimension: 4,
        context_length: 1024,
        original_context_length: 1024,
        frequency_base: 10_000.0,
        frequency_scale: 1.0,
        extension_factor: 0.0,
        attention_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    }
}

#[test]
fn neox_rotation_pairs_first_and_second_head_halves() {
    let mut values = [1.0_f32, 2.0, 3.0, 4.0];
    apply_neox_yarn_rope_in_place(&mut values, 1, 1, plain_params()).unwrap();

    let theta0 = 1.0_f32;
    let theta1 = 0.01_f32;
    let expected = [
        1.0 * theta0.cos() - 3.0 * theta0.sin(),
        2.0 * theta1.cos() - 4.0 * theta1.sin(),
        1.0 * theta0.sin() + 3.0 * theta0.cos(),
        2.0 * theta1.sin() + 4.0 * theta1.cos(),
    ];
    for (actual, expected) in values.into_iter().zip(expected) {
        close(actual, expected);
    }
}

#[test]
fn selected_hy3_params_match_the_pinned_yarn_defaults() {
    let config = Hy3Profile::selected_iq2_m();
    let params = Hy3RopeParams::from_config(config.config()).unwrap();
    assert_eq!(params.head_dimension, 128);
    assert_eq!(params.context_length, 1_048_576);
    assert_eq!(params.original_context_length, 262_144);
    assert_eq!(params.frequency_base, 11_158_840.0);
    assert_eq!(params.frequency_scale, 0.25);
    assert_eq!(params.extension_factor, 1.0);
    assert_eq!(params.attention_factor, 1.0);
    assert_eq!(params.beta_fast, 32.0);
    assert_eq!(params.beta_slow, 1.0);
}

#[test]
fn selected_yarn_applies_to_each_head_at_original_and_scaled_positions() {
    let params = Hy3RopeParams::from_config(Hy3Profile::selected_iq2_m().config()).unwrap();
    let mut at_zero = vec![1.0_f32; 256];
    apply_neox_yarn_rope_in_place(&mut at_zero, 2, 0, params).unwrap();
    let magnitude = 1.0_f32 + 0.1 * 4.0_f32.ln();
    for value in at_zero {
        close(value, magnitude);
    }

    let head: Vec<f32> = (0..128).map(|index| index as f32 / 129.0 - 0.5).collect();
    let input: Vec<f32> = head.iter().chain(&head).copied().collect();
    let mut at_boundary = input.clone();
    apply_neox_yarn_rope_in_place(&mut at_boundary, 2, params.original_context_length, params).unwrap();
    assert_ne!(at_boundary, input);
    assert_eq!(
        at_boundary[..128]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        at_boundary[128..]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut at_scaled = input;
    apply_neox_yarn_rope_in_place(&mut at_scaled, 2, params.context_length - 1, params).unwrap();
    assert!(at_scaled.iter().all(|value| value.is_finite()));
}

#[test]
fn rope_validation_is_atomic() {
    let sentinel = [f32::from_bits(0x7fc0_00a5), 1.0, 2.0, f32::from_bits(0x8000_0000)];
    let mut values = sentinel;
    assert!(matches!(
        apply_neox_yarn_rope_in_place(&mut values, 1, 1024, plain_params()),
        Err(KernelError::PositionOutOfRange {
            position: 1024,
            context_length: 1024,
        })
    ));
    assert_eq!(values.map(f32::to_bits), sentinel.map(f32::to_bits));

    let mut values = sentinel;
    assert!(matches!(
        apply_neox_yarn_rope_in_place(&mut values, 2, 0, plain_params()),
        Err(KernelError::DimensionMismatch {
            field: "RoPE values",
            expected: 8,
            actual: 4,
        })
    ));
    assert_eq!(values.map(f32::to_bits), sentinel.map(f32::to_bits));

    let mut nonfinite = [1.0_f32, f32::NAN, 2.0, 3.0];
    let before = nonfinite.map(f32::to_bits);
    assert!(matches!(
        apply_neox_yarn_rope_in_place(&mut nonfinite, 1, 0, plain_params()),
        Err(KernelError::NonFiniteValue {
            field: "RoPE input",
            index: 1,
            ..
        })
    ));
    assert_eq!(nonfinite.map(f32::to_bits), before);

    let mut overflow = [f32::MAX; 4];
    let before = overflow.map(f32::to_bits);
    assert!(apply_neox_yarn_rope_in_place(&mut overflow, 1, 1, plain_params()).is_err());
    assert_eq!(overflow.map(f32::to_bits), before);
}
