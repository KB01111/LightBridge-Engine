use bridge_kernels_reference::{causal_softmax_into, softmax_into, KernelError};

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 1.0e-6, "{actual} != {expected}");
}

#[test]
fn stable_softmax_handles_large_finite_logits() {
    let mut output = [f32::NAN; 3];
    softmax_into(&[10_000.0, 9_999.0, 9_998.0], &mut output).unwrap();
    let denominator = 1.0_f32 + (-1.0_f32).exp() + (-2.0_f32).exp();
    close(output[0], 1.0 / denominator);
    close(output[1], (-1.0_f32).exp() / denominator);
    close(output[2], (-2.0_f32).exp() / denominator);
    close(output.iter().sum(), 1.0);
}

#[test]
fn causal_mask_zeros_the_suffix_and_all_masked_rows() {
    let logits = [3.0_f32, 2.0, 1000.0, f32::NAN];
    let mut output = [f32::NAN; 4];
    causal_softmax_into(&logits, 2, &mut output).unwrap();
    let denominator = 1.0 + (-1.0_f32).exp();
    close(output[0], 1.0 / denominator);
    close(output[1], (-1.0_f32).exp() / denominator);
    assert_eq!(output[2..], [0.0, 0.0]);

    causal_softmax_into(&logits, 0, &mut output).unwrap();
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn validation_errors_do_not_mutate_output() {
    let sentinel = [f32::from_bits(0x7fc0_00a5), f32::from_bits(0x8000_0000)];
    let mut output = sentinel;
    assert!(matches!(
        softmax_into(&[1.0], &mut output),
        Err(KernelError::DimensionMismatch {
            field: "softmax output",
            expected: 1,
            actual: 2,
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    assert!(matches!(
        causal_softmax_into(&[1.0, 2.0], 3, &mut output),
        Err(KernelError::DimensionMismatch {
            field: "softmax unmasked prefix",
            expected: 2,
            actual: 3,
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    assert!(matches!(
        softmax_into(&[1.0, f32::INFINITY], &mut output),
        Err(KernelError::NonFiniteValue {
            field: "softmax logits",
            index: 1,
            ..
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));
}
