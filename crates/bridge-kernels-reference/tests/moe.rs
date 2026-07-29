use bridge_core::ggml_type::GgmlType;
use bridge_kernels_reference::{
    expert_swiglu_accumulate_into, moe_selected_into, swiglu_project_into, KernelError, PackedMatrix,
    PayloadEndian, ReferenceExecutionMode, SelectedExpert, SwiGluExpert, SwiGluScratch,
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn matrix<'a>(ty: GgmlType, input: usize, output: usize, bytes: &'a [u8]) -> PackedMatrix<'a> {
    PackedMatrix::from_parts(ty, PayloadEndian::Little, input, output, bytes).unwrap()
}

fn scalar_expert<'a>(gate_bytes: &'a [u8], up_bytes: &'a [u8], down_bytes: &'a [u8]) -> SwiGluExpert<'a> {
    SwiGluExpert::new(
        matrix(GgmlType::F32, 1, 1, gate_bytes),
        matrix(GgmlType::F32, 1, 1, up_bytes),
        matrix(GgmlType::F32, 1, 1, down_bytes),
    )
    .unwrap()
}

fn scalar_expected(input: f32, gate: f32, up: f32, down: f32) -> f32 {
    let gate = gate * input;
    let up = up * input;
    down * (gate / (1.0 + (-gate).exp()) * up)
}

#[test]
fn swiglu_projects_and_accumulates_with_caller_owned_scratch() {
    let gate = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
    let up = f32_bytes(&[1.0, 1.0, 1.0, -1.0]);
    let down = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
    let expert = SwiGluExpert::new(
        matrix(GgmlType::F32, 2, 2, &gate),
        matrix(GgmlType::F32, 2, 2, &up),
        matrix(GgmlType::F32, 2, 2, &down),
    )
    .unwrap();
    let input = [1.0_f32, 2.0];
    let expected = [
        1.0 / (1.0 + (-1.0_f32).exp()) * 3.0,
        -(2.0 / (1.0 + (-2.0_f32).exp())),
    ];
    let mut activation = [0.0; 4];
    let mut decoded = [0.0; 256];
    let mut q8 = [];
    let mut scratch = SwiGluScratch::new(&mut activation, &mut decoded, &mut q8);
    let mut output = [f32::NAN; 2];

    swiglu_project_into(
        ReferenceExecutionMode::DequantF32,
        expert,
        &input,
        &mut output,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));

    let mut destination = [10.0_f32, -4.0];
    expert_swiglu_accumulate_into(
        ReferenceExecutionMode::DequantF32,
        expert,
        &input,
        &mut destination,
        0.5,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(
        destination.map(f32::to_bits),
        [10.0 + expected[0] * 0.5, -4.0 + expected[1] * 0.5].map(f32::to_bits)
    );
}

#[test]
fn mixed_iq2_gate_up_and_iq3_down_execute_in_both_modes() {
    let gate = vec![0_u8; 82 * 256];
    let up = vec![0_u8; 82 * 256];
    let down = vec![0_u8; 110];
    let expert = SwiGluExpert::new(
        matrix(GgmlType::IQ2_S, 256, 256, &gate),
        matrix(GgmlType::IQ2_S, 256, 256, &up),
        matrix(GgmlType::IQ3_S, 256, 1, &down),
    )
    .unwrap();
    let input = [0.25_f32; 256];

    for mode in [
        ReferenceExecutionMode::DequantF32,
        ReferenceExecutionMode::LlamaQ8K,
    ] {
        let mut output = [f32::NAN];
        let mut activation = [0.0; 512];
        let mut decoded = [0.0; 256];
        let mut q8 = [0_u8; 292];
        let mut scratch = SwiGluScratch::new(&mut activation, &mut decoded, &mut q8);
        swiglu_project_into(mode, expert, &input, &mut output, &mut scratch).unwrap();
        assert_eq!(output, [0.0]);
    }
}

#[test]
fn routed_and_shared_experts_accumulate_in_declared_semantics() {
    let gate_a = f32_bytes(&[1.0]);
    let up_a = f32_bytes(&[2.0]);
    let down_a = f32_bytes(&[3.0]);
    let gate_b = f32_bytes(&[-1.0]);
    let up_b = f32_bytes(&[0.5]);
    let down_b = f32_bytes(&[4.0]);
    let gate_shared = f32_bytes(&[0.25]);
    let up_shared = f32_bytes(&[3.0]);
    let down_shared = f32_bytes(&[-2.0]);
    let expert_a = scalar_expert(&gate_a, &up_a, &down_a);
    let expert_b = scalar_expert(&gate_b, &up_b, &down_b);
    let shared = scalar_expert(&gate_shared, &up_shared, &down_shared);
    let routed = [
        SelectedExpert {
            expert_id: 2,
            coefficient: 0.25,
            expert: expert_a,
        },
        SelectedExpert {
            expert_id: 7,
            coefficient: 0.75,
            expert: expert_b,
        },
    ];
    let input = [2.0_f32];
    let expected = 0.25 * scalar_expected(2.0, 1.0, 2.0, 3.0)
        + 0.75 * scalar_expected(2.0, -1.0, 0.5, 4.0)
        + scalar_expected(2.0, 0.25, 3.0, -2.0);
    let mut output = [f32::from_bits(0x7fc0_00a5)];
    let mut activation = [0.0; 2];
    let mut preflight = [0.0; 1];
    let mut decoded = [0.0; 256];
    let mut q8 = [];
    let mut scratch = SwiGluScratch::new(&mut activation, &mut decoded, &mut q8);

    moe_selected_into(
        ReferenceExecutionMode::DequantF32,
        &routed,
        shared,
        &input,
        &mut output,
        &mut preflight,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(output[0].to_bits(), expected.to_bits());
}

#[test]
fn moe_rejects_order_duplicates_and_bad_scratch_before_output_mutation() {
    let one = f32_bytes(&[1.0]);
    let expert = scalar_expert(&one, &one, &one);
    let sentinel = f32::from_bits(0x7fc0_00a5);
    let mut activation = [0.0; 2];
    let mut preflight = [0.0; 1];
    let mut decoded = [0.0; 256];
    let mut q8 = [];
    let mut scratch = SwiGluScratch::new(&mut activation, &mut decoded, &mut q8);

    let out_of_order = [
        SelectedExpert {
            expert_id: 3,
            coefficient: 0.5,
            expert,
        },
        SelectedExpert {
            expert_id: 2,
            coefficient: 0.5,
            expert,
        },
    ];
    let mut output = [sentinel];
    assert!(matches!(
        moe_selected_into(
            ReferenceExecutionMode::DequantF32,
            &out_of_order,
            expert,
            &[1.0],
            &mut output,
            &mut preflight,
            &mut scratch,
        ),
        Err(KernelError::RoutedExpertOrder {
            previous: 3,
            current: 2,
        })
    ));
    assert_eq!(output[0].to_bits(), sentinel.to_bits());

    let duplicate = [
        SelectedExpert {
            expert_id: 2,
            coefficient: 0.5,
            expert,
        },
        SelectedExpert {
            expert_id: 2,
            coefficient: 0.5,
            expert,
        },
    ];
    assert!(matches!(
        moe_selected_into(
            ReferenceExecutionMode::DequantF32,
            &duplicate,
            expert,
            &[1.0],
            &mut output,
            &mut preflight,
            &mut scratch,
        ),
        Err(KernelError::DuplicateRoutedExpert { expert_id: 2 })
    ));
    assert_eq!(output[0].to_bits(), sentinel.to_bits());

    let valid = [SelectedExpert {
        expert_id: 2,
        coefficient: 1.0,
        expert,
    }];
    assert!(matches!(
        moe_selected_into(
            ReferenceExecutionMode::DequantF32,
            &valid,
            expert,
            &[1.0],
            &mut output,
            &mut [],
            &mut scratch,
        ),
        Err(KernelError::ScratchTooSmall {
            field: "MoE preflight output",
            required: 1,
            actual: 0,
        })
    ));
    assert_eq!(output[0].to_bits(), sentinel.to_bits());
}
