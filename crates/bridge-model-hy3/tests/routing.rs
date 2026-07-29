use bridge_model_hy3::{route_experts_into, Hy3Error, RouteCandidate, RoutedExpert};

fn candidate_sentinel() -> RouteCandidate {
    RouteCandidate {
        expert_id: u32::MAX,
        selection_score: f32::from_bits(0x7fc0_00a5),
        unbiased_weight: f32::from_bits(0xffc0_00b6),
    }
}

fn selected_sentinel() -> RoutedExpert {
    RoutedExpert {
        expert_id: u32::MAX,
        coefficient: f32::from_bits(0x7fc0_00c7),
    }
}

fn bits_candidates(values: &[RouteCandidate]) -> Vec<(u32, u32, u32)> {
    values
        .iter()
        .map(|value| {
            (
                value.expert_id,
                value.selection_score.to_bits(),
                value.unbiased_weight.to_bits(),
            )
        })
        .collect()
}

fn bits_selected(values: &[RoutedExpert]) -> Vec<(u32, u32)> {
    values
        .iter()
        .map(|value| (value.expert_id, value.coefficient.to_bits()))
        .collect()
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn equal_scores_use_ascending_ids_and_output_is_accumulation_sorted() {
    let logits = [0.0_f32; 4];
    let bias = [0.0_f32; 4];
    let mut candidates = [candidate_sentinel(); 4];
    let mut selected = [selected_sentinel(); 3];

    route_experts_into(&logits, &bias, 3, 2.826, &mut candidates, &mut selected).unwrap();

    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.expert_id)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        selected.iter().map(|expert| expert.expert_id).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    for expert in selected {
        assert_close(expert.coefficient, 2.826 / 3.0);
    }
}

#[test]
fn bias_changes_selection_but_never_routed_weights() {
    let logits = [3.0_f32, 1.0, -1.0, -3.0];
    let bias = [0.0_f32, 0.0, 2.0, 4.0];
    let mut candidates = [candidate_sentinel(); 4];
    let mut selected = [selected_sentinel(); 2];

    route_experts_into(&logits, &bias, 2, 2.0, &mut candidates, &mut selected).unwrap();
    assert_eq!(
        selected.iter().map(|expert| expert.expert_id).collect::<Vec<_>>(),
        [2, 3]
    );

    let weight2 = 1.0_f32 / (1.0 + 1.0_f32.exp());
    let weight3 = 1.0_f32 / (1.0 + 3.0_f32.exp());
    assert_close(selected[0].coefficient, weight2 / (weight2 + weight3) * 2.0);
    assert_close(selected[1].coefficient, weight3 / (weight2 + weight3) * 2.0);
}

#[test]
fn selected_output_is_sorted_even_when_score_order_is_not() {
    let logits = [0.0_f32, 4.0, -4.0, 2.0];
    let bias = [0.0_f32; 4];
    let mut candidates = [candidate_sentinel(); 4];
    let mut selected = [selected_sentinel(); 3];

    route_experts_into(&logits, &bias, 3, 1.0, &mut candidates, &mut selected).unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.expert_id)
            .collect::<Vec<_>>(),
        [1, 3, 0, 2]
    );
    assert_eq!(
        selected.iter().map(|expert| expert.expert_id).collect::<Vec<_>>(),
        [0, 1, 3]
    );
}

#[test]
fn underflowed_sigmoids_use_the_exact_sum_floor() {
    let logits = [f32::MIN, f32::MIN];
    let bias = [0.0_f32; 2];
    let mut candidates = [candidate_sentinel(); 2];
    let mut selected = [selected_sentinel(); 2];

    route_experts_into(&logits, &bias, 2, 2.826, &mut candidates, &mut selected).unwrap();
    assert!(selected.iter().all(|expert| expert.coefficient == 0.0));
    assert!(selected.iter().all(|expert| expert.coefficient.is_finite()));
}

#[test]
fn every_invalid_input_is_rejected_before_output_mutation() {
    let valid_logits = [0.0_f32, 1.0, 2.0];
    let valid_bias = [0.0_f32; 3];

    type RoutingCase<'a> = (&'a [f32], &'a [f32], usize, f32, Hy3Error);
    let cases: Vec<RoutingCase<'_>> = vec![
        (
            &valid_logits,
            &valid_bias[..2],
            2,
            1.0,
            Hy3Error::RoutingLength {
                field: "selection_bias",
                expected: 3,
                actual: 2,
            },
        ),
        (
            &valid_logits,
            &valid_bias,
            0,
            1.0,
            Hy3Error::RoutingTopK {
                expert_count: 3,
                expert_used_count: 0,
            },
        ),
        (
            &valid_logits,
            &valid_bias,
            4,
            1.0,
            Hy3Error::RoutingTopK {
                expert_count: 3,
                expert_used_count: 4,
            },
        ),
    ];

    for (logits, bias, top_k, scale, expected) in cases {
        let mut candidates = [candidate_sentinel(); 3];
        let mut selected = [selected_sentinel(); 2];
        let candidates_before = bits_candidates(&candidates);
        let selected_before = bits_selected(&selected);
        let error =
            route_experts_into(logits, bias, top_k, scale, &mut candidates, &mut selected).unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        assert_eq!(bits_candidates(&candidates), candidates_before);
        assert_eq!(bits_selected(&selected), selected_before);
    }

    let nonfinite_cases = [
        ("logits", 1, f32::INFINITY),
        ("selection_bias", 2, f32::from_bits(0x7fc0_1234)),
        ("weight_scale", 0, f32::NEG_INFINITY),
    ];
    for (field, index, value) in nonfinite_cases {
        let mut logits = valid_logits;
        let mut bias = valid_bias;
        let mut scale = 1.0_f32;
        match field {
            "logits" => logits[index] = value,
            "selection_bias" => bias[index] = value,
            "weight_scale" => scale = value,
            _ => unreachable!(),
        }
        let mut candidates = [candidate_sentinel(); 3];
        let mut selected = [selected_sentinel(); 2];
        let candidates_before = bits_candidates(&candidates);
        let selected_before = bits_selected(&selected);
        assert!(matches!(
            route_experts_into(&logits, &bias, 2, scale, &mut candidates, &mut selected),
            Err(Hy3Error::NonFiniteRoutingValue {
                field: actual_field,
                index: actual_index,
                bits,
            }) if actual_field == field && actual_index == index && bits == value.to_bits()
        ));
        assert_eq!(bits_candidates(&candidates), candidates_before);
        assert_eq!(bits_selected(&selected), selected_before);
    }
}

#[test]
fn scratch_length_errors_are_atomic() {
    let logits = [0.0_f32; 3];
    let bias = [0.0_f32; 3];

    let mut short_candidates = [candidate_sentinel(); 2];
    let mut selected = [selected_sentinel(); 2];
    let selected_before = bits_selected(&selected);
    assert!(matches!(
        route_experts_into(&logits, &bias, 2, 1.0, &mut short_candidates, &mut selected),
        Err(Hy3Error::RoutingLength {
            field: "candidates",
            expected: 3,
            actual: 2,
        })
    ));
    assert_eq!(bits_selected(&selected), selected_before);

    let mut candidates = [candidate_sentinel(); 3];
    let mut short_selected = [selected_sentinel(); 1];
    let candidates_before = bits_candidates(&candidates);
    assert!(matches!(
        route_experts_into(&logits, &bias, 2, 1.0, &mut candidates, &mut short_selected),
        Err(Hy3Error::RoutingLength {
            field: "selected",
            expected: 2,
            actual: 1,
        })
    ));
    assert_eq!(bits_candidates(&candidates), candidates_before);
}
