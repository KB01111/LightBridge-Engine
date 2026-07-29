use bridge_kernels_reference::{causal_gqa_attention_into, AttentionInput, AttentionScratch, KernelError};
use bridge_kv_gqa::{KvError, PagedKvCache};

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 1.0e-6, "{actual} != {expected}");
}

fn dense_attention(query: f32, keys: &[f32], values: &[f32]) -> f32 {
    let scores: Vec<f32> = keys.iter().map(|key| query * key).collect();
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = scores.iter().map(|score| (*score - maximum).exp()).collect();
    let denominator: f32 = exponentials.iter().sum();
    exponentials
        .iter()
        .zip(values)
        .map(|(score, value)| score / denominator * value)
        .sum()
}

#[test]
fn single_token_decode_maps_query_heads_to_shared_kv_heads() {
    let mut cache = PagedKvCache::new(1, 2, 2, 1, 2, 4).unwrap();
    let queries = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
    let keys = [1.0_f32, 0.0, 0.0, 1.0];
    let values = [10.0_f32, 20.0];
    let input = AttentionInput {
        token_count: 1,
        query_head_count: 4,
        queries: &queries,
        keys: &keys,
        values: &values,
    };
    let mut output = [f32::NAN; 4];
    let mut scores = [0.0; 4];
    let mut output_scratch = [0.0; 4];
    let mut scratch = AttentionScratch::new(&mut scores, &mut output_scratch);

    causal_gqa_attention_into(&mut cache, 0, input, &mut output, &mut scratch).unwrap();
    assert_eq!(output, [10.0, 10.0, 20.0, 20.0]);
    assert_eq!(cache.stored_tokens(0).unwrap(), 1);
}

#[test]
fn multi_token_prefill_matches_a_dense_causal_oracle_across_page_boundaries() {
    let mut cache = PagedKvCache::new(1, 1, 1, 1, 2, 5).unwrap();
    let queries = [1.0_f32, -1.0, 1.0, 0.0, -1.0, 1.0];
    let keys = [1.0_f32, 2.0, 3.0];
    let values = [10.0_f32, 20.0, 40.0];
    let input = AttentionInput {
        token_count: 3,
        query_head_count: 2,
        queries: &queries,
        keys: &keys,
        values: &values,
    };
    let mut output = [f32::NAN; 6];
    let mut scores = [0.0; 5];
    let mut output_scratch = [0.0; 6];
    let mut scratch = AttentionScratch::new(&mut scores, &mut output_scratch);

    causal_gqa_attention_into(&mut cache, 0, input, &mut output, &mut scratch).unwrap();
    for token in 0..3 {
        for head in 0..2 {
            let expected = dense_attention(queries[token * 2 + head], &keys[..=token], &values[..=token]);
            close(output[token * 2 + head], expected);
        }
    }
    assert_eq!(cache.stored_tokens(0).unwrap(), 3);
    assert_eq!(cache.key(0, 2, 0).unwrap(), [3.0]);
}

#[test]
fn decode_attends_to_existing_cache_and_then_appends() {
    let mut cache = PagedKvCache::new(1, 1, 1, 1, 2, 4).unwrap();
    cache.append_tokens(0, 2, &[1.0, 2.0], &[10.0, 20.0]).unwrap();
    let input = AttentionInput {
        token_count: 1,
        query_head_count: 1,
        queries: &[1.0],
        keys: &[3.0],
        values: &[40.0],
    };
    let mut output = [f32::NAN];
    let mut scores = [0.0; 4];
    let mut output_scratch = [0.0];
    let mut scratch = AttentionScratch::new(&mut scores, &mut output_scratch);

    causal_gqa_attention_into(&mut cache, 0, input, &mut output, &mut scratch).unwrap();
    close(
        output[0],
        dense_attention(1.0, &[1.0, 2.0, 3.0], &[10.0, 20.0, 40.0]),
    );
    assert_eq!(cache.stored_tokens(0).unwrap(), 3);
}

#[test]
fn selected_sixty_four_to_eight_head_ratio_is_generic() {
    let mut cache = PagedKvCache::new(1, 8, 1, 1, 1, 1).unwrap();
    let queries = [1.0_f32; 64];
    let keys: Vec<f32> = (0..8).map(|head| head as f32).collect();
    let values: Vec<f32> = (0..8).map(|head| head as f32 + 100.0).collect();
    let input = AttentionInput {
        token_count: 1,
        query_head_count: 64,
        queries: &queries,
        keys: &keys,
        values: &values,
    };
    let mut output = [0.0; 64];
    let mut scores = [0.0; 1];
    let mut output_scratch = [0.0; 64];
    let mut scratch = AttentionScratch::new(&mut scores, &mut output_scratch);
    causal_gqa_attention_into(&mut cache, 0, input, &mut output, &mut scratch).unwrap();

    for (head, value) in output.into_iter().enumerate() {
        assert_eq!(value, 100.0 + (head / 8) as f32);
    }
}

#[test]
fn validation_failures_mutate_neither_cache_nor_output() {
    let mut cache = PagedKvCache::new(1, 1, 1, 1, 2, 1).unwrap();
    let sentinel = [f32::from_bits(0x7fc0_00a5)];
    let valid = AttentionInput {
        token_count: 1,
        query_head_count: 1,
        queries: &[1.0],
        keys: &[1.0],
        values: &[1.0],
    };

    let mut output = sentinel;
    let mut no_scores = [];
    let mut output_scratch = [0.0];
    let mut scratch = AttentionScratch::new(&mut no_scores, &mut output_scratch);
    assert!(matches!(
        causal_gqa_attention_into(&mut cache, 0, valid, &mut output, &mut scratch),
        Err(KernelError::ScratchTooSmall {
            field: "attention scores",
            required: 1,
            actual: 0,
        })
    ));
    assert_eq!(cache.stored_tokens(0).unwrap(), 0);
    assert_eq!(output[0].to_bits(), sentinel[0].to_bits());

    let invalid = AttentionInput {
        values: &[f32::NAN],
        ..valid
    };
    let mut scores = [0.0; 1];
    let mut scratch = AttentionScratch::new(&mut scores, &mut output_scratch);
    assert!(matches!(
        causal_gqa_attention_into(&mut cache, 0, invalid, &mut output, &mut scratch),
        Err(KernelError::NonFiniteValue {
            field: "attention values",
            index: 0,
            ..
        })
    ));
    assert_eq!(cache.stored_tokens(0).unwrap(), 0);
    assert_eq!(output[0].to_bits(), sentinel[0].to_bits());

    cache.append(0, &[1.0], &[1.0]).unwrap();
    assert!(matches!(
        causal_gqa_attention_into(&mut cache, 0, valid, &mut output, &mut scratch),
        Err(KernelError::Kv(KvError::CapacityExhausted {
            layer: 0,
            stored: 1,
            additional: 1,
            capacity: 1,
        }))
    ));
    assert_eq!(cache.stored_tokens(0).unwrap(), 1);
    assert_eq!(output[0].to_bits(), sentinel[0].to_bits());
}
