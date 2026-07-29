use bridge_kv_gqa::PagedKvCache;

use crate::error::Result;
use crate::gemv::{validate_finite_slice, validate_finite_value};
use crate::KernelError;

#[derive(Debug, Clone, Copy)]
pub struct AttentionInput<'a> {
    pub token_count: usize,
    pub query_head_count: usize,
    pub queries: &'a [f32],
    pub keys: &'a [f32],
    pub values: &'a [f32],
}

pub struct AttentionScratch<'a> {
    pub scores: &'a mut [f32],
    pub output: &'a mut [f32],
}

impl<'a> AttentionScratch<'a> {
    pub fn new(scores: &'a mut [f32], output: &'a mut [f32]) -> Self {
        Self { scores, output }
    }
}

/// Computes causal grouped-query attention for a token batch, then atomically
/// appends the already-normalized and RoPE-rotated K plus V rows to the cache.
pub fn causal_gqa_attention_into(
    cache: &mut PagedKvCache,
    layer: usize,
    input: AttentionInput<'_>,
    output: &mut [f32],
    scratch: &mut AttentionScratch<'_>,
) -> Result<()> {
    let base_token = validate_call(cache, layer, input, output, scratch)?;
    let key_dimension = cache.key_dimension();
    let value_dimension = cache.value_dimension();
    let kv_head_count = cache.kv_head_count();
    let queries_per_kv_head = input.query_head_count / kv_head_count;
    let output_len = output.len();
    let scratch_output = &mut scratch.output[..output_len];
    let attention_scale = 1.0_f32 / (key_dimension as f32).sqrt();

    for token in 0..input.token_count {
        let allowed_tokens = base_token + token + 1;
        for query_head in 0..input.query_head_count {
            let kv_head = query_head / queries_per_kv_head;
            let query_start = (token * input.query_head_count + query_head) * key_dimension;
            let query = &input.queries[query_start..query_start + key_dimension];
            let scores = &mut scratch.scores[..allowed_tokens];

            for (source_token, score) in scores.iter_mut().enumerate() {
                let key = source_key(cache, layer, input, base_token, source_token, kv_head)?;
                let mut dot = 0.0_f32;
                for (&query_value, &key_value) in query.iter().zip(key) {
                    dot += query_value * key_value;
                }
                *score = dot * attention_scale;
                validate_finite_value("attention score", source_token, *score)?;
            }

            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0_f32;
            for score in scores.iter_mut() {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            validate_finite_value("attention softmax denominator", 0, denominator)?;

            for value_lane in 0..value_dimension {
                let mut attended = 0.0_f32;
                for (source_token, &unnormalized) in scores.iter().enumerate() {
                    let value = source_value(cache, layer, input, base_token, source_token, kv_head)?;
                    attended += (unnormalized / denominator) * value[value_lane];
                }
                let destination =
                    (token * input.query_head_count + query_head) * value_dimension + value_lane;
                validate_finite_value("attention output", destination, attended)?;
                scratch_output[destination] = attended;
            }
        }
    }

    cache.append_tokens(layer, input.token_count, input.keys, input.values)?;
    output.copy_from_slice(scratch_output);
    Ok(())
}

fn validate_call(
    cache: &PagedKvCache,
    layer: usize,
    input: AttentionInput<'_>,
    output: &[f32],
    scratch: &AttentionScratch<'_>,
) -> Result<usize> {
    let base_token = cache.stored_tokens(layer)?;
    if input.token_count == 0 {
        return Err(KernelError::InvalidParameter {
            field: "attention token_count",
            reason: "must be greater than zero",
        });
    }
    if input.query_head_count == 0 || input.query_head_count % cache.kv_head_count() != 0 {
        return Err(KernelError::InvalidParameter {
            field: "attention query heads",
            reason: "must be a positive multiple of KV heads",
        });
    }
    let expected_queries = checked_product(
        input.token_count,
        input.query_head_count,
        cache.key_dimension(),
        "attention query length",
    )?;
    let expected_keys = checked_product(
        input.token_count,
        cache.kv_head_count(),
        cache.key_dimension(),
        "attention key length",
    )?;
    let expected_values = checked_product(
        input.token_count,
        cache.kv_head_count(),
        cache.value_dimension(),
        "attention value length",
    )?;
    let expected_output = checked_product(
        input.token_count,
        input.query_head_count,
        cache.value_dimension(),
        "attention output length",
    )?;
    require_length("attention queries", expected_queries, input.queries.len())?;
    require_length("attention keys", expected_keys, input.keys.len())?;
    require_length("attention values", expected_values, input.values.len())?;
    require_length("attention output", expected_output, output.len())?;
    validate_finite_slice("attention queries", input.queries)?;
    validate_finite_slice("attention keys", input.keys)?;
    validate_finite_slice("attention values", input.values)?;

    let total_tokens = base_token
        .checked_add(input.token_count)
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "attention total token count",
        })?;
    if total_tokens > cache.token_capacity() {
        return Err(bridge_kv_gqa::KvError::CapacityExhausted {
            layer,
            stored: base_token,
            additional: input.token_count,
            capacity: cache.token_capacity(),
        }
        .into());
    }
    if scratch.scores.len() < total_tokens {
        return Err(KernelError::ScratchTooSmall {
            field: "attention scores",
            required: total_tokens,
            actual: scratch.scores.len(),
        });
    }
    if scratch.output.len() < expected_output {
        return Err(KernelError::ScratchTooSmall {
            field: "attention output",
            required: expected_output,
            actual: scratch.output.len(),
        });
    }
    Ok(base_token)
}

fn source_key<'a>(
    cache: &'a PagedKvCache,
    layer: usize,
    input: AttentionInput<'a>,
    base_token: usize,
    source_token: usize,
    kv_head: usize,
) -> Result<&'a [f32]> {
    if source_token < base_token {
        Ok(cache.key(layer, source_token, kv_head)?)
    } else {
        let local_token = source_token - base_token;
        let start = (local_token * cache.kv_head_count() + kv_head) * cache.key_dimension();
        Ok(&input.keys[start..start + cache.key_dimension()])
    }
}

fn source_value<'a>(
    cache: &'a PagedKvCache,
    layer: usize,
    input: AttentionInput<'a>,
    base_token: usize,
    source_token: usize,
    kv_head: usize,
) -> Result<&'a [f32]> {
    if source_token < base_token {
        Ok(cache.value(layer, source_token, kv_head)?)
    } else {
        let local_token = source_token - base_token;
        let start = (local_token * cache.kv_head_count() + kv_head) * cache.value_dimension();
        Ok(&input.values[start..start + cache.value_dimension()])
    }
}

fn checked_product(left: usize, middle: usize, right: usize, operation: &'static str) -> Result<usize> {
    left.checked_mul(middle)
        .and_then(|value| value.checked_mul(right))
        .ok_or(KernelError::ArithmeticOverflow { operation })
}

fn require_length(field: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(KernelError::DimensionMismatch {
            field,
            expected,
            actual,
        })
    }
}
