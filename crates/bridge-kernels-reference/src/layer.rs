use bridge_kv_gqa::PagedKvCache;
use bridge_model_hy3::{route_experts_into, RouteCandidate, RoutedExpert};

use crate::error::Result;
use crate::{
    apply_neox_yarn_rope_in_place, causal_gqa_attention_into, gemv_into, moe_routed_by_id_into,
    moe_selected_into, residual_add_in_place, swiglu_project_into, weighted_head_rms_norm_in_place,
    weighted_rms_norm_into, AttentionInput, AttentionScratch, Hy3RopeParams, KernelError, PackedMatrix,
    ReferenceExecutionMode, RoutedMoeSelection, SelectedExpert, SwiGluExpert, SwiGluScratch,
};

#[derive(Debug, Clone, Copy)]
pub struct Hy3AttentionWeights<'a> {
    pub input_norm: &'a [f32],
    pub query: PackedMatrix<'a>,
    pub query_norm: &'a [f32],
    pub key: PackedMatrix<'a>,
    pub key_norm: &'a [f32],
    pub value: PackedMatrix<'a>,
    pub output: PackedMatrix<'a>,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub key_dimension: usize,
    pub value_dimension: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Hy3MoeWeights<'a> {
    pub router: PackedMatrix<'a>,
    pub selection_bias: &'a [f32],
    pub routed_experts: &'a [SwiGluExpert<'a>],
    pub shared_expert: SwiGluExpert<'a>,
    pub expert_used_count: usize,
    pub weight_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Hy3StreamingMoeWeights<'a> {
    pub attention: Hy3AttentionWeights<'a>,
    pub ffn_norm: &'a [f32],
    pub router: PackedMatrix<'a>,
    pub selection_bias: &'a [f32],
    pub shared_expert: SwiGluExpert<'a>,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub weight_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum Hy3FeedForwardWeights<'a> {
    Dense(SwiGluExpert<'a>),
    Moe(Hy3MoeWeights<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct Hy3BlockWeights<'a> {
    pub attention: Hy3AttentionWeights<'a>,
    pub ffn_norm: &'a [f32],
    pub feed_forward: Hy3FeedForwardWeights<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct Hy3BlockExecution {
    pub mode: ReferenceExecutionMode,
    pub layer: usize,
    pub position: u64,
    pub rope: Hy3RopeParams,
    pub rms_epsilon: f32,
}

/// Reusable caller-owned storage for one complete Hy3 token/block evaluation.
///
/// Construction performs all allocations. `hy3_block_forward_token` performs
/// no heap allocation after this workspace and the KV cache exist.
#[derive(Debug)]
pub struct Hy3BlockScratch {
    attention_normalized: Vec<f32>,
    queries: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    attention_context: Vec<f32>,
    attention_staging: Vec<f32>,
    attention_delta: Vec<f32>,
    attention_residual: Vec<f32>,
    ffn_normalized: Vec<f32>,
    ffn_delta: Vec<f32>,
    preflight_output: Vec<f32>,
    swiglu_activation: Vec<f32>,
    decoded_block: Vec<f32>,
    q8: Vec<u8>,
    scores: Vec<f32>,
    router_logits: Vec<f32>,
    route_candidates: Vec<RouteCandidate>,
    routed: Vec<RoutedExpert>,
    routed_len: usize,
    streaming_moe_pending: bool,
}

impl Hy3BlockScratch {
    pub fn new(block: Hy3BlockWeights<'_>, context_capacity: usize) -> Result<Self> {
        validate_block(block)?;
        if context_capacity == 0 {
            return Err(KernelError::InvalidParameter {
                field: "block scratch context capacity",
                reason: "must be greater than zero",
            });
        }

        let attention = block.attention;
        let hidden = attention.query.input_width();
        let query_values = checked_product(
            attention.query_head_count,
            attention.key_dimension,
            "query scratch length",
        )?;
        let key_values = checked_product(
            attention.kv_head_count,
            attention.key_dimension,
            "key scratch length",
        )?;
        let value_values = checked_product(
            attention.kv_head_count,
            attention.value_dimension,
            "value scratch length",
        )?;
        let attention_values = checked_product(
            attention.query_head_count,
            attention.value_dimension,
            "attention scratch length",
        )?;
        let (ffn_hidden, expert_count, expert_used_count) = match block.feed_forward {
            Hy3FeedForwardWeights::Dense(expert) => (expert.hidden_width(), 0, 0),
            Hy3FeedForwardWeights::Moe(moe) => (
                moe.shared_expert.hidden_width(),
                moe.routed_experts.len(),
                moe.expert_used_count,
            ),
        };
        let activation_values = ffn_hidden.checked_mul(2).ok_or(KernelError::ArithmeticOverflow {
            operation: "SwiGLU activation scratch",
        })?;
        let maximum_input = [
            hidden,
            attention_values,
            ffn_hidden,
            block.feed_forward.router_input_width(),
        ]
        .into_iter()
        .max()
        .unwrap_or(hidden);
        let q8_bytes = if maximum_input >= 256 {
            crate::required_q8_k_bytes(maximum_input)?
        } else {
            0
        };

        Ok(Self {
            attention_normalized: zeroed_f32(hidden, "attention normalized")?,
            queries: zeroed_f32(query_values, "queries")?,
            keys: zeroed_f32(key_values, "keys")?,
            values: zeroed_f32(value_values, "values")?,
            attention_context: zeroed_f32(attention_values, "attention context")?,
            attention_staging: zeroed_f32(attention_values, "attention staging")?,
            attention_delta: zeroed_f32(hidden, "attention delta")?,
            attention_residual: zeroed_f32(hidden, "attention residual")?,
            ffn_normalized: zeroed_f32(hidden, "FFN normalized")?,
            ffn_delta: zeroed_f32(hidden, "FFN delta")?,
            preflight_output: zeroed_f32(hidden, "MoE preflight output")?,
            swiglu_activation: zeroed_f32(activation_values, "SwiGLU activation")?,
            decoded_block: zeroed_f32(256, "decoded quant block")?,
            q8: zeroed_u8(q8_bytes, "Q8_K activation")?,
            scores: zeroed_f32(context_capacity, "attention scores")?,
            router_logits: zeroed_f32(expert_count, "router logits")?,
            route_candidates: route_candidates(expert_count)?,
            routed: routed_experts(expert_used_count)?,
            routed_len: 0,
            streaming_moe_pending: false,
        })
    }

    pub fn new_streaming_moe(block: Hy3StreamingMoeWeights<'_>, context_capacity: usize) -> Result<Self> {
        validate_streaming_moe(block)?;
        if context_capacity == 0 {
            return Err(KernelError::InvalidParameter {
                field: "block scratch context capacity",
                reason: "must be greater than zero",
            });
        }

        let attention = block.attention;
        let hidden = attention.query.input_width();
        let query_values = checked_product(
            attention.query_head_count,
            attention.key_dimension,
            "query scratch length",
        )?;
        let key_values = checked_product(
            attention.kv_head_count,
            attention.key_dimension,
            "key scratch length",
        )?;
        let value_values = checked_product(
            attention.kv_head_count,
            attention.value_dimension,
            "value scratch length",
        )?;
        let attention_values = checked_product(
            attention.query_head_count,
            attention.value_dimension,
            "attention scratch length",
        )?;
        let ffn_hidden = block.shared_expert.hidden_width();
        let activation_values = ffn_hidden.checked_mul(2).ok_or(KernelError::ArithmeticOverflow {
            operation: "SwiGLU activation scratch",
        })?;
        let maximum_input = [hidden, attention_values, ffn_hidden, block.router.input_width()]
            .into_iter()
            .max()
            .unwrap_or(hidden);
        let q8_bytes = if maximum_input >= 256 {
            crate::required_q8_k_bytes(maximum_input)?
        } else {
            0
        };

        Ok(Self {
            attention_normalized: zeroed_f32(hidden, "attention normalized")?,
            queries: zeroed_f32(query_values, "queries")?,
            keys: zeroed_f32(key_values, "keys")?,
            values: zeroed_f32(value_values, "values")?,
            attention_context: zeroed_f32(attention_values, "attention context")?,
            attention_staging: zeroed_f32(attention_values, "attention staging")?,
            attention_delta: zeroed_f32(hidden, "attention delta")?,
            attention_residual: zeroed_f32(hidden, "attention residual")?,
            ffn_normalized: zeroed_f32(hidden, "FFN normalized")?,
            ffn_delta: zeroed_f32(hidden, "FFN delta")?,
            preflight_output: zeroed_f32(hidden, "MoE preflight output")?,
            swiglu_activation: zeroed_f32(activation_values, "SwiGLU activation")?,
            decoded_block: zeroed_f32(256, "decoded quant block")?,
            q8: zeroed_u8(q8_bytes, "Q8_K activation")?,
            scores: zeroed_f32(context_capacity, "attention scores")?,
            router_logits: zeroed_f32(block.expert_count, "router logits")?,
            route_candidates: route_candidates(block.expert_count)?,
            routed: routed_experts(block.expert_used_count)?,
            routed_len: 0,
            streaming_moe_pending: false,
        })
    }

    pub fn attention_normalized(&self) -> &[f32] {
        &self.attention_normalized
    }

    pub fn queries(&self) -> &[f32] {
        &self.queries
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn attention_context(&self) -> &[f32] {
        &self.attention_context
    }

    pub fn attention_delta(&self) -> &[f32] {
        &self.attention_delta
    }

    pub fn attention_residual(&self) -> &[f32] {
        &self.attention_residual
    }

    pub fn ffn_normalized(&self) -> &[f32] {
        &self.ffn_normalized
    }

    pub fn ffn_delta(&self) -> &[f32] {
        &self.ffn_delta
    }

    pub fn routed(&self) -> &[RoutedExpert] {
        &self.routed[..self.routed_len]
    }

    /// Clears phase state so a caller can safely reuse this workspace after a
    /// cancelled or failed token evaluation.
    pub fn reset(&mut self) {
        self.routed_len = 0;
        self.streaming_moe_pending = false;
    }
}

impl Hy3FeedForwardWeights<'_> {
    fn router_input_width(self) -> usize {
        match self {
            Self::Dense(expert) => expert.input_width(),
            Self::Moe(moe) => moe.router.input_width(),
        }
    }
}

/// Evaluates one token through attention, causal GQA/KV append, residual, and
/// either the dense or routed/shared Hy3 feed-forward graph.
pub fn hy3_block_forward_token(
    execution: Hy3BlockExecution,
    block: Hy3BlockWeights<'_>,
    cache: &mut PagedKvCache,
    hidden_state: &mut [f32],
    scratch: &mut Hy3BlockScratch,
) -> Result<()> {
    validate_runtime(
        block,
        cache,
        execution.layer,
        execution.position,
        hidden_state,
        scratch,
    )?;
    scratch.routed_len = 0;
    scratch.streaming_moe_pending = false;
    let attention = block.attention;

    weighted_rms_norm_into(
        hidden_state,
        attention.input_norm,
        execution.rms_epsilon,
        &mut scratch.attention_normalized,
    )?;
    gemv_into(
        execution.mode,
        attention.query,
        &scratch.attention_normalized,
        &mut scratch.queries,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    gemv_into(
        execution.mode,
        attention.key,
        &scratch.attention_normalized,
        &mut scratch.keys,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    gemv_into(
        execution.mode,
        attention.value,
        &scratch.attention_normalized,
        &mut scratch.values,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    weighted_head_rms_norm_in_place(
        &mut scratch.queries,
        attention.query_norm,
        attention.key_dimension,
        execution.rms_epsilon,
    )?;
    weighted_head_rms_norm_in_place(
        &mut scratch.keys,
        attention.key_norm,
        attention.key_dimension,
        execution.rms_epsilon,
    )?;
    apply_neox_yarn_rope_in_place(
        &mut scratch.queries,
        attention.query_head_count,
        execution.position,
        execution.rope,
    )?;
    apply_neox_yarn_rope_in_place(
        &mut scratch.keys,
        attention.kv_head_count,
        execution.position,
        execution.rope,
    )?;

    let mut attention_scratch = AttentionScratch::new(&mut scratch.scores, &mut scratch.attention_staging);
    causal_gqa_attention_into(
        cache,
        execution.layer,
        AttentionInput {
            token_count: 1,
            query_head_count: attention.query_head_count,
            queries: &scratch.queries,
            keys: &scratch.keys,
            values: &scratch.values,
        },
        &mut scratch.attention_context,
        &mut attention_scratch,
    )?;
    gemv_into(
        execution.mode,
        attention.output,
        &scratch.attention_context,
        &mut scratch.attention_delta,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    residual_add_in_place(hidden_state, &scratch.attention_delta)?;
    scratch.attention_residual.copy_from_slice(hidden_state);

    weighted_rms_norm_into(
        hidden_state,
        block.ffn_norm,
        execution.rms_epsilon,
        &mut scratch.ffn_normalized,
    )?;
    let mut swiglu_scratch = SwiGluScratch::new(
        &mut scratch.swiglu_activation,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    );
    match block.feed_forward {
        Hy3FeedForwardWeights::Dense(expert) => {
            swiglu_project_into(
                execution.mode,
                expert,
                &scratch.ffn_normalized,
                &mut scratch.ffn_delta,
                &mut swiglu_scratch,
            )?;
        }
        Hy3FeedForwardWeights::Moe(moe) => {
            gemv_into(
                execution.mode,
                moe.router,
                &scratch.ffn_normalized,
                &mut scratch.router_logits,
                swiglu_scratch.decoded_block,
                swiglu_scratch.q8,
            )?;
            route_experts_into(
                &scratch.router_logits,
                moe.selection_bias,
                moe.expert_used_count,
                moe.weight_scale,
                &mut scratch.route_candidates,
                &mut scratch.routed,
            )
            .map_err(|error| KernelError::Routing {
                message: error.to_string(),
            })?;
            scratch.routed_len = moe.expert_used_count;
            moe_routed_by_id_into(
                execution.mode,
                RoutedMoeSelection {
                    routed: &scratch.routed[..scratch.routed_len],
                    experts: moe.routed_experts,
                    shared: moe.shared_expert,
                },
                &scratch.ffn_normalized,
                &mut scratch.ffn_delta,
                &mut scratch.preflight_output,
                &mut swiglu_scratch,
            )?;
        }
    }
    residual_add_in_place(hidden_state, &scratch.ffn_delta)
}

/// Runs a MoE block through attention and authoritative routing without
/// requiring all routed expert payloads to be resident.
///
/// The selected expert IDs and coefficients remain in [`Hy3BlockScratch::routed`].
/// The caller loads only those expert payloads and completes the block with
/// [`hy3_moe_finish_token`].
pub fn hy3_moe_route_token(
    execution: Hy3BlockExecution,
    block: Hy3StreamingMoeWeights<'_>,
    cache: &mut PagedKvCache,
    hidden_state: &mut [f32],
    scratch: &mut Hy3BlockScratch,
) -> Result<()> {
    validate_streaming_runtime(
        block,
        cache,
        execution.layer,
        execution.position,
        hidden_state,
        scratch,
    )?;
    if scratch.streaming_moe_pending {
        return Err(KernelError::InvalidParameter {
            field: "streaming MoE phase",
            reason: "the previous routed phase must be completed before routing again",
        });
    }
    scratch.routed_len = 0;
    let attention = block.attention;

    weighted_rms_norm_into(
        hidden_state,
        attention.input_norm,
        execution.rms_epsilon,
        &mut scratch.attention_normalized,
    )?;
    gemv_into(
        execution.mode,
        attention.query,
        &scratch.attention_normalized,
        &mut scratch.queries,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    gemv_into(
        execution.mode,
        attention.key,
        &scratch.attention_normalized,
        &mut scratch.keys,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    gemv_into(
        execution.mode,
        attention.value,
        &scratch.attention_normalized,
        &mut scratch.values,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    weighted_head_rms_norm_in_place(
        &mut scratch.queries,
        attention.query_norm,
        attention.key_dimension,
        execution.rms_epsilon,
    )?;
    weighted_head_rms_norm_in_place(
        &mut scratch.keys,
        attention.key_norm,
        attention.key_dimension,
        execution.rms_epsilon,
    )?;
    apply_neox_yarn_rope_in_place(
        &mut scratch.queries,
        attention.query_head_count,
        execution.position,
        execution.rope,
    )?;
    apply_neox_yarn_rope_in_place(
        &mut scratch.keys,
        attention.kv_head_count,
        execution.position,
        execution.rope,
    )?;

    let mut attention_scratch = AttentionScratch::new(&mut scratch.scores, &mut scratch.attention_staging);
    causal_gqa_attention_into(
        cache,
        execution.layer,
        AttentionInput {
            token_count: 1,
            query_head_count: attention.query_head_count,
            queries: &scratch.queries,
            keys: &scratch.keys,
            values: &scratch.values,
        },
        &mut scratch.attention_context,
        &mut attention_scratch,
    )?;
    gemv_into(
        execution.mode,
        attention.output,
        &scratch.attention_context,
        &mut scratch.attention_delta,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    residual_add_in_place(hidden_state, &scratch.attention_delta)?;
    scratch.attention_residual.copy_from_slice(hidden_state);
    weighted_rms_norm_into(
        hidden_state,
        block.ffn_norm,
        execution.rms_epsilon,
        &mut scratch.ffn_normalized,
    )?;

    gemv_into(
        execution.mode,
        block.router,
        &scratch.ffn_normalized,
        &mut scratch.router_logits,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    )?;
    route_experts_into(
        &scratch.router_logits,
        block.selection_bias,
        block.expert_used_count,
        block.weight_scale,
        &mut scratch.route_candidates,
        &mut scratch.routed,
    )
    .map_err(|error| KernelError::Routing {
        message: error.to_string(),
    })?;
    scratch.routed_len = block.expert_used_count;
    scratch.streaming_moe_pending = true;
    Ok(())
}

/// Completes a routed MoE block after the selected expert payloads have been
/// loaded. IDs and coefficients must exactly match the authoritative routing
/// result from [`hy3_moe_route_token`].
pub fn hy3_moe_finish_token(
    mode: ReferenceExecutionMode,
    selected: &[SelectedExpert<'_>],
    shared: SwiGluExpert<'_>,
    hidden_state: &mut [f32],
    scratch: &mut Hy3BlockScratch,
) -> Result<()> {
    if !scratch.streaming_moe_pending {
        return Err(KernelError::InvalidParameter {
            field: "streaming MoE phase",
            reason: "routing must run exactly once before completion",
        });
    }
    let routed = &scratch.routed[..scratch.routed_len];
    if selected.len() != routed.len() {
        return Err(KernelError::DimensionMismatch {
            field: "streaming selected expert count",
            expected: routed.len(),
            actual: selected.len(),
        });
    }
    for (expected, actual) in routed.iter().zip(selected) {
        if expected.expert_id != actual.expert_id
            || expected.coefficient.to_bits() != actual.coefficient.to_bits()
        {
            return Err(KernelError::InvalidParameter {
                field: "streaming selected expert identity",
                reason: "IDs and coefficients must match authoritative routing exactly",
            });
        }
    }

    let mut swiglu_scratch = SwiGluScratch::new(
        &mut scratch.swiglu_activation,
        &mut scratch.decoded_block,
        &mut scratch.q8,
    );
    moe_selected_into(
        mode,
        selected,
        shared,
        &scratch.ffn_normalized,
        &mut scratch.ffn_delta,
        &mut scratch.preflight_output,
        &mut swiglu_scratch,
    )?;
    residual_add_in_place(hidden_state, &scratch.ffn_delta)?;
    scratch.streaming_moe_pending = false;
    Ok(())
}

fn validate_block(block: Hy3BlockWeights<'_>) -> Result<()> {
    let attention = block.attention;
    let hidden = attention.query.input_width();
    require_equal("attention norm weight", hidden, attention.input_norm.len())?;
    require_equal("attention key input", hidden, attention.key.input_width())?;
    require_equal("attention value input", hidden, attention.value.input_width())?;
    if attention.query_head_count == 0
        || attention.kv_head_count == 0
        || attention.query_head_count % attention.kv_head_count != 0
        || attention.key_dimension == 0
        || attention.value_dimension == 0
    {
        return Err(KernelError::InvalidParameter {
            field: "Hy3 attention dimensions",
            reason: "head counts and dimensions must be positive and query heads must group over KV heads",
        });
    }
    require_equal(
        "query projection output",
        checked_product(
            attention.query_head_count,
            attention.key_dimension,
            "query projection output",
        )?,
        attention.query.output_width(),
    )?;
    require_equal(
        "key projection output",
        checked_product(
            attention.kv_head_count,
            attention.key_dimension,
            "key projection output",
        )?,
        attention.key.output_width(),
    )?;
    require_equal(
        "value projection output",
        checked_product(
            attention.kv_head_count,
            attention.value_dimension,
            "value projection output",
        )?,
        attention.value.output_width(),
    )?;
    require_equal(
        "attention output projection input",
        checked_product(
            attention.query_head_count,
            attention.value_dimension,
            "attention output projection input",
        )?,
        attention.output.input_width(),
    )?;
    require_equal("attention output width", hidden, attention.output.output_width())?;
    require_equal(
        "query norm weight",
        attention.key_dimension,
        attention.query_norm.len(),
    )?;
    require_equal(
        "key norm weight",
        attention.key_dimension,
        attention.key_norm.len(),
    )?;
    require_equal("FFN norm weight", hidden, block.ffn_norm.len())?;

    match block.feed_forward {
        Hy3FeedForwardWeights::Dense(expert) => validate_expert_dimensions(hidden, expert),
        Hy3FeedForwardWeights::Moe(moe) => {
            require_equal("router input", hidden, moe.router.input_width())?;
            require_equal(
                "router output",
                moe.routed_experts.len(),
                moe.router.output_width(),
            )?;
            require_equal(
                "router selection bias",
                moe.routed_experts.len(),
                moe.selection_bias.len(),
            )?;
            if moe.expert_used_count == 0 || moe.expert_used_count > moe.routed_experts.len() {
                return Err(KernelError::InvalidParameter {
                    field: "MoE selected expert count",
                    reason: "must be within the available routed expert count",
                });
            }
            if !moe.weight_scale.is_finite() {
                return Err(KernelError::NonFiniteValue {
                    field: "MoE weight scale",
                    index: 0,
                    bits: moe.weight_scale.to_bits(),
                });
            }
            validate_expert_dimensions(hidden, moe.shared_expert)?;
            for expert in moe.routed_experts {
                validate_expert_dimensions(hidden, *expert)?;
                require_equal(
                    "routed/shared expert hidden width",
                    moe.shared_expert.hidden_width(),
                    expert.hidden_width(),
                )?;
            }
            Ok(())
        }
    }
}

fn validate_streaming_moe(block: Hy3StreamingMoeWeights<'_>) -> Result<()> {
    let attention = block.attention;
    let hidden = attention.query.input_width();
    require_equal("attention norm weight", hidden, attention.input_norm.len())?;
    require_equal("attention key input", hidden, attention.key.input_width())?;
    require_equal("attention value input", hidden, attention.value.input_width())?;
    if attention.query_head_count == 0
        || attention.kv_head_count == 0
        || attention.query_head_count % attention.kv_head_count != 0
        || attention.key_dimension == 0
        || attention.value_dimension == 0
    {
        return Err(KernelError::InvalidParameter {
            field: "Hy3 attention dimensions",
            reason: "head counts and dimensions must be positive and query heads must group over KV heads",
        });
    }
    require_equal(
        "query projection output",
        checked_product(
            attention.query_head_count,
            attention.key_dimension,
            "query projection output",
        )?,
        attention.query.output_width(),
    )?;
    require_equal(
        "key projection output",
        checked_product(
            attention.kv_head_count,
            attention.key_dimension,
            "key projection output",
        )?,
        attention.key.output_width(),
    )?;
    require_equal(
        "value projection output",
        checked_product(
            attention.kv_head_count,
            attention.value_dimension,
            "value projection output",
        )?,
        attention.value.output_width(),
    )?;
    require_equal(
        "attention output projection input",
        checked_product(
            attention.query_head_count,
            attention.value_dimension,
            "attention output projection input",
        )?,
        attention.output.input_width(),
    )?;
    require_equal("attention output width", hidden, attention.output.output_width())?;
    require_equal(
        "query norm weight",
        attention.key_dimension,
        attention.query_norm.len(),
    )?;
    require_equal(
        "key norm weight",
        attention.key_dimension,
        attention.key_norm.len(),
    )?;
    require_equal("FFN norm weight", hidden, block.ffn_norm.len())?;
    require_equal("router input", hidden, block.router.input_width())?;
    require_equal("router output", block.expert_count, block.router.output_width())?;
    require_equal(
        "router selection bias",
        block.expert_count,
        block.selection_bias.len(),
    )?;
    if block.expert_used_count == 0 || block.expert_used_count > block.expert_count {
        return Err(KernelError::InvalidParameter {
            field: "MoE selected expert count",
            reason: "must be within the available routed expert count",
        });
    }
    if !block.weight_scale.is_finite() {
        return Err(KernelError::NonFiniteValue {
            field: "MoE weight scale",
            index: 0,
            bits: block.weight_scale.to_bits(),
        });
    }
    validate_expert_dimensions(hidden, block.shared_expert)
}

fn validate_expert_dimensions(hidden: usize, expert: SwiGluExpert<'_>) -> Result<()> {
    require_equal("expert input width", hidden, expert.input_width())?;
    require_equal("expert output width", hidden, expert.output_width())
}

fn validate_streaming_runtime(
    block: Hy3StreamingMoeWeights<'_>,
    cache: &PagedKvCache,
    layer: usize,
    position: u64,
    hidden_state: &[f32],
    scratch: &Hy3BlockScratch,
) -> Result<()> {
    validate_streaming_moe(block)?;
    require_equal(
        "block hidden state",
        block.attention.query.input_width(),
        hidden_state.len(),
    )?;
    require_equal(
        "KV query heads",
        block.attention.kv_head_count,
        cache.kv_head_count(),
    )?;
    require_equal(
        "KV key dimension",
        block.attention.key_dimension,
        cache.key_dimension(),
    )?;
    require_equal(
        "KV value dimension",
        block.attention.value_dimension,
        cache.value_dimension(),
    )?;
    let stored = cache.stored_tokens(layer)?;
    let actual_position = usize::try_from(position).map_err(|_| KernelError::ArithmeticOverflow {
        operation: "token position conversion",
    })?;
    require_equal("KV position", stored, actual_position)?;
    require_equal(
        "block scratch hidden width",
        hidden_state.len(),
        scratch.attention_normalized.len(),
    )?;
    require_equal(
        "block scratch expert count",
        block.expert_count,
        scratch.router_logits.len(),
    )?;
    require_equal(
        "block scratch selected expert count",
        block.expert_used_count,
        scratch.routed.len(),
    )
}

fn validate_runtime(
    block: Hy3BlockWeights<'_>,
    cache: &PagedKvCache,
    layer: usize,
    position: u64,
    hidden_state: &[f32],
    scratch: &Hy3BlockScratch,
) -> Result<()> {
    validate_block(block)?;
    require_equal(
        "block hidden state",
        block.attention.query.input_width(),
        hidden_state.len(),
    )?;
    require_equal(
        "KV query heads",
        block.attention.kv_head_count,
        cache.kv_head_count(),
    )?;
    require_equal(
        "KV key dimension",
        block.attention.key_dimension,
        cache.key_dimension(),
    )?;
    require_equal(
        "KV value dimension",
        block.attention.value_dimension,
        cache.value_dimension(),
    )?;
    let stored = cache.stored_tokens(layer)?;
    let actual_position = usize::try_from(position).map_err(|_| KernelError::ArithmeticOverflow {
        operation: "token position conversion",
    })?;
    require_equal("KV position", stored, actual_position)?;
    require_equal(
        "block scratch hidden width",
        hidden_state.len(),
        scratch.attention_normalized.len(),
    )
}

fn checked_product(left: usize, right: usize, operation: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(KernelError::ArithmeticOverflow { operation })
}

fn require_equal(field: &'static str, expected: usize, actual: usize) -> Result<()> {
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

fn zeroed_f32(length: usize, context: &'static str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| KernelError::AllocationFailed {
            context,
            requested: length,
        })?;
    values.resize(length, 0.0);
    Ok(values)
}

fn zeroed_u8(length: usize, context: &'static str) -> Result<Vec<u8>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| KernelError::AllocationFailed {
            context,
            requested: length,
        })?;
    values.resize(length, 0);
    Ok(values)
}

fn route_candidates(length: usize) -> Result<Vec<RouteCandidate>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| KernelError::AllocationFailed {
            context: "route candidates",
            requested: length,
        })?;
    values.resize(
        length,
        RouteCandidate {
            expert_id: 0,
            selection_score: 0.0,
            unbiased_weight: 0.0,
        },
    );
    Ok(values)
}

fn routed_experts(length: usize) -> Result<Vec<RoutedExpert>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| KernelError::AllocationFailed {
            context: "routed experts",
            requested: length,
        })?;
    values.resize(
        length,
        RoutedExpert {
            expert_id: 0,
            coefficient: 0.0,
        },
    );
    Ok(values)
}
