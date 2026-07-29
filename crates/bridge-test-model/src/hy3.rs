use bridge_core::ggml_type::GgmlType;
use bridge_core::tensor::TensorDesc;
use bridge_gguf::{GgufArray, GgufReader, GgufValue, GgufValueType};
use bridge_kernels_reference::{
    hy3_block_forward_token, hy3_moe_finish_token, hy3_moe_route_token, softmax_into, weighted_rms_norm_into,
    Hy3AttentionWeights, Hy3BlockExecution, Hy3BlockScratch, Hy3BlockWeights, Hy3FeedForwardWeights,
    Hy3RopeParams, Hy3StreamingMoeWeights, KernelError, PackedMatrix, PayloadEndian, ReferenceExecutionMode,
    SelectedExpert, SwiGluExpert,
};
use bridge_kv_gqa::PagedKvCache;
use bridge_model_hy3::{
    validate_model_with_profile, Hy3Config, Hy3Profile, Hy3TensorRole, TensorSpec, ValidatedHy3Model,
};
use bridge_quant_layout::layout;
use bridge_tokenizer::{
    ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN, REASONING_MODE_TOKEN, THINK_BEGIN_TOKEN, THINK_END_TOKEN,
    USER_TOKEN,
};
use half::f16;

pub const BLOCK_COUNT: usize = 2;
pub const CONTEXT_LENGTH: usize = 1_024;
pub const HIDDEN_WIDTH: usize = 256;
pub const QUERY_HEAD_COUNT: usize = 4;
pub const KV_HEAD_COUNT: usize = 2;
pub const HEAD_DIMENSION: usize = 64;
pub const DENSE_FFN_WIDTH: usize = 256;
pub const EXPERT_FFN_WIDTH: usize = 256;
pub const EXPERT_COUNT: usize = 4;
pub const EXPERT_USED_COUNT: usize = 2;
pub const VOCABULARY_SIZE: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum TestModelError {
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error(transparent)]
    Kv(#[from] bridge_kv_gqa::KvError),
    #[error(transparent)]
    Quant(#[from] bridge_quant_layout::QuantError),
    #[error(transparent)]
    Core(#[from] bridge_core::error::CoreError),
    #[error(transparent)]
    Gguf(#[from] bridge_gguf::GgufError),
    #[error(transparent)]
    Split(#[from] bridge_gguf_split::SplitError),
    #[error(transparent)]
    Model(#[from] bridge_model_hy3::Hy3Error),
    #[error("token ID {token_id} is outside vocabulary size {vocabulary_size}")]
    TokenOutOfRange { token_id: u32, vocabulary_size: usize },
    #[error("checked arithmetic overflow while constructing {operation}")]
    ArithmeticOverflow { operation: &'static str },
    #[error("allocation failed while reserving {requested} entries for {context}")]
    AllocationFailed { context: &'static str, requested: usize },
    #[error("invalid internal reduced GGUF fixture: {reason}")]
    InvalidFixture { reason: &'static str },
}

#[derive(Debug)]
struct OwnedMatrix {
    ty: GgmlType,
    input_width: usize,
    output_width: usize,
    bytes: Vec<u8>,
}

impl OwnedMatrix {
    fn quantized(
        ty: GgmlType,
        input_width: usize,
        output_width: usize,
        seed: u64,
    ) -> Result<Self, TestModelError> {
        let block = layout(ty)?;
        if input_width == 0 || input_width % block.block_elements != 0 {
            return Err(TestModelError::ArithmeticOverflow {
                operation: "block-aligned reduced matrix width",
            });
        }
        let blocks_per_row = input_width / block.block_elements;
        let row_bytes =
            blocks_per_row
                .checked_mul(block.block_bytes)
                .ok_or(TestModelError::ArithmeticOverflow {
                    operation: "reduced matrix row bytes",
                })?;
        let byte_count = output_width
            .checked_mul(row_bytes)
            .ok_or(TestModelError::ArithmeticOverflow {
                operation: "reduced matrix byte count",
            })?;
        let mut bytes = zeroed_u8(byte_count, "reduced matrix payload")?;
        let mut state = seed | 1;
        for row in 0..output_width {
            for block_index in 0..blocks_per_row {
                let start = row * row_bytes + block_index * block.block_bytes;
                let encoded = &mut bytes[start..start + block.block_bytes];
                for byte in encoded.iter_mut() {
                    state = xorshift64(state);
                    *byte = (state >> 24) as u8;
                }
                let magnitude = 0.000_35_f32
                    + ((row + block_index + usize::from((seed & 7) as u8)) % 11) as f32 * 0.000_025_f32;
                encoded[..2].copy_from_slice(&f16::from_f32(magnitude).to_bits().to_le_bytes());
                if matches!(ty, GgmlType::Q4_K | GgmlType::Q5_K) {
                    encoded[2..4].copy_from_slice(&f16::from_f32(magnitude * 0.125).to_bits().to_le_bytes());
                }
            }
        }
        PackedMatrix::from_parts(ty, PayloadEndian::Little, input_width, output_width, &bytes)?;
        Ok(Self {
            ty,
            input_width,
            output_width,
            bytes,
        })
    }

    fn view(&self) -> Result<PackedMatrix<'_>, TestModelError> {
        Ok(PackedMatrix::from_parts(
            self.ty,
            PayloadEndian::Little,
            self.input_width,
            self.output_width,
            &self.bytes,
        )?)
    }
}

#[derive(Debug)]
struct OwnedExpert {
    gate: OwnedMatrix,
    up: OwnedMatrix,
    down: OwnedMatrix,
}

impl OwnedExpert {
    fn new(types: [GgmlType; 3], seed: u64, ffn_width: usize) -> Result<Self, TestModelError> {
        Ok(Self {
            gate: OwnedMatrix::quantized(types[0], HIDDEN_WIDTH, ffn_width, seed)?,
            up: OwnedMatrix::quantized(types[1], HIDDEN_WIDTH, ffn_width, seed + 1)?,
            down: OwnedMatrix::quantized(types[2], ffn_width, HIDDEN_WIDTH, seed + 2)?,
        })
    }

    fn view(&self) -> Result<SwiGluExpert<'_>, TestModelError> {
        Ok(SwiGluExpert::new(
            self.gate.view()?,
            self.up.view()?,
            self.down.view()?,
        )?)
    }
}

#[derive(Debug)]
struct OwnedAttention {
    input_norm: Vec<f32>,
    query: OwnedMatrix,
    query_norm: Vec<f32>,
    key: OwnedMatrix,
    key_norm: Vec<f32>,
    value: OwnedMatrix,
    output: OwnedMatrix,
}

impl OwnedAttention {
    fn new(types: [GgmlType; 4], seed: u64) -> Result<Self, TestModelError> {
        Ok(Self {
            input_norm: norm_weights(HIDDEN_WIDTH, seed),
            query: OwnedMatrix::quantized(
                types[0],
                HIDDEN_WIDTH,
                QUERY_HEAD_COUNT * HEAD_DIMENSION,
                seed + 1,
            )?,
            query_norm: norm_weights(HEAD_DIMENSION, seed + 2),
            key: OwnedMatrix::quantized(types[1], HIDDEN_WIDTH, KV_HEAD_COUNT * HEAD_DIMENSION, seed + 3)?,
            key_norm: norm_weights(HEAD_DIMENSION, seed + 4),
            value: OwnedMatrix::quantized(types[2], HIDDEN_WIDTH, KV_HEAD_COUNT * HEAD_DIMENSION, seed + 5)?,
            output: OwnedMatrix::quantized(
                types[3],
                QUERY_HEAD_COUNT * HEAD_DIMENSION,
                HIDDEN_WIDTH,
                seed + 6,
            )?,
        })
    }

    fn view(&self) -> Result<Hy3AttentionWeights<'_>, TestModelError> {
        Ok(Hy3AttentionWeights {
            input_norm: &self.input_norm,
            query: self.query.view()?,
            query_norm: &self.query_norm,
            key: self.key.view()?,
            key_norm: &self.key_norm,
            value: self.value.view()?,
            output: self.output.view()?,
            query_head_count: QUERY_HEAD_COUNT,
            kv_head_count: KV_HEAD_COUNT,
            key_dimension: HEAD_DIMENSION,
            value_dimension: HEAD_DIMENSION,
        })
    }
}

#[derive(Debug)]
struct DenseBlock {
    attention: OwnedAttention,
    ffn_norm: Vec<f32>,
    expert: OwnedExpert,
}

impl DenseBlock {
    fn weights(&self) -> Result<Hy3BlockWeights<'_>, TestModelError> {
        Ok(Hy3BlockWeights {
            attention: self.attention.view()?,
            ffn_norm: &self.ffn_norm,
            feed_forward: Hy3FeedForwardWeights::Dense(self.expert.view()?),
        })
    }
}

#[derive(Debug)]
struct MoeBlock {
    attention: OwnedAttention,
    ffn_norm: Vec<f32>,
    router: OwnedMatrix,
    selection_bias: [f32; EXPERT_COUNT],
    experts: [OwnedExpert; EXPERT_COUNT],
    shared: OwnedExpert,
}

impl MoeBlock {
    fn expert_views(&self) -> Result<[SwiGluExpert<'_>; EXPERT_COUNT], TestModelError> {
        Ok([
            self.experts[0].view()?,
            self.experts[1].view()?,
            self.experts[2].view()?,
            self.experts[3].view()?,
        ])
    }

    fn streaming_weights(&self) -> Result<Hy3StreamingMoeWeights<'_>, TestModelError> {
        Ok(Hy3StreamingMoeWeights {
            attention: self.attention.view()?,
            ffn_norm: &self.ffn_norm,
            router: self.router.view()?,
            selection_bias: &self.selection_bias,
            shared_expert: self.shared.view()?,
            expert_count: EXPERT_COUNT,
            expert_used_count: EXPERT_USED_COUNT,
            weight_scale: 1.75,
        })
    }
}

#[derive(Debug)]
pub struct ReducedHy3Model {
    config: Hy3Config,
    embeddings: Vec<f32>,
    dense: DenseBlock,
    moe: MoeBlock,
    output_norm: Vec<f32>,
    output: OwnedMatrix,
}

impl ReducedHy3Model {
    pub fn new() -> Result<Self, TestModelError> {
        let quant = [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::IQ2_S, GgmlType::IQ3_S];
        Ok(Self {
            config: reduced_config(),
            embeddings: embeddings(),
            dense: DenseBlock {
                attention: OwnedAttention::new(quant, 0x1001)?,
                ffn_norm: norm_weights(HIDDEN_WIDTH, 0x1101),
                expert: OwnedExpert::new(
                    [GgmlType::IQ2_S, GgmlType::Q4_K, GgmlType::Q5_K],
                    0x1201,
                    DENSE_FFN_WIDTH,
                )?,
            },
            moe: MoeBlock {
                attention: OwnedAttention::new(
                    [GgmlType::IQ3_S, GgmlType::IQ2_S, GgmlType::Q5_K, GgmlType::Q4_K],
                    0x2001,
                )?,
                ffn_norm: norm_weights(HIDDEN_WIDTH, 0x2101),
                router: OwnedMatrix::quantized(GgmlType::Q4_K, HIDDEN_WIDTH, EXPERT_COUNT, 0x2201)?,
                selection_bias: [0.025, -0.0125, 0.0375, -0.02],
                experts: [
                    OwnedExpert::new(
                        [GgmlType::IQ2_S, GgmlType::Q5_K, GgmlType::IQ3_S],
                        0x2301,
                        EXPERT_FFN_WIDTH,
                    )?,
                    OwnedExpert::new(
                        [GgmlType::IQ2_S, GgmlType::Q5_K, GgmlType::IQ3_S],
                        0x2401,
                        EXPERT_FFN_WIDTH,
                    )?,
                    OwnedExpert::new(
                        [GgmlType::IQ2_S, GgmlType::Q5_K, GgmlType::IQ3_S],
                        0x2501,
                        EXPERT_FFN_WIDTH,
                    )?,
                    OwnedExpert::new(
                        [GgmlType::IQ2_S, GgmlType::Q5_K, GgmlType::IQ3_S],
                        0x2601,
                        EXPERT_FFN_WIDTH,
                    )?,
                ],
                shared: OwnedExpert::new(
                    [GgmlType::IQ2_S, GgmlType::IQ3_S, GgmlType::Q4_K],
                    0x2701,
                    EXPERT_FFN_WIDTH,
                )?,
            },
            output_norm: norm_weights(HIDDEN_WIDTH, 0x3001),
            output: OwnedMatrix::quantized(GgmlType::IQ3_S, HIDDEN_WIDTH, VOCABULARY_SIZE, 0x3101)?,
        })
    }

    pub const fn config(&self) -> &Hy3Config {
        &self.config
    }

    /// Returns the exact schema authority for this reduced fixture.
    pub fn profile(&self) -> Result<Hy3Profile, TestModelError> {
        let fixtures = self.tensor_fixtures()?;
        profile_from_fixtures(&self.config, &fixtures)
    }

    /// Serializes the deterministic model through native GGUF v3 records,
    /// including exact payload lengths and canonical 32-byte alignment.
    pub fn gguf_bytes(&self) -> Result<Vec<u8>, TestModelError> {
        serialize_gguf(&self.config, &self.tensor_fixtures()?)
    }

    /// Serializes the same tensor payload with a minimal valid Hy3 tokenizer.
    ///
    /// The canonical fixture produced by [`Self::gguf_bytes`] remains
    /// byte-stable for the pinned external oracles. This variant exists for
    /// deterministic chat-engine and HTTP integration tests.
    pub fn gguf_bytes_with_chat_tokenizer(&self) -> Result<Vec<u8>, TestModelError> {
        serialize_gguf_with_metadata(&self.tensor_fixtures()?, reduced_chat_metadata(&self.config)?)
    }

    /// Expands each checked fixture tensor into exact row-major F32 values for
    /// offline framework-oracle loading.
    pub fn dequantized_tensors(&self) -> Result<Vec<DequantizedTensor>, TestModelError> {
        let fixtures = self.tensor_fixtures()?;
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(fixtures.len())
            .map_err(|_| TestModelError::AllocationFailed {
                context: "dequantized fixture tensors",
                requested: fixtures.len(),
            })?;
        for fixture in fixtures {
            let logical_elements = fixture
                .shape
                .iter()
                .try_fold(1_usize, |total, &dimension| total.checked_mul(dimension as usize))
                .ok_or(TestModelError::ArithmeticOverflow {
                    operation: "dequantized tensor element count",
                })?;
            let mut values = zeroed_f32(logical_elements, "dequantized fixture tensor")?;
            if fixture.ty == GgmlType::F32 {
                for (destination, encoded) in values.iter_mut().zip(fixture.bytes.chunks_exact(4)) {
                    *destination = f32::from_bits(u32::from_le_bytes([
                        encoded[0], encoded[1], encoded[2], encoded[3],
                    ]));
                }
            } else {
                let row_width = fixture.shape[0] as usize;
                let row_layout = layout(fixture.ty)?;
                let row_bytes = row_width / row_layout.block_elements * row_layout.block_bytes;
                for (encoded, output) in fixture
                    .bytes
                    .chunks_exact(row_bytes)
                    .zip(values.chunks_exact_mut(row_width))
                {
                    bridge_quant_layout::decode_row_into(fixture.ty, encoded, row_width, output)?;
                }
            }
            tensors.push(DequantizedTensor {
                name: fixture.name,
                shape: fixture.shape,
                values,
            });
        }
        Ok(tensors)
    }

    /// Exercises the production GGUF reader, split directory, and explicit
    /// Hy3 profile authorization boundary.
    pub fn parse_and_validate_gguf(&self) -> Result<ValidatedHy3Model, TestModelError> {
        let fixtures = self.tensor_fixtures()?;
        let profile = profile_from_fixtures(&self.config, &fixtures)?;
        let bytes = serialize_gguf(&self.config, &fixtures)?;
        let parsed = GgufReader::new(std::io::Cursor::new(bytes)).read()?;
        let set = bridge_gguf_split::testing::from_file(parsed)?;
        Ok(validate_model_with_profile(&set, &profile)?)
    }

    pub fn new_session(&self) -> Result<ReducedHy3Session, TestModelError> {
        let dense_weights = self.dense.weights()?;
        Ok(ReducedHy3Session {
            cache: PagedKvCache::new(
                BLOCK_COUNT,
                KV_HEAD_COUNT,
                HEAD_DIMENSION,
                HEAD_DIMENSION,
                4,
                CONTEXT_LENGTH,
            )?,
            dense_scratch: Hy3BlockScratch::new(dense_weights, CONTEXT_LENGTH)?,
            moe_scratch: Hy3BlockScratch::new_streaming_moe(self.moe.streaming_weights()?, CONTEXT_LENGTH)?,
            hidden: zeroed_f32(HIDDEN_WIDTH, "reduced hidden state")?,
            final_normalized: zeroed_f32(HIDDEN_WIDTH, "reduced final norm")?,
            logits: zeroed_f32(VOCABULARY_SIZE, "reduced logits")?,
            probabilities: zeroed_f32(VOCABULARY_SIZE, "reduced probabilities")?,
            decoded_block: zeroed_f32(256, "reduced output decoded block")?,
            q8: zeroed_u8(
                bridge_kernels_reference::required_q8_k_bytes(HIDDEN_WIDTH)?,
                "reduced output Q8_K activation",
            )?,
            position: 0,
            greedy_id: 0,
        })
    }

    pub fn evaluate_token<'a>(
        &self,
        session: &'a mut ReducedHy3Session,
        mode: ReferenceExecutionMode,
        token_id: u32,
    ) -> Result<ReducedTokenOutput<'a>, TestModelError> {
        let token = usize::try_from(token_id).map_err(|_| TestModelError::TokenOutOfRange {
            token_id,
            vocabulary_size: VOCABULARY_SIZE,
        })?;
        if token >= VOCABULARY_SIZE {
            return Err(TestModelError::TokenOutOfRange {
                token_id,
                vocabulary_size: VOCABULARY_SIZE,
            });
        }
        if session.position >= CONTEXT_LENGTH {
            return Err(KernelError::PositionOutOfRange {
                position: session.position as u64,
                context_length: CONTEXT_LENGTH as u64,
            }
            .into());
        }

        let embedding_start = token * HIDDEN_WIDTH;
        session
            .hidden
            .copy_from_slice(&self.embeddings[embedding_start..embedding_start + HIDDEN_WIDTH]);
        let rope = Hy3RopeParams::from_config(&self.config)?;
        hy3_block_forward_token(
            Hy3BlockExecution {
                mode,
                layer: 0,
                position: session.position as u64,
                rope,
                rms_epsilon: self.config.rms_epsilon,
            },
            self.dense.weights()?,
            &mut session.cache,
            &mut session.hidden,
            &mut session.dense_scratch,
        )?;
        hy3_moe_route_token(
            Hy3BlockExecution {
                mode,
                layer: 1,
                position: session.position as u64,
                rope,
                rms_epsilon: self.config.rms_epsilon,
            },
            self.moe.streaming_weights()?,
            &mut session.cache,
            &mut session.hidden,
            &mut session.moe_scratch,
        )?;
        let routed = [session.moe_scratch.routed()[0], session.moe_scratch.routed()[1]];
        let expert_views = self.moe.expert_views()?;
        let selected = routed.map(|route| SelectedExpert {
            expert_id: route.expert_id,
            coefficient: route.coefficient,
            expert: expert_views[route.expert_id as usize],
        });
        hy3_moe_finish_token(
            mode,
            &selected,
            self.moe.shared.view()?,
            &mut session.hidden,
            &mut session.moe_scratch,
        )?;
        weighted_rms_norm_into(
            &session.hidden,
            &self.output_norm,
            self.config.rms_epsilon,
            &mut session.final_normalized,
        )?;
        bridge_kernels_reference::gemv_into(
            mode,
            self.output.view()?,
            &session.final_normalized,
            &mut session.logits,
            &mut session.decoded_block,
            &mut session.q8,
        )?;
        softmax_into(&session.logits, &mut session.probabilities)?;
        session.greedy_id = greedy_id(&session.logits);
        session.position += 1;

        let routed = session.moe_scratch.routed();
        let selected_experts = [routed[0].expert_id, routed[1].expert_id];
        Ok(ReducedTokenOutput {
            hidden: &session.hidden,
            final_normalized: &session.final_normalized,
            logits: &session.logits,
            probabilities: &session.probabilities,
            greedy_id: session.greedy_id,
            selected_experts,
        })
    }

    fn tensor_fixtures(&self) -> Result<Vec<TensorFixture>, TestModelError> {
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(30)
            .map_err(|_| TestModelError::AllocationFailed {
                context: "reduced GGUF tensor fixtures",
                requested: 30,
            })?;
        push_f32_tensor(
            &mut tensors,
            "token_embd.weight",
            Hy3TensorRole::TokenEmbedding,
            vec![HIDDEN_WIDTH as u64, VOCABULARY_SIZE as u64],
            &self.embeddings,
        );
        push_f32_tensor(
            &mut tensors,
            "output_norm.weight",
            Hy3TensorRole::OutputNorm,
            vec![HIDDEN_WIDTH as u64],
            &self.output_norm,
        );
        push_matrix_tensor(&mut tensors, "output.weight", Hy3TensorRole::Output, &self.output);
        push_attention_tensors(&mut tensors, 0, &self.dense.attention);
        push_f32_tensor(
            &mut tensors,
            "blk.0.ffn_norm.weight",
            Hy3TensorRole::FfnNorm { layer: 0 },
            vec![HIDDEN_WIDTH as u64],
            &self.dense.ffn_norm,
        );
        push_matrix_tensor(
            &mut tensors,
            "blk.0.ffn_gate.weight",
            Hy3TensorRole::DenseGate { layer: 0 },
            &self.dense.expert.gate,
        );
        push_matrix_tensor(
            &mut tensors,
            "blk.0.ffn_up.weight",
            Hy3TensorRole::DenseUp { layer: 0 },
            &self.dense.expert.up,
        );
        push_matrix_tensor(
            &mut tensors,
            "blk.0.ffn_down.weight",
            Hy3TensorRole::DenseDown { layer: 0 },
            &self.dense.expert.down,
        );
        push_attention_tensors(&mut tensors, 1, &self.moe.attention);
        push_f32_tensor(
            &mut tensors,
            "blk.1.ffn_norm.weight",
            Hy3TensorRole::FfnNorm { layer: 1 },
            vec![HIDDEN_WIDTH as u64],
            &self.moe.ffn_norm,
        );
        push_matrix_tensor(
            &mut tensors,
            "blk.1.ffn_gate_inp.weight",
            Hy3TensorRole::RouterInput { layer: 1 },
            &self.moe.router,
        );
        push_f32_tensor(
            &mut tensors,
            "blk.1.exp_probs_b",
            Hy3TensorRole::RouterSelectionBias { layer: 1 },
            vec![EXPERT_COUNT as u64],
            &self.moe.selection_bias,
        );
        push_expert_tensor(
            &mut tensors,
            "blk.1.ffn_gate_exps.weight",
            Hy3TensorRole::RoutedGate { layer: 1 },
            &self.moe.experts,
            |expert| &expert.gate,
        )?;
        push_expert_tensor(
            &mut tensors,
            "blk.1.ffn_up_exps.weight",
            Hy3TensorRole::RoutedUp { layer: 1 },
            &self.moe.experts,
            |expert| &expert.up,
        )?;
        push_expert_tensor(
            &mut tensors,
            "blk.1.ffn_down_exps.weight",
            Hy3TensorRole::RoutedDown { layer: 1 },
            &self.moe.experts,
            |expert| &expert.down,
        )?;
        push_matrix_tensor(
            &mut tensors,
            "blk.1.ffn_gate_shexp.weight",
            Hy3TensorRole::SharedGate { layer: 1 },
            &self.moe.shared.gate,
        );
        push_matrix_tensor(
            &mut tensors,
            "blk.1.ffn_up_shexp.weight",
            Hy3TensorRole::SharedUp { layer: 1 },
            &self.moe.shared.up,
        );
        push_matrix_tensor(
            &mut tensors,
            "blk.1.ffn_down_shexp.weight",
            Hy3TensorRole::SharedDown { layer: 1 },
            &self.moe.shared.down,
        );
        Ok(tensors)
    }
}

#[derive(Debug)]
pub struct ReducedHy3Session {
    cache: PagedKvCache,
    dense_scratch: Hy3BlockScratch,
    moe_scratch: Hy3BlockScratch,
    hidden: Vec<f32>,
    final_normalized: Vec<f32>,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
    decoded_block: Vec<f32>,
    q8: Vec<u8>,
    position: usize,
    greedy_id: u32,
}

impl ReducedHy3Session {
    pub const fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        self.position = 0;
        self.greedy_id = 0;
        self.hidden.fill(0.0);
        self.final_normalized.fill(0.0);
        self.logits.fill(0.0);
        self.probabilities.fill(0.0);
    }

    pub const fn cache(&self) -> &PagedKvCache {
        &self.cache
    }

    pub const fn dense_scratch(&self) -> &Hy3BlockScratch {
        &self.dense_scratch
    }

    pub const fn moe_scratch(&self) -> &Hy3BlockScratch {
        &self.moe_scratch
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReducedTokenOutput<'a> {
    pub hidden: &'a [f32],
    pub final_normalized: &'a [f32],
    pub logits: &'a [f32],
    pub probabilities: &'a [f32],
    pub greedy_id: u32,
    pub selected_experts: [u32; EXPERT_USED_COUNT],
}

#[derive(Debug)]
pub struct DequantizedTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub values: Vec<f32>,
}

pub fn reduced_config() -> Hy3Config {
    Hy3Config {
        block_count: BLOCK_COUNT as u32,
        context_length: CONTEXT_LENGTH as u64,
        embedding_length: HIDDEN_WIDTH as u32,
        dense_ffn_length: DENSE_FFN_WIDTH as u32,
        expert_ffn_length: EXPERT_FFN_WIDTH as u32,
        shared_expert_ffn_length: EXPERT_FFN_WIDTH as u32,
        attention_head_count: QUERY_HEAD_COUNT as u32,
        attention_kv_head_count: KV_HEAD_COUNT as u32,
        key_length: HEAD_DIMENSION as u32,
        value_length: HEAD_DIMENSION as u32,
        rms_epsilon: 1.0e-5,
        expert_count: EXPERT_COUNT as u32,
        expert_used_count: EXPERT_USED_COUNT as u32,
        expert_weights_norm: true,
        expert_gating_func: 2,
        expert_weights_scale: 1.75,
        rope_base: 10_000.0,
        rope_scaling_type: "yarn".into(),
        yarn_factor: 4.0,
        yarn_original_context: (CONTEXT_LENGTH / 4) as u64,
        vocabulary_size: VOCABULARY_SIZE as u32,
    }
}

fn embeddings() -> Vec<f32> {
    let mut values = Vec::with_capacity(VOCABULARY_SIZE * HIDDEN_WIDTH);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..VOCABULARY_SIZE * HIDDEN_WIDTH {
        state = xorshift64(state);
        let centered = ((state >> 40) as u32 & 0xffff) as f32 / 65_535.0 - 0.5;
        values.push(centered * 0.5);
    }
    values
}

fn norm_weights(length: usize, seed: u64) -> Vec<f32> {
    (0..length)
        .map(|index| 0.95 + ((index as u64 * 17 + seed) % 101) as f32 * 0.001)
        .collect()
}

fn greedy_id(logits: &[f32]) -> u32 {
    let mut selected = 0_usize;
    for index in 1..logits.len() {
        if logits[index] > logits[selected] {
            selected = index;
        }
    }
    selected as u32
}

fn xorshift64(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn zeroed_f32(length: usize, context: &'static str) -> Result<Vec<f32>, TestModelError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| TestModelError::AllocationFailed {
            context,
            requested: length,
        })?;
    values.resize(length, 0.0);
    Ok(values)
}

fn zeroed_u8(length: usize, context: &'static str) -> Result<Vec<u8>, TestModelError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| TestModelError::AllocationFailed {
            context,
            requested: length,
        })?;
    values.resize(length, 0);
    Ok(values)
}

#[derive(Debug)]
struct TensorFixture {
    name: String,
    role: Hy3TensorRole,
    shape: Vec<u64>,
    ty: GgmlType,
    bytes: Vec<u8>,
}

fn push_attention_tensors(tensors: &mut Vec<TensorFixture>, layer: u32, attention: &OwnedAttention) {
    let prefix = format!("blk.{layer}");
    push_f32_tensor(
        tensors,
        &format!("{prefix}.attn_norm.weight"),
        Hy3TensorRole::AttentionNorm { layer },
        vec![HIDDEN_WIDTH as u64],
        &attention.input_norm,
    );
    push_matrix_tensor(
        tensors,
        &format!("{prefix}.attn_q.weight"),
        Hy3TensorRole::AttentionQ { layer },
        &attention.query,
    );
    push_f32_tensor(
        tensors,
        &format!("{prefix}.attn_q_norm.weight"),
        Hy3TensorRole::AttentionQNorm { layer },
        vec![HEAD_DIMENSION as u64],
        &attention.query_norm,
    );
    push_matrix_tensor(
        tensors,
        &format!("{prefix}.attn_k.weight"),
        Hy3TensorRole::AttentionK { layer },
        &attention.key,
    );
    push_f32_tensor(
        tensors,
        &format!("{prefix}.attn_k_norm.weight"),
        Hy3TensorRole::AttentionKNorm { layer },
        vec![HEAD_DIMENSION as u64],
        &attention.key_norm,
    );
    push_matrix_tensor(
        tensors,
        &format!("{prefix}.attn_v.weight"),
        Hy3TensorRole::AttentionV { layer },
        &attention.value,
    );
    push_matrix_tensor(
        tensors,
        &format!("{prefix}.attn_output.weight"),
        Hy3TensorRole::AttentionOutput { layer },
        &attention.output,
    );
}

fn push_f32_tensor(
    tensors: &mut Vec<TensorFixture>,
    name: &str,
    role: Hy3TensorRole,
    shape: Vec<u64>,
    values: &[f32],
) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    tensors.push(TensorFixture {
        name: name.to_owned(),
        role,
        shape,
        ty: GgmlType::F32,
        bytes,
    });
}

fn push_matrix_tensor(
    tensors: &mut Vec<TensorFixture>,
    name: &str,
    role: Hy3TensorRole,
    matrix: &OwnedMatrix,
) {
    tensors.push(TensorFixture {
        name: name.to_owned(),
        role,
        shape: vec![matrix.input_width as u64, matrix.output_width as u64],
        ty: matrix.ty,
        bytes: matrix.bytes.clone(),
    });
}

fn push_expert_tensor(
    tensors: &mut Vec<TensorFixture>,
    name: &str,
    role: Hy3TensorRole,
    experts: &[OwnedExpert; EXPERT_COUNT],
    projection: impl Fn(&OwnedExpert) -> &OwnedMatrix,
) -> Result<(), TestModelError> {
    let first = projection(&experts[0]);
    let requested =
        first
            .bytes
            .len()
            .checked_mul(EXPERT_COUNT)
            .ok_or(TestModelError::ArithmeticOverflow {
                operation: "routed expert tensor bytes",
            })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(requested)
        .map_err(|_| TestModelError::AllocationFailed {
            context: "routed expert tensor",
            requested,
        })?;
    for expert in experts {
        let matrix = projection(expert);
        if matrix.ty != first.ty
            || matrix.input_width != first.input_width
            || matrix.output_width != first.output_width
        {
            return Err(TestModelError::InvalidFixture {
                reason: "routed expert projection layouts differ",
            });
        }
        bytes.extend_from_slice(&matrix.bytes);
    }
    tensors.push(TensorFixture {
        name: name.to_owned(),
        role,
        shape: vec![
            first.input_width as u64,
            first.output_width as u64,
            EXPERT_COUNT as u64,
        ],
        ty: first.ty,
        bytes,
    });
    Ok(())
}

fn profile_from_fixtures(
    config: &Hy3Config,
    fixtures: &[TensorFixture],
) -> Result<Hy3Profile, TestModelError> {
    let schema = fixtures
        .iter()
        .map(|tensor| TensorSpec::new(tensor.name.clone(), tensor.role, tensor.shape.clone(), tensor.ty))
        .collect();
    Ok(Hy3Profile::explicit(config.clone(), schema)?)
}

fn serialize_gguf(config: &Hy3Config, fixtures: &[TensorFixture]) -> Result<Vec<u8>, TestModelError> {
    serialize_gguf_with_metadata(fixtures, reduced_metadata(config))
}

fn serialize_gguf_with_metadata(
    fixtures: &[TensorFixture],
    metadata: Vec<(String, GgufValue)>,
) -> Result<Vec<u8>, TestModelError> {
    let mut relative_offset = 0_u64;
    let mut descriptors = Vec::with_capacity(fixtures.len());
    for tensor in fixtures {
        let descriptor = TensorDesc::new(tensor.name.clone(), &tensor.shape, tensor.ty, relative_offset)?;
        if descriptor.encoded_bytes()? != tensor.bytes.len() as u64 {
            return Err(TestModelError::InvalidFixture {
                reason: "tensor payload length does not match its GGUF descriptor",
            });
        }
        relative_offset = align_up(
            relative_offset.checked_add(tensor.bytes.len() as u64).ok_or(
                TestModelError::ArithmeticOverflow {
                    operation: "GGUF tensor relative end",
                },
            )?,
            32,
        )?;
        descriptors.push(descriptor);
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(fixtures.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in &metadata {
        write_string(&mut bytes, key);
        bytes.extend_from_slice(&(value.value_type() as u32).to_le_bytes());
        write_metadata_value(&mut bytes, value)?;
    }
    for descriptor in &descriptors {
        write_string(&mut bytes, descriptor.name());
        bytes.extend_from_slice(&descriptor.n_dims().to_le_bytes());
        for dimension in descriptor.shape() {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&descriptor.ty().discriminant().to_le_bytes());
        bytes.extend_from_slice(&descriptor.relative_offset().to_le_bytes());
    }
    let data_offset = align_up(bytes.len() as u64, 32)? as usize;
    bytes.resize(data_offset, 0);
    for (tensor, descriptor) in fixtures.iter().zip(&descriptors) {
        let expected = data_offset + descriptor.relative_offset() as usize;
        if bytes.len() != expected {
            return Err(TestModelError::InvalidFixture {
                reason: "serialized tensor offset is not canonical",
            });
        }
        bytes.extend_from_slice(&tensor.bytes);
        let padded = align_up(bytes.len() as u64, 32)? as usize;
        bytes.resize(padded, 0);
    }
    Ok(bytes)
}

fn reduced_metadata(config: &Hy3Config) -> Vec<(String, GgufValue)> {
    let u32_value = |value| GgufValue::U32(value);
    vec![
        ("general.architecture".into(), GgufValue::String("hy_v3".into())),
        ("general.alignment".into(), u32_value(32)),
        ("hy_v3.block_count".into(), u32_value(config.block_count)),
        (
            "hy_v3.context_length".into(),
            u32_value(config.context_length as u32),
        ),
        (
            "hy_v3.embedding_length".into(),
            u32_value(config.embedding_length),
        ),
        (
            "hy_v3.feed_forward_length".into(),
            u32_value(config.dense_ffn_length),
        ),
        (
            "hy_v3.expert_feed_forward_length".into(),
            u32_value(config.expert_ffn_length),
        ),
        (
            "hy_v3.expert_shared_feed_forward_length".into(),
            u32_value(config.shared_expert_ffn_length),
        ),
        (
            "hy_v3.attention.head_count".into(),
            u32_value(config.attention_head_count),
        ),
        (
            "hy_v3.attention.head_count_kv".into(),
            u32_value(config.attention_kv_head_count),
        ),
        ("hy_v3.attention.key_length".into(), u32_value(config.key_length)),
        (
            "hy_v3.attention.value_length".into(),
            u32_value(config.value_length),
        ),
        (
            "hy_v3.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(config.rms_epsilon),
        ),
        ("hy_v3.expert_count".into(), u32_value(config.expert_count)),
        (
            "hy_v3.expert_used_count".into(),
            u32_value(config.expert_used_count),
        ),
        (
            "hy_v3.expert_weights_norm".into(),
            GgufValue::Bool(config.expert_weights_norm),
        ),
        (
            "hy_v3.expert_gating_func".into(),
            u32_value(config.expert_gating_func),
        ),
        (
            "hy_v3.expert_weights_scale".into(),
            GgufValue::F32(config.expert_weights_scale),
        ),
        ("hy_v3.rope.freq_base".into(), GgufValue::F32(config.rope_base)),
        (
            "hy_v3.rope.scaling.type".into(),
            GgufValue::String(config.rope_scaling_type.clone()),
        ),
        (
            "hy_v3.rope.scaling.factor".into(),
            GgufValue::F32(config.yarn_factor),
        ),
        (
            "hy_v3.rope.scaling.original_context_length".into(),
            u32_value(config.yarn_original_context as u32),
        ),
        ("hy_v3.vocab_size".into(), u32_value(config.vocabulary_size)),
        ("tokenizer.ggml.model".into(), GgufValue::String("none".into())),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: (0..config.vocabulary_size)
                    .map(|token| GgufValue::String(format!("<reduced-{token:02}>")))
                    .collect(),
            }),
        ),
    ]
}

fn reduced_chat_metadata(config: &Hy3Config) -> Result<Vec<(String, GgufValue)>, TestModelError> {
    let mut metadata = reduced_metadata(config);
    let model = metadata
        .iter_mut()
        .find(|(key, _)| key == "tokenizer.ggml.model")
        .ok_or(TestModelError::InvalidFixture {
            reason: "reduced metadata is missing the tokenizer model",
        })?;
    model.1 = GgufValue::String("gpt2".into());

    let mut tokens = vec![
        "a".to_owned(),
        BOS_TOKEN.to_owned(),
        EOS_TOKEN.to_owned(),
        "<|hy_pad:opensource|>".to_owned(),
        "<|hy_separator:opensource|>".to_owned(),
        USER_TOKEN.to_owned(),
        ASSISTANT_TOKEN.to_owned(),
        THINK_BEGIN_TOKEN.to_owned(),
        THINK_END_TOKEN.to_owned(),
        REASONING_MODE_TOKEN.to_owned(),
        "reasoning_effort:no_think".to_owned(),
    ];
    for token in tokens.len()..config.vocabulary_size as usize {
        tokens.push(format!("<reduced-{token:02}>"));
    }
    if tokens.len() != config.vocabulary_size as usize {
        return Err(TestModelError::InvalidFixture {
            reason: "chat tokenizer vocabulary does not match the reduced model",
        });
    }

    let token_entry = metadata
        .iter_mut()
        .find(|(key, _)| key == "tokenizer.ggml.tokens")
        .ok_or(TestModelError::InvalidFixture {
            reason: "reduced metadata is missing the tokenizer vocabulary",
        })?;
    token_entry.1 = string_array(tokens);

    let mut token_types = vec![GgufValue::I32(1); config.vocabulary_size as usize];
    for token_type in &mut token_types[1..=6] {
        *token_type = GgufValue::I32(3);
    }
    token_types[7] = GgufValue::I32(4);
    for token_type in &mut token_types[8..=9] {
        *token_type = GgufValue::I32(3);
    }
    token_types[10] = GgufValue::I32(4);
    metadata.extend([
        (
            "tokenizer.ggml.pre".into(),
            GgufValue::String("hunyuan-dense".into()),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(GgufArray {
                element_type: GgufValueType::I32,
                values: token_types,
            }),
        ),
        ("tokenizer.ggml.merges".into(), string_array(Vec::new())),
        ("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(1)),
        ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(2)),
        ("tokenizer.ggml.padding_token_id".into(), GgufValue::U32(3)),
        ("tokenizer.ggml.separator_token_id".into(), GgufValue::U32(4)),
        (
            "tokenizer.chat_template".into(),
            GgufValue::String("LightBridge deterministic reduced Hy3 fixture".into()),
        ),
    ]);
    Ok(metadata)
}

fn string_array(values: Vec<String>) -> GgufValue {
    GgufValue::Array(GgufArray {
        element_type: GgufValueType::String,
        values: values.into_iter().map(GgufValue::String).collect(),
    })
}

fn write_metadata_value(bytes: &mut Vec<u8>, value: &GgufValue) -> Result<(), TestModelError> {
    match value {
        GgufValue::U32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
        GgufValue::F32(value) => bytes.extend_from_slice(&value.to_bits().to_le_bytes()),
        GgufValue::Bool(value) => bytes.push(u8::from(*value)),
        GgufValue::String(value) => write_string(bytes, value),
        GgufValue::Array(array) if array.element_type == GgufValueType::String => {
            bytes.extend_from_slice(&(array.element_type as u32).to_le_bytes());
            bytes.extend_from_slice(&(array.values.len() as u64).to_le_bytes());
            for value in &array.values {
                if let GgufValue::String(value) = value {
                    write_string(bytes, value);
                } else {
                    return Err(TestModelError::InvalidFixture {
                        reason: "token array contains a non-string value",
                    });
                }
            }
        }
        GgufValue::Array(array) if array.element_type == GgufValueType::I32 => {
            bytes.extend_from_slice(&(array.element_type as u32).to_le_bytes());
            bytes.extend_from_slice(&(array.values.len() as u64).to_le_bytes());
            for value in &array.values {
                if let GgufValue::I32(value) = value {
                    bytes.extend_from_slice(&value.to_le_bytes());
                } else {
                    return Err(TestModelError::InvalidFixture {
                        reason: "token type array contains a non-i32 value",
                    });
                }
            }
        }
        _ => {
            return Err(TestModelError::InvalidFixture {
                reason: "unsupported metadata value in reduced fixture",
            });
        }
    }
    Ok(())
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn align_up(value: u64, alignment: u64) -> Result<u64, TestModelError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(TestModelError::ArithmeticOverflow {
            operation: "GGUF alignment",
        })
}
