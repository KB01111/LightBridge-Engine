use crate::Hy3Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hy3TensorRole {
    TokenEmbedding,
    OutputNorm,
    Output,
    AttentionNorm { layer: u32 },
    AttentionQ { layer: u32 },
    AttentionQNorm { layer: u32 },
    AttentionK { layer: u32 },
    AttentionKNorm { layer: u32 },
    AttentionV { layer: u32 },
    AttentionOutput { layer: u32 },
    FfnNorm { layer: u32 },
    DenseGate { layer: u32 },
    DenseUp { layer: u32 },
    DenseDown { layer: u32 },
    RouterInput { layer: u32 },
    RouterSelectionBias { layer: u32 },
    RoutedGate { layer: u32 },
    RoutedUp { layer: u32 },
    RoutedDown { layer: u32 },
    SharedGate { layer: u32 },
    SharedUp { layer: u32 },
    SharedDown { layer: u32 },
}

impl Hy3TensorRole {
    pub fn classify(name: &str, block_count: u32) -> Result<Self, Hy3Error> {
        match name {
            "token_embd.weight" => return Ok(Self::TokenEmbedding),
            "output_norm.weight" => return Ok(Self::OutputNorm),
            "output.weight" => return Ok(Self::Output),
            _ => {}
        }

        let invalid = || Hy3Error::InvalidTensorName {
            name: name.to_owned(),
            expected: "an exact complete Hy3 tensor name in the configured layer regime",
        };
        let remainder = name.strip_prefix("blk.").ok_or_else(invalid)?;
        let (layer_text, suffix) = remainder.split_once('.').ok_or_else(invalid)?;
        if layer_text.is_empty()
            || !layer_text.bytes().all(|byte| byte.is_ascii_digit())
            || (layer_text.len() > 1 && layer_text.starts_with('0'))
        {
            return Err(invalid());
        }
        let layer = layer_text.parse::<u32>().map_err(|_| invalid())?;
        if layer >= block_count {
            return Err(invalid());
        }

        let role = match suffix {
            "attn_norm.weight" => Self::AttentionNorm { layer },
            "attn_q.weight" => Self::AttentionQ { layer },
            "attn_q_norm.weight" => Self::AttentionQNorm { layer },
            "attn_k.weight" => Self::AttentionK { layer },
            "attn_k_norm.weight" => Self::AttentionKNorm { layer },
            "attn_v.weight" => Self::AttentionV { layer },
            "attn_output.weight" => Self::AttentionOutput { layer },
            "ffn_norm.weight" => Self::FfnNorm { layer },
            "ffn_gate.weight" if layer == 0 => Self::DenseGate { layer },
            "ffn_up.weight" if layer == 0 => Self::DenseUp { layer },
            "ffn_down.weight" if layer == 0 => Self::DenseDown { layer },
            "ffn_gate_inp.weight" if layer > 0 => Self::RouterInput { layer },
            "exp_probs_b" if layer > 0 => Self::RouterSelectionBias { layer },
            "ffn_gate_exps.weight" if layer > 0 => Self::RoutedGate { layer },
            "ffn_up_exps.weight" if layer > 0 => Self::RoutedUp { layer },
            "ffn_down_exps.weight" if layer > 0 => Self::RoutedDown { layer },
            "ffn_gate_shexp.weight" if layer > 0 => Self::SharedGate { layer },
            "ffn_up_shexp.weight" if layer > 0 => Self::SharedUp { layer },
            "ffn_down_shexp.weight" if layer > 0 => Self::SharedDown { layer },
            _ => return Err(invalid()),
        };
        Ok(role)
    }

    pub const fn is_routed_expert(self) -> bool {
        matches!(
            self,
            Self::RoutedGate { .. } | Self::RoutedUp { .. } | Self::RoutedDown { .. }
        )
    }
}
