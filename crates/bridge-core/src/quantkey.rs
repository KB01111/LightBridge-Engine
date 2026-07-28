//! The kernel dispatch key.
//!
//! Every optimized routine in the engine is selected by a [`QuantKernelKey`]. The key is
//! deliberately explicit: it carries the *actual* weight `ggml_type`, the activation encoding,
//! the exact shape, the batch class, the device, and the physical storage layout. Autotuning
//! results (`bridge tune-kernels`) are persisted against this key plus a
//! [`HardwareFingerprint`], and invalidated when any component changes.

use crate::ggml_type::GgmlType;

/// How activations are fed to a kernel. Chosen by measurement, never assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum ActivationType {
    /// Plain f32 activations, f32 accumulation. Always available, always the reference.
    F32,
    /// f16 activations (CUDA `half`).
    F16,
    /// bf16 activations.
    Bf16,
    /// Per-32-value Q8_0-style activation blocks feeding integer dot products.
    Q8Block,
    /// Per-32-value Q8_1-style blocks (carries a sum, needed by asymmetric weight types).
    Q8BlockSum,
}

impl ActivationType {
    pub const fn name(self) -> &'static str {
        match self {
            ActivationType::F32 => "f32",
            ActivationType::F16 => "f16",
            ActivationType::Bf16 => "bf16",
            ActivationType::Q8Block => "q8",
            ActivationType::Q8BlockSum => "q8s",
        }
    }
}

/// The operation a kernel implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum KernelOp {
    /// `y[out] = sum_in W[out, in] * x[in]` — batch-one matrix-vector.
    Gemv,
    /// Fused routed-expert gate+up: one pass over `x`, two weight matrices, SiLU(gate) * up.
    ExpertGateUpGemv,
    /// Routed-expert down projection with router-weight scaling accumulated into the MoE output.
    ExpertDownGemvAccum,
    /// Router projection followed by exact top-k selection.
    RouterTopK,
    /// Grouped expert GEMM used by prefill and speculative verification.
    GroupedExpertGemm,
    /// Dense matrix-matrix product.
    Gemm,
    /// RMS normalization.
    RmsNorm,
    /// Plain (mean/variance) layer norm with weight and bias.
    LayerNorm,
    /// Rotary position embedding.
    Rope,
    /// Exact top-k selection.
    TopK,
    /// Residual add.
    ResidualAdd,
    /// Final logits projection.
    Logits,
}

impl KernelOp {
    pub const fn name(self) -> &'static str {
        match self {
            KernelOp::Gemv => "gemv",
            KernelOp::ExpertGateUpGemv => "expert_gate_up_gemv",
            KernelOp::ExpertDownGemvAccum => "expert_down_gemv_accumulate",
            KernelOp::RouterTopK => "router_topk",
            KernelOp::GroupedExpertGemm => "grouped_expert_gemm",
            KernelOp::Gemm => "gemm",
            KernelOp::RmsNorm => "rms_norm",
            KernelOp::LayerNorm => "layer_norm",
            KernelOp::Rope => "rope",
            KernelOp::TopK => "top_k",
            KernelOp::ResidualAdd => "residual_add",
            KernelOp::Logits => "logits",
        }
    }
}

/// Coarse batch bucket. Decode (1) and prefill (many) want fundamentally different kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum BatchClass {
    /// Exactly one token: GEMV territory.
    One,
    /// 2..=8 tokens, e.g. MTP verification.
    Tiny,
    /// 9..=64 tokens.
    Small,
    /// 65..=512 tokens.
    Medium,
    /// > 512 tokens: full prefill, GEMM territory.
    Large,
}

impl BatchClass {
    pub fn of(n_tokens: usize) -> BatchClass {
        match n_tokens {
            0 | 1 => BatchClass::One,
            2..=8 => BatchClass::Tiny,
            9..=64 => BatchClass::Small,
            65..=512 => BatchClass::Medium,
            _ => BatchClass::Large,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            BatchClass::One => "b1",
            BatchClass::Tiny => "b2-8",
            BatchClass::Small => "b9-64",
            BatchClass::Medium => "b65-512",
            BatchClass::Large => "b512+",
        }
    }
}

/// Which physical device executes the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceKind {
    /// Scalar Rust, no intrinsics. The numerical reference.
    CpuScalar,
    CpuAvx2,
    CpuAvx512,
    /// AVX-512 with VNNI integer dot products.
    CpuAvx512Vnni,
    Cuda,
    /// Feature-gated, disabled by default, never a mandatory transfer bridge.
    IgpuVulkan,
}

impl DeviceKind {
    pub const fn name(self) -> &'static str {
        match self {
            DeviceKind::CpuScalar => "cpu-scalar",
            DeviceKind::CpuAvx2 => "cpu-avx2",
            DeviceKind::CpuAvx512 => "cpu-avx512",
            DeviceKind::CpuAvx512Vnni => "cpu-avx512-vnni",
            DeviceKind::Cuda => "cuda",
            DeviceKind::IgpuVulkan => "igpu-vulkan",
        }
    }

    pub const fn is_cpu(self) -> bool {
        matches!(
            self,
            DeviceKind::CpuScalar | DeviceKind::CpuAvx2 | DeviceKind::CpuAvx512 | DeviceKind::CpuAvx512Vnni
        )
    }
}

/// Physical arrangement of the weight bytes the kernel will read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum StorageLayout {
    /// Blocks exactly as they appear in the GGUF file.
    GgufNative,
    /// `.bridge` sidecar, `[gate][up][down]` per expert.
    ExpertMajorSequential,
    /// `.bridge` sidecar, gate/up blocks interleaved into paired tiles, then down.
    ExpertMajorFusedGateUp,
}

impl StorageLayout {
    pub const fn name(self) -> &'static str {
        match self {
            StorageLayout::GgufNative => "gguf",
            StorageLayout::ExpertMajorSequential => "sequential",
            StorageLayout::ExpertMajorFusedGateUp => "fused-gate-up",
        }
    }
}

/// Full dispatch key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct QuantKernelKey {
    pub weight_type: GgmlType,
    pub activation_type: ActivationType,
    pub operation: KernelOp,
    /// Reduction width (`ne[0]` of the weight).
    pub input_width: u32,
    /// Number of output rows (`ne[1]` of the weight).
    pub output_width: u32,
    pub batch_class: BatchClass,
    pub device: DeviceKind,
    pub storage_layout: StorageLayout,
}

impl QuantKernelKey {
    /// Stable string form used for autotune cache keys and diagnostics.
    pub fn cache_key(&self) -> String {
        format!(
            "{}/{}/{}/{}x{}/{}/{}/{}",
            self.operation.name(),
            self.weight_type.name(),
            self.activation_type.name(),
            self.input_width,
            self.output_width,
            self.batch_class.name(),
            self.device.name(),
            self.storage_layout.name()
        )
    }
}

impl std::fmt::Display for QuantKernelKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.cache_key())
    }
}

/// Everything an autotune record must be invalidated against.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct HardwareFingerprint {
    pub cpu_brand: String,
    pub cpu_isa: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub gpu_name: String,
    pub compute_capability: Option<(u32, u32)>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub engine_format_version: u32,
}

impl HardwareFingerprint {
    pub fn digest(&self) -> String {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        self.hash(&mut h);
        format!("{:016x}", h.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_classes() {
        assert_eq!(BatchClass::of(1), BatchClass::One);
        assert_eq!(BatchClass::of(8), BatchClass::Tiny);
        assert_eq!(BatchClass::of(64), BatchClass::Small);
        assert_eq!(BatchClass::of(512), BatchClass::Medium);
        assert_eq!(BatchClass::of(513), BatchClass::Large);
    }

    #[test]
    fn hy3_dispatch_key_is_stable_and_distinguishes_layouts() {
        let base = QuantKernelKey {
            weight_type: GgmlType::IQ2_XXS,
            activation_type: ActivationType::Q8Block,
            operation: KernelOp::TopK,
            input_width: 6144,
            output_width: 2048,
            batch_class: BatchClass::One,
            device: DeviceKind::CpuAvx2,
            storage_layout: StorageLayout::GgufNative,
        };
        assert_eq!(base.cache_key(), "top_k/IQ2_XXS/q8/6144x2048/b1/cpu-avx2/gguf");
        let mut other = base;
        other.storage_layout = StorageLayout::ExpertMajorFusedGateUp;
        assert_ne!(base.cache_key(), other.cache_key());
    }
}
