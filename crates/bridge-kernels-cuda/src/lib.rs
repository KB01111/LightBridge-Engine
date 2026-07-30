//! Optional CUDA packed-kernel ABI.
//!
//! The default build deliberately has no CUDA link dependency. A native
//! implementation must expose this exact ABI and pass canary/oracle gates
//! before the scheduler can select it.

mod nvrtc;

use serde::{Deserialize, Serialize};

pub use nvrtc::{
    packed_q8k_gemv_batch_into, packed_q8k_gemv_into, packed_q8k_gemv_pair_into, runtime_nvrtc_canary,
    runtime_packed_q8k_oracle, runtime_reusable_packed_q8k_canary, CudaNvrtcCanary,
    CudaPackedQ8KBatchExecution, CudaPackedQ8KBatchItem, CudaPackedQ8KExecution, CudaPackedQ8KFormatOracle,
    CudaPackedQ8KOracle, CudaPackedQ8KPairExecution, CudaReusablePackedQ8KCanary, CudaRuntimeError,
};

pub const CUDA_KERNEL_ABI_VERSION: u32 = 1;
pub const CUDA_TARGET_ARCHITECTURE: &str = "sm_89";
pub const CUDA_PTX_FALLBACK: &str = "compute_89";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaBuildCapabilities {
    pub abi_version: u32,
    pub native_canary_compiled: bool,
    pub runtime_nvrtc_compiled: bool,
    pub runtime_packed_oracle_compiled: bool,
    pub reusable_packed_executor_compiled: bool,
    pub packed_kernels_compiled: bool,
    pub target_architecture: String,
    pub ptx_fallback: String,
    pub strict_fp32: bool,
    pub rejection_reason: Option<String>,
}

pub fn build_capabilities() -> CudaBuildCapabilities {
    let native_canary_compiled = cfg!(feature = "cuda-native");
    CudaBuildCapabilities {
        abi_version: CUDA_KERNEL_ABI_VERSION,
        native_canary_compiled,
        runtime_nvrtc_compiled: cfg!(windows),
        runtime_packed_oracle_compiled: cfg!(windows),
        reusable_packed_executor_compiled: cfg!(windows),
        packed_kernels_compiled: cfg!(windows),
        target_architecture: CUDA_TARGET_ARCHITECTURE.to_owned(),
        ptx_fallback: CUDA_PTX_FALLBACK.to_owned(),
        strict_fp32: true,
        rejection_reason: Some(if cfg!(windows) {
            "the dynamically loaded NVRTC/Driver packed kernels and explicit streaming model \
             backend compile without an MSVC link dependency; the backend remains opt-in until \
             the multi-prompt correctness corpus and automatic-selection performance gates pass"
                .to_owned()
        } else if native_canary_compiled {
            "the sm_89/PTX CUDA runtime canary is compiled, but packed GEMV kernels and full-model \
             correctness evidence are not"
                .to_owned()
        } else {
            "bridge-kernels-cuda was built without the opt-in cuda-native canary; packed kernels \
             are not compiled"
                .to_owned()
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaRuntimeCanary {
    pub compute_major: i32,
    pub compute_minor: i32,
    pub global_memory_bytes: u64,
}

#[cfg(feature = "cuda-native")]
pub fn runtime_canary() -> Result<CudaRuntimeCanary, i32> {
    unsafe extern "C" {
        fn bridge_cuda_canary_v1(
            abi_version: u32,
            major: *mut i32,
            minor: *mut i32,
            global_memory_bytes: *mut u64,
        ) -> i32;
    }

    let mut compute_major = 0;
    let mut compute_minor = 0;
    let mut global_memory_bytes = 0;
    let status = unsafe {
        bridge_cuda_canary_v1(
            CUDA_KERNEL_ABI_VERSION,
            &mut compute_major,
            &mut compute_minor,
            &mut global_memory_bytes,
        )
    };
    if status != 0 {
        return Err(status);
    }
    Ok(CudaRuntimeCanary {
        compute_major,
        compute_minor,
        global_memory_bytes,
    })
}

#[cfg(not(feature = "cuda-native"))]
pub fn runtime_canary() -> Result<CudaRuntimeCanary, i32> {
    Err(-1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaArenaConfig {
    pub pinned_read_slots: usize,
    pub device_staging_arenas: usize,
    pub slot_bytes: usize,
    pub reserved_vram_bytes: u64,
}

impl CudaArenaConfig {
    pub fn validate(self) -> Result<(), &'static str> {
        if self.pinned_read_slots == 0 {
            return Err("CUDA requires non-zero pinned_read_slots");
        }
        if self.device_staging_arenas != 2 {
            return Err("CUDA requires exactly two device_staging_arenas");
        }
        if self.slot_bytes == 0 {
            return Err("CUDA requires non-zero slot_bytes");
        }
        if self.reserved_vram_bytes < 1_280 * 1024 * 1024 {
            return Err("CUDA must reserve at least 1.25 GiB of VRAM");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_build_is_an_explicit_non_native_capability() {
        let capabilities = build_capabilities();
        assert_eq!(capabilities.abi_version, CUDA_KERNEL_ABI_VERSION);
        assert_eq!(capabilities.native_canary_compiled, cfg!(feature = "cuda-native"));
        assert_eq!(capabilities.runtime_nvrtc_compiled, cfg!(windows));
        assert_eq!(capabilities.runtime_packed_oracle_compiled, cfg!(windows));
        assert_eq!(capabilities.reusable_packed_executor_compiled, cfg!(windows));
        assert_eq!(capabilities.packed_kernels_compiled, cfg!(windows));
        assert!(capabilities.rejection_reason.is_some());
    }

    #[test]
    fn arenas_enforce_double_buffering_and_vram_reserve() {
        assert!(CudaArenaConfig {
            pinned_read_slots: 8,
            device_staging_arenas: 2,
            slot_bytes: 8 * 1024 * 1024,
            reserved_vram_bytes: 1_280 * 1024 * 1024,
        }
        .validate()
        .is_ok());
        assert!(CudaArenaConfig {
            pinned_read_slots: 8,
            device_staging_arenas: 1,
            slot_bytes: 8 * 1024 * 1024,
            reserved_vram_bytes: 1_280 * 1024 * 1024,
        }
        .validate()
        .is_err());
    }

    #[cfg(windows)]
    #[test]
    fn reusable_executor_rejects_malformed_input_before_initializing_cuda() {
        let mut output = [f32::from_bits(0x7fc0_1234)];
        let error = packed_q8k_gemv_into(
            bridge_quant_layout::GgmlType::Q4_K,
            &[],
            &[0_u8; bridge_quant_layout::Q8_K_BLOCK_BYTES],
            bridge_quant_layout::Q8_K_BLOCK_ELEMENTS,
            &mut output,
        )
        .unwrap_err();
        assert!(matches!(error, CudaRuntimeError::PackedValidation { .. }));
        assert_eq!(output[0].to_bits(), 0x7fc0_1234);
    }

    #[cfg(windows)]
    #[test]
    fn reusable_batch_is_atomic_and_executes_multiple_validated_items_when_available() {
        use bridge_quant_layout::{GgmlType, Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMENTS};

        let first_weights = [0_u8; 144];
        let second_weights = [0_u8; 288];
        let q8 = [0_u8; Q8_K_BLOCK_BYTES];
        let sentinel = [
            f32::from_bits(0x7fc0_1111),
            f32::from_bits(0x7fc0_2222),
            f32::from_bits(0x7fc0_3333),
        ];
        let mut malformed_output = sentinel;
        let malformed = [
            CudaPackedQ8KBatchItem {
                weight_type: GgmlType::Q4_K,
                weights: &first_weights,
                q8: &q8,
                logical_elements: Q8_K_BLOCK_ELEMENTS,
                rows: 1,
            },
            CudaPackedQ8KBatchItem {
                weight_type: GgmlType::Q4_K,
                weights: &[],
                q8: &q8,
                logical_elements: Q8_K_BLOCK_ELEMENTS,
                rows: 2,
            },
        ];
        assert!(matches!(
            packed_q8k_gemv_batch_into(&malformed, &mut malformed_output),
            Err(CudaRuntimeError::PackedValidation { .. })
        ));
        assert_eq!(malformed_output.map(f32::to_bits), sentinel.map(f32::to_bits));

        if runtime_reusable_packed_q8k_canary().is_err() {
            return;
        }
        let items = [
            CudaPackedQ8KBatchItem {
                weight_type: GgmlType::Q4_K,
                weights: &first_weights,
                q8: &q8,
                logical_elements: Q8_K_BLOCK_ELEMENTS,
                rows: 1,
            },
            CudaPackedQ8KBatchItem {
                weight_type: GgmlType::Q4_K,
                weights: &second_weights,
                q8: &q8,
                logical_elements: Q8_K_BLOCK_ELEMENTS,
                rows: 2,
            },
        ];
        let mut output = sentinel;
        let execution = packed_q8k_gemv_batch_into(&items, &mut output).unwrap();
        assert_eq!(execution.items, 2);
        assert_eq!(execution.rows, 3);
        assert_eq!(output.map(f32::to_bits), [0.0_f32.to_bits(); 3]);
    }
}
