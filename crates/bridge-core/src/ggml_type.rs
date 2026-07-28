//! GGML type descriptors used by the GGUF reader.

use crate::error::{CoreError, Result};

/// A `ggml_type` discriminant as stored in a GGUF tensor record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl GgmlType {
    pub const ALL: &'static [Self] = &[
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
        Self::IQ2_XXS,
        Self::IQ2_XS,
        Self::IQ3_XXS,
        Self::IQ1_S,
        Self::IQ4_NL,
        Self::IQ3_S,
        Self::IQ2_S,
        Self::IQ4_XS,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::IQ1_M,
        Self::BF16,
        Self::TQ1_0,
        Self::TQ2_0,
        Self::MXFP4,
        Self::NVFP4,
        Self::Q1_0,
        Self::Q2_0,
    ];

    /// Preferred spelling for the GGML Q4_K type.
    pub const Q4K: Self = Self::Q4_K;
    /// Preferred spelling for NVIDIA's FP4 block type.
    #[allow(non_upper_case_globals)]
    pub const Nvfp4: Self = Self::NVFP4;

    pub fn from_discriminant(value: u32) -> Result<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|ty| ty.discriminant() == value)
            .ok_or(CoreError::UnknownGgmlType(value))
    }

    pub const fn discriminant(self) -> u32 {
        self as u32
    }

    pub const fn block_size(self) -> u64 {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4_NL | MXFP4 => 32,
            Q1_0 => 128,
            Q2_0 | NVFP4 => 64,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K | IQ2_XXS | IQ2_XS | IQ3_XXS | IQ1_S | IQ3_S | IQ2_S
            | IQ4_XS | IQ1_M | TQ1_0 | TQ2_0 => 256,
        }
    }

    pub const fn type_size(self) -> u64 {
        use GgmlType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            F64 | I64 => 8,
            I8 => 1,
            Q4_0 | IQ4_NL => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 => 36,
            Q2_K => 84,
            Q3_K | IQ3_S => 110,
            Q4_K => 144,
            Q5_K => 176,
            Q6_K => 210,
            Q8_K => 292,
            IQ2_XXS | TQ2_0 => 66,
            IQ2_XS => 74,
            IQ2_S => 82,
            IQ3_XXS => 98,
            IQ1_S => 50,
            IQ4_XS => 136,
            IQ1_M => 56,
            TQ1_0 => 54,
            MXFP4 => 17,
            NVFP4 => 36,
            Q1_0 | Q2_0 => 18,
        }
    }

    /// Encoded bytes in one row whose leading dimension is `ne0`.
    pub fn row_size(self, ne0: u64) -> Result<u64> {
        let block = self.block_size();
        if ne0 % block != 0 {
            return Err(CoreError::NotBlockAligned {
                ne: ne0,
                block,
                ty: self.name(),
            });
        }
        (ne0 / block)
            .checked_mul(self.type_size())
            .ok_or(CoreError::ArithmeticOverflow("GGML row size"))
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
            Self::NVFP4 => "NVFP4",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_ggml_abi_sizes_are_exact() {
        assert_eq!(GgmlType::Q1_0.block_size(), 128);
        assert_eq!(GgmlType::Q1_0.type_size(), 18);
        assert_eq!(GgmlType::Q2_0.block_size(), 64);
        assert_eq!(GgmlType::Q2_0.type_size(), 18);
        assert_eq!(GgmlType::Nvfp4.block_size(), 64);
        assert_eq!(GgmlType::Nvfp4.type_size(), 36);
    }

    #[test]
    fn unknown_discriminants_are_rejected() {
        assert!(matches!(
            GgmlType::from_discriminant(u32::MAX),
            Err(CoreError::UnknownGgmlType(_))
        ));
    }
}
