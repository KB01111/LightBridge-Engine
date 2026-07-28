//! Checked tensor descriptors for GGUF tensor records.

use std::ops::Range;

use crate::error::{CoreError, Result};
use crate::ggml_type::GgmlType;

/// A validated tensor location and encoding. Its fields stay private so every instance has
/// passed the same shape, stride, and overflow checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    name: String,
    shape: [u64; 4],
    n_dims: u32,
    ty: GgmlType,
    relative_offset: u64,
    strides: [u64; 4],
}

impl TensorDesc {
    pub fn new(name: impl Into<String>, shape: &[u64], ty: GgmlType, relative_offset: u64) -> Result<Self> {
        let n_dims = u32::try_from(shape.len()).map_err(|_| CoreError::InvalidTensorRank(u32::MAX))?;
        if !(1..=4).contains(&n_dims) {
            return Err(CoreError::InvalidTensorRank(n_dims));
        }
        let mut fixed_shape = [1; 4];
        for (index, &dimension) in shape.iter().enumerate() {
            if dimension == 0 {
                return Err(CoreError::ZeroTensorDimension { dimension: index });
            }
            fixed_shape[index] = dimension;
        }
        let strides = compute_strides(fixed_shape, n_dims, ty)?;
        Ok(Self {
            name: name.into(),
            shape: fixed_shape,
            n_dims,
            ty,
            relative_offset,
            strides,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &[u64] {
        &self.shape[..self.n_dims as usize]
    }
    pub const fn n_dims(&self) -> u32 {
        self.n_dims
    }
    pub const fn ty(&self) -> GgmlType {
        self.ty
    }
    pub const fn relative_offset(&self) -> u64 {
        self.relative_offset
    }
    pub const fn strides(&self) -> [u64; 4] {
        self.strides
    }

    pub fn element_count(&self) -> Result<u64> {
        self.shape().iter().try_fold(1_u64, |total, &dimension| {
            total
                .checked_mul(dimension)
                .ok_or(CoreError::ArithmeticOverflow("tensor element count"))
        })
    }

    pub fn row_bytes(&self) -> Result<u64> {
        self.ty.row_size(self.shape[0])
    }

    pub fn encoded_bytes(&self) -> Result<u64> {
        if self.n_dims == 1 {
            return self.row_bytes();
        }
        let last = self.n_dims as usize - 1;
        self.strides[last]
            .checked_mul(self.shape[last])
            .ok_or(CoreError::ArithmeticOverflow("tensor encoded byte length"))
    }

    pub fn checked_absolute_range(&self, data_offset: u64, file_len: u64) -> Result<Range<u64>> {
        let start = data_offset
            .checked_add(self.relative_offset)
            .ok_or(CoreError::ArithmeticOverflow("tensor absolute offset"))?;
        let end = start
            .checked_add(self.encoded_bytes()?)
            .ok_or(CoreError::ArithmeticOverflow("tensor absolute end offset"))?;
        if end > file_len {
            return Err(CoreError::TensorOutOfBounds { start, end, file_len });
        }
        Ok(start..end)
    }
}

/// Compute GGML's byte strides (`nb[]`) after validating the leading block dimension.
pub fn compute_strides(ne: [u64; 4], n_dims: u32, ty: GgmlType) -> Result<[u64; 4]> {
    if !(1..=4).contains(&n_dims) {
        return Err(CoreError::InvalidTensorRank(n_dims));
    }
    for (index, &dimension) in ne[..n_dims as usize].iter().enumerate() {
        if dimension == 0 {
            return Err(CoreError::ZeroTensorDimension { dimension: index });
        }
    }
    let mut strides = [0; 4];
    strides[0] = ty.type_size();
    if n_dims == 1 {
        ty.row_size(ne[0])?;
        return Ok(strides);
    }
    strides[1] = ty.row_size(ne[0])?;
    for index in 2..n_dims as usize {
        strides[index] = strides[index - 1]
            .checked_mul(ne[index - 1])
            .ok_or(CoreError::ArithmeticOverflow("GGML tensor stride"))?;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_rows_require_the_first_dimension_to_be_block_aligned() {
        let strides = compute_strides([256, 3, 1, 1], 2, GgmlType::Q4K).unwrap();
        assert_eq!(strides[0], GgmlType::Q4K.type_size());
        assert_eq!(strides[1], GgmlType::Q4K.row_size(256).unwrap());
        assert!(compute_strides([255, 256, 1, 1], 2, GgmlType::Q4K).is_err());
    }

    #[test]
    fn descriptor_rejects_overflowing_file_ranges() {
        let tensor = TensorDesc::new("weights", &[256, 2], GgmlType::Q4K, u64::MAX - 8).unwrap();
        assert!(matches!(
            tensor.checked_absolute_range(0, u64::MAX),
            Err(CoreError::ArithmeticOverflow(_))
        ));
    }

    #[test]
    fn rank_one_quantized_tensor_uses_one_encoded_row_for_length_and_range() {
        let tensor = TensorDesc::new("vector", &[256], GgmlType::Q4K, 32).unwrap();
        assert_eq!(tensor.encoded_bytes().unwrap(), 144);
        assert_eq!(tensor.checked_absolute_range(8, 184).unwrap(), 40..184);
    }
}
