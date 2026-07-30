use bridge_core::ggml_type::GgmlType;

use crate::error::Result;
use crate::KernelError;

const MAX_TENSOR_RANK: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadEndian {
    Little,
    Big,
}

/// A non-owning tensor payload whose exact shape and encoded length have been
/// checked. Dimensions use GGML order (`ne[0]` first).
#[derive(Debug, Clone, Copy)]
pub struct EncodedTensorView<'a> {
    ty: GgmlType,
    endian: PayloadEndian,
    shape: [usize; MAX_TENSOR_RANK],
    rank: usize,
    bytes: &'a [u8],
}

impl<'a> EncodedTensorView<'a> {
    pub fn new(ty: GgmlType, endian: PayloadEndian, shape: &[usize], bytes: &'a [u8]) -> Result<Self> {
        if shape.is_empty() || shape.len() > MAX_TENSOR_RANK {
            return Err(KernelError::ShapeRankTooLarge {
                maximum: MAX_TENSOR_RANK,
                actual: shape.len(),
            });
        }
        let mut fixed_shape = [1_usize; MAX_TENSOR_RANK];
        for (dimension, &length) in shape.iter().enumerate() {
            if length == 0 {
                return Err(KernelError::ZeroDimension { dimension });
            }
            fixed_shape[dimension] = length;
        }

        let row_bytes_u64 = ty
            .row_size(
                u64::try_from(shape[0]).map_err(|_| KernelError::ArithmeticOverflow {
                    operation: "leading tensor dimension conversion",
                })?,
            )
            .map_err(|_| KernelError::DimensionMismatch {
                field: "block-aligned leading tensor dimension",
                expected: usize::try_from(ty.block_size()).unwrap_or(usize::MAX),
                actual: shape[0],
            })?;
        let row_bytes = usize::try_from(row_bytes_u64).map_err(|_| KernelError::ArithmeticOverflow {
            operation: "tensor row byte conversion",
        })?;
        let row_count = shape[1..].iter().try_fold(1_usize, |total, &dimension| {
            total
                .checked_mul(dimension)
                .ok_or(KernelError::ArithmeticOverflow {
                    operation: "tensor row count",
                })
        })?;
        let expected = row_bytes
            .checked_mul(row_count)
            .ok_or(KernelError::ArithmeticOverflow {
                operation: "tensor encoded byte length",
            })?;
        if bytes.len() != expected {
            return Err(KernelError::EncodedLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }

        Ok(Self {
            ty,
            endian,
            shape: fixed_shape,
            rank: shape.len(),
            bytes,
        })
    }

    pub const fn ty(self) -> GgmlType {
        self.ty
    }

    pub const fn endian(self) -> PayloadEndian {
        self.endian
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// A checked GGML `[input_width, output_width]` packed matrix.
#[derive(Debug, Clone, Copy)]
pub struct PackedMatrix<'a> {
    ty: GgmlType,
    input_width: usize,
    output_width: usize,
    row_bytes: usize,
    bytes: &'a [u8],
}

impl<'a> PackedMatrix<'a> {
    pub fn new(view: EncodedTensorView<'a>) -> Result<Self> {
        if view.endian() != PayloadEndian::Little {
            return Err(KernelError::BigEndianPayload);
        }
        if view.shape().len() != 2 {
            return Err(KernelError::TensorRank {
                expected: 2,
                actual: view.shape().len(),
            });
        }
        if !matches!(
            view.ty(),
            GgmlType::F32 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ2_S | GgmlType::IQ3_S
        ) {
            return Err(KernelError::UnsupportedType { ty: view.ty() });
        }
        let row_bytes_u64 = view
            .ty()
            .row_size(
                u64::try_from(view.shape()[0]).map_err(|_| KernelError::ArithmeticOverflow {
                    operation: "matrix input width conversion",
                })?,
            )
            .map_err(|_| KernelError::DimensionMismatch {
                field: "block-aligned matrix input width",
                expected: usize::try_from(view.ty().block_size()).unwrap_or(usize::MAX),
                actual: view.shape()[0],
            })?;
        let row_bytes = usize::try_from(row_bytes_u64).map_err(|_| KernelError::ArithmeticOverflow {
            operation: "matrix row byte conversion",
        })?;

        Ok(Self {
            ty: view.ty(),
            input_width: view.shape()[0],
            output_width: view.shape()[1],
            row_bytes,
            bytes: view.bytes(),
        })
    }

    pub fn from_parts(
        ty: GgmlType,
        endian: PayloadEndian,
        input_width: usize,
        output_width: usize,
        bytes: &'a [u8],
    ) -> Result<Self> {
        Self::new(EncodedTensorView::new(
            ty,
            endian,
            &[input_width, output_width],
            bytes,
        )?)
    }

    pub const fn ty(self) -> GgmlType {
        self.ty
    }

    pub const fn input_width(self) -> usize {
        self.input_width
    }

    pub const fn output_width(self) -> usize {
        self.output_width
    }

    /// Gets the encoded size of each matrix row in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let matrix = PackedMatrix::from_parts(
    ///     GgmlType::F32,
    ///     PayloadEndian::Little,
    ///     2,
    ///     1,
    ///     &[0; 8],
    /// ).unwrap();
    /// assert_eq!(matrix.row_bytes(), 8);
    /// ```
    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }

    /// Provides access to the encoded matrix bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let matrix = PackedMatrix::from_parts(
    ///     GgmlType::F32,
    ///     PayloadEndian::Little,
    ///     1,
    ///     1,
    ///     &[0, 0, 0, 0],
    /// ).unwrap();
    /// assert_eq!(matrix.bytes(), &[0, 0, 0, 0]);
    /// ```
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Provides the encoded bytes for a matrix row.
    ///
    /// # Panics
    ///
    /// Panics if `index` does not identify a complete row in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// let bytes: Vec<u8> = (0..16).collect();
    /// let matrix = PackedMatrix::from_parts(
    ///     GgmlType::F32,
    ///     PayloadEndian::Little,
    ///     2,
    ///     2,
    ///     &bytes,
    /// ).unwrap();
    ///
    /// assert_eq!(matrix.row(1), &[8, 9, 10, 11]);
    /// ```
    pub fn row(self, index: usize) -> &'a [u8] {
        let start = index * self.row_bytes;
        &self.bytes[start..start + self.row_bytes]
    }
}
