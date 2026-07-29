use bridge_core::ggml_type::GgmlType;
use bridge_kernels_reference::{EncodedTensorView, KernelError, PackedMatrix, PayloadEndian};

#[test]
fn checked_view_and_matrix_preserve_exact_ggml_orientation() {
    let bytes = [0_u8; 3 * 2 * 4];
    let view = EncodedTensorView::new(GgmlType::F32, PayloadEndian::Little, &[3, 2], &bytes).unwrap();
    assert_eq!(view.shape(), [3, 2]);
    assert_eq!(view.bytes(), bytes);

    let matrix = PackedMatrix::new(view).unwrap();
    assert_eq!(matrix.ty(), GgmlType::F32);
    assert_eq!(matrix.input_width(), 3);
    assert_eq!(matrix.output_width(), 2);
    assert_eq!(matrix.row_bytes(), 12);
    assert_eq!(matrix.row(0), &bytes[..12]);
    assert_eq!(matrix.row(1), &bytes[12..]);
}

#[test]
fn view_rejects_bad_rank_dimensions_alignment_and_length() {
    assert!(matches!(
        EncodedTensorView::new(GgmlType::F32, PayloadEndian::Little, &[], &[]),
        Err(KernelError::ShapeRankTooLarge {
            maximum: 3,
            actual: 0,
        })
    ));
    assert!(matches!(
        EncodedTensorView::new(GgmlType::F32, PayloadEndian::Little, &[1, 1, 1, 1], &[0; 4]),
        Err(KernelError::ShapeRankTooLarge {
            maximum: 3,
            actual: 4,
        })
    ));
    assert!(matches!(
        EncodedTensorView::new(GgmlType::F32, PayloadEndian::Little, &[0, 1], &[]),
        Err(KernelError::ZeroDimension { dimension: 0 })
    ));
    assert!(matches!(
        EncodedTensorView::new(GgmlType::Q4_K, PayloadEndian::Little, &[255, 1], &[]),
        Err(KernelError::DimensionMismatch {
            field: "block-aligned leading tensor dimension",
            expected: 256,
            actual: 255,
        })
    ));
    assert!(matches!(
        EncodedTensorView::new(GgmlType::Q4_K, PayloadEndian::Little, &[256, 2], &[0; 287]),
        Err(KernelError::EncodedLengthMismatch {
            expected: 288,
            actual: 287,
        })
    ));
}

#[test]
fn packed_matrix_rejects_rank_big_endian_and_unsupported_types() {
    let vector = EncodedTensorView::new(GgmlType::F32, PayloadEndian::Little, &[2], &[0; 8]).unwrap();
    assert!(matches!(
        PackedMatrix::new(vector),
        Err(KernelError::TensorRank {
            expected: 2,
            actual: 1,
        })
    ));

    let big_endian = EncodedTensorView::new(GgmlType::F32, PayloadEndian::Big, &[2, 1], &[0; 8]).unwrap();
    assert_eq!(
        PackedMatrix::new(big_endian).unwrap_err(),
        KernelError::BigEndianPayload
    );

    let f16 = EncodedTensorView::new(GgmlType::F16, PayloadEndian::Little, &[2, 1], &[0; 4]).unwrap();
    assert_eq!(
        PackedMatrix::new(f16).unwrap_err(),
        KernelError::UnsupportedType { ty: GgmlType::F16 }
    );
}
