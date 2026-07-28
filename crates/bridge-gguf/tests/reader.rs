use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bridge_gguf::{Endianness, GgufError, GgufReader, GgufValue, GgufValueType, MetadataError, ReaderLimits};

struct ObservingAllocator;

static OBSERVE_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static LARGEST_ALLOCATION: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Every operation is forwarded unchanged to `System`; the observer only records
// requested sizes in atomics and does not alter allocation semantics.
unsafe impl GlobalAlloc for ObservingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        observe_allocation(layout.size());
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        observe_allocation(layout.size());
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` came from the system allocator through this wrapper.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        observe_allocation(new_size);
        // SAFETY: the original allocation and layout came from `System`, and `new_size` is
        // forwarded unchanged.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: ObservingAllocator = ObservingAllocator;

fn observe_allocation(size: usize) {
    if OBSERVE_ALLOCATIONS.load(Ordering::Relaxed) {
        LARGEST_ALLOCATION.fetch_max(size, Ordering::Relaxed);
    }
}

struct AllocationObservation;

impl AllocationObservation {
    fn start() -> Self {
        LARGEST_ALLOCATION.store(0, Ordering::Relaxed);
        OBSERVE_ALLOCATIONS.store(true, Ordering::Release);
        Self
    }

    fn finish(self) -> usize {
        OBSERVE_ALLOCATIONS.store(false, Ordering::Release);
        let largest = LARGEST_ALLOCATION.load(Ordering::Acquire);
        std::mem::forget(self);
        largest
    }
}

impl Drop for AllocationObservation {
    fn drop(&mut self) {
        OBSERVE_ALLOCATIONS.store(false, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    fn magic(self) -> [u8; 4] {
        match self {
            Self::Little => *b"GGUF",
            Self::Big => *b"FUGG",
        }
    }

    fn u32(self, value: u32) -> [u8; 4] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }

    fn u64(self, value: u64) -> [u8; 8] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }
}

struct Fixture {
    order: ByteOrder,
    version: u32,
    metadata: Vec<Vec<u8>>,
    tensors: Vec<Vec<u8>>,
    payload: Vec<u8>,
    alignment: u64,
}

impl Fixture {
    fn new(version: u32, order: ByteOrder) -> Self {
        Self {
            order,
            version,
            metadata: Vec::new(),
            tensors: Vec::new(),
            payload: Vec::new(),
            alignment: 32,
        }
    }

    fn metadata(mut self, key: &[u8], ty: u32, value: Vec<u8>) -> Self {
        let mut record = string(self.order, key);
        record.extend(self.order.u32(ty));
        record.extend(value);
        self.metadata.push(record);
        self
    }

    fn tensor(mut self, name: &[u8], dimensions: &[u64], ty: u32, offset: u64) -> Self {
        let mut record = string(self.order, name);
        record.extend(self.order.u32(u32::try_from(dimensions.len()).unwrap()));
        for dimension in dimensions {
            record.extend(self.order.u64(*dimension));
        }
        record.extend(self.order.u32(ty));
        record.extend(self.order.u64(offset));
        self.tensors.push(record);
        self
    }

    fn payload(mut self, bytes: &[u8]) -> Self {
        self.payload.extend(bytes);
        self
    }

    fn bytes(self) -> Vec<u8> {
        let tensor_count = self.tensors.len() as u64;
        let metadata_count = self.metadata.len() as u64;
        self.bytes_with_counts(tensor_count, metadata_count)
    }

    fn bytes_with_counts(self, tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.order.magic());
        bytes.extend(self.order.u32(self.version));
        bytes.extend(self.order.u64(tensor_count));
        bytes.extend(self.order.u64(metadata_count));
        for metadata in self.metadata {
            bytes.extend(metadata);
        }
        for tensor in self.tensors {
            bytes.extend(tensor);
        }
        let data_offset = align_up(bytes.len() as u64, self.alignment) as usize;
        bytes.resize(data_offset, 0);
        bytes.extend(self.payload);
        bytes
    }
}

fn string(order: ByteOrder, bytes: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::new();
    encoded.extend(order.u64(bytes.len() as u64));
    encoded.extend(bytes);
    encoded
}

fn array(order: ByteOrder, element_type: u32, elements: &[u8], count: u64) -> Vec<u8> {
    let mut encoded = Vec::new();
    encoded.extend(order.u32(element_type));
    encoded.extend(order.u64(count));
    encoded.extend(elements);
    encoded
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

fn read(bytes: Vec<u8>) -> Result<bridge_gguf::GgufFile, GgufError> {
    GgufReader::new(Cursor::new(bytes)).read()
}

fn read_with_limits(bytes: Vec<u8>, limits: ReaderLimits) -> Result<bridge_gguf::GgufFile, GgufError> {
    GgufReader::with_limits(Cursor::new(bytes), limits).read()
}

#[test]
fn reads_minimal_little_endian_v3_without_touching_payload() {
    let bytes = Fixture::new(3, ByteOrder::Little)
        .tensor(b"x", &[1], 0, 0)
        .payload(&[0; 32])
        .bytes();
    let reader = PayloadGuard::new(bytes, 64);

    let file = GgufReader::new(reader).read().unwrap();

    assert_eq!(file.version, 3);
    assert_eq!(file.endianness, Endianness::Little);
    assert_eq!(file.alignment, 32);
    assert_eq!(file.data_offset, 64);
    assert_eq!(file.file_len, 96);
    assert_eq!(file.tensors.len(), 1);
    assert_eq!(file.tensors[0].name(), "x");
    assert_eq!(file.tensors[0].shape(), &[1]);
}

#[test]
fn reads_exactly_aligned_tensor_directory_without_payload_seek() {
    let bytes = Fixture::new(3, ByteOrder::Little)
        .tensor(b"12345678", &[1], 0, 0)
        .payload(&[0; 32])
        .bytes();
    let reader = PayloadGuard::new(bytes, 64);

    let file = GgufReader::new(reader).read().unwrap();

    assert_eq!(file.data_offset, 64);
    assert_eq!(file.file_len, 96);
    assert_eq!(file.tensors[0].name(), "12345678");
}

#[test]
fn rejects_distinct_tensor_names_that_alias_the_same_payload() {
    let error = read(
        Fixture::new(3, ByteOrder::Little)
            .tensor(b"first", &[1], 0, 0)
            .tensor(b"alias", &[1], 0, 0)
            .payload(&[0; 64])
            .bytes(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "tensor \"alias\" has relative payload offset 0, expected 32 for 32-byte alignment"
    );
}

#[test]
fn rejects_aligned_partial_overlap_below_the_padded_extent() {
    let error = read(
        Fixture::new(3, ByteOrder::Little)
            .tensor(b"wide", &[9], 0, 0)
            .tensor(b"overlap", &[1], 0, 32)
            .payload(&[0; 96])
            .bytes(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "tensor \"overlap\" has relative payload offset 32, expected 64 for 32-byte alignment"
    );
}

#[test]
fn rejects_aligned_gap_above_the_padded_extent() {
    let error = read(
        Fixture::new(3, ByteOrder::Little)
            .tensor(b"first", &[1], 0, 0)
            .tensor(b"after-gap", &[1], 0, 64)
            .payload(&[0; 96])
            .bytes(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "tensor \"after-gap\" has relative payload offset 64, expected 32 for 32-byte alignment"
    );
}

#[test]
fn rejects_nonzero_first_tensor_offset() {
    let error = read(
        Fixture::new(3, ByteOrder::Little)
            .tensor(b"first", &[1], 0, 32)
            .payload(&[0; 64])
            .bytes(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "tensor \"first\" has relative payload offset 32, expected 0 for 32-byte alignment"
    );
}

#[test]
fn rejects_reversed_directory_order_even_when_sorted_ranges_would_be_disjoint() {
    let error = read(
        Fixture::new(3, ByteOrder::Little)
            .tensor(b"second", &[1], 0, 32)
            .tensor(b"first", &[1], 0, 0)
            .payload(&[0; 64])
            .bytes(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "tensor \"second\" has relative payload offset 32, expected 0 for 32-byte alignment"
    );
}

#[test]
fn rejects_truncated_final_tensor_padding() {
    let error = read(
        Fixture::new(3, ByteOrder::Little)
            .tensor(b"x", &[1], 0, 0)
            .payload(&[0; 4])
            .bytes(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "GGUF tensor data section requires padded range 64..96, but physical file length is 68"
    );
}

#[test]
fn rejects_overflow_while_padding_a_tensor_extent() {
    let error = read(
        Fixture::new(3, ByteOrder::Little)
            .tensor(b"huge", &[u64::MAX], 24, 0)
            .bytes(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "tensor \"huge\" encoded byte length 18446744073709551615 overflows 32-byte padding"
    );
}

#[test]
fn accepts_mixed_types_at_exact_padded_offsets_and_trailing_bytes() {
    let bytes = Fixture::new(3, ByteOrder::Little)
        .tensor(b"float", &[1], 0, 0)
        .tensor(b"quant", &[256], 12, 32)
        .tensor(b"bytes", &[7], 24, 192)
        .payload(&[0; 231])
        .bytes();

    let file = read(bytes).unwrap();

    assert_eq!(file.tensors[0].encoded_bytes().unwrap(), 4);
    assert_eq!(file.tensors[1].encoded_bytes().unwrap(), 144);
    assert_eq!(file.tensors[2].encoded_bytes().unwrap(), 7);
    assert_eq!(file.file_len, file.data_offset + 224 + 7);
}

#[test]
fn reads_minimal_v2() {
    let file = read(Fixture::new(2, ByteOrder::Little).bytes()).unwrap();
    assert_eq!(file.version, 2);
    assert_eq!(file.endianness, Endianness::Little);
}

#[test]
fn reads_byte_swapped_big_endian_file() {
    let file = read(
        Fixture::new(3, ByteOrder::Big)
            .metadata(b"answer", 4, ByteOrder::Big.u32(42).to_vec())
            .bytes(),
    )
    .unwrap();

    assert_eq!(file.endianness, Endianness::Big);
    assert_eq!(file.get_u32("answer"), Ok(42));
}

#[test]
fn rejects_bad_magic_and_unsupported_versions() {
    let mut bad_magic = Fixture::new(3, ByteOrder::Little).bytes();
    bad_magic[..4].copy_from_slice(b"NOPE");
    assert!(matches!(read(bad_magic), Err(GgufError::BadMagic(_))));

    for version in [1, 4] {
        assert!(matches!(
            read(Fixture::new(version, ByteOrder::Little).bytes()),
            Err(GgufError::UnsupportedVersion(found)) if found == version
        ));
    }
}

#[test]
fn rejects_truncated_scalars_strings_arrays_and_tensor_records() {
    let mut scalar = Fixture::new(3, ByteOrder::Little)
        .metadata(b"k", 10, vec![0; 7])
        .bytes();
    scalar.truncate(44);
    let mut string_value = Fixture::new(3, ByteOrder::Little)
        .metadata(b"k", 8, {
            let mut value = ByteOrder::Little.u64(4).to_vec();
            value.extend(b"abc");
            value
        })
        .bytes();
    string_value.truncate(48);
    let mut array_value = Fixture::new(3, ByteOrder::Little)
        .metadata(b"k", 9, array(ByteOrder::Little, 4, &[0; 3], 1))
        .bytes();
    array_value.truncate(52);
    let tensor = {
        let fixture = Fixture::new(3, ByteOrder::Little).tensor(b"x", &[1], 0, 0);
        let mut bytes = fixture.bytes();
        bytes.truncate(56);
        bytes
    };

    for bytes in [scalar, string_value, array_value, tensor] {
        assert!(matches!(read(bytes), Err(GgufError::Truncated { .. })));
    }
}

#[test]
fn enforces_metadata_tensor_string_and_array_limits_before_allocation() {
    let low_metadata = ReaderLimits {
        max_metadata_entries: 1,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(
            Fixture::new(3, ByteOrder::Little).bytes_with_counts(0, 2),
            low_metadata
        ),
        Err(GgufError::LimitExceeded {
            kind: "metadata entries",
            ..
        })
    ));

    let low_tensors = ReaderLimits {
        max_tensors: 1,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(
            Fixture::new(3, ByteOrder::Little).bytes_with_counts(2, 0),
            low_tensors
        ),
        Err(GgufError::LimitExceeded { kind: "tensors", .. })
    ));

    let low_strings = ReaderLimits {
        max_string_bytes: 3,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"four", 4, ByteOrder::Little.u32(1).to_vec())
                .bytes(),
            low_strings
        ),
        Err(GgufError::LimitExceeded {
            kind: "string bytes",
            ..
        })
    ));

    let low_arrays = ReaderLimits {
        max_array_elements: 1,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"k", 9, array(ByteOrder::Little, 0, &[1, 2], 2))
                .bytes(),
            low_arrays
        ),
        Err(GgufError::LimitExceeded {
            kind: "array elements",
            ..
        })
    ));
}

#[test]
fn metadata_budget_charges_string_and_array_payloads() {
    let limits = ReaderLimits {
        max_metadata_bytes: 3,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"k", 8, string(ByteOrder::Little, b"abc"))
                .bytes(),
            limits
        ),
        Err(GgufError::LimitExceeded {
            kind: "metadata bytes",
            ..
        })
    ));

    let limits = ReaderLimits {
        max_metadata_bytes: 8,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"k", 9, array(ByteOrder::Little, 4, &[0, 0, 0, 0, 1, 0, 0, 0], 2))
                .bytes(),
            limits
        ),
        Err(GgufError::LimitExceeded {
            kind: "metadata bytes",
            ..
        })
    ));

    let limits = ReaderLimits {
        max_metadata_bytes: 11,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(
            Fixture::new(3, ByteOrder::Little)
                .tensor(b"name", &[1], 0, 0)
                .payload(&[0; 4])
                .bytes(),
            limits
        ),
        Err(GgufError::LimitExceeded {
            kind: "metadata bytes",
            ..
        })
    ));
}

#[test]
fn array_element_limit_is_aggregate_across_the_file() {
    let limits = ReaderLimits {
        max_array_elements: 3,
        ..ReaderLimits::default()
    };
    let bytes = Fixture::new(3, ByteOrder::Little)
        .metadata(b"a", 9, array(ByteOrder::Little, 0, &[1, 2], 2))
        .metadata(b"b", 9, array(ByteOrder::Little, 0, &[3, 4], 2))
        .bytes();

    assert!(matches!(
        read_with_limits(bytes, limits),
        Err(GgufError::LimitExceeded {
            kind: "array elements",
            limit: 3,
            actual: 4,
        })
    ));
}

#[test]
fn array_minimum_encoding_must_fit_remaining_metadata_budget() {
    let mut bytes = Fixture::new(3, ByteOrder::Little)
        .metadata(b"k", 9, array(ByteOrder::Little, 10, &[], 4))
        .bytes();
    bytes.truncate(49);
    let limits = ReaderLimits {
        max_metadata_bytes: 50,
        ..ReaderLimits::default()
    };

    assert!(matches!(
        read_with_limits(bytes, limits),
        Err(GgufError::LimitExceeded {
            kind: "metadata bytes",
            limit: 50,
            actual: 57,
        })
    ));
}

#[test]
fn truncated_maximum_array_does_not_prereserve_decoded_element_storage() {
    let mut bytes = Fixture::new(3, ByteOrder::Little)
        .metadata(b"k", 9, array(ByteOrder::Little, 0, &[], 1_000_000))
        .bytes();
    bytes.truncate(49);
    let forbidden_full_reservation = std::mem::size_of::<GgufValue>() * 1_000_000;

    let observation = AllocationObservation::start();
    let result = read(bytes);
    let largest_allocation = observation.finish();

    assert!(matches!(result, Err(GgufError::Truncated { .. })));
    assert!(
        largest_allocation < forbidden_full_reservation,
        "parser requested a {largest_allocation}-byte allocation; a full decoded reservation is \
         {forbidden_full_reservation} bytes"
    );
}

#[test]
fn rejects_invalid_utf8_boolean_and_metadata_type() {
    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .metadata(&[0xff], 4, ByteOrder::Little.u32(1).to_vec())
                .bytes()
        ),
        Err(GgufError::InvalidUtf8 { .. })
    ));
    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"k", 7, vec![2])
                .bytes()
        ),
        Err(GgufError::InvalidBoolean(2))
    ));
    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"k", 99, Vec::new())
                .bytes()
        ),
        Err(GgufError::UnknownValueType(99))
    ));
}

#[test]
fn rejects_recursive_array_type() {
    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"k", 9, array(ByteOrder::Little, 9, &[], 0))
                .bytes()
        ),
        Err(GgufError::NestedArray)
    ));
}

#[test]
fn rejects_unknown_ggml_type_and_invalid_dimension_counts() {
    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .tensor(b"x", &[1], 5, 0)
                .payload(&[0; 4])
                .bytes()
        ),
        Err(GgufError::Core(bridge_core::error::CoreError::UnknownGgmlType(5)))
    ));

    let mut zero_rank = Vec::new();
    zero_rank.extend(string(ByteOrder::Little, b"x"));
    zero_rank.extend(ByteOrder::Little.u32(0));
    zero_rank.extend(ByteOrder::Little.u32(0));
    zero_rank.extend(ByteOrder::Little.u64(0));
    let mut fixture = Fixture::new(3, ByteOrder::Little);
    fixture.tensors.push(zero_rank);
    assert!(matches!(
        read(fixture.bytes()),
        Err(GgufError::InvalidDimensionCount(0))
    ));

    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .tensor(b"x", &[1, 1, 1, 1, 1], 0, 0)
                .payload(&[0; 4])
                .bytes()
        ),
        Err(GgufError::InvalidDimensionCount(5))
    ));
}

#[test]
fn rejects_quantized_row_block_mismatch() {
    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .tensor(b"q", &[31], 2, 0)
                .bytes()
        ),
        Err(GgufError::Core(
            bridge_core::error::CoreError::NotBlockAligned { .. }
        ))
    ));
}

#[test]
fn rejects_data_offset_overflow() {
    let header = Fixture::new(3, ByteOrder::Little).bytes();
    assert!(matches!(
        GgufReader::new(OverflowPositionReader::new(header)).read(),
        Err(GgufError::ArithmeticOverflow("GGUF data offset"))
    ));
}

#[test]
fn rejects_tensor_range_beyond_physical_file_length() {
    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .tensor(b"x", &[1], 0, 0)
                .bytes()
        ),
        Err(GgufError::Core(
            bridge_core::error::CoreError::TensorOutOfBounds { .. }
        ))
    ));
}

#[test]
fn rejects_duplicate_metadata_and_invalid_alignment() {
    let duplicate = Fixture::new(3, ByteOrder::Little)
        .metadata(b"k", 4, ByteOrder::Little.u32(1).to_vec())
        .metadata(b"k", 4, ByteOrder::Little.u32(2).to_vec())
        .bytes();
    assert!(matches!(
        read(duplicate),
        Err(GgufError::DuplicateMetadataKey(key)) if key == "k"
    ));

    for alignment in [0, 3] {
        assert!(matches!(
            read(
                Fixture::new(3, ByteOrder::Little)
                    .metadata(
                        b"general.alignment",
                        4,
                        ByteOrder::Little.u32(alignment).to_vec()
                    )
                    .bytes()
            ),
            Err(GgufError::InvalidAlignment(found)) if found == alignment
        ));
    }

    assert!(matches!(
        read(
            Fixture::new(3, ByteOrder::Little)
                .metadata(b"general.alignment", 10, ByteOrder::Little.u64(32).to_vec())
                .bytes()
        ),
        Err(GgufError::AlignmentWrongType(GgufValueType::U64))
    ));
}

#[test]
fn typed_getters_distinguish_missing_keys_from_wrong_types() {
    let file = read(
        Fixture::new(3, ByteOrder::Little)
            .metadata(b"name", 8, string(ByteOrder::Little, b"hy3"))
            .metadata(b"enabled", 7, vec![1])
            .bytes(),
    )
    .unwrap();

    assert_eq!(file.get_string("name"), Ok("hy3"));
    assert_eq!(file.get_bool("enabled"), Ok(true));
    assert_eq!(
        file.get_u32("missing"),
        Err(MetadataError::Missing {
            key: "missing".to_owned()
        })
    );
    assert_eq!(
        file.get_u32("name"),
        Err(MetadataError::WrongType {
            key: "name".to_owned(),
            expected: GgufValueType::U32,
            actual: GgufValueType::String,
        })
    );
    assert!(matches!(
        file.metadata[0].1,
        GgufValue::String(ref value) if value == "hy3"
    ));
}

#[test]
fn rejects_caller_dimension_limit_above_ggml_maximum() {
    let limits = ReaderLimits {
        max_dimensions: 5,
        ..ReaderLimits::default()
    };
    assert!(matches!(
        read_with_limits(Fixture::new(3, ByteOrder::Little).bytes(), limits),
        Err(GgufError::InvalidDimensionLimit(5))
    ));
}

struct PayloadGuard {
    inner: Cursor<Vec<u8>>,
    data_offset: u64,
}

impl PayloadGuard {
    fn new(bytes: Vec<u8>, data_offset: u64) -> Self {
        Self {
            inner: Cursor::new(bytes),
            data_offset,
        }
    }
}

impl Read for PayloadGuard {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.inner.position() >= self.data_offset {
            return Err(io::Error::other("attempted tensor payload read"));
        }
        self.inner.read(buffer)
    }
}

impl Seek for PayloadGuard {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        if matches!(position, SeekFrom::Start(target) if target >= self.data_offset) {
            return Err(io::Error::other("attempted tensor payload seek"));
        }
        self.inner.seek(position)
    }
}

struct OverflowPositionReader {
    inner: Cursor<Vec<u8>>,
}

impl OverflowPositionReader {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            inner: Cursor::new(bytes),
        }
    }
}

impl Read for OverflowPositionReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buffer)
    }
}

impl Seek for OverflowPositionReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        if matches!(position, SeekFrom::Current(0)) && self.inner.position() == 24 {
            return Ok(u64::MAX - 15);
        }
        self.inner.seek(position)
    }
}
