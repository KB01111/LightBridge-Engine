use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use bridge_core::ggml_type::GgmlType;
use bridge_core::tensor::TensorDesc;

use crate::error::{GgufError, MetadataError, Result};
use crate::value::{GgufArray, GgufValue, GgufValueType};

const DEFAULT_ALIGNMENT: u64 = 32;
const GGML_MAX_DIMENSIONS: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReaderLimits {
    pub max_dimensions: u32,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_tensors: u64,
    pub max_metadata_entries: u64,
    pub max_metadata_bytes: u64,
}

impl Default for ReaderLimits {
    fn default() -> Self {
        Self {
            max_dimensions: GGML_MAX_DIMENSIONS,
            max_string_bytes: 1024 * 1024,
            max_array_elements: 1_000_000,
            max_tensors: 4_000_000,
            max_metadata_entries: 1_000_000,
            max_metadata_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub version: u32,
    pub endianness: Endianness,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorDesc>,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_len: u64,
}

impl GgufFile {
    pub fn get(&self, key: &str) -> Result<&GgufValue, MetadataError> {
        self.metadata
            .iter()
            .find_map(|(candidate, value)| (candidate == key).then_some(value))
            .ok_or_else(|| MetadataError::Missing { key: key.to_owned() })
    }

    fn expect_type(&self, key: &str, expected: GgufValueType) -> Result<&GgufValue, MetadataError> {
        let value = self.get(key)?;
        let actual = value.value_type();
        if actual == expected {
            Ok(value)
        } else {
            Err(MetadataError::WrongType {
                key: key.to_owned(),
                expected,
                actual,
            })
        }
    }

    pub fn get_u8(&self, key: &str) -> Result<u8, MetadataError> {
        match self.expect_type(key, GgufValueType::U8)? {
            GgufValue::U8(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_i8(&self, key: &str) -> Result<i8, MetadataError> {
        match self.expect_type(key, GgufValueType::I8)? {
            GgufValue::I8(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_u16(&self, key: &str) -> Result<u16, MetadataError> {
        match self.expect_type(key, GgufValueType::U16)? {
            GgufValue::U16(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_i16(&self, key: &str) -> Result<i16, MetadataError> {
        match self.expect_type(key, GgufValueType::I16)? {
            GgufValue::I16(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_u32(&self, key: &str) -> Result<u32, MetadataError> {
        match self.expect_type(key, GgufValueType::U32)? {
            GgufValue::U32(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_i32(&self, key: &str) -> Result<i32, MetadataError> {
        match self.expect_type(key, GgufValueType::I32)? {
            GgufValue::I32(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_f32(&self, key: &str) -> Result<f32, MetadataError> {
        match self.expect_type(key, GgufValueType::F32)? {
            GgufValue::F32(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, MetadataError> {
        match self.expect_type(key, GgufValueType::Bool)? {
            GgufValue::Bool(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_string(&self, key: &str) -> Result<&str, MetadataError> {
        match self.expect_type(key, GgufValueType::String)? {
            GgufValue::String(value) => Ok(value),
            _ => unreachable!(),
        }
    }

    pub fn get_array(&self, key: &str) -> Result<&GgufArray, MetadataError> {
        match self.expect_type(key, GgufValueType::Array)? {
            GgufValue::Array(value) => Ok(value),
            _ => unreachable!(),
        }
    }

    pub fn get_u64(&self, key: &str) -> Result<u64, MetadataError> {
        match self.expect_type(key, GgufValueType::U64)? {
            GgufValue::U64(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_i64(&self, key: &str) -> Result<i64, MetadataError> {
        match self.expect_type(key, GgufValueType::I64)? {
            GgufValue::I64(value) => Ok(*value),
            _ => unreachable!(),
        }
    }

    pub fn get_f64(&self, key: &str) -> Result<f64, MetadataError> {
        match self.expect_type(key, GgufValueType::F64)? {
            GgufValue::F64(value) => Ok(*value),
            _ => unreachable!(),
        }
    }
}

pub struct GgufReader<R> {
    reader: R,
    limits: ReaderLimits,
    endianness: Endianness,
    metadata_bytes: u64,
    array_elements: u64,
    account_metadata: bool,
}

impl<R: Read + Seek> GgufReader<R> {
    pub fn new(reader: R) -> Self {
        Self::with_limits(reader, ReaderLimits::default())
    }

    pub fn with_limits(reader: R, limits: ReaderLimits) -> Self {
        Self {
            reader,
            limits,
            endianness: Endianness::Little,
            metadata_bytes: 0,
            array_elements: 0,
            account_metadata: false,
        }
    }

    pub fn read(mut self) -> Result<GgufFile> {
        if self.limits.max_dimensions > GGML_MAX_DIMENSIONS {
            return Err(GgufError::InvalidDimensionLimit(self.limits.max_dimensions));
        }

        let mut magic = [0; 4];
        self.read_exact(&mut magic, "magic")?;
        self.endianness = match &magic {
            b"GGUF" => Endianness::Little,
            b"FUGG" => Endianness::Big,
            _ => return Err(GgufError::BadMagic(magic)),
        };

        let version = self.read_u32("version")?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = self.read_u64("tensor count")?;
        let metadata_count = self.read_u64("metadata count")?;
        let tensor_count = self.checked_count(tensor_count, self.limits.max_tensors, "tensors")?;
        let metadata_count = self.checked_count(
            metadata_count,
            self.limits.max_metadata_entries,
            "metadata entries",
        )?;

        let mut metadata = Vec::new();
        let mut keys = HashSet::new();
        self.account_metadata = true;
        for _ in 0..metadata_count {
            let key = self.read_string("metadata key")?;
            let value_type = GgufValueType::try_from(self.read_u32("metadata value type")?)?;
            let value = self.read_value(value_type)?;
            keys.try_reserve(1).map_err(|_| GgufError::AllocationFailed {
                kind: "metadata key set",
            })?;
            if !keys.insert(self.try_clone_string(&key, "metadata key set")?) {
                return Err(GgufError::DuplicateMetadataKey(key));
            }
            self.try_push(&mut metadata, (key, value), "metadata entries")?;
        }
        self.account_metadata = false;

        let alignment = match metadata
            .iter()
            .find_map(|(key, value)| (key == "general.alignment").then_some(value))
        {
            None => DEFAULT_ALIGNMENT,
            Some(GgufValue::U32(value)) if *value > 0 && value.is_power_of_two() => u64::from(*value),
            Some(GgufValue::U32(value)) => return Err(GgufError::InvalidAlignment(*value)),
            Some(value) => return Err(GgufError::AlignmentWrongType(value.value_type())),
        };

        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            self.account_metadata = true;
            let name = self.read_string("tensor name")?;
            self.account_metadata = false;
            let n_dims = self.read_u32("tensor dimension count")?;
            if !(1..=GGML_MAX_DIMENSIONS).contains(&n_dims) {
                return Err(GgufError::InvalidDimensionCount(n_dims));
            }
            if n_dims > self.limits.max_dimensions {
                return Err(GgufError::LimitExceeded {
                    kind: "dimensions",
                    limit: u64::from(self.limits.max_dimensions),
                    actual: u64::from(n_dims),
                });
            }
            let n_dims_usize = usize::try_from(n_dims).map_err(|_| GgufError::CountDoesNotFit {
                kind: "dimensions",
                value: u64::from(n_dims),
            })?;
            let mut shape = Vec::new();
            shape
                .try_reserve_exact(n_dims_usize)
                .map_err(|_| GgufError::AllocationFailed {
                    kind: "tensor dimensions",
                })?;
            for _ in 0..n_dims_usize {
                shape.push(self.read_u64("tensor dimension")?);
            }
            let ty = GgmlType::from_discriminant(self.read_u32("GGML type")?)?;
            let relative_offset = self.read_u64("tensor payload offset")?;
            let descriptor = TensorDesc::new(name, &shape, ty, relative_offset)?;
            self.try_push(&mut tensors, descriptor, "tensor descriptors")?;
        }

        let directory_end = self.reader.stream_position().map_err(GgufError::Io)?;
        let data_offset = align_up(directory_end, alignment)?;
        let file_len = self.reader.seek(SeekFrom::End(0)).map_err(GgufError::Io)?;
        if data_offset > file_len {
            return Err(GgufError::DataOffsetBeyondFile {
                data_offset,
                file_len,
            });
        }
        let mut padded_extent = 0_u64;
        for tensor in &tensors {
            let actual_offset = tensor.relative_offset();
            if actual_offset != padded_extent {
                return Err(GgufError::TensorOffsetMismatch {
                    name: tensor.name().to_owned(),
                    actual_offset,
                    expected_offset: padded_extent,
                    alignment,
                });
            }
            let encoded_bytes = tensor.encoded_bytes()?;
            let padded_bytes = encoded_bytes
                .checked_add(alignment - 1)
                .map(|sum| sum & !(alignment - 1))
                .ok_or_else(|| GgufError::TensorPaddingOverflow {
                    name: tensor.name().to_owned(),
                    encoded_bytes,
                    alignment,
                })?;
            padded_extent =
                padded_extent
                    .checked_add(padded_bytes)
                    .ok_or_else(|| GgufError::TensorExtentOverflow {
                        name: tensor.name().to_owned(),
                        expected_offset: padded_extent,
                        padded_bytes,
                    })?;
            tensor.checked_absolute_range(data_offset, file_len)?;
        }
        let required_end =
            data_offset
                .checked_add(padded_extent)
                .ok_or(GgufError::TensorDataEndOverflow {
                    data_offset,
                    padded_extent,
                })?;
        if required_end > file_len {
            return Err(GgufError::TensorDataSectionTruncated {
                data_offset,
                required_end,
                file_len,
            });
        }

        Ok(GgufFile {
            version,
            endianness: self.endianness,
            metadata,
            tensors,
            alignment,
            data_offset,
            file_len,
        })
    }

    fn read_value(&mut self, value_type: GgufValueType) -> Result<GgufValue> {
        Ok(match value_type {
            GgufValueType::Array => {
                let element_type = GgufValueType::try_from(self.read_u32("array element type")?)?;
                if element_type == GgufValueType::Array {
                    return Err(GgufError::NestedArray);
                }
                let count = self.read_u64("array element count")?;
                let count = self.check_array_budget(element_type, count)?;
                let mut values = Vec::new();
                for _ in 0..count {
                    let value = self.read_value(element_type)?;
                    self.try_push(&mut values, value, "array elements")?;
                }
                GgufValue::Array(GgufArray { element_type, values })
            }
            GgufValueType::U8 => GgufValue::U8(self.read_u8("u8 metadata value")?),
            GgufValueType::I8 => GgufValue::I8(self.read_u8("i8 metadata value")? as i8),
            GgufValueType::U16 => GgufValue::U16(self.read_u16("u16 metadata value")?),
            GgufValueType::I16 => GgufValue::I16(self.read_u16("i16 metadata value")? as i16),
            GgufValueType::U32 => GgufValue::U32(self.read_u32("u32 metadata value")?),
            GgufValueType::I32 => GgufValue::I32(self.read_u32("i32 metadata value")? as i32),
            GgufValueType::F32 => GgufValue::F32(f32::from_bits(self.read_u32("f32 metadata value")?)),
            GgufValueType::Bool => match self.read_u8("boolean metadata value")? {
                0 => GgufValue::Bool(false),
                1 => GgufValue::Bool(true),
                other => return Err(GgufError::InvalidBoolean(other)),
            },
            GgufValueType::String => GgufValue::String(self.read_string("string metadata value")?),
            GgufValueType::U64 => GgufValue::U64(self.read_u64("u64 metadata value")?),
            GgufValueType::I64 => GgufValue::I64(self.read_u64("i64 metadata value")? as i64),
            GgufValueType::F64 => GgufValue::F64(f64::from_bits(self.read_u64("f64 metadata value")?)),
        })
    }

    fn check_array_budget(&mut self, element_type: GgufValueType, count: u64) -> Result<usize> {
        let aggregate = self
            .array_elements
            .checked_add(count)
            .ok_or(GgufError::ArithmeticOverflow("aggregate array elements"))?;
        if aggregate > self.limits.max_array_elements {
            return Err(GgufError::LimitExceeded {
                kind: "array elements",
                limit: self.limits.max_array_elements,
                actual: aggregate,
            });
        }

        let minimum_element_bytes = match element_type {
            GgufValueType::U8 | GgufValueType::I8 | GgufValueType::Bool => 1_u64,
            GgufValueType::U16 | GgufValueType::I16 => 2,
            GgufValueType::U32 | GgufValueType::I32 | GgufValueType::F32 => 4,
            GgufValueType::String | GgufValueType::U64 | GgufValueType::I64 | GgufValueType::F64 => 8,
            GgufValueType::Array => return Err(GgufError::NestedArray),
        };
        let minimum_payload_bytes = count
            .checked_mul(minimum_element_bytes)
            .ok_or(GgufError::ArithmeticOverflow("minimum encoded array bytes"))?;
        let minimum_metadata_bytes = self
            .metadata_bytes
            .checked_add(minimum_payload_bytes)
            .ok_or(GgufError::ArithmeticOverflow("metadata bytes"))?;
        if minimum_metadata_bytes > self.limits.max_metadata_bytes {
            return Err(GgufError::LimitExceeded {
                kind: "metadata bytes",
                limit: self.limits.max_metadata_bytes,
                actual: minimum_metadata_bytes,
            });
        }

        let count = usize::try_from(count).map_err(|_| GgufError::CountDoesNotFit {
            kind: "array elements",
            value: count,
        })?;
        self.array_elements = aggregate;
        Ok(count)
    }

    fn read_string(&mut self, context: &'static str) -> Result<String> {
        let length = self.read_u64("string length")?;
        if length > self.limits.max_string_bytes {
            return Err(GgufError::LimitExceeded {
                kind: "string bytes",
                limit: self.limits.max_string_bytes,
                actual: length,
            });
        }
        let length = usize::try_from(length).map_err(|_| GgufError::CountDoesNotFit {
            kind: "string bytes",
            value: length,
        })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| GgufError::AllocationFailed { kind: "string bytes" })?;
        bytes.resize(length, 0);
        self.read_exact(&mut bytes, context)?;
        String::from_utf8(bytes).map_err(|source| GgufError::InvalidUtf8 { context, source })
    }

    fn checked_count(&self, actual: u64, limit: u64, kind: &'static str) -> Result<usize> {
        if actual > limit {
            return Err(GgufError::LimitExceeded { kind, limit, actual });
        }
        usize::try_from(actual).map_err(|_| GgufError::CountDoesNotFit { kind, value: actual })
    }

    fn try_push<T>(&self, values: &mut Vec<T>, value: T, kind: &'static str) -> Result<()> {
        values
            .try_reserve(1)
            .map_err(|_| GgufError::AllocationFailed { kind })?;
        values.push(value);
        Ok(())
    }

    fn try_clone_string(&self, value: &str, kind: &'static str) -> Result<String> {
        let mut clone = String::new();
        clone
            .try_reserve_exact(value.len())
            .map_err(|_| GgufError::AllocationFailed { kind })?;
        clone.push_str(value);
        Ok(clone)
    }

    fn read_exact(&mut self, bytes: &mut [u8], context: &'static str) -> Result<()> {
        if self.account_metadata {
            let amount =
                u64::try_from(bytes.len()).map_err(|_| GgufError::ArithmeticOverflow("metadata bytes"))?;
            let next = self
                .metadata_bytes
                .checked_add(amount)
                .ok_or(GgufError::ArithmeticOverflow("metadata bytes"))?;
            if next > self.limits.max_metadata_bytes {
                return Err(GgufError::LimitExceeded {
                    kind: "metadata bytes",
                    limit: self.limits.max_metadata_bytes,
                    actual: next,
                });
            }
            self.metadata_bytes = next;
        }
        self.reader
            .read_exact(bytes)
            .map_err(|error| GgufError::from_io(error, context))
    }

    fn read_u8(&mut self, context: &'static str) -> Result<u8> {
        let mut bytes = [0; 1];
        self.read_exact(&mut bytes, context)?;
        Ok(bytes[0])
    }

    fn read_u16(&mut self, context: &'static str) -> Result<u16> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes, context)?;
        Ok(match self.endianness {
            Endianness::Little => u16::from_le_bytes(bytes),
            Endianness::Big => u16::from_be_bytes(bytes),
        })
    }

    fn read_u32(&mut self, context: &'static str) -> Result<u32> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes, context)?;
        Ok(match self.endianness {
            Endianness::Little => u32::from_le_bytes(bytes),
            Endianness::Big => u32::from_be_bytes(bytes),
        })
    }

    fn read_u64(&mut self, context: &'static str) -> Result<u64> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes, context)?;
        Ok(match self.endianness {
            Endianness::Little => u64::from_le_bytes(bytes),
            Endianness::Big => u64::from_be_bytes(bytes),
        })
    }
}

pub fn open(path: impl AsRef<Path>) -> Result<GgufFile> {
    GgufReader::new(File::open(path).map_err(GgufError::Io)?).read()
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(GgufError::ArithmeticOverflow("GGUF data offset"))
}
