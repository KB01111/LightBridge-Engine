//! Versioned, source-bound, lossless expert sidecar metadata and reader.

use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use bridge_io_windows::{PositionedFile, ReadCancellation, ReadError, ReadLimits};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SIDECAR_FORMAT: &str = "lightbridge-expert-sidecar";
pub const SIDECAR_FORMAT_VERSION: u32 = 1;
pub const QUANT_ABI_VERSION: u32 = 1;
pub const SIDECAR_HEADER_BYTES: usize = 64;
pub const MAX_MANIFEST_BYTES: u64 = 32 * 1024 * 1024;
const SIDECAR_MAGIC: &[u8; 16] = b"LIGHTBRIDGE.EXP\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExpertLayout {
    Sequential,
    FusedGateUp,
}

impl ExpertLayout {
    const fn code(self) -> u32 {
        match self {
            Self::Sequential => 0,
            Self::FusedGateUp => 1,
        }
    }

    fn from_code(code: u32) -> Result<Self, FormatError> {
        match code {
            0 => Ok(Self::Sequential),
            1 => Ok(Self::FusedGateUp),
            _ => Err(FormatError::UnknownLayout(code)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub offset: u64,
    pub length: u64,
    pub ggml_type: String,
}

impl Segment {
    pub fn range(&self) -> Result<Range<u64>, FormatError> {
        let end = self
            .offset
            .checked_add(self.length)
            .ok_or(FormatError::ArithmeticOverflow)?;
        Ok(self.offset..end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertRecord {
    pub key: ExpertKey,
    pub offset: u64,
    pub length: u64,
    pub gate: Segment,
    pub up: Segment,
    pub down: Segment,
}

impl ExpertRecord {
    pub fn range(&self) -> Result<Range<u64>, FormatError> {
        let end = self
            .offset
            .checked_add(self.length)
            .ok_or(FormatError::ArithmeticOverflow)?;
        Ok(self.offset..end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFileIdentity {
    pub ordinal: u32,
    pub path: String,
    pub length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarFileIdentity {
    pub length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarManifest {
    pub format: String,
    pub format_version: u32,
    pub engine_version: String,
    pub quant_abi_version: u32,
    pub alignment: u64,
    pub layout: ExpertLayout,
    pub source_files: Vec<SourceFileIdentity>,
    pub tensor_directory_sha256: String,
    pub sidecar: SidecarFileIdentity,
    pub records: Vec<ExpertRecord>,
}

impl SidecarManifest {
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.format != SIDECAR_FORMAT {
            return Err(FormatError::WrongFormat(self.format.clone()));
        }
        if self.format_version != SIDECAR_FORMAT_VERSION {
            return Err(FormatError::WrongVersion {
                expected: SIDECAR_FORMAT_VERSION,
                actual: self.format_version,
            });
        }
        if self.quant_abi_version != QUANT_ABI_VERSION {
            return Err(FormatError::WrongQuantAbi {
                expected: QUANT_ABI_VERSION,
                actual: self.quant_abi_version,
            });
        }
        if self.alignment < 32 || !self.alignment.is_power_of_two() {
            return Err(FormatError::InvalidAlignment(self.alignment));
        }
        validate_sha256(&self.tensor_directory_sha256, "tensor directory")?;
        validate_sha256(&self.sidecar.sha256, "sidecar")?;
        if self.sidecar.length < SIDECAR_HEADER_BYTES as u64 {
            return Err(FormatError::SidecarTooShort(self.sidecar.length));
        }
        if self.source_files.is_empty() {
            return Err(FormatError::NoSourceFiles);
        }
        for (index, source) in self.source_files.iter().enumerate() {
            if source.ordinal as usize != index {
                return Err(FormatError::SourceOrdinal {
                    index,
                    actual: source.ordinal,
                });
            }
            validate_sha256(&source.sha256, "source file")?;
        }

        let mut previous_key = None;
        let mut previous_end = align_up(SIDECAR_HEADER_BYTES as u64, self.alignment)?;
        for record in &self.records {
            if previous_key.is_some_and(|key| key >= record.key) {
                return Err(FormatError::RecordsNotStrictlyOrdered(record.key));
            }
            if record.offset % self.alignment != 0 {
                return Err(FormatError::UnalignedRecord {
                    key: record.key,
                    offset: record.offset,
                    alignment: self.alignment,
                });
            }
            let record_range = record.range()?;
            if record_range.start < previous_end || record_range.end > self.sidecar.length {
                return Err(FormatError::RecordOutOfBounds {
                    key: record.key,
                    start: record_range.start,
                    end: record_range.end,
                    minimum: previous_end,
                    sidecar_length: self.sidecar.length,
                });
            }
            validate_record_segments(record, self.layout, self.alignment)?;
            previous_key = Some(record.key);
            previous_end = record_range.end;
        }
        Ok(())
    }

    pub fn record(&self, key: ExpertKey) -> Option<&ExpertRecord> {
        self.records
            .binary_search_by_key(&key, |record| record.key)
            .ok()
            .and_then(|index| self.records.get(index))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidecarHeader {
    pub format_version: u32,
    pub layout: ExpertLayout,
    pub alignment: u64,
    pub record_count: u64,
}

impl SidecarHeader {
    pub fn new(layout: ExpertLayout, alignment: u64, record_count: u64) -> Self {
        Self {
            format_version: SIDECAR_FORMAT_VERSION,
            layout,
            alignment,
            record_count,
        }
    }

    pub fn encode(self) -> [u8; SIDECAR_HEADER_BYTES] {
        let mut bytes = [0_u8; SIDECAR_HEADER_BYTES];
        bytes[..16].copy_from_slice(SIDECAR_MAGIC);
        bytes[16..20].copy_from_slice(&self.format_version.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.layout.code().to_le_bytes());
        bytes[24..32].copy_from_slice(&self.alignment.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.record_count.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() != SIDECAR_HEADER_BYTES {
            return Err(FormatError::HeaderLength {
                expected: SIDECAR_HEADER_BYTES,
                actual: bytes.len(),
            });
        }
        if &bytes[..16] != SIDECAR_MAGIC {
            return Err(FormatError::BadMagic);
        }
        if bytes[40..].iter().any(|&byte| byte != 0) {
            return Err(FormatError::NonZeroReservedHeader);
        }
        let format_version = u32::from_le_bytes(bytes[16..20].try_into().expect("fixed range"));
        let layout_code = u32::from_le_bytes(bytes[20..24].try_into().expect("fixed range"));
        let alignment = u64::from_le_bytes(bytes[24..32].try_into().expect("fixed range"));
        let record_count = u64::from_le_bytes(bytes[32..40].try_into().expect("fixed range"));
        Ok(Self {
            format_version,
            layout: ExpertLayout::from_code(layout_code)?,
            alignment,
            record_count,
        })
    }
}

#[derive(Debug)]
pub struct Sidecar {
    data: PositionedFile,
    manifest: SidecarManifest,
}

impl Sidecar {
    pub fn open(data_path: impl AsRef<Path>, manifest_path: impl AsRef<Path>) -> Result<Self, SidecarError> {
        let manifest_path = manifest_path.as_ref();
        let metadata = fs::metadata(manifest_path).map_err(|source| SidecarError::ManifestIo {
            path: manifest_path.to_owned(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(SidecarError::ManifestNotAFile(manifest_path.to_owned()));
        }
        if metadata.len() > MAX_MANIFEST_BYTES {
            return Err(SidecarError::ManifestTooLarge {
                actual: metadata.len(),
                maximum: MAX_MANIFEST_BYTES,
            });
        }
        let bytes = fs::read(manifest_path).map_err(|source| SidecarError::ManifestIo {
            path: manifest_path.to_owned(),
            source,
        })?;
        let manifest: SidecarManifest = serde_json::from_slice(&bytes).map_err(SidecarError::ManifestJson)?;
        manifest.validate()?;

        let maximum_record = manifest
            .records
            .iter()
            .map(|record| record.length)
            .max()
            .unwrap_or(SIDECAR_HEADER_BYTES as u64)
            .max(SIDECAR_HEADER_BYTES as u64);
        let maximum_record = usize::try_from(maximum_record).map_err(|_| FormatError::ArithmeticOverflow)?;
        let data = PositionedFile::open(
            data_path,
            ReadLimits {
                max_request_bytes: maximum_record,
            },
        )?;
        if data.length() != manifest.sidecar.length {
            return Err(SidecarError::DataLength {
                expected: manifest.sidecar.length,
                actual: data.length(),
            });
        }
        let cancellation = ReadCancellation::new();
        let header_bytes = data.read_exact_at(0..SIDECAR_HEADER_BYTES as u64, &cancellation)?;
        let header = SidecarHeader::decode(&header_bytes)?;
        if header.format_version != manifest.format_version
            || header.layout != manifest.layout
            || header.alignment != manifest.alignment
            || header.record_count != manifest.records.len() as u64
        {
            return Err(SidecarError::HeaderManifestMismatch);
        }
        Ok(Self { data, manifest })
    }

    pub const fn manifest(&self) -> &SidecarManifest {
        &self.manifest
    }

    pub fn read_expert(
        &self,
        key: ExpertKey,
        cancellation: &ReadCancellation,
    ) -> Result<ExpertBytes, SidecarError> {
        let record = self
            .manifest
            .record(key)
            .ok_or(SidecarError::MissingExpert(key))?;
        let bytes = self.data.read_exact_at(record.range()?, cancellation)?;
        Ok(ExpertBytes {
            gate: relative_range(record, &record.gate)?,
            up: relative_range(record, &record.up)?,
            down: relative_range(record, &record.down)?,
            bytes,
        })
    }

    pub fn verify_data_hash(&self, cancellation: &ReadCancellation) -> Result<(), SidecarError> {
        let mut hasher = Sha256::new();
        let chunk_size = self.data.limits().max_request_bytes.min(8 * 1024 * 1024);
        let mut offset = 0_u64;
        while offset < self.data.length() {
            let remaining = self.data.length() - offset;
            let length = remaining.min(chunk_size as u64);
            let end = offset
                .checked_add(length)
                .ok_or(FormatError::ArithmeticOverflow)?;
            hasher.update(self.data.read_exact_at(offset..end, cancellation)?);
            offset = end;
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != self.manifest.sidecar.sha256 {
            return Err(SidecarError::DataHash {
                expected: self.manifest.sidecar.sha256.clone(),
                actual,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBytes {
    bytes: Vec<u8>,
    gate: Range<usize>,
    up: Range<usize>,
    down: Range<usize>,
}

impl ExpertBytes {
    pub fn gate(&self) -> &[u8] {
        &self.bytes[self.gate.clone()]
    }

    pub fn up(&self) -> &[u8] {
        &self.bytes[self.up.clone()]
    }

    pub fn down(&self) -> &[u8] {
        &self.bytes[self.down.clone()]
    }

    pub fn record_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub fn align_up(value: u64, alignment: u64) -> Result<u64, FormatError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(FormatError::InvalidAlignment(alignment));
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|rounded| rounded & !mask)
        .ok_or(FormatError::ArithmeticOverflow)
}

fn validate_record_segments(
    record: &ExpertRecord,
    layout: ExpertLayout,
    alignment: u64,
) -> Result<(), FormatError> {
    let record_range = record.range()?;
    let gate = record.gate.range()?;
    let up = record.up.range()?;
    let down = record.down.range()?;
    for (name, segment) in [("gate", &gate), ("up", &up), ("down", &down)] {
        if segment.is_empty() || segment.start < record_range.start || segment.end > record_range.end {
            return Err(FormatError::SegmentOutOfBounds {
                key: record.key,
                segment: name,
            });
        }
    }
    if !(gate.end <= up.start && up.end <= down.start) {
        return Err(FormatError::OverlappingSegments(record.key));
    }
    match layout {
        ExpertLayout::Sequential => {
            if [gate.start, up.start, down.start]
                .iter()
                .any(|offset| offset % alignment != 0)
            {
                return Err(FormatError::UnalignedSegment(record.key));
            }
        }
        ExpertLayout::FusedGateUp => {
            if gate.start != record.offset || up.start != gate.end || down.start % alignment != 0 {
                return Err(FormatError::InvalidFusedLayout(record.key));
            }
        }
    }
    Ok(())
}

fn relative_range(record: &ExpertRecord, segment: &Segment) -> Result<Range<usize>, FormatError> {
    let start = segment
        .offset
        .checked_sub(record.offset)
        .ok_or(FormatError::ArithmeticOverflow)?;
    let end = start
        .checked_add(segment.length)
        .ok_or(FormatError::ArithmeticOverflow)?;
    Ok(
        usize::try_from(start).map_err(|_| FormatError::ArithmeticOverflow)?
            ..usize::try_from(end).map_err(|_| FormatError::ArithmeticOverflow)?,
    )
}

fn validate_sha256(value: &str, context: &'static str) -> Result<(), FormatError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FormatError::InvalidSha256 {
            context,
            value: value.to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("unexpected sidecar format {0:?}")]
    WrongFormat(String),
    #[error("sidecar format version is {actual}, expected {expected}")]
    WrongVersion { expected: u32, actual: u32 },
    #[error("quant ABI version is {actual}, expected {expected}")]
    WrongQuantAbi { expected: u32, actual: u32 },
    #[error("invalid sidecar alignment {0}; expected a power of two of at least 32")]
    InvalidAlignment(u64),
    #[error("invalid {context} SHA-256 {value:?}; expected 64 lowercase hexadecimal characters")]
    InvalidSha256 { context: &'static str, value: String },
    #[error("sidecar length {0} is shorter than its header")]
    SidecarTooShort(u64),
    #[error("manifest contains no source files")]
    NoSourceFiles,
    #[error("source file at index {index} declares ordinal {actual}")]
    SourceOrdinal { index: usize, actual: u32 },
    #[error("expert records are duplicated or out of order at {0:?}")]
    RecordsNotStrictlyOrdered(ExpertKey),
    #[error("expert record {key:?} offset {offset} is not aligned to {alignment}")]
    UnalignedRecord {
        key: ExpertKey,
        offset: u64,
        alignment: u64,
    },
    #[error("expert record {key:?} range {start}..{end} is outside {minimum}..{sidecar_length}")]
    RecordOutOfBounds {
        key: ExpertKey,
        start: u64,
        end: u64,
        minimum: u64,
        sidecar_length: u64,
    },
    #[error("expert record {key:?} {segment} segment is empty or outside its record")]
    SegmentOutOfBounds { key: ExpertKey, segment: &'static str },
    #[error("expert record {0:?} segments overlap or are out of order")]
    OverlappingSegments(ExpertKey),
    #[error("expert record {0:?} sequential segment is unaligned")]
    UnalignedSegment(ExpertKey),
    #[error("expert record {0:?} does not satisfy fused gate-up layout")]
    InvalidFusedLayout(ExpertKey),
    #[error("sidecar header length is {actual}, expected {expected}")]
    HeaderLength { expected: usize, actual: usize },
    #[error("sidecar header magic is invalid")]
    BadMagic,
    #[error("sidecar header contains non-zero reserved bytes")]
    NonZeroReservedHeader,
    #[error("sidecar header contains unknown layout code {0}")]
    UnknownLayout(u32),
    #[error("checked arithmetic overflow while validating sidecar format")]
    ArithmeticOverflow,
}

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error("failed to read sidecar manifest {path:?}: {source}")]
    ManifestIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("sidecar manifest path is not a regular file: {0:?}")]
    ManifestNotAFile(PathBuf),
    #[error("sidecar manifest is {actual} bytes, maximum is {maximum}")]
    ManifestTooLarge { actual: u64, maximum: u64 },
    #[error("invalid sidecar manifest JSON: {0}")]
    ManifestJson(serde_json::Error),
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    Read(#[from] ReadError),
    #[error("sidecar data length is {actual}, manifest declares {expected}")]
    DataLength { expected: u64, actual: u64 },
    #[error("sidecar header does not match its manifest")]
    HeaderManifestMismatch,
    #[error("sidecar does not contain expert {0:?}")]
    MissingExpert(ExpertKey),
    #[error("sidecar SHA-256 is {actual}, manifest declares {expected}")]
    DataHash { expected: String, actual: String },
}
