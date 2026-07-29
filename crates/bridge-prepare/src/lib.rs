//! Validated direct expert reads and transactional expert-sidecar preparation.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bridge_format::{
    align_up, ExpertKey, ExpertLayout, ExpertRecord, FormatError, Segment, Sidecar, SidecarFileIdentity,
    SidecarHeader, SidecarManifest, SourceFileIdentity, MAX_MANIFEST_BYTES, QUANT_ABI_VERSION,
    SIDECAR_FORMAT, SIDECAR_FORMAT_VERSION,
};
use bridge_gguf_split::GgufSet;
use bridge_io_windows::{PositionedFile, ReadCancellation, ReadError, ReadLimits};
use bridge_model_hy3::{Hy3Error, Hy3Tensor, Hy3TensorRole, ValidatedHy3Model};
use sha2::{Digest, Sha256};

const DEFAULT_HASH_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const MAX_HASH_CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSegment {
    pub shard_index: usize,
    pub range: Range<u64>,
    pub ggml_type: String,
}

impl SourceSegment {
    pub fn length(&self) -> u64 {
        self.range.end - self.range.start
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectExpertRecord {
    pub key: ExpertKey,
    pub gate: SourceSegment,
    pub up: SourceSegment,
    pub down: SourceSegment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectExpertIndex {
    records: Vec<DirectExpertRecord>,
}

impl DirectExpertIndex {
    pub fn build(model: &ValidatedHy3Model) -> Result<Self, PrepareError> {
        let config = model.config();
        let moe_layer_count = config
            .block_count
            .checked_sub(1)
            .ok_or(PrepareError::NoMoeLayers)?;
        let capacity_u64 = u64::from(moe_layer_count)
            .checked_mul(u64::from(config.expert_count))
            .ok_or(PrepareError::ArithmeticOverflow)?;
        let capacity = usize::try_from(capacity_u64).map_err(|_| PrepareError::ArithmeticOverflow)?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(capacity)
            .map_err(|_| PrepareError::AllocationFailed { requested: capacity })?;

        for layer in 1..config.block_count {
            let gate = required_tensor(model, Hy3TensorRole::RoutedGate { layer })?;
            let up = required_tensor(model, Hy3TensorRole::RoutedUp { layer })?;
            let down = required_tensor(model, Hy3TensorRole::RoutedDown { layer })?;
            for expert in 0..config.expert_count {
                records.push(DirectExpertRecord {
                    key: ExpertKey { layer, expert },
                    gate: source_segment(gate, config.expert_count, expert)?,
                    up: source_segment(up, config.expert_count, expert)?,
                    down: source_segment(down, config.expert_count, expert)?,
                });
            }
        }
        Ok(Self { records })
    }

    pub fn records(&self) -> &[DirectExpertRecord] {
        &self.records
    }

    pub fn get(&self, key: ExpertKey) -> Option<&DirectExpertRecord> {
        self.records
            .binary_search_by_key(&key, |record| record.key)
            .ok()
            .and_then(|index| self.records.get(index))
    }

    pub fn maximum_segment_bytes(&self) -> Result<usize, PrepareError> {
        let maximum = self
            .records
            .iter()
            .flat_map(|record| [&record.gate, &record.up, &record.down])
            .map(SourceSegment::length)
            .max()
            .unwrap_or(0);
        usize::try_from(maximum).map_err(|_| PrepareError::ArithmeticOverflow)
    }
}

#[derive(Debug)]
pub struct DirectExpertStore {
    readers: Vec<PositionedFile>,
    index: DirectExpertIndex,
}

impl DirectExpertStore {
    pub fn open(set: &GgufSet, model: &ValidatedHy3Model) -> Result<Self, PrepareError> {
        let index = DirectExpertIndex::build(model)?;
        let maximum = index.maximum_segment_bytes()?.max(1);
        let mut readers = Vec::new();
        readers
            .try_reserve_exact(set.files().len())
            .map_err(|_| PrepareError::AllocationFailed {
                requested: set.files().len(),
            })?;
        for shard in set.files() {
            readers.push(PositionedFile::open(
                shard.path(),
                ReadLimits {
                    max_request_bytes: maximum,
                },
            )?);
        }
        Ok(Self { readers, index })
    }

    pub const fn index(&self) -> &DirectExpertIndex {
        &self.index
    }

    pub fn read_expert(
        &self,
        key: ExpertKey,
        cancellation: &ReadCancellation,
    ) -> Result<DirectExpertBytes, PrepareError> {
        let record = self.index.get(key).ok_or(PrepareError::MissingExpert(key))?;
        Ok(DirectExpertBytes {
            gate: self.read_segment(&record.gate, cancellation)?,
            up: self.read_segment(&record.up, cancellation)?,
            down: self.read_segment(&record.down, cancellation)?,
            gate_type: record.gate.ggml_type.clone(),
            up_type: record.up.ggml_type.clone(),
            down_type: record.down.ggml_type.clone(),
        })
    }

    fn read_segment(
        &self,
        segment: &SourceSegment,
        cancellation: &ReadCancellation,
    ) -> Result<Vec<u8>, PrepareError> {
        let reader = self
            .readers
            .get(segment.shard_index)
            .ok_or(PrepareError::ShardIndexOutOfRange(segment.shard_index))?;
        Ok(reader.read_exact_at(segment.range.clone(), cancellation)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectExpertBytes {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
    pub gate_type: String,
    pub up_type: String,
    pub down_type: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepareOptions {
    pub layout: ExpertLayout,
    pub alignment: u64,
    pub overwrite: bool,
    pub verify_after_write: bool,
    pub hash_chunk_bytes: usize,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            layout: ExpertLayout::FusedGateUp,
            alignment: 4096,
            overwrite: false,
            verify_after_write: true,
            hash_chunk_bytes: DEFAULT_HASH_CHUNK_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareReport {
    pub data_path: PathBuf,
    pub manifest_path: PathBuf,
    pub source_bytes_hashed: u64,
    pub expert_payload_bytes: u64,
    pub sidecar_bytes: u64,
    pub record_count: usize,
    pub sidecar_sha256: String,
    pub tensor_directory_sha256: String,
}

pub fn prepare_sidecar(
    set: &GgufSet,
    model: &ValidatedHy3Model,
    data_path: impl AsRef<Path>,
    manifest_path: impl AsRef<Path>,
    options: PrepareOptions,
    cancellation: &ReadCancellation,
) -> Result<PrepareReport, PrepareError> {
    validate_options(options)?;
    let data_path = absolute_output_path(data_path.as_ref())?;
    let manifest_path = absolute_output_path(manifest_path.as_ref())?;
    validate_output_targets(set, &data_path, &manifest_path, options.overwrite)?;
    if cancellation.is_cancelled() {
        return Err(PrepareError::Cancelled);
    }

    let direct = DirectExpertStore::open(set, model)?;
    let directory_hash = tensor_directory_sha256(set)?;
    let source_files = hash_source_files(set, options.hash_chunk_bytes, cancellation)?;
    let source_bytes_hashed = source_files.iter().try_fold(0_u64, |total, source| {
        total
            .checked_add(source.length)
            .ok_or(PrepareError::ArithmeticOverflow)
    })?;

    let nonce = unique_nonce()?;
    let temporary_data = temporary_path(&data_path, "data", nonce)?;
    let temporary_manifest = temporary_path(&manifest_path, "manifest", nonce)?;
    let mut temporary = TemporaryOutputs::new(temporary_data.clone(), temporary_manifest.clone());
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_data)
        .map_err(|source| PrepareError::OutputIo {
            path: temporary_data.clone(),
            operation: "create temporary sidecar",
            source,
        })?;
    let record_count =
        u64::try_from(direct.index.records.len()).map_err(|_| PrepareError::ArithmeticOverflow)?;
    let header = SidecarHeader::new(options.layout, options.alignment, record_count).encode();
    let mut sidecar_hasher = Sha256::new();
    write_hashed(&mut output, &temporary_data, &header, &mut sidecar_hasher)?;
    let mut position = header.len() as u64;
    position = pad_to_alignment(
        &mut output,
        &temporary_data,
        position,
        options.alignment,
        &mut sidecar_hasher,
    )?;

    let mut records = Vec::new();
    records
        .try_reserve_exact(direct.index.records.len())
        .map_err(|_| PrepareError::AllocationFailed {
            requested: direct.index.records.len(),
        })?;
    let mut expert_payload_bytes = 0_u64;
    for source in &direct.index.records {
        if cancellation.is_cancelled() {
            return Err(PrepareError::Cancelled);
        }
        position = pad_to_alignment(
            &mut output,
            &temporary_data,
            position,
            options.alignment,
            &mut sidecar_hasher,
        )?;
        let record_start = position;

        let (gate, next) = {
            let mut sink = SidecarSink {
                output: &mut output,
                path: &temporary_data,
                hasher: &mut sidecar_hasher,
            };
            copy_segment(
                &direct,
                source.key,
                &source.gate,
                &mut sink,
                position,
                cancellation,
            )?
        };
        position = next;
        if options.layout == ExpertLayout::Sequential {
            position = pad_to_alignment(
                &mut output,
                &temporary_data,
                position,
                options.alignment,
                &mut sidecar_hasher,
            )?;
        }
        let (up, next) = {
            let mut sink = SidecarSink {
                output: &mut output,
                path: &temporary_data,
                hasher: &mut sidecar_hasher,
            };
            copy_segment(&direct, source.key, &source.up, &mut sink, position, cancellation)?
        };
        position = next;
        position = pad_to_alignment(
            &mut output,
            &temporary_data,
            position,
            options.alignment,
            &mut sidecar_hasher,
        )?;
        let (down, next) = {
            let mut sink = SidecarSink {
                output: &mut output,
                path: &temporary_data,
                hasher: &mut sidecar_hasher,
            };
            copy_segment(
                &direct,
                source.key,
                &source.down,
                &mut sink,
                position,
                cancellation,
            )?
        };
        position = next;
        position = pad_to_alignment(
            &mut output,
            &temporary_data,
            position,
            options.alignment,
            &mut sidecar_hasher,
        )?;
        let length = position
            .checked_sub(record_start)
            .ok_or(PrepareError::ArithmeticOverflow)?;
        expert_payload_bytes = expert_payload_bytes
            .checked_add(gate.length)
            .and_then(|total| total.checked_add(up.length))
            .and_then(|total| total.checked_add(down.length))
            .ok_or(PrepareError::ArithmeticOverflow)?;
        records.push(ExpertRecord {
            key: source.key,
            offset: record_start,
            length,
            gate,
            up,
            down,
        });
    }
    output
        .flush()
        .and_then(|()| output.sync_all())
        .map_err(|source| PrepareError::OutputIo {
            path: temporary_data.clone(),
            operation: "flush temporary sidecar",
            source,
        })?;
    drop(output);

    let sidecar_sha256 = format!("{:x}", sidecar_hasher.finalize());
    let manifest = SidecarManifest {
        format: SIDECAR_FORMAT.into(),
        format_version: SIDECAR_FORMAT_VERSION,
        engine_version: env!("CARGO_PKG_VERSION").into(),
        quant_abi_version: QUANT_ABI_VERSION,
        alignment: options.alignment,
        layout: options.layout,
        source_files,
        tensor_directory_sha256: directory_hash.clone(),
        sidecar: SidecarFileIdentity {
            length: position,
            sha256: sidecar_sha256.clone(),
        },
        records,
    };
    manifest.validate()?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(PrepareError::ManifestSerialization)?;
    if manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(PrepareError::ManifestTooLarge {
            actual: manifest_bytes.len(),
            maximum: MAX_MANIFEST_BYTES as usize,
        });
    }
    let mut manifest_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_manifest)
        .map_err(|source| PrepareError::OutputIo {
            path: temporary_manifest.clone(),
            operation: "create temporary manifest",
            source,
        })?;
    manifest_file
        .write_all(&manifest_bytes)
        .and_then(|()| manifest_file.flush())
        .and_then(|()| manifest_file.sync_all())
        .map_err(|source| PrepareError::OutputIo {
            path: temporary_manifest.clone(),
            operation: "write temporary manifest",
            source,
        })?;
    drop(manifest_file);

    if options.verify_after_write {
        let sidecar = Sidecar::open(&temporary_data, &temporary_manifest)?;
        sidecar.verify_data_hash(cancellation)?;
    }
    commit_outputs(
        &temporary_data,
        &temporary_manifest,
        &data_path,
        &manifest_path,
        options.overwrite,
        nonce,
    )?;
    temporary.disarm();

    Ok(PrepareReport {
        data_path,
        manifest_path,
        source_bytes_hashed,
        expert_payload_bytes,
        sidecar_bytes: position,
        record_count: direct.index.records.len(),
        sidecar_sha256,
        tensor_directory_sha256: directory_hash,
    })
}

pub fn verify_source_bindings(
    set: &GgufSet,
    manifest: &SidecarManifest,
    hash_chunk_bytes: usize,
    cancellation: &ReadCancellation,
) -> Result<(), PrepareError> {
    validate_hash_chunk(hash_chunk_bytes)?;
    let directory = tensor_directory_sha256(set)?;
    if directory != manifest.tensor_directory_sha256 {
        return Err(PrepareError::DirectoryHashMismatch {
            expected: manifest.tensor_directory_sha256.clone(),
            actual: directory,
        });
    }
    let actual = hash_source_files(set, hash_chunk_bytes, cancellation)?;
    if actual.len() != manifest.source_files.len() {
        return Err(PrepareError::SourceCountMismatch {
            expected: manifest.source_files.len(),
            actual: actual.len(),
        });
    }
    for (expected, actual) in manifest.source_files.iter().zip(actual) {
        if expected.ordinal != actual.ordinal
            || expected.length != actual.length
            || expected.sha256 != actual.sha256
        {
            return Err(PrepareError::SourceIdentityMismatch {
                ordinal: expected.ordinal,
                expected_length: expected.length,
                actual_length: actual.length,
                expected_sha256: expected.sha256.clone(),
                actual_sha256: actual.sha256,
            });
        }
    }
    Ok(())
}

pub fn tensor_directory_sha256(set: &GgufSet) -> Result<String, PrepareError> {
    let mut hasher = Sha256::new();
    hasher.update(b"lightbridge-tensor-directory-v1\0");
    for location in set.tensors().ordered() {
        let descriptor = location.descriptor();
        hash_bytes(&mut hasher, descriptor.name().as_bytes())?;
        hash_u64(
            &mut hasher,
            u64::try_from(location.shard_index()).map_err(|_| PrepareError::ArithmeticOverflow)?,
        );
        hash_u64(&mut hasher, location.absolute_range().start);
        hash_u64(&mut hasher, location.absolute_range().end);
        hash_u64(&mut hasher, u64::from(descriptor.n_dims()));
        for &dimension in descriptor.shape() {
            hash_u64(&mut hasher, dimension);
        }
        for stride in descriptor.strides() {
            hash_u64(&mut hasher, stride);
        }
        hash_bytes(&mut hasher, descriptor.ty().name().as_bytes())?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn required_tensor(model: &ValidatedHy3Model, role: Hy3TensorRole) -> Result<&Hy3Tensor, PrepareError> {
    model
        .tensor_for_role(role)
        .ok_or(PrepareError::MissingTensorRole(role))
}

fn source_segment(tensor: &Hy3Tensor, expert_count: u32, expert: u32) -> Result<SourceSegment, PrepareError> {
    let slab = tensor.expert_slab(expert_count, expert)?;
    let tensor_start = tensor.location().absolute_range().start;
    let start = tensor_start
        .checked_add(slab.relative_range.start)
        .ok_or(PrepareError::ArithmeticOverflow)?;
    let end = tensor_start
        .checked_add(slab.relative_range.end)
        .ok_or(PrepareError::ArithmeticOverflow)?;
    Ok(SourceSegment {
        shard_index: tensor.location().shard_index(),
        range: start..end,
        ggml_type: tensor.location().descriptor().ty().name().into(),
    })
}

fn copy_segment(
    direct: &DirectExpertStore,
    key: ExpertKey,
    source: &SourceSegment,
    sink: &mut SidecarSink<'_>,
    position: u64,
    cancellation: &ReadCancellation,
) -> Result<(Segment, u64), PrepareError> {
    let reader = direct
        .readers
        .get(source.shard_index)
        .ok_or(PrepareError::ShardIndexOutOfRange(source.shard_index))?;
    let bytes = reader.read_exact_at(source.range.clone(), cancellation)?;
    let length = u64::try_from(bytes.len()).map_err(|_| PrepareError::ArithmeticOverflow)?;
    if length != source.length() {
        return Err(PrepareError::SegmentLengthMismatch {
            key,
            expected: source.length(),
            actual: length,
        });
    }
    sink.write(&bytes)?;
    let next = position
        .checked_add(length)
        .ok_or(PrepareError::ArithmeticOverflow)?;
    Ok((
        Segment {
            offset: position,
            length,
            ggml_type: source.ggml_type.clone(),
        },
        next,
    ))
}

struct SidecarSink<'a> {
    output: &'a mut File,
    path: &'a Path,
    hasher: &'a mut Sha256,
}

impl SidecarSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> Result<(), PrepareError> {
        write_hashed(self.output, self.path, bytes, self.hasher)
    }
}

fn hash_source_files(
    set: &GgufSet,
    chunk_size: usize,
    cancellation: &ReadCancellation,
) -> Result<Vec<SourceFileIdentity>, PrepareError> {
    validate_hash_chunk(chunk_size)?;
    let mut identities = Vec::new();
    identities
        .try_reserve_exact(set.files().len())
        .map_err(|_| PrepareError::AllocationFailed {
            requested: set.files().len(),
        })?;
    for (index, shard) in set.files().iter().enumerate() {
        let path = fs::canonicalize(shard.path()).map_err(|source| PrepareError::SourceIo {
            path: shard.path().to_owned(),
            operation: "canonicalize source",
            source,
        })?;
        let reader = PositionedFile::open(
            &path,
            ReadLimits {
                max_request_bytes: chunk_size,
            },
        )?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(chunk_size)
            .map_err(|_| PrepareError::AllocationFailed {
                requested: chunk_size,
            })?;
        buffer.resize(chunk_size, 0);
        let mut hasher = Sha256::new();
        let mut offset = 0_u64;
        while offset < reader.length() {
            if cancellation.is_cancelled() {
                return Err(PrepareError::Cancelled);
            }
            let remaining = reader.length() - offset;
            let length = usize::try_from(remaining.min(chunk_size as u64))
                .map_err(|_| PrepareError::ArithmeticOverflow)?;
            reader.read_exact_at_into(offset, &mut buffer[..length], cancellation)?;
            hasher.update(&buffer[..length]);
            offset = offset
                .checked_add(length as u64)
                .ok_or(PrepareError::ArithmeticOverflow)?;
        }
        identities.push(SourceFileIdentity {
            ordinal: u32::try_from(index).map_err(|_| PrepareError::ArithmeticOverflow)?,
            path: path.to_string_lossy().into_owned(),
            length: reader.length(),
            sha256: format!("{:x}", hasher.finalize()),
        });
    }
    Ok(identities)
}

fn validate_options(options: PrepareOptions) -> Result<(), PrepareError> {
    align_up(0, options.alignment)?;
    validate_hash_chunk(options.hash_chunk_bytes)
}

fn validate_hash_chunk(chunk_size: usize) -> Result<(), PrepareError> {
    if chunk_size == 0 || chunk_size > MAX_HASH_CHUNK_BYTES {
        return Err(PrepareError::InvalidHashChunk {
            actual: chunk_size,
            maximum: MAX_HASH_CHUNK_BYTES,
        });
    }
    Ok(())
}

fn validate_output_targets(
    set: &GgufSet,
    data: &Path,
    manifest: &Path,
    overwrite: bool,
) -> Result<(), PrepareError> {
    if data == manifest {
        return Err(PrepareError::OutputCollision(data.to_owned()));
    }
    for shard in set.files() {
        let source = fs::canonicalize(shard.path()).map_err(|source| PrepareError::SourceIo {
            path: shard.path().to_owned(),
            operation: "canonicalize source",
            source,
        })?;
        if data == source || manifest == source {
            return Err(PrepareError::OutputOverwritesSource(source));
        }
    }
    for target in [data, manifest] {
        if target.exists() {
            let metadata = fs::metadata(target).map_err(|source| PrepareError::OutputIo {
                path: target.to_owned(),
                operation: "inspect output target",
                source,
            })?;
            if !metadata.is_file() {
                return Err(PrepareError::OutputNotAFile(target.to_owned()));
            }
            if !overwrite {
                return Err(PrepareError::OutputExists(target.to_owned()));
            }
        }
    }
    Ok(())
}

fn absolute_output_path(path: &Path) -> Result<PathBuf, PrepareError> {
    let file_name = path
        .file_name()
        .ok_or_else(|| PrepareError::InvalidOutputPath(path.to_owned()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|source| PrepareError::OutputIo {
        path: parent.to_owned(),
        operation: "canonicalize output directory",
        source,
    })?;
    Ok(parent.join(file_name))
}

fn temporary_path(target: &Path, kind: &str, nonce: u128) -> Result<PathBuf, PrepareError> {
    let name = target
        .file_name()
        .ok_or_else(|| PrepareError::InvalidOutputPath(target.to_owned()))?
        .to_string_lossy();
    Ok(target.with_file_name(format!(".{name}.lightbridge-{kind}-{nonce}.tmp")))
}

fn unique_nonce() -> Result<u128, PrepareError> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PrepareError::ClockBeforeEpoch)?
        .as_nanos();
    Ok(nanos ^ u128::from(std::process::id()))
}

fn write_hashed(
    output: &mut File,
    path: &Path,
    bytes: &[u8],
    hasher: &mut Sha256,
) -> Result<(), PrepareError> {
    output.write_all(bytes).map_err(|source| PrepareError::OutputIo {
        path: path.to_owned(),
        operation: "write sidecar data",
        source,
    })?;
    hasher.update(bytes);
    Ok(())
}

fn pad_to_alignment(
    output: &mut File,
    path: &Path,
    position: u64,
    alignment: u64,
    hasher: &mut Sha256,
) -> Result<u64, PrepareError> {
    let aligned = align_up(position, alignment)?;
    let mut remaining = aligned - position;
    const ZEROES: [u8; 8192] = [0; 8192];
    while remaining > 0 {
        let length = usize::try_from(remaining.min(ZEROES.len() as u64))
            .map_err(|_| PrepareError::ArithmeticOverflow)?;
        write_hashed(output, path, &ZEROES[..length], hasher)?;
        remaining -= length as u64;
    }
    Ok(aligned)
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), PrepareError> {
    hash_u64(
        hasher,
        u64::try_from(bytes.len()).map_err(|_| PrepareError::ArithmeticOverflow)?,
    );
    hasher.update(bytes);
    Ok(())
}

fn commit_outputs(
    temporary_data: &Path,
    temporary_manifest: &Path,
    data: &Path,
    manifest: &Path,
    overwrite: bool,
    nonce: u128,
) -> Result<(), PrepareError> {
    let backup_data = temporary_path(data, "data-backup", nonce)?;
    let backup_manifest = temporary_path(manifest, "manifest-backup", nonce)?;
    let mut data_backed_up = false;
    let mut manifest_backed_up = false;
    if overwrite && data.exists() {
        fs::rename(data, &backup_data).map_err(|source| PrepareError::CommitIo {
            operation: "back up existing sidecar",
            path: data.to_owned(),
            source,
        })?;
        data_backed_up = true;
    }
    if overwrite && manifest.exists() {
        if let Err(source) = fs::rename(manifest, &backup_manifest) {
            if data_backed_up {
                let _ = fs::rename(&backup_data, data);
            }
            return Err(PrepareError::CommitIo {
                operation: "back up existing manifest",
                path: manifest.to_owned(),
                source,
            });
        }
        manifest_backed_up = true;
    }

    if let Err(source) = fs::rename(temporary_data, data) {
        restore_backups(
            data,
            manifest,
            &backup_data,
            &backup_manifest,
            data_backed_up,
            manifest_backed_up,
        );
        return Err(PrepareError::CommitIo {
            operation: "commit sidecar",
            path: data.to_owned(),
            source,
        });
    }
    if let Err(source) = fs::rename(temporary_manifest, manifest) {
        let _ = fs::remove_file(data);
        restore_backups(
            data,
            manifest,
            &backup_data,
            &backup_manifest,
            data_backed_up,
            manifest_backed_up,
        );
        return Err(PrepareError::CommitIo {
            operation: "commit manifest",
            path: manifest.to_owned(),
            source,
        });
    }
    if data_backed_up {
        let _ = fs::remove_file(backup_data);
    }
    if manifest_backed_up {
        let _ = fs::remove_file(backup_manifest);
    }
    Ok(())
}

fn restore_backups(
    data: &Path,
    manifest: &Path,
    backup_data: &Path,
    backup_manifest: &Path,
    data_backed_up: bool,
    manifest_backed_up: bool,
) {
    if data_backed_up {
        let _ = fs::rename(backup_data, data);
    }
    if manifest_backed_up {
        let _ = fs::rename(backup_manifest, manifest);
    }
}

struct TemporaryOutputs {
    data: PathBuf,
    manifest: PathBuf,
    armed: bool,
}

impl TemporaryOutputs {
    fn new(data: PathBuf, manifest: PathBuf) -> Self {
        Self {
            data,
            manifest,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryOutputs {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.data);
            let _ = fs::remove_file(&self.manifest);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error(transparent)]
    Hy3(#[from] Hy3Error),
    #[error(transparent)]
    Read(#[from] ReadError),
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    Sidecar(#[from] bridge_format::SidecarError),
    #[error("validated model contains no MoE layers")]
    NoMoeLayers,
    #[error("validated model is missing required tensor role {0:?}")]
    MissingTensorRole(Hy3TensorRole),
    #[error("direct expert index does not contain {0:?}")]
    MissingExpert(ExpertKey),
    #[error("direct expert source references missing shard index {0}")]
    ShardIndexOutOfRange(usize),
    #[error("expert {key:?} segment length is {actual}, expected {expected}")]
    SegmentLengthMismatch {
        key: ExpertKey,
        expected: u64,
        actual: u64,
    },
    #[error("hash chunk size is {actual}, expected 1..={maximum}")]
    InvalidHashChunk { actual: usize, maximum: usize },
    #[error("source file operation {operation} failed for {path:?}: {source}")]
    SourceIo {
        path: PathBuf,
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("output operation {operation} failed for {path:?}: {source}")]
    OutputIo {
        path: PathBuf,
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("commit operation {operation} failed for {path:?}: {source}")]
    CommitIo {
        path: PathBuf,
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("sidecar data and manifest output paths collide at {0:?}")]
    OutputCollision(PathBuf),
    #[error("sidecar output would overwrite source GGUF {0:?}")]
    OutputOverwritesSource(PathBuf),
    #[error("output target already exists: {0:?}")]
    OutputExists(PathBuf),
    #[error("output target is not a regular file: {0:?}")]
    OutputNotAFile(PathBuf),
    #[error("invalid output path {0:?}")]
    InvalidOutputPath(PathBuf),
    #[error("manifest serialization failed: {0}")]
    ManifestSerialization(serde_json::Error),
    #[error("serialized manifest is {actual} bytes, maximum is {maximum}")]
    ManifestTooLarge { actual: usize, maximum: usize },
    #[error("source binding count is {actual}, expected {expected}")]
    SourceCountMismatch { expected: usize, actual: usize },
    #[error(
        "source {ordinal} identity mismatch: length {actual_length} vs {expected_length}, SHA-256 {actual_sha256} vs {expected_sha256}"
    )]
    SourceIdentityMismatch {
        ordinal: u32,
        expected_length: u64,
        actual_length: u64,
        expected_sha256: String,
        actual_sha256: String,
    },
    #[error("tensor directory SHA-256 is {actual}, expected {expected}")]
    DirectoryHashMismatch { expected: String, actual: String },
    #[error("sidecar preparation was cancelled")]
    Cancelled,
    #[error("system clock is before the Unix epoch")]
    ClockBeforeEpoch,
    #[error("checked arithmetic overflow during sidecar preparation")]
    ArithmeticOverflow,
    #[error("allocation failed while reserving {requested} preparation entries")]
    AllocationFailed { requested: usize },
}
