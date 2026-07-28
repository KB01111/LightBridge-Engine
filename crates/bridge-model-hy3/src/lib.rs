//! Authoritative Hy3 GGUF metadata and tensor-schema validation.

mod config;
mod profile;
mod schema;
mod tensor_role;

use std::ops::Range;

use bridge_gguf::GgufValueType;
use bridge_gguf_split::{GgufSet, TensorLocation};
use config::{validate_exact_replica, SELECTED_METADATA_KEYS};
use schema::{generate_schema_for_validated_selected_iq2_m, validate_tensor_locations};

pub use config::{resolve_config, Hy3Config};
pub use profile::Hy3Profile;
pub use schema::{
    checked_expert_slab, generate_selected_iq2_m_schema, validate_selected_iq2_m_tensor_descriptors,
    TensorSpec,
};
pub use tensor_role::Hy3TensorRole;

const SELECTED_TENSOR_COUNT: usize = 1_278;

#[derive(Debug, thiserror::Error)]
pub enum Hy3Error {
    #[error("GGUF metadata key {key:?} has expected {expected:?}, actual missing")]
    MissingMetadataType { key: String, expected: GgufValueType },
    #[error("GGUF metadata key {key:?} has expected {expected:?}, actual {actual:?}")]
    MetadataStoredType {
        key: String,
        expected: GgufValueType,
        actual: GgufValueType,
    },
    #[error("GGUF metadata key {key:?} has actual value {actual}, expected {expected}")]
    MetadataValue {
        key: &'static str,
        expected: String,
        actual: String,
    },
    #[error("GGUF metadata key {key:?} has actual array element type {actual:?}, expected {expected:?}")]
    MetadataArrayElementType {
        key: &'static str,
        expected: GgufValueType,
        actual: GgufValueType,
    },
    #[error("GGUF metadata key {key:?} has non-finite actual value {actual}, expected finite {expected}")]
    NonFiniteMetadata {
        key: &'static str,
        expected: f32,
        actual: f32,
    },
    #[error("tensor name {name:?} is invalid; expected {expected}")]
    InvalidTensorName { name: String, expected: &'static str },
    #[error(
        "tensor {name:?} has actual state missing, expected shape {expected_shape:?} and type {expected_type}"
    )]
    MissingTensor {
        name: String,
        expected_shape: Vec<u64>,
        expected_type: &'static str,
    },
    #[error("unexpected tensor {name:?}; expected no tensor outside the generated Hy3 schema")]
    UnexpectedTensor { name: String },
    #[error("tensor {name:?} has actual duplicate descriptors, expected exactly one descriptor")]
    DuplicateTensor { name: String },
    #[error("tensor {name:?} has actual shape {actual:?}, expected shape {expected:?}")]
    TensorShape {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("tensor {name:?} has actual physical type {actual}, expected physical type {expected}")]
    TensorType {
        name: String,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("tensor {name:?} has actual rank {actual}, expected rank {expected}")]
    TensorRank {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("tensor {name:?} has actual dimension 2 value {actual}, expected expert_count {expected}")]
    ExpertDimension {
        name: String,
        expected: u32,
        actual: u64,
    },
    #[error(
        "tensor {name:?} has actual payload length {payload_bytes}, expected bytes divisible by expert_count {expert_count}"
    )]
    ExpertPayloadNotDivisible {
        name: String,
        payload_bytes: u64,
        expert_count: u32,
    },
    #[error("tensor {name:?} expert index has actual value {expert}, expected less than {expert_count}")]
    ExpertIndexOutOfRange {
        name: String,
        expert: u32,
        expert_count: u32,
    },
    #[error(
        "tensor {name:?} arithmetic overflow while computing {operation}; expected a representable checked range"
    )]
    Arithmetic { name: String, operation: &'static str },
    #[error("checked GGUF set has actual shard count 0, expected at least one metadata-bearing shard")]
    EmptySet,
    #[error(
        "GGUF shard {shard_index} has partial selected metadata with actual missing key {missing_key:?}, expected all selected keys or none"
    )]
    PartialShardMetadata {
        shard_index: usize,
        missing_key: &'static str,
    },
    #[error("GGUF shard {shard_index} selected metadata conflicts with shard 0: {source}")]
    ShardMetadata {
        shard_index: usize,
        #[source]
        source: Box<Hy3Error>,
    },
    #[error("config field {field} has actual value {actual}, expected range {minimum}..={maximum}")]
    ConfigOutOfRange {
        field: &'static str,
        minimum: u64,
        maximum: u64,
        actual: u64,
    },
    #[error("config field {field} has actual value {actual}, expected {expected}")]
    ConfigRelation {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("tensor directory count has actual value {actual}, expected {expected}")]
    TensorDirectoryCount { expected: usize, actual: usize },
    #[error("allocation failed while reserving {requested} entries for {context}")]
    AllocationFailed { context: &'static str, requested: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSlab {
    pub expert: u32,
    pub relative_range: Range<u64>,
}

#[derive(Debug, Clone)]
pub struct Hy3Tensor {
    role: Hy3TensorRole,
    location: TensorLocation,
}

impl Hy3Tensor {
    pub const fn role(&self) -> Hy3TensorRole {
        self.role
    }

    pub fn location(&self) -> &TensorLocation {
        &self.location
    }

    pub fn expert_slab(&self, expert_count: u32, expert: u32) -> Result<ExpertSlab, Hy3Error> {
        let range = self.location.absolute_range();
        let payload_bytes = range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| Hy3Error::Arithmetic {
                name: self.location.descriptor().name().to_owned(),
                operation: "tensor payload length",
            })?;
        checked_expert_slab(
            self.location.descriptor().name(),
            self.location.descriptor().shape(),
            payload_bytes,
            expert_count,
            expert,
        )
    }
}

#[derive(Debug)]
pub struct ValidatedHy3Model {
    config: Hy3Config,
    tensors: Vec<Hy3Tensor>,
    has_mtp: bool,
}

impl ValidatedHy3Model {
    pub const fn config(&self) -> &Hy3Config {
        &self.config
    }

    pub fn tensors(&self) -> &[Hy3Tensor] {
        &self.tensors
    }

    pub const fn has_mtp(&self) -> bool {
        self.has_mtp
    }
}

pub fn validate_selected_model(set: &GgufSet) -> Result<ValidatedHy3Model, Hy3Error> {
    let metadata = set.files().first().ok_or(Hy3Error::EmptySet)?.parsed();
    let config = resolve_config(metadata)?;
    Hy3Profile::selected_iq2_m().validate(&config)?;
    validate_later_shard_metadata(set, &config)?;

    let directory = set.tensors().ordered();
    if directory.len() != SELECTED_TENSOR_COUNT {
        return Err(Hy3Error::TensorDirectoryCount {
            expected: SELECTED_TENSOR_COUNT,
            actual: directory.len(),
        });
    }
    validate_tensor_locations(&config, directory)?;

    let schema = generate_schema_for_validated_selected_iq2_m(&config)?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(schema.len())
        .map_err(|_| Hy3Error::AllocationFailed {
            context: "validated semantic tensors",
            requested: schema.len(),
        })?;
    for spec in schema {
        let location = set
            .tensors()
            .get(spec.name())
            .ok_or_else(|| Hy3Error::MissingTensor {
                name: spec.name().to_owned(),
                expected_shape: spec.shape().to_vec(),
                expected_type: spec.ty().name(),
            })?;
        tensors.push(Hy3Tensor {
            role: spec.role(),
            location: location.clone(),
        });
    }

    Ok(ValidatedHy3Model {
        config,
        tensors,
        has_mtp: false,
    })
}

fn validate_later_shard_metadata(set: &GgufSet, authoritative: &Hy3Config) -> Result<(), Hy3Error> {
    for (shard_index, shard) in set.files().iter().enumerate().skip(1) {
        let mut present_count = 0;
        let mut first_missing = None;
        for &key in SELECTED_METADATA_KEYS {
            if shard
                .parsed()
                .metadata
                .iter()
                .any(|(candidate, _)| candidate == key)
            {
                present_count += 1;
            } else if first_missing.is_none() {
                first_missing = Some(key);
            }
        }
        if present_count == 0 {
            continue;
        }
        if let Some(missing_key) = first_missing {
            return Err(Hy3Error::PartialShardMetadata {
                shard_index,
                missing_key,
            });
        }

        let replica = resolve_config(shard.parsed()).map_err(|source| Hy3Error::ShardMetadata {
            shard_index,
            source: Box::new(source),
        })?;
        validate_exact_replica(authoritative, &replica).map_err(|source| Hy3Error::ShardMetadata {
            shard_index,
            source: Box::new(source),
        })?;
    }
    Ok(())
}
