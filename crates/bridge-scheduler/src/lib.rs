//! Versioned, drift-sensitive hardware tuning and backend policy.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HARDWARE_FINGERPRINT_VERSION: u32 = 1;
pub const TUNING_PROFILE_VERSION: u32 = 1;
pub const TUNING_PROFILE_FORMAT: &str = "lightbridge-hardware-tuning";
pub const DEFAULT_MINIMUM_IMPROVEMENT_BPS: u32 = 1_000;
pub const DEFAULT_HOST_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const DEFAULT_GPU_RESERVE_BYTES: u64 = 1_280 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareFingerprintV1 {
    pub version: u32,
    pub engine_build: String,
    pub operating_system: String,
    pub architecture: String,
    pub cpu_signature: String,
    pub logical_processors: usize,
    pub total_physical_memory: u64,
    pub devices: Vec<DeviceFingerprint>,
    pub power_state: String,
    pub pcie_links: Vec<String>,
    pub runtimes: BTreeMap<String, String>,
}

impl HardwareFingerprintV1 {
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.version != HARDWARE_FINGERPRINT_VERSION {
            return Err(ProfileError::HardwareVersion {
                actual: self.version,
                expected: HARDWARE_FINGERPRINT_VERSION,
            });
        }
        require_non_empty("engine build", &self.engine_build)?;
        require_non_empty("operating system", &self.operating_system)?;
        require_non_empty("architecture", &self.architecture)?;
        require_non_empty("CPU signature", &self.cpu_signature)?;
        require_non_empty("power state", &self.power_state)?;
        if self.logical_processors == 0 {
            return Err(ProfileError::ZeroLogicalProcessors);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ProfileError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.devices.sort_by(|a, b| {
            a.backend
                .cmp(&b.backend)
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.device_uuid.cmp(&b.device_uuid))
        });
        canonical.pcie_links.sort();
        let encoded = serde_json::to_vec(&canonical).map_err(ProfileError::Json)?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFingerprint {
    pub backend: BackendKind,
    pub name: String,
    pub device_uuid: Option<String>,
    pub driver: Option<String>,
    pub memory_bytes: Option<u64>,
    pub capability: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFingerprint {
    pub canonical_path: String,
    pub length: u64,
    pub sha256: String,
}

impl ArtifactFingerprint {
    fn validate(&self, field: &'static str) -> Result<(), ProfileError> {
        require_non_empty(field, &self.canonical_path)?;
        validate_sha256(field, &self.sha256)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    CpuScalar,
    CpuAvx2,
    CpuAvxVnni,
    CpuAvx512Vnni,
    Cuda,
    Vulkan,
    WindowsMlNpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageMode {
    Buffered,
    UnbufferedOverlapped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoragePolicy {
    pub mode: StorageMode,
    pub queue_depth: usize,
    pub read_slots: usize,
    pub slot_bytes: usize,
    pub hot_cache_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertSplit {
    pub cpu_experts: usize,
    pub cuda_experts: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    pub authoritative_backend: BackendKind,
    pub fallback_backend: BackendKind,
    pub cpu_threads: usize,
    pub cpu_set_ids: Vec<u32>,
    pub storage: StoragePolicy,
    pub prefill_chunk: usize,
    pub expert_split: ExpertSplit,
    pub speculative_ngram_t: Option<usize>,
    pub minimum_improvement_bps: u32,
    pub host_memory_reserve_bytes: u64,
    pub gpu_memory_reserve_bytes: u64,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            authoritative_backend: BackendKind::CpuAvx2,
            fallback_backend: BackendKind::CpuScalar,
            cpu_threads: 1,
            cpu_set_ids: Vec::new(),
            storage: StoragePolicy {
                mode: StorageMode::Buffered,
                queue_depth: 1,
                read_slots: 8,
                slot_bytes: 8 * 1024 * 1024,
                hot_cache_bytes: 0,
            },
            prefill_chunk: 1,
            expert_split: ExpertSplit {
                cpu_experts: 8,
                cuda_experts: 0,
            },
            speculative_ngram_t: None,
            minimum_improvement_bps: DEFAULT_MINIMUM_IMPROVEMENT_BPS,
            host_memory_reserve_bytes: DEFAULT_HOST_RESERVE_BYTES,
            gpu_memory_reserve_bytes: DEFAULT_GPU_RESERVE_BYTES,
        }
    }
}

impl ExecutionPolicy {
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.cpu_threads == 0 {
            return Err(ProfileError::InvalidPolicy("CPU threads must be non-zero"));
        }
        if !self.cpu_set_ids.is_empty() && self.cpu_set_ids.len() != self.cpu_threads {
            return Err(ProfileError::InvalidPolicy(
                "CPU affinity requires one logical CPU ID per worker",
            ));
        }
        if self.cpu_set_ids.iter().copied().collect::<BTreeSet<_>>().len() != self.cpu_set_ids.len() {
            return Err(ProfileError::InvalidPolicy("CPU affinity IDs must be unique"));
        }
        if self.storage.queue_depth == 0 || self.storage.read_slots == 0 || self.storage.slot_bytes == 0 {
            return Err(ProfileError::InvalidPolicy(
                "storage queue depth, read slots, and slot bytes must be non-zero",
            ));
        }
        if !matches!(self.prefill_chunk, 1 | 2 | 4 | 8) {
            return Err(ProfileError::InvalidPolicy("prefill chunk must be 1, 2, 4, or 8"));
        }
        if let Some(t) = self.speculative_ngram_t {
            if t != 2 {
                return Err(ProfileError::InvalidPolicy(
                    "only lossless T=2 n-gram speculation is supported",
                ));
            }
            if self.prefill_chunk < t {
                return Err(ProfileError::InvalidPolicy(
                    "T=2 speculation requires grouped prefill of at least two positions",
                ));
            }
        }
        if self.minimum_improvement_bps > 10_000 {
            return Err(ProfileError::InvalidPolicy(
                "minimum improvement basis points cannot exceed 10000",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectnessEvidence {
    pub packed_dot_oracle: bool,
    pub routes_match: bool,
    pub greedy_tokens_match: bool,
    pub deterministic_repeats: bool,
    pub maximum_absolute_logit_error: f64,
    pub maximum_relative_logit_error: f64,
}

impl CorrectnessEvidence {
    pub fn passes(&self) -> bool {
        self.packed_dot_oracle
            && self.routes_match
            && self.greedy_tokens_match
            && self.deterministic_repeats
            && self.maximum_absolute_logit_error <= 1.0e-3
            && self.maximum_relative_logit_error <= 1.0e-4
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendDecision {
    pub backend: BackendKind,
    pub automatic: bool,
    pub authoritative: bool,
    pub median_complete_token_ns: Option<u64>,
    pub next_best_median_ns: Option<u64>,
    pub improvement_bps: Option<u32>,
    pub correctness: Option<CorrectnessEvidence>,
    pub reason: String,
}

impl BackendDecision {
    pub fn measured(
        backend: BackendKind,
        authoritative: bool,
        median_complete_token_ns: u64,
        next_best_median_ns: u64,
        correctness: CorrectnessEvidence,
        minimum_improvement_bps: u32,
    ) -> Self {
        let improvement_bps = improvement_bps(median_complete_token_ns, next_best_median_ns);
        let correctness_passes = correctness.passes();
        let automatic = authoritative
            && correctness_passes
            && improvement_bps.is_some_and(|value| value > 0 && value >= minimum_improvement_bps);
        let reason = if !authoritative {
            "backend is advisory-only".to_owned()
        } else if !correctness_passes {
            "correctness gate did not pass".to_owned()
        } else if !automatic {
            format!(
                "measured improvement is {} bps; {} bps required",
                improvement_bps.unwrap_or(0),
                minimum_improvement_bps
            )
        } else {
            format!(
                "correctness passed and measured improvement is {} bps",
                improvement_bps.unwrap_or(0)
            )
        };
        Self {
            backend,
            automatic,
            authoritative,
            median_complete_token_ns: Some(median_complete_token_ns),
            next_best_median_ns: Some(next_best_median_ns),
            improvement_bps,
            correctness: Some(correctness),
            reason,
        }
    }

    pub fn rejected(backend: BackendKind, authoritative: bool, reason: impl Into<String>) -> Self {
        Self {
            backend,
            automatic: false,
            authoritative,
            median_complete_token_ns: None,
            next_best_median_ns: None,
            improvement_bps: None,
            correctness: None,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningMeasurement {
    pub name: String,
    pub backend: Option<BackendKind>,
    pub samples_ns: Vec<u64>,
    pub median_ns: u64,
    pub bytes: Option<u64>,
    pub detail: String,
}

impl TuningMeasurement {
    pub fn new(
        name: impl Into<String>,
        backend: Option<BackendKind>,
        samples: Vec<Duration>,
        bytes: Option<u64>,
        detail: impl Into<String>,
    ) -> Result<Self, ProfileError> {
        if samples.is_empty() {
            return Err(ProfileError::EmptyMeasurement);
        }
        let mut samples_ns = samples
            .into_iter()
            .map(|sample| u64::try_from(sample.as_nanos()).unwrap_or(u64::MAX))
            .collect::<Vec<_>>();
        samples_ns.sort_unstable();
        let median_ns = samples_ns[samples_ns.len() / 2];
        Ok(Self {
            name: name.into(),
            backend,
            samples_ns,
            median_ns,
            bytes,
            detail: detail.into(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningProfileV1 {
    pub format: String,
    pub version: u32,
    pub profile: String,
    pub hardware: HardwareFingerprintV1,
    pub hardware_sha256: String,
    pub model: ArtifactFingerprint,
    pub sidecar: Option<ArtifactFingerprint>,
    pub policy: ExecutionPolicy,
    pub measurements: Vec<TuningMeasurement>,
    pub decisions: Vec<BackendDecision>,
    #[serde(default)]
    pub rejections: Vec<CandidateRejection>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRejection {
    pub candidate: String,
    pub reason: String,
}

impl TuningProfileV1 {
    pub fn new(
        profile: impl Into<String>,
        hardware: HardwareFingerprintV1,
        model: ArtifactFingerprint,
        sidecar: Option<ArtifactFingerprint>,
        policy: ExecutionPolicy,
        measurements: Vec<TuningMeasurement>,
        decisions: Vec<BackendDecision>,
    ) -> Result<Self, ProfileError> {
        let hardware_sha256 = hardware.digest()?;
        let value = Self {
            format: TUNING_PROFILE_FORMAT.to_owned(),
            version: TUNING_PROFILE_VERSION,
            profile: profile.into(),
            hardware,
            hardware_sha256,
            model,
            sidecar,
            policy,
            measurements,
            decisions,
            rejections: Vec::new(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.format != TUNING_PROFILE_FORMAT {
            return Err(ProfileError::Format(self.format.clone()));
        }
        if self.version != TUNING_PROFILE_VERSION {
            return Err(ProfileError::ProfileVersion {
                actual: self.version,
                expected: TUNING_PROFILE_VERSION,
            });
        }
        require_non_empty("profile", &self.profile)?;
        self.hardware.validate()?;
        let actual = self.hardware.digest()?;
        if actual != self.hardware_sha256 {
            return Err(ProfileError::HardwareDigest {
                expected: self.hardware_sha256.clone(),
                actual,
            });
        }
        self.model.validate("model")?;
        if let Some(sidecar) = &self.sidecar {
            sidecar.validate("sidecar")?;
        }
        self.policy.validate()
    }

    pub fn validate_current(
        &self,
        hardware: &HardwareFingerprintV1,
        model: &ArtifactFingerprint,
        sidecar: Option<&ArtifactFingerprint>,
    ) -> Result<(), ProfileError> {
        self.validate()?;
        let current_hardware = hardware.digest()?;
        if current_hardware != self.hardware_sha256 {
            return Err(ProfileError::Drift("hardware fingerprint changed"));
        }
        if model != &self.model {
            return Err(ProfileError::Drift("model identity changed"));
        }
        if sidecar != self.sidecar.as_ref() {
            return Err(ProfileError::Drift("sidecar identity changed"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChromeTrace {
    #[serde(rename = "traceEvents")]
    pub trace_events: Vec<TraceEvent>,
    #[serde(rename = "displayTimeUnit")]
    pub display_time_unit: String,
}

impl Default for ChromeTrace {
    fn default() -> Self {
        Self {
            trace_events: Vec::new(),
            display_time_unit: "ms".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String,
    pub ts: u64,
    pub dur: u64,
    pub pid: u32,
    pub tid: u32,
    pub args: BTreeMap<String, serde_json::Value>,
}

impl ChromeTrace {
    pub fn push_complete(
        &mut self,
        name: impl Into<String>,
        category: impl Into<String>,
        start: Duration,
        duration: Duration,
        args: BTreeMap<String, serde_json::Value>,
    ) {
        self.trace_events.push(TraceEvent {
            name: name.into(),
            cat: category.into(),
            ph: "X".to_owned(),
            ts: micros(start),
            dur: micros(duration),
            pid: std::process::id(),
            tid: 0,
            args,
        });
    }
}

fn improvement_bps(candidate_ns: u64, reference_ns: u64) -> Option<u32> {
    if reference_ns == 0 || candidate_ns >= reference_ns {
        return Some(0);
    }
    let saved = u128::from(reference_ns - candidate_ns);
    let basis_points = saved.checked_mul(10_000)?.checked_div(u128::from(reference_ns))?;
    u32::try_from(basis_points).ok()
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), ProfileError> {
    if value.trim().is_empty() {
        Err(ProfileError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), ProfileError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(ProfileError::Sha256(field))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("unexpected hardware fingerprint version {actual}; expected {expected}")]
    HardwareVersion { actual: u32, expected: u32 },
    #[error("unexpected tuning profile version {actual}; expected {expected}")]
    ProfileVersion { actual: u32, expected: u32 },
    #[error("unexpected tuning profile format {0:?}")]
    Format(String),
    #[error("profile field {0} must not be empty")]
    EmptyField(&'static str),
    #[error("profile field {0} is not a lowercase SHA-256")]
    Sha256(&'static str),
    #[error("hardware fingerprint declares zero logical processors")]
    ZeroLogicalProcessors,
    #[error("hardware fingerprint digest is {actual}, expected {expected}")]
    HardwareDigest { expected: String, actual: String },
    #[error("invalid execution policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("tuning measurement has no samples")]
    EmptyMeasurement,
    #[error("tuning profile is stale: {0}")]
    Drift(&'static str),
    #[error("failed to encode or decode tuning JSON: {0}")]
    Json(serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware() -> HardwareFingerprintV1 {
        HardwareFingerprintV1 {
            version: HARDWARE_FINGERPRINT_VERSION,
            engine_build: "test-build".to_owned(),
            operating_system: "windows".to_owned(),
            architecture: "x86_64".to_owned(),
            cpu_signature: "family-model-stepping".to_owned(),
            logical_processors: 24,
            total_physical_memory: 32 * 1024 * 1024 * 1024,
            devices: Vec::new(),
            power_state: "ac".to_owned(),
            pcie_links: Vec::new(),
            runtimes: BTreeMap::new(),
        }
    }

    fn artifact(name: &str) -> ArtifactFingerprint {
        ArtifactFingerprint {
            canonical_path: name.to_owned(),
            length: 42,
            sha256: "a".repeat(64),
        }
    }

    fn correctness() -> CorrectnessEvidence {
        CorrectnessEvidence {
            packed_dot_oracle: true,
            routes_match: true,
            greedy_tokens_match: true,
            deterministic_repeats: true,
            maximum_absolute_logit_error: 1.0e-4,
            maximum_relative_logit_error: 1.0e-5,
        }
    }

    #[test]
    fn automatic_backend_requires_correctness_and_ten_percent_gain() {
        let accepted = BackendDecision::measured(
            BackendKind::Cuda,
            true,
            800,
            1_000,
            correctness(),
            DEFAULT_MINIMUM_IMPROVEMENT_BPS,
        );
        assert!(accepted.automatic);
        assert_eq!(accepted.improvement_bps, Some(2_000));

        let too_close = BackendDecision::measured(
            BackendKind::CpuAvx512Vnni,
            true,
            950,
            1_000,
            correctness(),
            DEFAULT_MINIMUM_IMPROVEMENT_BPS,
        );
        assert!(!too_close.automatic);

        let mut failed = correctness();
        failed.routes_match = false;
        assert!(
            !BackendDecision::measured(
                BackendKind::Vulkan,
                true,
                500,
                1_000,
                failed,
                DEFAULT_MINIMUM_IMPROVEMENT_BPS,
            )
            .automatic
        );
    }

    #[test]
    fn profile_detects_hardware_and_artifact_drift() {
        let model = artifact("model.gguf");
        let profile = TuningProfileV1::new(
            "performance",
            hardware(),
            model.clone(),
            None,
            ExecutionPolicy {
                cpu_threads: 12,
                ..ExecutionPolicy::default()
            },
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        profile.validate_current(&hardware(), &model, None).unwrap();

        let mut changed = hardware();
        changed.power_state = "battery".to_owned();
        assert!(matches!(
            profile.validate_current(&changed, &model, None),
            Err(ProfileError::Drift("hardware fingerprint changed"))
        ));
        assert!(matches!(
            profile.validate_current(&hardware(), &artifact("other.gguf"), None),
            Err(ProfileError::Drift("model identity changed"))
        ));
    }

    #[test]
    fn policy_rejects_invalid_affinity_and_speculation_coupling() {
        let mut policy = ExecutionPolicy {
            cpu_threads: 2,
            cpu_set_ids: vec![0],
            ..ExecutionPolicy::default()
        };
        assert!(matches!(
            policy.validate(),
            Err(ProfileError::InvalidPolicy(
                "CPU affinity requires one logical CPU ID per worker"
            ))
        ));

        policy.cpu_set_ids = vec![0, 0];
        assert!(matches!(
            policy.validate(),
            Err(ProfileError::InvalidPolicy("CPU affinity IDs must be unique"))
        ));

        policy.cpu_set_ids = vec![0, 1];
        policy.speculative_ngram_t = Some(2);
        assert!(matches!(
            policy.validate(),
            Err(ProfileError::InvalidPolicy(
                "T=2 speculation requires grouped prefill of at least two positions"
            ))
        ));
        policy.prefill_chunk = 2;
        policy.validate().unwrap();
    }

    #[test]
    fn trace_uses_complete_events_and_microseconds() {
        let mut trace = ChromeTrace::default();
        trace.push_complete(
            "prefill",
            "model",
            Duration::from_millis(2),
            Duration::from_millis(3),
            BTreeMap::new(),
        );
        assert_eq!(trace.trace_events[0].ph, "X");
        assert_eq!(trace.trace_events[0].ts, 2_000);
        assert_eq!(trace.trace_events[0].dur, 3_000);
    }
}
