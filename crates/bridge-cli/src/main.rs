use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::mem;
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use bridge_cache::{CacheConfig, CompressedCache};
use bridge_cli::{build_report, render_json, render_text};
use bridge_core::ggml_type::GgmlType;
use bridge_core::sys::{memory_status, CpuTopology};
use bridge_format::{ExpertKey, ExpertLayout, Sidecar};
use bridge_io_windows::{file_storage, FileStorage, PositionedFile, ReadCancellation, ReadLimits};
use bridge_kernels_cpu::{recommended_thread_count, CpuBackend, CpuBackendConfig, CpuCapabilities};
use bridge_kernels_reference::{
    gemv_into, required_q8_k_bytes, PackedMatrix, PayloadEndian, ReferenceExecutionMode,
};
use bridge_model_hy3::validate_selected_model;
use bridge_prepare::{prepare_sidecar, DirectExpertIndex, PrepareOptions};
use bridge_quant_layout::{
    layout as quant_layout, quantize_row_q8_k_into, vec_dot_q8_k, CpuDotBackend, ValidatedQ8KMatrix,
};
use bridge_runtime::{
    validate_selected_payload, CancellationToken, ExpertSourceOptions, Hy3ChatEngine, Hy3ChatSession,
    Hy3ScalarOptions, SamplingConfig,
};
use bridge_scheduler::{
    ArtifactFingerprint, BackendDecision, BackendKind, CandidateRejection, ChromeTrace, DeviceFingerprint,
    ExecutionPolicy, HardwareFingerprintV1, StorageMode, TuningMeasurement, TuningProfileV1,
    HARDWARE_FINGERPRINT_VERSION,
};
use bridge_tokenizer::{ChatMessage, ChatTemplateOptions, Hy3Tokenizer, ReasoningEffort};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(windows)]
use bridge_io_windows::{FileBuffering, OverlappedFile, OverlappedRead, ReadSlotPool};

const MAX_JSON_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_CACHE_HEAT_ENTRIES: usize = 65_536;
const MAX_BENCHMARK_CORPUS_PROMPTS: usize = 8;
const MAX_BENCHMARK_CORPUS_REPEATS: usize = 5;
const MAX_BENCHMARK_PROMPT_BYTES: usize = 64 * 1024;

#[derive(Debug, Parser)]
#[command(
    name = "bridge",
    version,
    about = "Run, validate, prepare, and serve the selected Hy3 model"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect and validate one selected-profile Hy3 GGUF model set.
    InspectGguf {
        /// GGUF file or numbered shard in the model set.
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        /// Emit one deterministic pretty-printed JSON report.
        #[arg(long)]
        json: bool,
    },
    /// Inspect host resources and report executable engine capabilities.
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// Build a deterministic storage and memory plan without reading weights.
    Plan {
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        #[arg(long, default_value_t = 2_048)]
        context: usize,
        #[arg(long, default_value_t = 2_048)]
        cache_mib: usize,
        #[arg(long, default_value_t = 64)]
        kv_page_tokens: usize,
        #[arg(long, default_value_t = 512)]
        memory_headroom_mib: usize,
        #[arg(long)]
        json: bool,
    },
    /// Validate schema, physical completeness, and optionally the full payload hash.
    Validate {
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        /// Read and authenticate every payload byte against the pinned SHA-256.
        #[arg(long)]
        payload: bool,
        #[arg(long)]
        json: bool,
    },
    /// Prepare a lossless expert-major sidecar and bound manifest.
    Prepare {
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        #[arg(long, value_name = "PATH")]
        output: PathBuf,
        #[arg(long, value_name = "PATH")]
        manifest: PathBuf,
        #[arg(long, value_enum, default_value_t = LayoutArg::FusedGateUp)]
        layout: LayoutArg,
        #[arg(long, default_value_t = 4_096)]
        alignment: u64,
        #[arg(long)]
        overwrite: bool,
        #[arg(long)]
        no_verify: bool,
        #[arg(long)]
        json: bool,
    },
    /// Encode plain text or a JSON chat transcript with the embedded tokenizer.
    Tokenize {
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        #[arg(long, conflicts_with = "chat_json")]
        text: Option<String>,
        #[arg(long, value_name = "PATH", conflicts_with = "text")]
        chat_json: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ReasoningArg::NoThink)]
        reasoning_effort: ReasoningArg,
        #[arg(long)]
        json: bool,
    },
    /// Decode comma-separated token IDs with the embedded tokenizer.
    Detokenize {
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        tokens: Vec<u32>,
        #[arg(long)]
        include_special_tokens: bool,
        #[arg(long)]
        json: bool,
    },
    /// Generate one streamed assistant response.
    Chat {
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[arg(long, conflicts_with = "chat_json")]
        prompt: Option<String>,
        #[arg(long, value_name = "PATH", conflicts_with_all = ["prompt", "system"])]
        chat_json: Option<PathBuf>,
        #[arg(long, conflicts_with = "chat_json")]
        system: Option<String>,
        #[arg(long, value_enum, default_value_t = ReasoningArg::NoThink)]
        reasoning_effort: ReasoningArg,
        #[arg(long, value_name = "PATH")]
        session_in: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        session_out: Option<PathBuf>,
        #[arg(long, default_value_t = 2_048)]
        session_max_mib: usize,
        /// Buffer the completion and emit one JSON object instead of streaming text.
        #[arg(long)]
        json: bool,
    },
    /// Run the bounded local OpenAI-compatible HTTP server.
    Serve {
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        #[arg(long, default_value_t = 1)]
        max_concurrent_requests: usize,
        #[arg(long, default_value_t = 2 * 1024 * 1024)]
        max_request_bytes: usize,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
    },
    /// Measure an authenticated prompt and decode run; never fabricates throughput.
    Bench {
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[arg(long, default_value = "Hello")]
        prompt: String,
        /// Versioned JSON prompt corpus evaluated in one authenticated engine.
        #[arg(long, value_name = "PATH", conflicts_with = "cold_warm")]
        prompt_corpus: Option<PathBuf>,
        /// Deterministic repetitions of every corpus prompt.
        #[arg(long, default_value_t = 2, requires = "prompt_corpus")]
        corpus_repeats: usize,
        /// Measure cold, admission, and repeated warm-state runs in one authenticated engine.
        #[arg(long)]
        cold_warm: bool,
        /// Validate and bind this drift-sensitive tuning profile before the run.
        #[arg(long, value_name = "PATH")]
        hardware_profile: Option<PathBuf>,
        /// Write measured Chrome/Perfetto complete events atomically.
        #[arg(long, value_name = "PATH")]
        trace: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Measure hardware candidates and write a versioned, drift-sensitive profile.
    Tune {
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        /// Optional prepared expert sidecar data file.
        #[arg(long, value_name = "PATH", requires = "sidecar_manifest")]
        sidecar: Option<PathBuf>,
        /// Manifest bound to --sidecar.
        #[arg(long, value_name = "PATH", requires = "sidecar")]
        sidecar_manifest: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = TuneProfileArg::Performance)]
        profile: TuneProfileArg,
        #[arg(long, value_name = "PATH")]
        output: PathBuf,
        /// Repeated samples per CPU and storage candidate.
        #[arg(long, default_value_t = 5)]
        samples: usize,
    },
    /// Inspect or remove persisted bounded-cache heat state.
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
}

#[derive(Debug, Subcommand)]
enum CacheCommand {
    /// Validate and normalize a bounded expert heat snapshot.
    InspectHeat {
        #[arg(long, value_name = "PATH")]
        path: PathBuf,
        #[arg(long, default_value_t = 65_536)]
        max_entries: usize,
    },
    /// Remove an explicitly named persisted heat snapshot.
    ClearHeat {
        #[arg(long, value_name = "PATH")]
        path: PathBuf,
    },
}

#[derive(Debug, Clone, Args)]
struct RuntimeArgs {
    #[arg(long, value_name = "PATH")]
    model: PathBuf,
    #[arg(long, default_value_t = 2_048)]
    context: usize,
    #[arg(long, default_value_t = 2_048)]
    cache_mib: usize,
    #[arg(long, default_value_t = 64)]
    kv_page_tokens: usize,
    #[arg(
        long = "backend",
        visible_alias = "reference-mode",
        value_enum,
        default_value_t = ExecutionModeArg::CpuQ8K
    )]
    backend: ExecutionModeArg,
    /// Number of bounded CPU worker threads; zero selects half the logical processors.
    #[arg(long, default_value_t = 0)]
    cpu_threads: usize,
    /// Pin persistent workers to these comma-separated logical CPU IDs.
    #[arg(long, value_delimiter = ',', value_name = "ID,...")]
    cpu_set_ids: Vec<u32>,
    /// Layer-major prompt positions per prefill chunk (1, 2, 4, or 8).
    #[arg(long, default_value_t = 1)]
    prefill_chunk: usize,
    /// Enable lossless greedy n-gram speculation at this width; only 2 is supported.
    #[arg(long, value_name = "TOKENS")]
    speculative_ngram_t: Option<usize>,
    /// Physical memory kept free after resident weights, expert cache, and the first KV page.
    #[arg(long, default_value_t = 512)]
    memory_headroom_mib: usize,
    /// Import and atomically persist expert-cache heat at this path.
    #[arg(long, value_name = "PATH")]
    cache_heat: Option<PathBuf>,
    #[arg(long, default_value_t = DEFAULT_CACHE_HEAT_ENTRIES)]
    cache_heat_max_entries: usize,
    #[arg(long, value_name = "PATH", requires = "sidecar_manifest")]
    sidecar_data: Option<PathBuf>,
    #[arg(long, value_name = "PATH", requires = "sidecar_data")]
    sidecar_manifest: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
struct SamplingArgs {
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LayoutArg {
    Sequential,
    FusedGateUp,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ExecutionModeArg {
    #[value(name = "cpu-q8-k")]
    CpuQ8K,
    /// Explicit streaming CUDA packed GEMV with transactional AVX2 fallback.
    #[value(name = "cuda-q8-k")]
    CudaQ8K,
    /// Opt-in Zen 5 AVX-VNNI packed dots; never selected without tuning evidence.
    #[value(name = "cpu-avx-vnni-q8-k")]
    CpuAvxVnniQ8K,
    /// Opt-in Zen 5 AVX-512/VNNI packed dots; never selected without tuning evidence.
    #[value(name = "cpu-avx512-vnni-q8-k")]
    CpuAvx512VnniQ8K,
    #[value(name = "scalar-q8-k", alias = "q8-k")]
    ScalarQ8K,
    DequantF32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ReasoningArg {
    High,
    Low,
    NoThink,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TuneProfileArg {
    Performance,
}

/// Runs the command-line application and maps its result to a process exit code.
///
/// Errors are formatted and written to standard error before returning failure.
///
/// # Examples
///
/// ```no_run
/// // Run the compiled `bridge` executable from a shell.
/// ```
///
#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Parses command-line arguments and dispatches the selected subcommand.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> anyhow::Result<()> {
/// run().await?;
/// # Ok(())
/// # }
/// ```
async fn run() -> Result<()> {
async fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::InspectGguf { model, json } => inspect_gguf(&model, json),
        Command::Doctor { json } => doctor(json),
        Command::Plan {
            model,
            context,
            cache_mib,
            kv_page_tokens,
            memory_headroom_mib,
            json,
        } => plan(
            &model,
            context,
            cache_mib,
            kv_page_tokens,
            memory_headroom_mib,
            json,
        ),
        Command::Validate { model, payload, json } => validate(&model, payload, json),
        Command::Prepare {
            model,
            output,
            manifest,
            layout,
            alignment,
            overwrite,
            no_verify,
            json,
        } => prepare(
            &model, &output, &manifest, layout, alignment, overwrite, no_verify, json,
        ),
        Command::Tokenize {
            model,
            text,
            chat_json,
            reasoning_effort,
            json,
        } => tokenize(&model, text, chat_json.as_deref(), reasoning_effort, json),
        Command::Detokenize {
            model,
            tokens,
            include_special_tokens,
            json,
        } => detokenize(&model, &tokens, include_special_tokens, json),
        Command::Chat {
            runtime,
            sampling,
            prompt,
            chat_json,
            system,
            reasoning_effort,
            session_in,
            session_out,
            session_max_mib,
            json,
        } => chat(
            runtime,
            sampling,
            prompt,
            chat_json.as_deref(),
            system,
            reasoning_effort,
            session_in.as_deref(),
            session_out.as_deref(),
            session_max_mib,
            json,
        ),
        Command::Serve {
            runtime,
            bind,
            max_concurrent_requests,
            max_request_bytes,
            max_new_tokens,
        } => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "bridge_server=info".into()),
                )
                .try_init()
                .ok();
            let engine = Arc::new(open_engine(&runtime)?);
            bridge_server::serve_shared(
                Arc::clone(&engine),
                bridge_server::ServerConfig {
                    bind,
                    max_concurrent_requests,
                    max_request_bytes,
                    max_new_tokens,
                },
            )
            .await?;
            persist_cache_heat(&runtime, &engine)?;
            Ok(())
        }
        Command::Bench {
            runtime,
            sampling,
            prompt,
            prompt_corpus,
            corpus_repeats,
            cold_warm,
            hardware_profile,
            trace,
            json,
        } => bench(BenchRunArgs {
            runtime,
            sampling,
            prompt,
            prompt_corpus,
            corpus_repeats,
            cold_warm,
            hardware_profile,
            trace_path: trace,
            json,
        }),
        Command::Tune {
            model,
            sidecar,
            sidecar_manifest,
            profile,
            output,
            samples,
        } => tune(
            &model,
            sidecar.as_deref(),
            sidecar_manifest.as_deref(),
            profile,
            &output,
            samples,
        ),
        Command::Cache { command } => cache(command),
    }
}

fn inspect_gguf(model: &Path, json: bool) -> Result<()> {
    let set = bridge_gguf_split::open_set(model)
        .with_context(|| format!("failed to open GGUF set {}", model.display()))?;
    let report =
        build_report(&set).with_context(|| format!("failed to validate Hy3 model {}", model.display()))?;
    let rendered = if json {
        render_json(&report).context("failed to serialize inspection report")?
    } else {
        render_text(&report)
    };
    write_stdout(rendered.as_bytes())
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    engine_version: &'static str,
    operating_system: &'static str,
    architecture: &'static str,
    cpu: CpuTopology,
    memory: bridge_core::sys::MemoryStatus,
    nvidia: NvidiaStatus,
    cuda_toolchain: ProbeStatus,
    cuda_nvrtc: CudaNvrtcStatus,
    cuda_packed_oracle: CudaPackedOracleStatus,
    vulkan: ProbeStatus,
    windows_ml_npu: ProbeStatus,
    npu_feasibility: NpuFeasibilityReport,
    capabilities: CapabilityReport,
    backend_status: Vec<BackendCapabilityStatus>,
}

#[derive(Debug, Serialize)]
struct NvidiaStatus {
    available: bool,
    detail: String,
    name: Option<String>,
    memory_mib: Option<u64>,
    driver: Option<String>,
    uuid: Option<String>,
    pci_bus_id: Option<String>,
    pcie_link: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeStatus {
    available: bool,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct CudaNvrtcStatus {
    available: bool,
    detail: String,
    canary: Option<bridge_kernels_cuda::CudaNvrtcCanary>,
}

#[derive(Debug, Clone, Serialize)]
struct CudaPackedOracleStatus {
    available: bool,
    detail: String,
    oracle: Option<bridge_kernels_cuda::CudaPackedQ8KOracle>,
    reusable: Option<bridge_kernels_cuda::CudaReusablePackedQ8KCanary>,
}

#[derive(Debug, Serialize)]
struct NpuFeasibilityReport {
    device_detected: bool,
    bridge_compiled: bool,
    authoritative_model_backend: bool,
    advisory_router_candidate: bool,
    weight_conversion_permitted: bool,
    reason: String,
}

#[derive(Debug, Serialize)]
struct BackendCapabilityStatus {
    backend: BackendKind,
    compiled: bool,
    runtime_available: bool,
    authoritative: bool,
    automatic: bool,
    reason: String,
}

#[derive(Debug, Serialize)]
struct CapabilityReport {
    scalar_reference: bool,
    q8_k_activation_path: bool,
    cpu_parallel_backend: bool,
    cpu_parallel_default_threads: usize,
    cpu_simd_backend: bool,
    parallel_expert_prefetch: bool,
    persistent_expert_heat: bool,
    cuda_backend: bool,
    cuda_runtime_compiler: bool,
    cuda_packed_dot_oracle: bool,
    cuda_reusable_packed_executor: bool,
    grouped_prefill: bool,
    speculative_ngram_t2: bool,
    persistent_kv: bool,
    mtp_acceleration: bool,
    experimental_igpu: bool,
    server: bool,
    selected_model_required_for_chat: bool,
}

/// Reports host hardware, runtime probes, and backend capability status in JSON or human-readable form.
///
/// # Examples
///
/// ```no_run
/// doctor(true)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
fn doctor(json: bool) -> Result<()> {
    let nvidia = probe_nvidia();
    let cuda_toolchain = probe_cuda_toolchain();
    let cuda_nvrtc = probe_cuda_nvrtc();
    let cuda_packed_oracle = probe_cuda_packed_oracle(cuda_nvrtc.available);
    let vulkan = probe_vulkan();
    let windows_ml_npu = probe_windows_ml_npu();
    let cpu = CpuTopology::detect();
    let cpu_capabilities = CpuCapabilities::detect();
    let cuda_build = bridge_kernels_cuda::build_capabilities();
    let npu_feasibility = NpuFeasibilityReport {
        device_detected: windows_ml_npu.available,
        bridge_compiled: false,
        authoritative_model_backend: false,
        advisory_router_candidate: windows_ml_npu.available,
        weight_conversion_permitted: false,
        reason: "the selected IQ2_S/IQ3_S/Q4_K/Q5_K GGUF arithmetic is not representable by \
                 the W4/BF16-oriented NPU path; only a separately validated auxiliary router \
                 predictor may use the NPU"
            .to_owned(),
    };
    let backend_status = vec![
        BackendCapabilityStatus {
            backend: BackendKind::CpuScalar,
            compiled: true,
            runtime_available: true,
            authoritative: true,
            automatic: true,
            reason: "authoritative fallback".to_owned(),
        },
        BackendCapabilityStatus {
            backend: BackendKind::CpuAvx2,
            compiled: true,
            runtime_available: cpu_capabilities.avx2,
            authoritative: true,
            automatic: cpu_capabilities.avx2,
            reason: if cpu_capabilities.avx2 {
                "packed-dot oracle is compiled and AVX2 is present".to_owned()
            } else {
                "CPU does not report AVX2".to_owned()
            },
        },
        BackendCapabilityStatus {
            backend: BackendKind::CpuAvxVnni,
            compiled: true,
            runtime_available: cpu_capabilities.avx_vnni_dot_kernel_available(),
            authoritative: false,
            automatic: false,
            reason: if cpu_capabilities.avx_vnni_dot_kernel_available() {
                "bit-exact AVX-VNNI packed-dot kernel is compiled; full-token 10% qualification is still required"
                    .to_owned()
            } else {
                "CPU does not report AVX2 plus AVX-VNNI".to_owned()
            },
        },
        BackendCapabilityStatus {
            backend: BackendKind::CpuAvx512Vnni,
            compiled: true,
            runtime_available: cpu_capabilities.avx512_dot_kernel_available(),
            authoritative: false,
            automatic: false,
            reason: if cpu_capabilities.avx512_dot_kernel_available() {
                "bit-exact packed-dot kernel is compiled; full-token 10% qualification is still required"
                    .to_owned()
            } else {
                "CPU does not report AVX-512F plus AVX-512 VNNI".to_owned()
            },
        },
        BackendCapabilityStatus {
            backend: BackendKind::Cuda,
            compiled: cuda_build.packed_kernels_compiled,
            runtime_available: nvidia.available && cuda_packed_oracle.available,
            authoritative: false,
            automatic: false,
            reason: format!(
                "{}; native toolchain: {}; runtime compiler: {}; packed oracle: {}",
                cuda_build
                    .rejection_reason
                    .clone()
                    .unwrap_or_else(|| "CUDA packed-kernel qualification is missing".to_owned()),
                cuda_toolchain.detail,
                cuda_nvrtc.detail,
                cuda_packed_oracle.detail,
            ),
        },
        BackendCapabilityStatus {
            backend: BackendKind::Vulkan,
            compiled: false,
            runtime_available: vulkan.available,
            authoritative: false,
            automatic: false,
            reason: "Vulkan device probing is available; packed compute kernels are research-only".to_owned(),
        },
        BackendCapabilityStatus {
            backend: BackendKind::WindowsMlNpu,
            compiled: false,
            runtime_available: windows_ml_npu.available,
            authoritative: false,
            automatic: false,
            reason: "NPU is restricted to an advisory next-router predictor; no IQ2/IQ3 model backend"
                .to_owned(),
        },
    ];
    let cuda_nvrtc_available = cuda_nvrtc.available;
    let cuda_packed_oracle_available = cuda_packed_oracle.available;
    let cuda_reusable_packed_executor_available = cuda_packed_oracle
        .reusable
        .as_ref()
        .is_some_and(|canary| canary.bit_exact && canary.deterministic);
    let report = DoctorReport {
        engine_version: env!("CARGO_PKG_VERSION"),
        operating_system: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        cpu,
        memory: memory_status(),
        nvidia,
        cuda_toolchain,
        cuda_nvrtc,
        cuda_packed_oracle,
        vulkan,
        windows_ml_npu,
        npu_feasibility,
        capabilities: CapabilityReport {
            scalar_reference: true,
            q8_k_activation_path: true,
            cpu_parallel_backend: true,
            cpu_parallel_default_threads: recommended_thread_count(),
            cpu_simd_backend: cpu_capabilities.avx2_dot_kernel_available(),
            parallel_expert_prefetch: true,
            persistent_expert_heat: true,
            cuda_backend: cuda_build.packed_kernels_compiled,
            cuda_runtime_compiler: cuda_nvrtc_available,
            cuda_packed_dot_oracle: cuda_packed_oracle_available,
            cuda_reusable_packed_executor: cuda_reusable_packed_executor_available,
            grouped_prefill: true,
            speculative_ngram_t2: true,
            persistent_kv: true,
            mtp_acceleration: false,
            experimental_igpu: false,
            server: true,
            selected_model_required_for_chat: true,
        },
        backend_status,
    };
    if json {
        write_json(&report)
    } else {
        let text = format!(
            "LightBridge {}\nOS: {} {}\nCPU: {}\nPhysical cores: {}\nLogical processors: {}\nISA: {}\nRAM: {} total, {} available\nNVIDIA: {}\nCUDA native toolchain: {}\nCUDA NVRTC/Driver: {}\nCUDA packed Q8_K oracle: {}\nVulkan: {}\nWindows ML/NPU: {}\nExecution: {} with {} bounded threads (AVX2 detected: {}, AVX-512 VNNI detected: {})\nExpert prefetch: parallel\nExpert heat persistence: available\nCUDA backend: {}\nGrouped prefill: opt-in chunks 2/4/8; token-serial default\nT=2 n-gram speculation: opt-in greedy verifier with lossless KV rewind\nPersistent KV: model-bound checksummed sessions available\nMTP: not applicable to selected model\nExperimental iGPU: research-only\nServer: available\n",
            report.engine_version,
            report.operating_system,
            report.architecture,
            report.cpu.brand,
            report.cpu.n_physical(),
            report.cpu.logical_processors,
            report.cpu.isa.tag(),
            human_bytes(report.memory.total_physical),
            human_bytes(report.memory.available_physical),
            report.nvidia.detail,
            report.cuda_toolchain.detail,
            report.cuda_nvrtc.detail,
            report.cuda_packed_oracle.detail,
            report.vulkan.detail,
            report.windows_ml_npu.detail,
            cpu_capabilities.backend_name(),
            report.capabilities.cpu_parallel_default_threads,
            cpu_capabilities.avx2,
            cpu_capabilities.avx512_vnni,
            cuda_build
                .rejection_reason
                .as_deref()
                .unwrap_or("awaiting correctness and performance qualification"),
        );
        write_stdout(text.as_bytes())
    }
}

/// Probes the host for NVIDIA GPU availability and device details.
///
/// Uses `nvidia-smi` when available and falls back to Windows Plug and Play
/// device discovery when the command fails or reports no devices.
///
/// # Examples
///
/// ```
/// let status = probe_nvidia();
/// println!("{}", status.detail);
/// assert!(status.available || !status.available);
/// ```
fn probe_nvidia() -> NvidiaStatus
fn probe_nvidia() -> NvidiaStatus {
    let output = ProcessCommand::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version,uuid,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let detail = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if detail.is_empty() {
                return probe_nvidia_pnp("nvidia-smi returned no device rows")
                    .unwrap_or_else(|| unavailable_nvidia("nvidia-smi returned no device rows"));
            }
            let fields = detail.split(',').map(str::trim).collect::<Vec<_>>();
            NvidiaStatus {
                available: true,
                detail: detail.clone(),
                name: fields.first().map(|value| (*value).to_owned()),
                memory_mib: fields.get(1).and_then(|value| value.parse().ok()),
                driver: fields.get(2).map(|value| (*value).to_owned()),
                uuid: fields.get(3).map(|value| (*value).to_owned()),
                pci_bus_id: fields.get(4).map(|value| (*value).to_owned()),
                pcie_link: fields
                    .get(5)
                    .zip(fields.get(6))
                    .map(|(generation, width)| format!("Gen{generation} x{width}")),
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let detail = if stderr.is_empty() {
                format!("nvidia-smi exited with {}", output.status)
            } else {
                stderr
            };
            probe_nvidia_pnp(&detail).unwrap_or_else(|| unavailable_nvidia(detail))
        }
        Err(error) => {
            let detail = error.to_string();
            probe_nvidia_pnp(&detail).unwrap_or_else(|| unavailable_nvidia(detail))
        }
    }
}

/// Creates an unavailable NVIDIA status with the provided diagnostic detail.
///
/// # Examples
///
/// ```
/// let status = unavailable_nvidia("NVIDIA tooling is unavailable");
/// assert!(!status.available);
/// assert_eq!(status.detail, "NVIDIA tooling is unavailable");
/// ```
///
/// # Arguments
///
/// * `detail` - Diagnostic information explaining why NVIDIA is unavailable.
fn unavailable_nvidia(detail: impl Into<String>) -> NvidiaStatus {
    NvidiaStatus {
        available: false,
        detail: detail.into(),
        name: None,
        memory_mib: None,
        driver: None,
        uuid: None,
        pci_bus_id: None,
        pcie_link: None,
    }
}

/// Probes Windows display devices for an NVIDIA adapter when `nvidia-smi` cannot provide complete details.
///
/// Returns an unavailable status containing the adapter name, PnP status, driver package, and
/// `nvidia-smi` detail when an NVIDIA display device is found. Returns `None` if the PnP query
/// fails or no NVIDIA display device is detected.
///
/// # Examples
///
/// ```
/// #[cfg(windows)]
/// {
///     let status = probe_nvidia_pnp("nvidia-smi was unavailable");
///     if let Some(status) = status {
///         assert!(!status.available);
///     }
/// }
/// ```
fn probe_nvidia_pnp(nvidia_smi_detail: &str) -> Option<NvidiaStatus> {
    let output = ProcessCommand::new("pnputil")
        .args(["/enum-devices", "/class", "Display"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let (name, status, driver) = parse_nvidia_pnp(&text)?;
    Some(NvidiaStatus {
        available: false,
        detail: format!(
            "{name}; Windows PnP status: {status}; driver package: {driver}; nvidia-smi: \
             {nvidia_smi_detail}. The RTX must be reconnected in the laptop GPU-mode control \
             before CUDA qualification"
        ),
        name: Some(name),
        memory_mib: None,
        driver: None,
        uuid: None,
        pci_bus_id: None,
        pcie_link: None,
    })
}

/// Provides the NVIDIA Plug and Play fallback probe on non-Windows platforms.
///
/// # Examples
///
/// ```
/// assert!(probe_nvidia_pnp("").is_none());
/// ```
///
/// # Returns
///
/// `None` because NVIDIA Plug and Play probing is unavailable on non-Windows platforms.
#[cfg(not(windows))]
fn probe_nvidia_pnp(_nvidia_smi_detail: &str) -> Option<NvidiaStatus> {
    None
}

/// Extracts device description, status, and driver name from an NVIDIA PnP device block.
///
/// # Examples
///
/// ```
/// let text = "Device Description: NVIDIA GPU\nStatus: Started\nDriver Name: display.inf\nVEN_10DE";
/// let device = parse_nvidia_pnp(text).unwrap();
///
/// assert_eq!(device.0, "NVIDIA GPU");
/// assert_eq!(device.1, "Started");
/// assert_eq!(device.2, "display.inf");
/// ```
///
/// Returns `None` when the input contains no block identifying an NVIDIA device.
/// Missing or empty fields are reported as `"unknown"`.
///
/// # Returns
///
/// The device description, status, and driver name when an NVIDIA device block is found.
fn parse_nvidia_pnp(text: &str) -> Option<(String, String, String)> {
    text.replace('\r', "")
        .split("\n\n")
        .find(|block| block.to_ascii_uppercase().contains("VEN_10DE"))
        .map(|block| {
            let value = |label: &str| {
                block
                    .lines()
                    .find_map(|line| line.trim().strip_prefix(label).map(str::trim))
                    .filter(|value| !value.is_empty())
                    .unwrap_or("unknown")
                    .to_owned()
            };
            (
                value("Device Description:"),
                value("Status:"),
                value("Driver Name:"),
            )
        })
}

/// Probes the CUDA toolchain and reports whether the available host compiler is compatible.
///
/// # Examples
///
/// ```
/// let status = probe_cuda_toolchain();
/// assert!(!status.detail.is_empty());
/// ```
///
/// # Returns
///
/// A status containing toolchain availability and diagnostic details.
fn probe_cuda_toolchain() -> ProbeStatus {
    let Some(nvcc) = probe_version("nvcc", &["--version"]) else {
        return ProbeStatus {
            available: false,
            detail: "NVCC is not available on PATH".to_owned(),
        };
    };
    #[cfg(windows)]
    if let Some(version) = probe_visual_studio_version() {
        let major = version
            .split('.')
            .next()
            .and_then(|value| value.parse::<u32>().ok());
        if major.is_some_and(|major| major >= 18) {
            return ProbeStatus {
                available: false,
                detail: format!(
                    "{nvcc}; Visual Studio {version} is outside CUDA 13.1's supported 2019-2022 \
                     host-compiler range"
                ),
            };
        }
        return ProbeStatus {
            available: true,
            detail: format!("{nvcc}; Visual Studio {version}"),
        };
    }
    #[cfg(not(windows))]
    return ProbeStatus {
        available: true,
        detail: nvcc,
    };
    ProbeStatus {
        available: false,
        detail: format!("{nvcc}; no compatible MSVC installation was discovered"),
    }
}

/// Probes CUDA NVRTC support and records the runtime canary result.
///
/// # Examples
///
/// ```no_run
/// let status = probe_cuda_nvrtc();
/// println!("{}", status.detail);
/// ```
///
/// The probe reports NVRTC and GPU execution details when successful; otherwise it
/// records the failure detail.
///
/// # Returns
///
/// A [`CudaNvrtcStatus`] describing NVRTC availability and, when available, the
/// successful canary result.
fn probe_cuda_nvrtc() -> CudaNvrtcStatus {
    match bridge_kernels_cuda::runtime_nvrtc_canary() {
        Ok(canary) => CudaNvrtcStatus {
            available: true,
            detail: format!(
                "NVRTC {}.{} compiled {} bytes of compute_89 PTX; GPU compute {}.{} completed \
                 pinned async H2D/kernel/D2H in {:.3} ms",
                canary.nvrtc_major,
                canary.nvrtc_minor,
                canary.ptx_bytes,
                canary.compute_major,
                canary.compute_minor,
                canary.elapsed_milliseconds,
            ),
            canary: Some(canary),
        },
        Err(error) => CudaNvrtcStatus {
            available: false,
            detail: error.to_string(),
            canary: None,
        },
    }
}

/// Qualifies the CUDA packed Q8K oracle and reusable executor when the runtime canary succeeds.
///
/// If the runtime canary fails, qualification is skipped and the returned status explains why.
/// Otherwise, the status includes oracle and reusable-executor results, timing details, and
/// staging-arena information.
///
/// # Examples
///
/// ```
/// let status = probe_cuda_packed_oracle(false);
/// assert!(!status.available);
/// assert!(status.oracle.is_none());
/// assert!(status.reusable.is_none());
/// ```
fn probe_cuda_packed_oracle(runtime_canary_available: bool) -> CudaPackedOracleStatus
fn probe_cuda_packed_oracle(runtime_canary_available: bool) -> CudaPackedOracleStatus {
    if !runtime_canary_available {
        return CudaPackedOracleStatus {
            available: false,
            detail: "not attempted because the NVRTC/Driver runtime canary failed".to_owned(),
            oracle: None,
            reusable: None,
        };
    }
    match bridge_kernels_cuda::runtime_packed_q8k_oracle() {
        Ok(oracle) => {
            let timings = oracle
                .formats
                .iter()
                .map(|format| format!("{}={:.3}ms", format.weight_type, format.elapsed_milliseconds))
                .collect::<Vec<_>>()
                .join(", ");
            match bridge_kernels_cuda::runtime_reusable_packed_q8k_canary() {
                Ok(reusable) => {
                    let arenas = reusable
                        .executions
                        .iter()
                        .map(|execution| execution.staging_arena)
                        .collect::<BTreeSet<_>>();
                    CudaPackedOracleStatus {
                        available: true,
                        detail: format!(
                            "all {} packed formats are bit-exact for {}x{} GEMV; NVRTC emitted {} \
                             bytes of PTX; reusable executor passed {} deterministic operations \
                             across {} staging arenas; {}",
                            oracle.formats.len(),
                            oracle.formats.first().map_or(0, |format| format.rows),
                            oracle.formats.first().map_or(0, |format| format.logical_elements),
                            oracle.ptx_bytes,
                            reusable.executions.len(),
                            arenas.len(),
                            timings,
                        ),
                        oracle: Some(oracle),
                        reusable: Some(reusable),
                    }
                }
                Err(error) => CudaPackedOracleStatus {
                    available: false,
                    detail: format!(
                        "one-shot packed arithmetic passed, but the reusable executor was rejected: \
                         {error}"
                    ),
                    oracle: Some(oracle),
                    reusable: None,
                },
            }
        }
        Err(error) => CudaPackedOracleStatus {
            available: false,
            detail: error.to_string(),
            oracle: None,
            reusable: None,
        },
    }
}

/// Finds the latest installed Visual Studio version with C++ build tools.
///
/// # Examples
///
/// ```
/// if let Some(version) = probe_visual_studio_version() {
///     println!("Visual Studio version: {version}");
/// }
/// ```
///
/// Returns the installation version, or `None` when the required tools or
/// version probe are unavailable.
#[cfg(windows)]
fn probe_visual_studio_version() -> Option<String> {
    let root = std::env::var_os("ProgramFiles(x86)")?;
    let vswhere = PathBuf::from(root)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let output = ProcessCommand::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationVersion",
        ])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Probes the Vulkan loader and reports the available physical devices.
///
/// # Examples
///
/// ```
/// let status = probe_vulkan();
/// assert!(!status.detail.is_empty());
/// ```
fn probe_vulkan() -> ProbeStatus {
fn probe_vulkan() -> ProbeStatus {
    let output = ProcessCommand::new("vulkaninfo").arg("--summary").output();
    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let devices = stdout
                .lines()
                .filter_map(|line| line.trim().strip_prefix("deviceName"))
                .filter_map(|line| line.split_once('=').map(|(_, value)| value.trim()))
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>();
            ProbeStatus {
                available: !devices.is_empty(),
                detail: if devices.is_empty() {
                    "Vulkan loader responded but reported no physical devices".to_owned()
                } else {
                    devices.join("; ")
                },
            }
        }
        Ok(output) => ProbeStatus {
            available: false,
            detail: probe_failure_detail(&output.stderr, "Vulkan loader/device unavailable"),
        },
        Err(error) => ProbeStatus {
            available: false,
            detail: format!("Vulkan loader/device unavailable: {error}"),
        },
    }
}

/// Probes connected Windows ComputeAccelerator devices for a started NPU.
///
/// # Examples
///
/// ```no_run
/// let status = probe_windows_ml_npu();
/// println!("{}", status.detail);
/// ```
#[cfg(windows)]
fn probe_windows_ml_npu() -> ProbeStatus {
    let output = ProcessCommand::new("pnputil")
        .args(["/enum-devices", "/connected", "/class", "ComputeAccelerator"])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let field = |name: &str| {
                stdout.lines().find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.trim().eq_ignore_ascii_case(name).then(|| value.trim())
                })
            };
            let description = field("Device Description");
            let manufacturer = field("Manufacturer Name");
            let status = field("Status");
            let driver = field("Driver Name");
            let detected = description.is_some_and(|value| {
                value.contains("NPU") || value.contains("Neural") || value.contains("Compute Accelerator")
            });
            let started = status.is_some_and(|value| value.eq_ignore_ascii_case("Started"));
            let mut details = [description, manufacturer, status, driver]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            details.dedup();
            ProbeStatus {
                available: detected && started,
                detail: if details.is_empty() {
                    "no connected ComputeAccelerator device was reported".to_owned()
                } else if detected && !started {
                    format!("{} (device is not started)", details.join(", "))
                } else {
                    details.join(", ")
                },
            }
        }
        Ok(output) => ProbeStatus {
            available: false,
            detail: probe_failure_detail(
                &output.stderr,
                "no connected ComputeAccelerator device was reported",
            ),
        },
        Err(error) => ProbeStatus {
            available: false,
            detail: format!("NPU device probe failed: {error}"),
        },
    }
}

/// Reports Windows ML/NPU availability on non-Windows platforms.

///

/// # Examples

///

/// ```

/// let status = probe_windows_ml_npu();

/// assert!(!status.available);

/// ```

///

/// #[cfg(not(windows))]
fn probe_windows_ml_npu() -> ProbeStatus {
    ProbeStatus {
        available: false,
        detail: "Windows ML is only available on Windows".to_owned(),
    }
}

/// Selects diagnostic text from command output, falling back to a default message when the output is empty.
///
/// # Examples
///
/// ```
/// let detail = probe_failure_detail(b"driver unavailable\n", "probe failed");
/// assert_eq!(detail, "driver unavailable");
///
/// let fallback = probe_failure_detail(b"", "probe failed");
/// assert_eq!(fallback, "probe failed");
/// ```
///
/// # Arguments
///
/// * `stderr` - Command output containing diagnostic text.
/// * `unavailable` - Message used when `stderr` is empty after trimming.
///
/// # Returns
///
/// The trimmed diagnostic text, or `unavailable` when no diagnostic text is present.
fn probe_failure_detail(stderr: &[u8], unavailable: &str) -> String {
    let detail = String::from_utf8_lossy(stderr).trim().to_owned();
    if detail.is_empty() {
        unavailable.to_owned()
    } else {
        detail
    }
}

#[derive(Debug, Serialize)]
struct PlanReport {
    model: String,
    files: Vec<StorageReport>,
    context_capacity: usize,
    kv_page_tokens: usize,
    kv_bytes_per_token: u64,
    first_kv_page_bytes: u64,
    maximum_logical_kv_bytes: u64,
    resident_weight_bytes: u64,
    routed_expert_payload_bytes: u64,
    maximum_expert_record_bytes: u64,
    expert_cache_ceiling_bytes: u64,
    available_physical_memory: u64,
    memory_headroom_bytes: u64,
    minimum_startup_memory_bytes: u64,
    memory_preflight_passes: bool,
    execution_backend: &'static str,
    execution_threads: usize,
    cold_expert_path: &'static str,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct StorageReport {
    path: String,
    logical_bytes: u64,
    allocated_bytes: u64,
    sparse: bool,
    compressed: bool,
}

fn plan(
    model: &Path,
    context: usize,
    cache_mib: usize,
    kv_page_tokens: usize,
    memory_headroom_mib: usize,
    json: bool,
) -> Result<()> {
    let set = bridge_gguf_split::open_set(model)?;
    let validated = validate_selected_model(&set)?;
    if context == 0 || context as u64 > validated.config().context_length {
        bail!("context must be within 1..={}", validated.config().context_length);
    }
    if kv_page_tokens == 0 {
        bail!("kv-page-tokens must be greater than zero");
    }
    let cache_bytes = mib(cache_mib)?;
    let memory_headroom_bytes = mib(memory_headroom_mib)?;
    let resident_weight_bytes = validated
        .tensors()
        .iter()
        .filter(|tensor| !tensor.role().is_routed_expert())
        .try_fold(0_u64, |total, tensor| {
            total
                .checked_add(
                    tensor.location().absolute_range().end - tensor.location().absolute_range().start,
                )
                .context("resident weight byte count overflow")
        })?;
    let routed_expert_payload_bytes = validated
        .tensors()
        .iter()
        .filter(|tensor| tensor.role().is_routed_expert())
        .try_fold(0_u64, |total, tensor| {
            total
                .checked_add(
                    tensor.location().absolute_range().end - tensor.location().absolute_range().start,
                )
                .context("routed weight byte count overflow")
        })?;
    let index = DirectExpertIndex::build(&validated)?;
    let maximum_expert_record_bytes = index
        .records()
        .iter()
        .map(|record| record.gate.length() + record.up.length() + record.down.length())
        .max()
        .unwrap_or(0);
    if cache_bytes < maximum_expert_record_bytes {
        bail!(
            "cache is {}, smaller than one complete expert record {}",
            human_bytes(cache_bytes),
            human_bytes(maximum_expert_record_bytes)
        );
    }
    let config = validated.config();
    let kv_bytes_per_token = u64::from(config.block_count)
        .checked_mul(u64::from(config.attention_kv_head_count))
        .and_then(|value| value.checked_mul(u64::from(config.key_length) + u64::from(config.value_length)))
        .and_then(|value| value.checked_mul(4))
        .context("KV byte count overflow")?;
    let first_kv_page_bytes = kv_bytes_per_token
        .checked_mul(kv_page_tokens as u64)
        .context("KV page byte count overflow")?;
    let maximum_logical_kv_bytes = kv_bytes_per_token
        .checked_mul(context as u64)
        .context("logical KV byte count overflow")?;
    let minimum_startup_memory_bytes = resident_weight_bytes
        .checked_add(cache_bytes)
        .and_then(|value| value.checked_add(first_kv_page_bytes))
        .and_then(|value| value.checked_add(memory_headroom_bytes))
        .context("startup memory byte count overflow")?;
    let files = set
        .files()
        .iter()
        .map(|shard| storage_report(shard.path()))
        .collect::<Result<Vec<_>>>()?;
    let mut warnings = Vec::new();
    if files
        .iter()
        .any(|file| file.sparse && file.allocated_bytes < file.logical_bytes)
    {
        warnings.push(
            "the model is a sparse header mirror and cannot pass payload authentication or inference".into(),
        );
    }
    let cpu_capabilities = CpuCapabilities::detect();
    if !cpu_capabilities.avx2_dot_kernel_available() {
        warnings.push("AVX2 is unavailable; the CPU-parallel backend is using exact scalar dots".into());
    }
    warnings.push("CUDA backend is unavailable; CPU execution remains fully functional".into());
    let available_physical_memory = memory_status().available_physical;
    let memory_preflight_passes =
        available_physical_memory == 0 || available_physical_memory >= minimum_startup_memory_bytes;
    if !memory_preflight_passes {
        warnings.push(format!(
            "startup memory preflight fails: {} required, {} currently available",
            human_bytes(minimum_startup_memory_bytes),
            human_bytes(available_physical_memory)
        ));
    }
    let report = PlanReport {
        model: model.display().to_string(),
        files,
        context_capacity: context,
        kv_page_tokens,
        kv_bytes_per_token,
        first_kv_page_bytes,
        maximum_logical_kv_bytes,
        resident_weight_bytes,
        routed_expert_payload_bytes,
        maximum_expert_record_bytes,
        expert_cache_ceiling_bytes: cache_bytes,
        available_physical_memory,
        memory_headroom_bytes,
        minimum_startup_memory_bytes,
        memory_preflight_passes,
        execution_backend: cpu_capabilities.backend_name(),
        execution_threads: recommended_thread_count(),
        cold_expert_path: "positioned_disk_read_then_cpu",
        warnings,
    };
    if json {
        write_json(&report)
    } else {
        let mut text = format!(
            "Model: {}\nContext capacity: {}\nBackend: {} ({} threads)\nResident weights: {}\nExpert cache ceiling: {}\nMaximum expert record: {}\nKV per token: {}\nFirst lazy KV page: {}\nMaximum logical KV: {}\nMemory headroom: {}\nMinimum startup RAM: {}\nAvailable RAM: {}\nMemory preflight: {}\n",
            report.model,
            report.context_capacity,
            report.execution_backend,
            report.execution_threads,
            human_bytes(report.resident_weight_bytes),
            human_bytes(report.expert_cache_ceiling_bytes),
            human_bytes(report.maximum_expert_record_bytes),
            human_bytes(report.kv_bytes_per_token),
            human_bytes(report.first_kv_page_bytes),
            human_bytes(report.maximum_logical_kv_bytes),
            human_bytes(report.memory_headroom_bytes),
            human_bytes(report.minimum_startup_memory_bytes),
            human_bytes(report.available_physical_memory),
            if report.memory_preflight_passes {
                "pass"
            } else {
                "fail"
            },
        );
        for warning in &report.warnings {
            text.push_str("Warning: ");
            text.push_str(warning);
            text.push('\n');
        }
        write_stdout(text.as_bytes())
    }
}

#[derive(Debug, Serialize)]
struct HeaderValidationReport {
    schema_valid: bool,
    payload_authenticated: bool,
    files: Vec<StorageReport>,
}

fn validate(model: &Path, payload: bool, json: bool) -> Result<()> {
    if payload {
        let report = validate_selected_payload(model)?;
        if json {
            return write_json(&report);
        }
        let file = &report.files[0];
        return write_stdout(
            format!(
                "Schema: valid\nPayload: authenticated\nSHA-256: {}\nLogical bytes: {}\nAllocated bytes: {}\n",
                file.sha256, file.logical_bytes, file.allocated_bytes
            )
            .as_bytes(),
        );
    }
    let set = bridge_gguf_split::open_set(model)?;
    validate_selected_model(&set)?;
    let files = set
        .files()
        .iter()
        .map(|shard| storage_report(shard.path()))
        .collect::<Result<Vec<_>>>()?;
    let report = HeaderValidationReport {
        schema_valid: true,
        payload_authenticated: false,
        files,
    };
    if json {
        write_json(&report)
    } else {
        let sparse = report
            .files
            .iter()
            .any(|file| file.sparse && file.allocated_bytes < file.logical_bytes);
        write_stdout(
            format!(
                "Schema: valid\nPayload: not authenticated\nPhysical completeness: {}\n",
                if sparse {
                    "sparse/incomplete"
                } else {
                    "not checked"
                }
            )
            .as_bytes(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    model: &Path,
    output: &Path,
    manifest: &Path,
    layout: LayoutArg,
    alignment: u64,
    overwrite: bool,
    no_verify: bool,
    json: bool,
) -> Result<()> {
    let set = bridge_gguf_split::open_set(model)?;
    let validated = validate_selected_model(&set)?;
    reject_sparse_sources(&set)?;
    let report = prepare_sidecar(
        &set,
        &validated,
        output,
        manifest,
        PrepareOptions {
            layout: layout.into(),
            alignment,
            overwrite,
            verify_after_write: !no_verify,
            ..PrepareOptions::default()
        },
        &ReadCancellation::new(),
    )?;
    let value = serde_json::json!({
        "data_path": report.data_path,
        "manifest_path": report.manifest_path,
        "source_bytes_hashed": report.source_bytes_hashed,
        "expert_payload_bytes": report.expert_payload_bytes,
        "sidecar_bytes": report.sidecar_bytes,
        "record_count": report.record_count,
        "sidecar_sha256": report.sidecar_sha256,
        "tensor_directory_sha256": report.tensor_directory_sha256,
    });
    if json {
        write_json(&value)
    } else {
        write_stdout(
            format!(
                "Prepared: {}\nManifest: {}\nRecords: {}\nSidecar bytes: {}\nSHA-256: {}\n",
                report.data_path.display(),
                report.manifest_path.display(),
                report.record_count,
                report.sidecar_bytes,
                report.sidecar_sha256
            )
            .as_bytes(),
        )
    }
}

fn tokenize(
    model: &Path,
    text: Option<String>,
    chat_json: Option<&Path>,
    effort: ReasoningArg,
    json: bool,
) -> Result<()> {
    let tokenizer = load_tokenizer(model)?;
    let token_ids = match (text, chat_json) {
        (Some(text), None) => tokenizer.encode(&text)?,
        (None, Some(path)) => {
            let messages: Vec<ChatMessage> = read_json_file(path)?;
            tokenizer.format_and_encode(
                &messages,
                &ChatTemplateOptions {
                    reasoning_effort: effort.into(),
                    ..ChatTemplateOptions::default()
                },
            )?
        }
        _ => bail!("provide exactly one of --text or --chat-json"),
    };
    if json {
        write_json(&serde_json::json!({
            "count": token_ids.len(),
            "token_ids": token_ids,
        }))
    } else {
        let text = token_ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
        write_stdout(format!("{text}\n").as_bytes())
    }
}

fn detokenize(model: &Path, tokens: &[u32], include_special_tokens: bool, json: bool) -> Result<()> {
    let tokenizer = load_tokenizer(model)?;
    let text = tokenizer.decode(tokens, !include_special_tokens)?;
    if json {
        write_json(&serde_json::json!({"text": text}))
    } else {
        write_stdout(format!("{text}\n").as_bytes())
    }
}

/// Runs a single chat completion from a prompt or a JSON chat transcript.
///
/// The completion is streamed as text or emitted as JSON. An optional session can
/// be restored before completion and saved afterward.
///
/// # Examples
///
/// ```text
/// bridge chat --model model.gguf --prompt "Explain ownership in Rust."
/// ```
///
/// # Errors
///
/// Returns an error if the input mode is invalid, the model or session cannot be
/// opened, completion fails, or output cannot be written.
fn chat(
    runtime: RuntimeArgs,
    sampling: SamplingArgs,
    prompt: Option<String>,
    chat_json: Option<&Path>,
    system: Option<String>,
    effort: ReasoningArg,
    session_in: Option<&Path>,
    session_out: Option<&Path>,
    session_max_mib: usize,
    json: bool,
) -> Result<()> {
    let engine = open_engine(&runtime)?;
    let messages = match (prompt, chat_json) {
        (Some(prompt), None) => {
            let mut messages = Vec::new();
            if let Some(system) = system {
                messages.push(ChatMessage::system(system));
            }
            messages.push(ChatMessage::user(prompt));
            messages
        }
        (None, Some(path)) => read_json_file(path)?,
        _ => bail!("provide exactly one of --prompt or --chat-json"),
    };
    let session_max_bytes = usize::try_from(mib(session_max_mib)?)
        .context("session limit is not representable on this platform")?;
    let mut session = if let Some(path) = session_in {
        let bytes = read_bounded_with_limit(path, session_max_bytes as u64)?;
        engine.restore_session(&bytes, session_max_bytes)?
    } else {
        engine.new_session()?
    };
    let template = ChatTemplateOptions {
        reasoning_effort: effort.into(),
        ..ChatTemplateOptions::default()
    };
    let sampling = sampling_config(&sampling);
    if json {
        let completion = engine.complete_in_session(
            &mut session,
            &messages,
            &template,
            sampling,
            &CancellationToken::new(),
            |_| ControlFlow::Continue(()),
        )?;
        let response = serde_json::json!({
            "text": completion.text,
            "assistant": completion.assistant,
            "raw_text": completion.raw_text,
            "structured_output_error": completion.structured_output_error,
            "prompt_token_ids": completion.prompt_token_ids,
            "token_ids": completion.generation.token_ids,
            "stop_reason": completion.generation.stop_reason,
            "prompt_tokens": completion.generation.stats.prompt_tokens,
            "cached_prompt_tokens": completion.cached_prompt_tokens,
            "generated_tokens": completion.generation.stats.generated_tokens,
            "prefill_milliseconds": completion.generation.stats.prefill_duration.as_millis(),
            "decode_milliseconds": completion.generation.stats.decode_duration.as_millis(),
            "total_milliseconds": completion.generation.stats.total_duration.as_millis(),
        });
        persist_chat_session(&engine, &session, session_out, session_max_bytes)?;
        persist_cache_heat(&runtime, &engine)?;
        return write_json(&response);
    }

    let mut stdout = io::stdout().lock();
    let mut write_error = None;
    let buffer_structured_output = template.reasoning_effort != ReasoningEffort::NoThink;
    let completion = engine.complete_in_session(
        &mut session,
        &messages,
        &template,
        sampling,
        &CancellationToken::new(),
        |chunk| {
            if buffer_structured_output {
                return ControlFlow::Continue(());
            }
            if let Err(error) = stdout.write_all(chunk.as_bytes()).and_then(|()| stdout.flush()) {
                write_error = Some(error);
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    )?;
    if let Some(error) = write_error {
        return Err(error).context("failed to stream completion to stdout");
    }
    if buffer_structured_output {
        if let Some(reasoning) = &completion.assistant.reasoning {
            eprintln!("[reasoning]\n{reasoning}\n[/reasoning]");
        }
        stdout.write_all(completion.text.as_bytes())?;
    }
    stdout.write_all(b"\n")?;
    eprintln!(
        "{} prompt tokens ({} cached), {} generated tokens, {:.3} tokens/s",
        completion.generation.stats.prompt_tokens,
        completion.cached_prompt_tokens,
        completion.generation.stats.generated_tokens,
        tokens_per_second(
            completion.generation.stats.generated_tokens,
            completion.generation.stats.decode_duration
        )
    );
    persist_chat_session(&engine, &session, session_out, session_max_bytes)?;
    persist_cache_heat(&runtime, &engine)?;
    Ok(())
}

struct BenchRunArgs {
    runtime: RuntimeArgs,
    sampling: SamplingArgs,
    prompt: String,
    prompt_corpus: Option<PathBuf>,
    corpus_repeats: usize,
    cold_warm: bool,
    hardware_profile: Option<PathBuf>,
    trace_path: Option<PathBuf>,
    json: bool,
}

/// Benchmarks model completion performance using a single prompt, a validated prompt corpus, or cold/admission/warm phases.
///
/// The benchmark records completion timing, throughput, backend information, cache statistics, and optionally
/// Chrome trace output. Cold/admission/warm runs and corpus repetitions verify deterministic generated token IDs.
/// Cache heat is persisted after a successful run.
///
/// # Examples
///
/// ```no_run
/// # // Construct `BenchRunArgs` from the CLI configuration.
/// # let args = todo!();
/// bench(args).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the engine cannot be opened, the hardware profile or prompt corpus is invalid, a
/// completion fails, generated tokens are nondeterministic across repeated runs, or benchmark output cannot
/// be persisted.
fn bench(args: BenchRunArgs) -> Result<()> {
    let BenchRunArgs {
        runtime,
        sampling,
        prompt,
        prompt_corpus,
        corpus_repeats,
        cold_warm,
        hardware_profile,
        trace_path,
        json,
    } = args;
    let benchmark_started = Instant::now();
    let engine = open_engine(&runtime)?;
    let engine_open_duration = benchmark_started.elapsed();
    let initial_backend = engine.model().backend_name();
    if let Some(path) = hardware_profile.as_deref() {
        validate_hardware_profile(path, &runtime, &engine)?;
    }
    let mut trace = ChromeTrace::default();
    trace.push_complete(
        "engine_open",
        "allocation,model",
        Duration::ZERO,
        engine_open_duration,
        BTreeMap::new(),
    );
    if let Some(path) = prompt_corpus.as_deref() {
        let corpus = read_benchmark_corpus(path)?;
        return bench_corpus(
            &runtime,
            &sampling,
            &engine,
            initial_backend,
            corpus,
            corpus_repeats,
            benchmark_started,
            trace,
            trace_path.as_deref(),
            json,
        );
    }
    if cold_warm {
        let mut runs = Vec::new();
        let mut expected_tokens = None;
        for phase in ["cold", "admission", "warm"] {
            let cache_before = engine.model().cache_stats()?;
            let run_started_at = benchmark_started.elapsed();
            let completion = engine.complete(
                &[ChatMessage::user(prompt.clone())],
                &ChatTemplateOptions::default(),
                sampling_config(&sampling),
                &CancellationToken::new(),
                |_| ControlFlow::Continue(()),
            )?;
            append_completion_trace(&mut trace, phase, run_started_at, completion.generation.stats);
            let cache_after = engine.model().cache_stats()?;
            let token_ids = &completion.generation.token_ids;
            if let Some(expected) = &expected_tokens {
                if token_ids != expected {
                    bail!(
                        "benchmark output changed between identical deterministic runs: \
                         expected {expected:?}, got {token_ids:?}"
                    );
                }
            } else {
                expected_tokens = Some(token_ids.clone());
            }
            let stats = completion.generation.stats;
            runs.push(serde_json::json!({
                "phase": phase,
                "prompt_tokens": stats.prompt_tokens,
                "generated_tokens": stats.generated_tokens,
                "token_ids": token_ids,
                "text": &completion.text,
                "raw_text": &completion.raw_text,
                "prefill_milliseconds": stats.prefill_duration.as_millis(),
                "decode_milliseconds": stats.decode_duration.as_millis(),
                "total_milliseconds": stats.total_duration.as_millis(),
                "decode_tokens_per_second": tokens_per_second(
                    stats.generated_tokens,
                    stats.decode_duration,
                ),
                "cache_delta": {
                    "hits": cache_after.hits.saturating_sub(cache_before.hits),
                    "misses": cache_after.misses.saturating_sub(cache_before.misses),
                    "loads": cache_after.loads.saturating_sub(cache_before.loads),
                    "waits": cache_after.waits.saturating_sub(cache_before.waits),
                    "evictions": cache_after.evictions.saturating_sub(cache_before.evictions),
                },
                "cache_before": cache_before,
                "cache_after": cache_after,
                "backend_after": engine.model().backend_name(),
                "cuda_fallback_active": engine.model().cuda_fallback_active(),
                "stop_reason": completion.generation.stop_reason,
            }));
        }
        let backend = engine.model().backend_name();
        let cpu_threads = engine.model().cpu_threads();
        let cpu_set_ids = engine.model().cpu_set_ids();
        persist_cache_heat(&runtime, &engine)?;
        persist_trace(trace_path.as_deref(), &trace)?;
        let report = serde_json::json!({
            "model": runtime.model,
            "backend": backend,
            "backend_initial": initial_backend,
            "backend_final": backend,
            "cuda_fallback_active": engine.model().cuda_fallback_active(),
            "cpu_threads": cpu_threads,
            "cpu_set_ids": cpu_set_ids,
            "mode": "cold_admission_warm",
            "cold": runs[0],
            "admission": runs[1],
            "warm": runs[2],
            "cache": engine.model().cache_stats()?,
        });
        if json {
            return write_json(&report);
        }
        return write_stdout(
            format!(
                "Backend: {}{}\nCold decode: {:.3} tokens/s\nAdmission decode: {:.3} tokens/s\nWarm decode: {:.3} tokens/s\n",
                backend,
                cpu_threads.map(|threads| format!(" ({threads} threads)")).unwrap_or_default(),
                report["cold"]["decode_tokens_per_second"].as_f64().unwrap_or(0.0),
                report["admission"]["decode_tokens_per_second"]
                    .as_f64()
                    .unwrap_or(0.0),
                report["warm"]["decode_tokens_per_second"].as_f64().unwrap_or(0.0),
            )
            .as_bytes(),
        );
    }

    let run_started_at = benchmark_started.elapsed();
    let completion = engine.complete(
        &[ChatMessage::user(prompt)],
        &ChatTemplateOptions::default(),
        sampling_config(&sampling),
        &CancellationToken::new(),
        |_| ControlFlow::Continue(()),
    )?;
    append_completion_trace(&mut trace, "single", run_started_at, completion.generation.stats);
    let stats = completion.generation.stats;
    let backend = engine.model().backend_name();
    let cpu_threads = engine.model().cpu_threads();
    let cpu_set_ids = engine.model().cpu_set_ids();
    persist_cache_heat(&runtime, &engine)?;
    persist_trace(trace_path.as_deref(), &trace)?;
    let report = serde_json::json!({
        "model": runtime.model,
        "backend": backend,
        "backend_initial": initial_backend,
        "backend_final": backend,
        "cuda_fallback_active": engine.model().cuda_fallback_active(),
        "cpu_threads": cpu_threads,
        "cpu_set_ids": cpu_set_ids,
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": stats.generated_tokens,
        "token_ids": &completion.generation.token_ids,
        "text": &completion.text,
        "raw_text": &completion.raw_text,
        "prefill_milliseconds": stats.prefill_duration.as_millis(),
        "decode_milliseconds": stats.decode_duration.as_millis(),
        "total_milliseconds": stats.total_duration.as_millis(),
        "decode_tokens_per_second": tokens_per_second(stats.generated_tokens, stats.decode_duration),
        "cache": engine.model().cache_stats()?,
        "stop_reason": completion.generation.stop_reason,
    });
    if json {
        write_json(&report)
    } else {
        write_stdout(
            format!(
                "Backend: {}{}\nPrompt tokens: {}\nGenerated tokens: {}\nToken IDs: {:?}\nPrefill: {} ms\nDecode: {} ms\nDecode throughput: {:.3} tokens/s\n",
                backend,
                cpu_threads.map(|threads| format!(" ({threads} threads)")).unwrap_or_default(),
                stats.prompt_tokens,
                stats.generated_tokens,
                completion.generation.token_ids,
                stats.prefill_duration.as_millis(),
                stats.decode_duration.as_millis(),
                tokens_per_second(stats.generated_tokens, stats.decode_duration),
            )
            .as_bytes(),
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkCorpusV1 {
    format: String,
    version: u32,
    prompts: Vec<String>,
}

/// Reads and validates a versioned benchmark corpus from a JSON file.
///
/// # Examples
///
/// ```
/// let path = std::path::Path::new("benchmark-corpus.json");
/// let corpus = read_benchmark_corpus(path)?;
/// assert!(!corpus.prompts.is_empty());
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Errors
///
/// Returns an error if the file cannot be read, exceeds the maximum JSON input
/// size, or contains an invalid benchmark corpus.
///
/// # Arguments
///
/// * `path` - Path to the benchmark corpus JSON file.
///
/// # Returns
///
/// The validated benchmark corpus.
fn read_benchmark_corpus(path: &Path) -> Result<BenchmarkCorpusV1> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read benchmark corpus {}", path.display()))?;
    if bytes.len() as u64 > MAX_JSON_INPUT_BYTES {
        bail!(
            "benchmark corpus is {} bytes, maximum {}",
            bytes.len(),
            MAX_JSON_INPUT_BYTES
        );
    }
    parse_benchmark_corpus(&bytes)
}

/// Parses and validates a version 1 benchmark corpus from JSON bytes.
///
/// # Errors
///
/// Returns an error when the JSON is invalid, the corpus format or version is unsupported,
/// or the corpus contains invalid, oversized, or duplicate prompts.
///
/// # Examples
///
/// ```
/// let corpus = parse_benchmark_corpus(
///     br#"{"format":"lightbridge-benchmark-corpus","version":1,"prompts":["Hello"]}"#,
/// )
/// .unwrap();
///
/// assert_eq!(corpus.prompts, vec!["Hello"]);
/// ```
///
/// `bytes` must contain a JSON-encoded [`BenchmarkCorpusV1`].
fn parse_benchmark_corpus(bytes: &[u8]) -> Result<BenchmarkCorpusV1> {
fn parse_benchmark_corpus(bytes: &[u8]) -> Result<BenchmarkCorpusV1> {
    let corpus: BenchmarkCorpusV1 =
        serde_json::from_slice(bytes).context("benchmark corpus is not valid bounded JSON")?;
    if corpus.format != "lightbridge-benchmark-corpus" || corpus.version != 1 {
        bail!("benchmark corpus must use format lightbridge-benchmark-corpus version 1");
    }
    if corpus.prompts.is_empty() || corpus.prompts.len() > MAX_BENCHMARK_CORPUS_PROMPTS {
        bail!(
            "benchmark corpus has {} prompts, expected 1..={}",
            corpus.prompts.len(),
            MAX_BENCHMARK_CORPUS_PROMPTS
        );
    }
    let mut unique = BTreeSet::new();
    for (index, prompt) in corpus.prompts.iter().enumerate() {
        if prompt.trim().is_empty() {
            bail!("benchmark corpus prompt {index} is empty");
        }
        if prompt.len() > MAX_BENCHMARK_PROMPT_BYTES {
            bail!(
                "benchmark corpus prompt {index} is {} bytes, maximum {}",
                prompt.len(),
                MAX_BENCHMARK_PROMPT_BYTES
            );
        }
        if !unique.insert(prompt) {
            bail!("benchmark corpus prompt {index} duplicates an earlier prompt");
        }
    }
    Ok(corpus)
}

/// Benchmarks each prompt in a corpus for the requested number of repeats and reports median performance metrics.
///
/// Each prompt must produce identical token IDs across repeats. The benchmark also records per-run
/// timing, cache deltas, backend state, completion output, and optional Chrome trace data.
///
/// # Errors
///
/// Returns an error if `repeats` is outside the supported range, a repeated prompt produces
/// different token IDs, or benchmarking or report persistence fails.
///
/// # Examples
///
/// ```ignore
/// // Run a validated benchmark corpus and emit its report.
/// bench_corpus(
///     &runtime,
///     &sampling,
///     &engine,
///     initial_backend,
///     corpus,
///     3,
///     benchmark_started,
///     trace,
///     None,
///     true,
/// )?;
/// # Ok::<(), anyhow::Error>(())
/// ```
fn bench_corpus(
    runtime: &RuntimeArgs,
    sampling: &SamplingArgs,
    engine: &Hy3ChatEngine,
    initial_backend: &'static str,
    corpus: BenchmarkCorpusV1,
    repeats: usize,
    benchmark_started: Instant,
    mut trace: ChromeTrace,
    trace_path: Option<&Path>,
    json: bool,
) -> Result<()> {
    if repeats == 0 || repeats > MAX_BENCHMARK_CORPUS_REPEATS {
        bail!("corpus repeats is {repeats}, expected 1..={MAX_BENCHMARK_CORPUS_REPEATS}");
    }
    let mut expected_tokens = vec![None::<Vec<u32>>; corpus.prompts.len()];
    let mut total_milliseconds = Vec::new();
    let mut prefill_milliseconds = Vec::new();
    let mut decode_rates = Vec::new();
    let mut runs = Vec::new();
    for repeat in 0..repeats {
        for (prompt_index, prompt) in corpus.prompts.iter().enumerate() {
            let cache_before = engine.model().cache_stats()?;
            let run_started_at = benchmark_started.elapsed();
            let completion = engine.complete(
                &[ChatMessage::user(prompt.clone())],
                &ChatTemplateOptions::default(),
                sampling_config(sampling),
                &CancellationToken::new(),
                |_| ControlFlow::Continue(()),
            )?;
            append_completion_trace(
                &mut trace,
                &format!("corpus_{repeat}_{prompt_index}"),
                run_started_at,
                completion.generation.stats,
            );
            let cache_after = engine.model().cache_stats()?;
            let token_ids = &completion.generation.token_ids;
            if let Some(expected) = &expected_tokens[prompt_index] {
                if token_ids != expected {
                    bail!(
                        "corpus prompt {prompt_index} output changed on repeat {repeat}: \
                         expected {expected:?}, got {token_ids:?}"
                    );
                }
            } else {
                expected_tokens[prompt_index] = Some(token_ids.clone());
            }
            let stats = completion.generation.stats;
            let total_ms = u64::try_from(stats.total_duration.as_millis()).unwrap_or(u64::MAX);
            let prefill_ms = u64::try_from(stats.prefill_duration.as_millis()).unwrap_or(u64::MAX);
            let decode_rate = tokens_per_second(stats.generated_tokens, stats.decode_duration);
            total_milliseconds.push(total_ms);
            prefill_milliseconds.push(prefill_ms);
            decode_rates.push(decode_rate);
            runs.push(serde_json::json!({
                "repeat": repeat,
                "prompt_index": prompt_index,
                "prompt_sha256": format!("{:x}", Sha256::digest(prompt.as_bytes())),
                "prompt_bytes": prompt.len(),
                "prompt_tokens": stats.prompt_tokens,
                "generated_tokens": stats.generated_tokens,
                "token_ids": token_ids,
                "text": &completion.text,
                "raw_text": &completion.raw_text,
                "prefill_milliseconds": prefill_ms,
                "decode_milliseconds": stats.decode_duration.as_millis(),
                "total_milliseconds": total_ms,
                "decode_tokens_per_second": decode_rate,
                "backend_after": engine.model().backend_name(),
                "cuda_fallback_active": engine.model().cuda_fallback_active(),
                "cache_delta": {
                    "hits": cache_after.hits.saturating_sub(cache_before.hits),
                    "misses": cache_after.misses.saturating_sub(cache_before.misses),
                    "loads": cache_after.loads.saturating_sub(cache_before.loads),
                    "waits": cache_after.waits.saturating_sub(cache_before.waits),
                    "evictions": cache_after.evictions.saturating_sub(cache_before.evictions),
                },
                "stop_reason": completion.generation.stop_reason,
            }));
        }
    }
    total_milliseconds.sort_unstable();
    prefill_milliseconds.sort_unstable();
    decode_rates.sort_by(f64::total_cmp);
    let backend_final = engine.model().backend_name();
    persist_cache_heat(runtime, engine)?;
    persist_trace(trace_path, &trace)?;
    let report = serde_json::json!({
        "format": "lightbridge-benchmark-corpus-report",
        "version": 1,
        "model": runtime.model,
        "backend": backend_final,
        "backend_initial": initial_backend,
        "backend_final": backend_final,
        "cuda_fallback_active": engine.model().cuda_fallback_active(),
        "prompt_count": corpus.prompts.len(),
        "repeats": repeats,
        "deterministic": true,
        "median_total_milliseconds": total_milliseconds[total_milliseconds.len() / 2],
        "median_prefill_milliseconds": prefill_milliseconds[prefill_milliseconds.len() / 2],
        "median_decode_tokens_per_second": decode_rates[decode_rates.len() / 2],
        "runs": runs,
        "cache": engine.model().cache_stats()?,
    });
    if json {
        write_json(&report)
    } else {
        write_stdout(
            format!(
                "Backend: {backend_final}\nCorpus prompts: {}\nRepeats: {repeats}\n\
                 Deterministic: yes\nMedian total: {} ms\nMedian prefill: {} ms\n\
                 Median decode: {:.3} tokens/s\n",
                corpus.prompts.len(),
                report["median_total_milliseconds"].as_u64().unwrap_or(u64::MAX),
                report["median_prefill_milliseconds"].as_u64().unwrap_or(u64::MAX),
                report["median_decode_tokens_per_second"].as_f64().unwrap_or(0.0),
            )
            .as_bytes(),
        )
    }
}

/// Builds an authenticated, drift-sensitive tuning profile from model, hardware, and microbenchmark data.
///
/// The model payload is authenticated before tuning. When provided, the sidecar data and manifest
/// are authenticated together. The resulting profile is written atomically to `output`; measured
/// candidates are recorded, while execution-policy changes require full-token correctness evidence.
///
/// # Errors
///
/// Returns an error if authentication, probing, benchmarking, profile serialization, or output
/// writing fails. Also errors if `samples` is zero or only one sidecar path is provided.
///
/// # Examples
///
/// ```no_run
/// # use std::path::Path;
/// # let model = Path::new("model.gguf");
/// # let output = Path::new("tuning-profile.json");
/// let result = tune(
///     model,
///     None,
///     None,
///     TuneProfileArg::Performance,
///     output,
///     3,
/// );
/// assert!(result.is_ok());
/// ```
fn tune(
    model: &Path,
    sidecar_path: Option<&Path>,
    sidecar_manifest: Option<&Path>,
    profile: TuneProfileArg,
    output: &Path,
    samples: usize,
) -> Result<()> {
    if samples == 0 {
        bail!("--samples must be greater than zero");
    }

    let authenticated = validate_selected_payload(model)
        .with_context(|| format!("failed to authenticate {}", model.display()))?;
    let model_file = authenticated
        .files
        .first()
        .context("authenticated model report contains no files")?;
    let model_artifact = ArtifactFingerprint {
        canonical_path: canonical_string(&model_file.path)?,
        length: model_file.logical_bytes,
        sha256: model_file.sha256.clone(),
    };

    let sidecar_artifact = match (sidecar_path, sidecar_manifest) {
        (Some(data), Some(manifest)) => {
            let sidecar = Sidecar::open(data, manifest)
                .with_context(|| format!("failed to open sidecar {}", data.display()))?;
            sidecar
                .verify_data_hash(&ReadCancellation::new())
                .with_context(|| format!("failed to authenticate sidecar {}", data.display()))?;
            Some(ArtifactFingerprint {
                canonical_path: canonical_string(data)?,
                length: sidecar.manifest().sidecar.length,
                sha256: sidecar.manifest().sidecar.sha256.clone(),
            })
        }
        (None, None) => None,
        _ => bail!("--sidecar and --sidecar-manifest must be provided together"),
    };

    let hardware = current_hardware_fingerprint()?;
    let topology = CpuTopology::detect();
    let (mut measurements, candidate_cpu_threads, candidate_cpu_set_ids) =
        tune_cpu_threads(&topology, samples)?;
    let (dot_measurements, mut candidate_rejections) = tune_cpu_dot_backends(samples)?;
    measurements.extend(dot_measurements);
    let (cuda_measurements, mut cuda_rejections) = tune_cuda_packed_executor(samples)?;
    measurements.extend(cuda_measurements);
    candidate_rejections.append(&mut cuda_rejections);
    let storage_target = sidecar_path.unwrap_or(model);
    let (storage_measurements, candidate_queue_depth, mut storage_rejections) =
        tune_buffered_storage(storage_target, samples)?;
    measurements.extend(storage_measurements);
    candidate_rejections.append(&mut storage_rejections);
    candidate_rejections.push(CandidateRejection {
        candidate: format!(
            "cpu_workers_{candidate_cpu_threads}_{}",
            if candidate_cpu_set_ids.is_empty() {
                "unbound"
            } else {
                "pinned"
            }
        ),
        reason: "microbenchmark winner is recorded for explicit full-token qualification and is \
                 not applied to the execution policy"
            .to_owned(),
    });
    candidate_rejections.push(CandidateRejection {
        candidate: format!("buffered_queue_depth_{candidate_queue_depth}"),
        reason: "storage microbenchmark winner is recorded for explicit full-token qualification \
                 and is not applied to the execution policy"
            .to_owned(),
    });

    let mut policy = ExecutionPolicy {
        cpu_threads: recommended_thread_count(),
        cpu_set_ids: Vec::new(),
        ..ExecutionPolicy::default()
    };
    policy.storage.queue_depth = 1;
    policy.storage.mode = StorageMode::Buffered;

    let cpu_capabilities = CpuCapabilities::detect();
    let cuda = bridge_kernels_cuda::build_capabilities();
    let decisions = vec![
        BackendDecision::rejected(
            BackendKind::CpuAvx2,
            true,
            "accepted CPU backend retained; micro-tuned worker and storage settings require \
             full-token 10% qualification before policy application",
        ),
        BackendDecision::rejected(
            BackendKind::CpuAvxVnni,
            true,
            if cpu_capabilities.avx_vnni_dot_kernel_available() {
                "packed-dot oracle passed during tuning; routes, tokens, and full-token 10% qualification remain"
            } else {
                "runtime CPU lacks AVX2 plus AVX-VNNI"
            },
        ),
        BackendDecision::rejected(
            BackendKind::CpuAvx512Vnni,
            true,
            if cpu_capabilities.avx512_dot_kernel_available() {
                "packed-dot oracle passed during tuning; routes, tokens, and full-token 10% qualification remain"
            } else {
                "runtime CPU lacks AVX-512F plus AVX-512 VNNI"
            },
        ),
        cuda_tuning_decision(cuda.rejection_reason),
        BackendDecision::rejected(
            BackendKind::Vulkan,
            false,
            "Vulkan is research-only until integer-dot, float-control, route, and token gates pass",
        ),
        BackendDecision::rejected(
            BackendKind::WindowsMlNpu,
            false,
            "Windows ML/NPU is advisory-only because W4/BF16 cannot preserve IQ2/IQ3 arithmetic",
        ),
    ];

    let profile_name = match profile {
        TuneProfileArg::Performance => "performance",
    };
    let mut profile = TuningProfileV1::new(
        profile_name,
        hardware,
        model_artifact,
        sidecar_artifact,
        policy,
        measurements,
        decisions,
    )?;
    profile.rejections = candidate_rejections;
    let mut bytes = serde_json::to_vec_pretty(&profile)?;
    bytes.push(b'\n');
    atomic_write(output, &bytes)?;
    write_stdout(
        format!(
            "Wrote authenticated {profile_name} tuning profile to {}\nMicrobenchmark-only CPU candidate: {} workers, affinity {:?}\nMicrobenchmark-only buffered queue-depth candidate: {}\nRetained execution policy: {} unbound CPU workers, buffered queue depth {}\nNo microbenchmark candidate or accelerator was auto-enabled without full-token correctness and >=10% evidence.\n",
            output.display(),
            candidate_cpu_threads,
            candidate_cpu_set_ids,
            candidate_queue_depth,
            profile.policy.cpu_threads,
            profile.policy.storage.queue_depth,
        )
        .as_bytes(),
    )
}

/// Creates a non-authoritative CUDA backend decision with a supplied or default rejection reason.
///
/// # Examples
///
/// ```
/// let decision = cuda_tuning_decision(Some("runtime qualification failed".to_owned()));
/// let _ = decision;
/// ```
fn cuda_tuning_decision...
fn cuda_tuning_decision(rejection_reason: Option<String>) -> BackendDecision {
    BackendDecision::rejected(
        BackendKind::Cuda,
        false,
        rejection_reason.unwrap_or_else(|| {
            "CUDA full-model correctness and complete-token timing are missing".to_owned()
        }),
    )
}

/// Benchmarks CPU thread-count and affinity configurations for packed GEMV execution.
///
/// Returns all measurements together with the fastest thread count and its CPU affinity set.
///
/// # Examples
///
/// ```
/// let topology = CpuTopology::detect();
/// let (measurements, threads, cpu_set_ids) = tune_cpu_threads(&topology, 1).unwrap();
///
/// assert!(!measurements.is_empty());
/// assert!(threads > 0);
/// assert!(cpu_set_ids.is_empty() || cpu_set_ids.len() == threads);
/// ```
fn tune_cpu_threads(
    topology: &CpuTopology,
    samples: usize,
) -> Result<(Vec<TuningMeasurement>, usize, Vec<u32>)> {
    let mut candidates = vec![
        1,
        topology.n_physical().max(1),
        recommended_thread_count(),
        topology.logical_processors as usize,
    ];
    candidates.sort_unstable();
    candidates.dedup();

    let rows = 512;
    let weights = vec![0_u8; 82 * rows];
    let matrix = PackedMatrix::from_parts(GgmlType::IQ2_S, PayloadEndian::Little, 256, rows, &weights)?;
    let input = (0..256)
        .map(|index| ((index % 31) as f32 - 15.0) / 16.0)
        .collect::<Vec<_>>();
    let physical_cpu_ids = topology.one_thread_per_core();
    let mut configurations = Vec::new();
    for threads in candidates {
        configurations.push((threads, Vec::new()));
        if threads <= physical_cpu_ids.len() {
            configurations.push((threads, physical_cpu_ids[..threads].to_vec()));
        }
    }
    let mut measurements = Vec::new();
    let mut best: Option<(usize, Vec<u32>, u64)> = None;

    for (threads, cpu_set_ids) in configurations {
        let backend = CpuBackend::new_with_cpu_set(CpuBackendConfig { threads }, &cpu_set_ids)?;
        let mut durations = Vec::with_capacity(samples);
        let mut expected = None;
        for _ in 0..samples {
            let mut output = vec![f32::NAN; rows];
            let mut q8 = vec![0_u8; required_q8_k_bytes(input.len())?];
            let started = Instant::now();
            backend.install(|| {
                for _ in 0..16 {
                    gemv_into(
                        ReferenceExecutionMode::CpuParallelQ8K,
                        matrix,
                        &input,
                        &mut output,
                        &mut [],
                        &mut q8,
                    )?;
                }
                Ok::<(), bridge_kernels_reference::KernelError>(())
            })?;
            durations.push(started.elapsed());
            let bits = output.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            if let Some(expected) = &expected {
                if &bits != expected {
                    bail!("CPU tuning candidate {threads} was not deterministic");
                }
            } else {
                expected = Some(bits);
            }
        }
        let measurement = TuningMeasurement::new(
            format!(
                "cpu_packed_gemv_threads_{threads}_{}",
                if cpu_set_ids.is_empty() {
                    "unbound"
                } else {
                    "one_thread_per_core"
                }
            ),
            Some(if CpuCapabilities::detect().avx2 {
                BackendKind::CpuAvx2
            } else {
                BackendKind::CpuScalar
            }),
            durations,
            None,
            if cpu_set_ids.is_empty() {
                "16 deterministic IQ2_S x Q8_K GEMVs with OS placement; microbenchmark only"
            } else {
                "16 deterministic IQ2_S x Q8_K GEMVs pinned one thread per core; microbenchmark only"
            },
        )?;
        if best
            .as_ref()
            .is_none_or(|(_, _, median)| measurement.median_ns < *median)
        {
            best = Some((threads, cpu_set_ids, measurement.median_ns));
        }
        measurements.push(measurement);
    }
    let (threads, cpu_set_ids, _) = best.context("CPU tuner produced no candidates")?;
    Ok((measurements, threads, cpu_set_ids))
}

/// Benchmarks available CPU packed-dot backends against mixed quantized weight formats.
///
/// # Parameters
///
/// * `samples` - Number of timing samples collected for each available backend.
///
/// # Examples
///
/// ```
/// let (measurements, rejections) = tune_cpu_dot_backends(1)?;
/// assert!(!measurements.is_empty() || !rejections.is_empty());
/// # Ok::<(), anyhow::Error>(())
/// ```
fn tune_cpu_dot_backends(samples: usize) -> Result<(Vec<TuningMeasurement>, Vec<CandidateRejection>)> {
    let logical_elements = 1_024;
    let input = (0..logical_elements)
        .map(|index| ((index % 47) as f32 - 23.0) / 24.0)
        .collect::<Vec<_>>();
    let mut q8 = vec![0_u8; required_q8_k_bytes(logical_elements)?];
    quantize_row_q8_k_into(&input, &mut q8)?;
    let mut cases = Vec::new();
    for ty in [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::IQ2_S, GgmlType::IQ3_S] {
        let block_bytes = quant_layout(ty)?.block_bytes;
        let mut weights = (0..block_bytes * 4)
            .map(|index| (index as u8).wrapping_mul(73).wrapping_add(19))
            .collect::<Vec<_>>();
        for block in weights.chunks_exact_mut(block_bytes) {
            block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            if matches!(ty, GgmlType::Q4_K | GgmlType::Q5_K) {
                block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
            }
        }
        let expected = vec_dot_q8_k(ty, &weights, &q8, logical_elements)?;
        cases.push((ty, weights, expected));
    }

    let mut measurements = Vec::new();
    let mut rejections = Vec::new();
    for backend in [
        CpuDotBackend::Scalar,
        CpuDotBackend::Avx2,
        CpuDotBackend::AvxVnni,
        CpuDotBackend::Avx512Vnni,
    ] {
        if !backend.available() {
            rejections.push(CandidateRejection {
                candidate: format!("cpu_dot_{}", backend.name()),
                reason: "required runtime ISA is not present".to_owned(),
            });
            continue;
        }
        let prepared = cases
            .iter()
            .map(|(ty, weights, expected)| {
                Ok((
                    ValidatedQ8KMatrix::new(*ty, weights, &q8, logical_elements, 1, backend)?,
                    *expected,
                ))
            })
            .collect::<std::result::Result<Vec<_>, bridge_quant_layout::QuantError>>()?;
        let mut durations = Vec::with_capacity(samples);
        for _ in 0..samples {
            let started = Instant::now();
            let mut digest = 0_u32;
            for _ in 0..128 {
                for (prepared, expected) in &prepared {
                    let actual = prepared.dot_row(0)?;
                    if actual.to_bits() != expected.to_bits() {
                        bail!(
                            "{} {:?} result {:#010x} differs from scalar {:#010x}",
                            backend.name(),
                            prepared.backend(),
                            actual.to_bits(),
                            expected.to_bits()
                        );
                    }
                    digest ^= actual.to_bits();
                }
            }
            std::hint::black_box(digest);
            durations.push(started.elapsed());
        }
        measurements.push(TuningMeasurement::new(
            format!("cpu_dot_{}", backend.name()),
            Some(match backend {
                CpuDotBackend::Scalar => BackendKind::CpuScalar,
                CpuDotBackend::Avx2 => BackendKind::CpuAvx2,
                CpuDotBackend::AvxVnni => BackendKind::CpuAvxVnni,
                CpuDotBackend::Avx512Vnni => BackendKind::CpuAvx512Vnni,
            }),
            durations,
            None,
            "512 mixed-format validated packed dots; common activation and matrix scales checked once",
        )?);
    }
    Ok((measurements, rejections))
}

/// Benchmarks CUDA packed GEMV candidates and verifies their outputs against scalar results.
///
/// The candidates cover several quantized weight formats and an IQ2_S paired submission.
/// Failed or non-bit-exact candidates are returned as rejections rather than causing the
/// benchmark to fail.
///
/// # Examples
///
/// ```no_run
/// let (measurements, rejections) = tune_cuda_packed_executor(3)?;
/// println!("{} candidates measured", measurements.len());
/// println!("{} candidates rejected", rejections.len());
/// # Ok::<(), anyhow::Error>(())
/// ```
fn tune_cuda_packed_executor(samples: usize) -> Result<(Vec<TuningMeasurement>, Vec<CandidateRejection>)> {
    const LOGICAL_ELEMENTS: usize = 4_096;
    const ROWS: usize = 1_344;

    let input = (0..LOGICAL_ELEMENTS)
        .map(|index| (((index * 29 + 7) % 251) as f32 - 125.0) / 23.0)
        .collect::<Vec<_>>();
    let mut q8 = vec![0_u8; required_q8_k_bytes(LOGICAL_ELEMENTS)?];
    quantize_row_q8_k_into(&input, &mut q8)?;
    let block_count = LOGICAL_ELEMENTS / bridge_quant_layout::Q8_K_BLOCK_ELEMENTS;
    let mut measurements = Vec::new();
    let mut rejections = Vec::new();

    for (format_index, weight_type) in [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::IQ2_S, GgmlType::IQ3_S]
        .into_iter()
        .enumerate()
    {
        let block_bytes = quant_layout(weight_type)?.block_bytes;
        let mut weights = vec![0_u8; ROWS * block_count * block_bytes];
        for (index, byte) in weights.iter_mut().enumerate() {
            *byte = ((index * 73 + format_index * 41 + 19) % 251) as u8;
        }
        for block in weights.chunks_exact_mut(block_bytes) {
            block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            if matches!(weight_type, GgmlType::Q4_K | GgmlType::Q5_K) {
                block[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
            }
        }
        let scalar = ValidatedQ8KMatrix::new(
            weight_type,
            &weights,
            &q8,
            LOGICAL_ELEMENTS,
            ROWS,
            CpuDotBackend::Scalar,
        )?;
        let expected = (0..ROWS)
            .map(|row| scalar.dot_row(row))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut output = vec![f32::NAN; ROWS];
        let warmup = match bridge_kernels_cuda::packed_q8k_gemv_into(
            weight_type,
            &weights,
            &q8,
            LOGICAL_ELEMENTS,
            &mut output,
        ) {
            Ok(execution) => execution,
            Err(error) => {
                rejections.push(CandidateRejection {
                    candidate: format!("cuda_reusable_packed_{weight_type:?}"),
                    reason: error.to_string(),
                });
                break;
            }
        };
        if output
            .iter()
            .zip(&expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            rejections.push(CandidateRejection {
                candidate: format!("cuda_reusable_packed_{weight_type:?}"),
                reason: "warmup output differs from the scalar oracle".to_owned(),
            });
            break;
        }

        let mut durations = Vec::with_capacity(samples);
        let mut last_execution = warmup;
        for _ in 0..samples {
            output.fill(f32::NAN);
            let started = Instant::now();
            last_execution = match bridge_kernels_cuda::packed_q8k_gemv_into(
                weight_type,
                &weights,
                &q8,
                LOGICAL_ELEMENTS,
                &mut output,
            ) {
                Ok(execution) => execution,
                Err(error) => {
                    rejections.push(CandidateRejection {
                        candidate: format!("cuda_reusable_packed_{weight_type:?}"),
                        reason: error.to_string(),
                    });
                    return Ok((measurements, rejections));
                }
            };
            durations.push(started.elapsed());
            if output
                .iter()
                .zip(&expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                rejections.push(CandidateRejection {
                    candidate: format!("cuda_reusable_packed_{weight_type:?}"),
                    reason: "sample output differs from the scalar oracle".to_owned(),
                });
                return Ok((measurements, rejections));
            }
        }
        measurements.push(TuningMeasurement::new(
            format!("cuda_reusable_packed_{weight_type:?}"),
            Some(BackendKind::Cuda),
            durations,
            Some(
                u64::try_from(
                    weights
                        .len()
                        .saturating_add(q8.len())
                        .saturating_add(output.len() * mem::size_of::<f32>()),
                )
                .unwrap_or(u64::MAX),
            ),
            format!(
                "{ROWS}x{LOGICAL_ELEMENTS} bit-exact reusable GEMV; last staging arena {}; host \
                 staging {:.3} ms; CUDA events {:.3} ms; diagnostic only",
                last_execution.staging_arena,
                last_execution.host_staging_milliseconds,
                last_execution.device_elapsed_milliseconds,
            ),
        )?);
    }
    if rejections.is_empty() {
        let weight_type = GgmlType::IQ2_S;
        let block_bytes = quant_layout(weight_type)?.block_bytes;
        let mut first_weights = vec![0_u8; ROWS * block_count * block_bytes];
        let mut second_weights = vec![0_u8; ROWS * block_count * block_bytes];
        for (index, byte) in first_weights.iter_mut().enumerate() {
            *byte = ((index * 73 + 101) % 251) as u8;
        }
        for (index, byte) in second_weights.iter_mut().enumerate() {
            *byte = ((index * 67 + 149) % 251) as u8;
        }
        for weights in [&mut first_weights, &mut second_weights] {
            for block in weights.chunks_exact_mut(block_bytes) {
                block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            }
        }
        let expected = [&first_weights, &second_weights]
            .into_iter()
            .map(|weights| {
                let scalar = ValidatedQ8KMatrix::new(
                    weight_type,
                    weights,
                    &q8,
                    LOGICAL_ELEMENTS,
                    ROWS,
                    CpuDotBackend::Scalar,
                )?;
                (0..ROWS)
                    .map(|row| scalar.dot_row(row))
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .collect::<std::result::Result<Vec<_>, bridge_quant_layout::QuantError>>()?;
        let mut first_output = vec![f32::NAN; ROWS];
        let mut second_output = vec![f32::NAN; ROWS];
        let mut last_execution = match bridge_kernels_cuda::packed_q8k_gemv_pair_into(
            [weight_type, weight_type],
            [&first_weights, &second_weights],
            &q8,
            LOGICAL_ELEMENTS,
            [&mut first_output, &mut second_output],
        ) {
            Ok(execution) => execution,
            Err(error) => {
                rejections.push(CandidateRejection {
                    candidate: "cuda_reusable_packed_pair_IQ2_S".to_owned(),
                    reason: error.to_string(),
                });
                return Ok((measurements, rejections));
            }
        };
        let outputs_match = |first: &[f32], second: &[f32]| {
            first
                .iter()
                .zip(&expected[0])
                .chain(second.iter().zip(&expected[1]))
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        };
        if !outputs_match(&first_output, &second_output) {
            rejections.push(CandidateRejection {
                candidate: "cuda_reusable_packed_pair_IQ2_S".to_owned(),
                reason: "warmup paired output differs from the scalar oracle".to_owned(),
            });
            return Ok((measurements, rejections));
        }
        let mut durations = Vec::with_capacity(samples);
        for _ in 0..samples {
            first_output.fill(f32::NAN);
            second_output.fill(f32::NAN);
            let started = Instant::now();
            last_execution = match bridge_kernels_cuda::packed_q8k_gemv_pair_into(
                [weight_type, weight_type],
                [&first_weights, &second_weights],
                &q8,
                LOGICAL_ELEMENTS,
                [&mut first_output, &mut second_output],
            ) {
                Ok(execution) => execution,
                Err(error) => {
                    rejections.push(CandidateRejection {
                        candidate: "cuda_reusable_packed_pair_IQ2_S".to_owned(),
                        reason: error.to_string(),
                    });
                    return Ok((measurements, rejections));
                }
            };
            durations.push(started.elapsed());
            if !outputs_match(&first_output, &second_output) {
                rejections.push(CandidateRejection {
                    candidate: "cuda_reusable_packed_pair_IQ2_S".to_owned(),
                    reason: "sample paired output differs from the scalar oracle".to_owned(),
                });
                return Ok((measurements, rejections));
            }
        }
        measurements.push(TuningMeasurement::new(
            "cuda_reusable_packed_pair_IQ2_S",
            Some(BackendKind::Cuda),
            durations,
            Some(
                u64::try_from(
                    first_weights
                        .len()
                        .saturating_add(second_weights.len())
                        .saturating_add(q8.len())
                        .saturating_add((first_output.len() + second_output.len()) * mem::size_of::<f32>()),
                )
                .unwrap_or(u64::MAX),
            ),
            format!(
                "two {ROWS}x{LOGICAL_ELEMENTS} bit-exact GEMVs in one submission; arenas {:?}; \
                 host staging {:.3} ms; CUDA events {:.3} ms; diagnostic only",
                last_execution.staging_arenas,
                last_execution.host_staging_milliseconds,
                last_execution.device_elapsed_milliseconds,
            ),
        )?);
    }
    Ok((measurements, rejections))
}

/// Benchmarks buffered file reads across several queue depths and records any additional platform-specific tuning failures.
///
/// `samples` controls the number of measurements collected for each queue depth.
///
/// # Examples
///
/// ```
/// # fn main() -> anyhow::Result<()> {
/// let path = std::env::temp_dir().join(format!(
///     "bridge-storage-tuning-{}",
///     std::process::id()
/// ));
/// std::fs::write(&path, vec![0_u8; 4096])?;
///
/// let (measurements, best_queue_depth, rejections) =
///     tune_buffered_storage(&path, 1)?;
///
/// assert!(!measurements.is_empty());
/// assert!((1..=4).contains(&best_queue_depth));
/// assert!(rejections.is_empty());
///
/// std::fs::remove_file(path)?;
/// # Ok(())
/// # }
/// ```
///
/// # Returns
///
/// A tuple containing the queue-depth measurements, the best buffered queue depth, and any rejected platform-specific candidates.
fn tune_buffered_storage(
    path: &Path,
    samples: usize,
) -> Result<(Vec<TuningMeasurement>, usize, Vec<CandidateRejection>)> {
    let length = fs::metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?
        .len();
    if length == 0 {
        bail!("cannot tune reads from an empty file: {}", path.display());
    }
    let read_bytes = usize::try_from(length.min(8 * 1024 * 1024))
        .context("storage sample is not representable on this platform")?;
    let file = Arc::new(PositionedFile::open(
        path,
        ReadLimits {
            max_request_bytes: read_bytes,
        },
    )?);
    let mut measurements = Vec::new();
    let mut best = None;
    for queue_depth in [1_usize, 2, 4] {
        let mut durations = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut buffers = (0..queue_depth)
                .map(|_| vec![0_u8; read_bytes])
                .collect::<Vec<_>>();
            let started = Instant::now();
            std::thread::scope(|scope| -> Result<()> {
                let mut handles = Vec::with_capacity(queue_depth);
                for (index, buffer) in buffers.iter_mut().enumerate() {
                    let file = Arc::clone(&file);
                    let window = length.saturating_sub(read_bytes as u64).saturating_add(1);
                    let ordinal = sample
                        .checked_mul(queue_depth)
                        .and_then(|value| value.checked_add(index))
                        .context("storage tuning ordinal overflow")?;
                    let offset = (ordinal as u64)
                        .saturating_mul(read_bytes as u64)
                        .checked_rem(window)
                        .unwrap_or(0);
                    handles
                        .push(scope.spawn(move || {
                            file.read_exact_at_into(offset, buffer, &ReadCancellation::new())
                        }));
                }
                for handle in handles {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("storage tuning worker panicked"))??;
                }
                Ok(())
            })?;
            durations.push(started.elapsed());
        }
        let total_bytes = (read_bytes as u64).saturating_mul(queue_depth as u64);
        let measurement = TuningMeasurement::new(
            format!("buffered_read_qd_{queue_depth}"),
            None,
            durations,
            Some(total_bytes),
            "positioned Windows-cache reads into reusable per-sample buffers",
        )?;
        if best.is_none_or(|(_, median)| measurement.median_ns < median) {
            best = Some((queue_depth, measurement.median_ns));
        }
        measurements.push(measurement);
    }
    let best_queue_depth = best.context("storage tuner produced no candidates")?.0;
    let mut rejections = Vec::new();
    #[cfg(windows)]
    {
        for buffering in [FileBuffering::Buffered, FileBuffering::Unbuffered] {
            match tune_iocp_storage(path, samples, buffering) {
                Ok(mut measured) => measurements.append(&mut measured),
                Err(error) => rejections.push(CandidateRejection {
                    candidate: match buffering {
                        FileBuffering::Buffered => "iocp_buffered",
                        FileBuffering::Unbuffered => "iocp_unbuffered",
                    }
                    .to_owned(),
                    reason: format!("{error:#}"),
                }),
            }
        }
    }
    Ok((measurements, best_queue_depth, rejections))
}

/// Benchmarks overlapped IOCP reads for several queue depths and buffering modes.
///
/// # Arguments
///
/// * `path` - File used as the benchmark source.
/// * `samples` - Number of read batches measured for each queue depth.
/// * `buffering` - File buffering mode used for the reads.
///
/// # Examples
///
/// ```no_run
/// let measurements = tune_iocp_storage(
///     std::path::Path::new("model.gguf"),
///     3,
///     FileBuffering::Buffered,
/// ).unwrap();
/// assert_eq!(measurements.len(), 3);
/// ```
#[cfg(windows)]
fn tune_iocp_storage(
    path: &Path,
    samples: usize,
    buffering: FileBuffering,
) -> Result<Vec<TuningMeasurement>> {
    let file = OverlappedFile::open(path, buffering)?;
    let alignment = file.alignment();
    let maximum = usize::try_from(file.length().min(8 * 1024 * 1024))
        .context("IOCP sample is not representable on this platform")?;
    let slot_bytes = maximum - (maximum % alignment);
    if slot_bytes == 0 {
        bail!(
            "{} is shorter than its {}-byte device alignment",
            path.display(),
            alignment
        );
    }

    let mut measurements = Vec::new();
    for queue_depth in [1_usize, 2, 4] {
        let pool = ReadSlotPool::new(queue_depth, slot_bytes, alignment)?;
        let mut durations = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut leases = (0..queue_depth)
                .map(|_| pool.acquire(&ReadCancellation::new()))
                .collect::<Result<Vec<_>, _>>()?;
            let positions = file
                .length()
                .saturating_sub(slot_bytes as u64)
                .checked_div(alignment as u64)
                .unwrap_or(0)
                .saturating_add(1);
            let mut requests = leases
                .iter_mut()
                .enumerate()
                .map(|(index, lease)| {
                    let ordinal = sample
                        .checked_mul(queue_depth)
                        .and_then(|value| value.checked_add(index))
                        .unwrap_or(usize::MAX);
                    let position = (ordinal as u64).checked_rem(positions).unwrap_or(0);
                    OverlappedRead {
                        offset: position.saturating_mul(alignment as u64),
                        buffer: lease.as_mut_slice(),
                    }
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            file.read_many(&mut requests, &ReadCancellation::new())?;
            durations.push(started.elapsed());
        }
        let mode = match buffering {
            FileBuffering::Buffered => "buffered",
            FileBuffering::Unbuffered => "unbuffered",
        };
        measurements.push(TuningMeasurement::new(
            format!("iocp_{mode}_qd_{queue_depth}"),
            None,
            durations,
            Some((slot_bytes as u64).saturating_mul(queue_depth as u64)),
            format!(
                "true overlapped IOCP batch; alignment {alignment}; explicit profile only until complete-token qualification"
            ),
        )?);
    }
    Ok(measurements)
}

/// Validates a persisted hardware tuning profile against the current run configuration.
///
/// The validation checks the current hardware, authenticated model payload, and optional
/// sidecar artifacts. The profile must reference the same model and sidecar artifacts and
/// remain compatible with the detected hardware.
///
/// # Errors
///
/// Returns an error if the profile cannot be read or parsed, required artifact authentication
/// data is unavailable, sidecar arguments are incomplete, or the profile is invalid.
///
/// # Examples
///
/// ```no_run
/// # use std::path::Path;
/// # let path = Path::new("tuning-profile.json");
/// # let runtime = runtime_args();
/// # let engine = open_engine(&runtime)?;
/// validate_hardware_profile(path, &runtime, &engine)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
fn validate_hardware_profile(path: &Path, runtime: &RuntimeArgs, engine: &Hy3ChatEngine) -> Result<()> {
    let bytes = read_bounded(path)?;
    let profile: TuningProfileV1 =
        serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))?;
    let model_hash = engine
        .model()
        .source_sha256()
        .first()
        .context("loaded engine has no authenticated source hash")?;
    let model = artifact_from_verified_hash(&runtime.model, model_hash)?;
    let sidecar = match (&runtime.sidecar_data, &runtime.sidecar_manifest) {
        (Some(data), Some(manifest)) => {
            let sidecar = Sidecar::open(data, manifest)?;
            Some(ArtifactFingerprint {
                canonical_path: canonical_string(data)?,
                length: sidecar.manifest().sidecar.length,
                sha256: sidecar.manifest().sidecar.sha256.clone(),
            })
        }
        (None, None) => None,
        _ => bail!("--sidecar-data and --sidecar-manifest must be provided together"),
    };
    profile
        .validate_current(&current_hardware_fingerprint()?, &model, sidecar.as_ref())
        .with_context(|| format!("hardware profile {} is not valid for this run", path.display()))
}

/// Adds prefill and decode events for a generation to a Chrome trace.
///
/// # Examples
///
/// ```
/// let mut trace = ChromeTrace::default();
/// let stats = bridge_runtime::GenerationStats {
///     prompt_tokens: 4,
///     generated_tokens: 2,
///     prefill_duration: Duration::from_millis(10),
///     decode_duration: Duration::from_millis(20),
/// };
///
/// append_completion_trace(&mut trace, "single", Duration::ZERO, stats);
/// ```
fn append_completion_trace(
    trace: &mut ChromeTrace,
    phase: &str,
    start: Duration,
    stats: bridge_runtime::GenerationStats,
) {
    let mut prefill_args = BTreeMap::new();
    prefill_args.insert("prompt_tokens".to_owned(), stats.prompt_tokens.into());
    trace.push_complete(
        format!("{phase}_prefill"),
        "attention,routing,kernels,output_head",
        start,
        stats.prefill_duration,
        prefill_args,
    );
    let mut decode_args = BTreeMap::new();
    decode_args.insert("generated_tokens".to_owned(), stats.generated_tokens.into());
    trace.push_complete(
        format!("{phase}_decode"),
        "disk_wait,cache,transfer,kernels",
        start + stats.prefill_duration,
        stats.decode_duration,
        decode_args,
    );
}

/// Persists a Chrome trace as newline-terminated JSON when a destination path is provided.
///
/// # Examples
///
/// ```
/// let trace = ChromeTrace::default();
/// assert!(persist_trace(None, &trace).is_ok());
/// ```
fn persist_trace(path: Option<&Path>, trace: &ChromeTrace) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let mut bytes = serde_json::to_vec(trace)?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

/// Collects the current hardware, runtime, and executable identity for tuning validation.
///
/// # Examples
///
/// ```
/// let fingerprint = current_hardware_fingerprint()?;
/// assert!(!fingerprint.operating_system.is_empty());
/// assert!(!fingerprint.architecture.is_empty());
/// # Ok::<(), anyhow::Error>(())
/// ```
fn current_hardware_fingerprint() -> Result<HardwareFingerprintV1> {
    let topology = CpuTopology::detect();
    let topology_bytes = serde_json::to_vec(&topology)?;
    let cpu_signature = format!("{}:{:x}", topology.brand, Sha256::digest(topology_bytes));
    let nvidia = probe_nvidia();
    let cuda_nvrtc = probe_cuda_nvrtc();
    let cuda_packed_oracle = probe_cuda_packed_oracle(cuda_nvrtc.available);
    let vulkan = probe_vulkan();
    let npu = probe_windows_ml_npu();
    let mut devices = Vec::new();
    let mut pcie_links = Vec::new();
    let mut runtimes = BTreeMap::new();
    if nvidia.available {
        if let Some(driver) = &nvidia.driver {
            runtimes.insert("nvidia_driver".to_owned(), driver.clone());
        }
        if let Some(link) = &nvidia.pcie_link {
            pcie_links.push(format!("nvidia:{link}"));
        }
        devices.push(DeviceFingerprint {
            backend: BackendKind::Cuda,
            name: nvidia.name.clone().unwrap_or_else(|| nvidia.detail.clone()),
            device_uuid: nvidia.uuid.clone(),
            driver: nvidia.driver.clone(),
            memory_bytes: nvidia.memory_mib.and_then(|mib| mib.checked_mul(1024 * 1024)),
            capability: Some(bridge_kernels_cuda::CUDA_TARGET_ARCHITECTURE.to_owned()),
        });
    }
    if vulkan.available {
        devices.push(DeviceFingerprint {
            backend: BackendKind::Vulkan,
            name: vulkan.detail.clone(),
            device_uuid: None,
            driver: None,
            memory_bytes: None,
            capability: Some("runtime probe only".to_owned()),
        });
    }
    if npu.available {
        devices.push(DeviceFingerprint {
            backend: BackendKind::WindowsMlNpu,
            name: npu.detail.clone(),
            device_uuid: None,
            driver: None,
            memory_bytes: None,
            capability: Some("advisory router only".to_owned()),
        });
    }
    if let Some(version) = probe_version("nvcc", &["--version"]) {
        runtimes.insert("cuda_toolkit".to_owned(), version);
    }
    if let Some(canary) = cuda_nvrtc.canary {
        runtimes.insert(
            "cuda_nvrtc_driver_canary".to_owned(),
            format!(
                "nvrtc={}.{};compute={}.{};ptx_bytes={};pinned_async={}",
                canary.nvrtc_major,
                canary.nvrtc_minor,
                canary.compute_major,
                canary.compute_minor,
                canary.ptx_bytes,
                canary.pinned_async_transfers,
            ),
        );
    }
    if let Some(oracle) = cuda_packed_oracle.oracle {
        let formats = oracle
            .formats
            .iter()
            .map(|format| format.weight_type.as_str())
            .collect::<Vec<_>>()
            .join(",");
        runtimes.insert(
            "cuda_packed_q8k_oracle".to_owned(),
            format!(
                "nvrtc={}.{};compute={}.{};ptx_bytes={};formats={};bit_exact=true",
                oracle.nvrtc_major,
                oracle.nvrtc_minor,
                oracle.compute_major,
                oracle.compute_minor,
                oracle.ptx_bytes,
                formats,
            ),
        );
    }
    if let Some(reusable) = cuda_packed_oracle.reusable {
        let arenas = reusable
            .executions
            .iter()
            .map(|execution| execution.staging_arena.to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(",");
        runtimes.insert(
            "cuda_reusable_packed_executor".to_owned(),
            format!(
                "passes={};formats={};operations={};arenas={};bit_exact={};deterministic={}",
                reusable.passes,
                reusable.formats,
                reusable.executions.len(),
                arenas,
                reusable.bit_exact,
                reusable.deterministic,
            ),
        );
    }
    #[cfg(windows)]
    if let Some(version) = probe_visual_studio_version() {
        runtimes.insert("msvc".to_owned(), version);
    }
    if vulkan.available {
        runtimes.insert("vulkan".to_owned(), vulkan.detail);
    }

    let executable = std::env::current_exe().context("failed to locate the running bridge executable")?;
    Ok(HardwareFingerprintV1 {
        version: HARDWARE_FINGERPRINT_VERSION,
        engine_build: sha256_file(&executable)?,
        operating_system: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        cpu_signature,
        logical_processors: topology.logical_processors as usize,
        total_physical_memory: memory_status().total_physical,
        devices,
        power_state: probe_power_state(),
        pcie_links,
        runtimes,
    })
}

/// Creates an artifact fingerprint from a regular file and its verified SHA-256 hash.
///
/// # Errors
///
/// Returns an error if the path cannot be inspected or does not refer to a regular file.
///
/// # Examples
///
/// ```
/// let path = std::path::Path::new("model.gguf");
/// let fingerprint = artifact_from_verified_hash(path, "abc123")?;
/// assert_eq!(fingerprint.sha256, "abc123");
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Returns
///
/// An artifact fingerprint containing the file's canonical path, length, and supplied SHA-256 hash.
fn artifact_from_verified_hash(path: &Path, sha256: &str) -> Result<ArtifactFingerprint> {
    let metadata = fs::metadata(path).with_context(|| format!("failed to inspect {}", path.display()))?;
    if !metadata.is_file() {
        bail!("artifact is not a regular file: {}", path.display());
    }
    Ok(ArtifactFingerprint {
        canonical_path: canonical_string(path)?,
        length: metadata.len(),
        sha256: sha256.to_owned(),
    })
}

/// Resolves a path to its canonical, lossless string representation.
///
/// # Examples
///
/// ```
/// let path = canonical_string(std::path::Path::new("."))?;
/// assert!(!path.is_empty());
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Parameters
///
/// * `path` - The path to canonicalize.
///
/// # Errors
///
/// Returns an error if the path cannot be canonicalized.
fn canonical_string(path: &Path) -> Result<String> {
    Ok(fs::canonicalize(path)
        .with_context(|| format!("failed to canonicalize {}", path.display()))?
        .to_string_lossy()
        .into_owned())
}

/// Computes the SHA-256 digest of a file as a lowercase hexadecimal string.
///
/// # Examples
///
/// ```
/// let path = std::env::temp_dir().join("sha256_file_example.txt");
/// std::fs::write(&path, b"hello")?;
///
/// let digest = sha256_file(&path)?;
/// assert_eq!(
///     digest,
///     "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
/// );
///
/// std::fs::remove_file(path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Reports the active Windows power scheme and battery status.
///
/// # Examples
///
/// ```
/// # #[cfg(windows)]
/// # {
/// let state = probe_power_state();
/// assert!(state.contains(';'));
/// # }
/// ```
#[cfg(windows)]
fn probe_power_state() -> String {
    let scheme =
        probe_version("powercfg", &["/getactivescheme"]).unwrap_or_else(|| "unknown-scheme".to_owned());
    let ac = ProcessCommand::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty BatteryStatus)",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
        .map_or("ac-or-no-battery".to_owned(), |status| format!("battery-status-{status}"));
    format!("{ac};{scheme}")
}

/// Reports the current power state on unsupported platforms.
///
/// # Examples
///
/// ```
/// assert_eq!(probe_power_state(), "unknown");
/// ```
fn probe_power_state() -> String {
    "unknown".to_owned()
}

/// Runs a program and captures its non-empty standard output when it exits successfully.
///
/// # Examples
///
/// ```
/// let version = probe_version("rustc", &["--version"]);
/// assert!(version.is_some());
/// ```
fn probe_version(program: &str, arguments: &[&str]) -> Option<String> {
    ProcessCommand::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Inspects or removes a persisted expert-cache heat snapshot.
///
/// # Examples
///
/// ```
/// use std::{fs, path::PathBuf};
///
/// let path = PathBuf::from(std::env::temp_dir()).join(format!(
///     "bridge-cache-example-{}",
///     std::process::id()
/// ));
/// fs::write(&path, b"snapshot").unwrap();
///
/// cache(CacheCommand::ClearHeat { path: path.clone() }).unwrap();
/// assert!(!path.exists());
/// ```
fn cache(command: CacheCommand) -> Result<()> {
    match command {
        CacheCommand::InspectHeat { path, max_entries } => {
            let bytes = read_bounded(&path)?;
            let cache = CompressedCache::<ExpertKey>::new(CacheConfig {
                capacity_bytes: 1,
                admit_after_requests: 1,
            })?;
            cache.import_heat_json(&bytes, MAX_JSON_INPUT_BYTES as usize, max_entries)?;
            let normalized = cache.export_heat_json(max_entries)?;
            write_stdout(&normalized)?;
            write_stdout(b"\n")
        }
        CacheCommand::ClearHeat { path } => {
            let metadata =
                fs::metadata(&path).with_context(|| format!("failed to inspect {}", path.display()))?;
            if !metadata.is_file() {
                bail!("heat snapshot is not a regular file: {}", path.display());
            }
            fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
            write_stdout(format!("Removed {}\n", path.display()).as_bytes())
        }
    }
}

fn load_tokenizer(model: &Path) -> Result<Hy3Tokenizer> {
    let set = bridge_gguf_split::open_set(model)?;
    let file = set.files().first().context("GGUF set contains no files")?;
    Ok(Hy3Tokenizer::from_gguf(file.parsed())?)
}

fn open_engine(args: &RuntimeArgs) -> Result<Hy3ChatEngine> {
    if args.cache_heat_max_entries == 0 {
        bail!("cache-heat-max-entries must be greater than zero");
    }
    let options = runtime_options(args)?;
    let engine = Hy3ChatEngine::open_selected(&args.model, options)
        .with_context(|| format!("failed to load {}", args.model.display()))?;
    if let Some(path) = &args.cache_heat {
        if path
            .try_exists()
            .with_context(|| format!("failed to inspect {}", path.display()))?
        {
            let bytes = read_bounded(path)?;
            engine.model().import_cache_heat(
                &bytes,
                usize::try_from(MAX_JSON_INPUT_BYTES).unwrap_or(usize::MAX),
                args.cache_heat_max_entries,
            )?;
        }
    }
    Ok(engine)
}

fn persist_cache_heat(args: &RuntimeArgs, engine: &Hy3ChatEngine) -> Result<()> {
    let Some(path) = &args.cache_heat else {
        return Ok(());
    };
    let bytes = engine.model().export_cache_heat(args.cache_heat_max_entries)?;
    atomic_write(path, &bytes)
}

fn persist_chat_session(
    engine: &Hy3ChatEngine,
    session: &Hy3ChatSession,
    path: Option<&Path>,
    maximum_bytes: usize,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let bytes = engine.export_session(session, maximum_bytes)?;
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("cache heat path must have a UTF-8 file name")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let mut temporary = None;
    for attempt in 0..16_u8 {
        let candidate = parent.join(format!(
            ".{file_name}.{}.{}.{}.tmp",
            std::process::id(),
            nonce,
            attempt
        ));
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to create cache heat file in {}", parent.display()));
            }
        }
    }
    let (temporary_path, mut output) =
        temporary.context("failed to allocate a unique temporary cache heat path")?;
    let write_result = output
        .write_all(bytes)
        .and_then(|()| output.sync_all())
        .with_context(|| format!("failed to write {}", temporary_path.display()));
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(error);
    }
    if let Err(error) = replace_file(&temporary_path, path) {
        let _ = fs::remove_file(&temporary_path);
        return Err(error).with_context(|| format!("failed to publish {}", path.display()));
    }
    Ok(())
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let mut source_wide = source.as_os_str().encode_wide().collect::<Vec<_>>();
    source_wide.push(0);
    let mut destination_wide = destination.as_os_str().encode_wide().collect::<Vec<_>>();
    destination_wide.push(0);
    // SAFETY: both paths are live NUL-terminated UTF-16 buffers for the
    // duration of the call.
    let result = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

/// Builds engine runtime options from the command-line runtime arguments.
///
/// Sidecar data and its manifest must be provided together. A zero CPU-thread
/// value selects the recommended thread count, while memory sizes are converted
/// from MiB and must fit the platform's `usize`.
///
/// # Errors
///
/// Returns an error if only one sidecar path is provided or a configured memory
/// size cannot be represented on the platform.
///
/// # Examples
///
/// ```no_run
/// let options = runtime_options(&args)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Arguments
///
/// * `args` — Runtime configuration used to construct the engine options.
///
/// # Returns
///
/// The configured scalar engine options.
fn runtime_options(args: &RuntimeArgs) -> Result<Hy3ScalarOptions> {
    let expert_source = match (&args.sidecar_data, &args.sidecar_manifest) {
        (Some(data_path), Some(manifest_path)) => ExpertSourceOptions::Sidecar {
            data_path: data_path.clone(),
            manifest_path: manifest_path.clone(),
            verify_data_hash: true,
            verify_source_bindings: true,
        },
        (None, None) => ExpertSourceOptions::Direct,
        _ => bail!("--sidecar-data and --sidecar-manifest must be provided together"),
    };
    Ok(Hy3ScalarOptions {
        context_capacity: args.context,
        kv_page_tokens: args.kv_page_tokens,
        expert_cache_bytes: usize::try_from(mib(args.cache_mib)?)
            .context("cache size is not representable on this platform")?,
        execution_mode: match args.backend {
            ExecutionModeArg::CpuQ8K => bridge_kernels_reference::ReferenceExecutionMode::CpuParallelQ8K,
            ExecutionModeArg::CudaQ8K => bridge_kernels_reference::ReferenceExecutionMode::CudaQ8K,
            ExecutionModeArg::CpuAvxVnniQ8K => {
                bridge_kernels_reference::ReferenceExecutionMode::CpuParallelAvxVnni
            }
            ExecutionModeArg::CpuAvx512VnniQ8K => {
                bridge_kernels_reference::ReferenceExecutionMode::CpuParallelAvx512Vnni
            }
            ExecutionModeArg::ScalarQ8K => bridge_kernels_reference::ReferenceExecutionMode::LlamaQ8K,
            ExecutionModeArg::DequantF32 => bridge_kernels_reference::ReferenceExecutionMode::DequantF32,
        },
        cpu_threads: if args.cpu_threads == 0 {
            recommended_thread_count()
        } else {
            args.cpu_threads
        },
        cpu_set_ids: args.cpu_set_ids.clone(),
        prefill_chunk: args.prefill_chunk,
        speculative_ngram_t: args.speculative_ngram_t,
        memory_headroom_bytes: usize::try_from(mib(args.memory_headroom_mib)?)
            .context("memory headroom is not representable on this platform")?,
        expert_source,
        ..Hy3ScalarOptions::default()
    })
}

fn sampling_config(args: &SamplingArgs) -> SamplingConfig {
    SamplingConfig {
        max_new_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        repeat_last_n: args.repeat_last_n,
        seed: args.seed,
        ..SamplingConfig::default()
    }
}

fn reject_sparse_sources(set: &bridge_gguf_split::GgufSet) -> Result<()> {
    for shard in set.files() {
        let storage = file_storage(shard.path())?;
        if storage.is_sparse && storage.allocated_bytes < storage.logical_bytes {
            bail!(
                "refusing sparse/incomplete source {}: {} allocated bytes for {} logical bytes",
                shard.path().display(),
                storage.allocated_bytes,
                storage.logical_bytes
            );
        }
    }
    Ok(())
}

fn storage_report(path: &Path) -> Result<StorageReport> {
    let FileStorage {
        logical_bytes,
        allocated_bytes,
        is_sparse,
        is_compressed,
    } = file_storage(path)?;
    Ok(StorageReport {
        path: path.display().to_string(),
        logical_bytes,
        allocated_bytes,
        sparse: is_sparse,
        compressed: is_compressed,
    })
}

fn read_json_file<T>(path: &Path) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let bytes = read_bounded(path)?;
    serde_json::from_slice(&bytes).with_context(|| format!("invalid JSON in {}", path.display()))
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    read_bounded_with_limit(path, MAX_JSON_INPUT_BYTES)
}

fn read_bounded_with_limit(path: &Path, maximum_bytes: u64) -> Result<Vec<u8>> {
    if maximum_bytes == 0 {
        bail!("input byte limit must be greater than zero");
    }
    let metadata = fs::metadata(path).with_context(|| format!("failed to inspect {}", path.display()))?;
    if !metadata.is_file() {
        bail!("input is not a regular file: {}", path.display());
    }
    if metadata.len() > maximum_bytes {
        bail!(
            "input {} is {} bytes, maximum is {}",
            path.display(),
            metadata.len(),
            maximum_bytes
        );
    }
    fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

fn write_json(value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_stdout(&bytes)
}

fn write_stdout(bytes: &[u8]) -> Result<()> {
    io::stdout()
        .lock()
        .write_all(bytes)
        .context("failed to write stdout")
}

fn mib(value: usize) -> Result<u64> {
    (value as u64)
        .checked_mul(1024 * 1024)
        .context("MiB byte count overflow")
}

fn human_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} bytes")
    }
}

fn tokens_per_second(tokens: usize, duration: std::time::Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

impl From<LayoutArg> for ExpertLayout {
    fn from(value: LayoutArg) -> Self {
        match value {
            LayoutArg::Sequential => Self::Sequential,
            LayoutArg::FusedGateUp => Self::FusedGateUp,
        }
    }
}

impl From<ReasoningArg> for ReasoningEffort {
    fn from(value: ReasoningArg) -> Self {
        match value {
            ReasoningArg::High => Self::High,
            ReasoningArg::Low => Self::Low,
            ReasoningArg::NoThink => Self::NoThink,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_cache_heat_write_replaces_the_complete_previous_snapshot() {
        let path = std::env::temp_dir().join(format!(
            "lightbridge-cache-heat-{}-{}.json",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        atomic_write(&path, b"{\"version\":1}").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"{\"version\":1}");
        atomic_write(&path, b"{\"version\":2,\"entries\":[]}").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"{\"version\":2,\"entries\":[]}");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn nvidia_pnp_parser_reports_a_disconnected_adapter() {
        let text = r#"Microsoft PnP Utility

Instance ID:                PCI\VEN_1002&DEV_150E
Device Description:         AMD Radeon(TM) 890M Graphics
Status:                     Started
Driver Name:                oem57.inf

Instance ID:                PCI\VEN_10DE&DEV_2860
Device Description:         NVIDIA GeForce RTX 4070 Laptop GPU
Status:                     Disconnected
Driver Name:                oem16.inf
"#;
        let parsed = parse_nvidia_pnp(text).unwrap();
        assert_eq!(parsed.0, "NVIDIA GeForce RTX 4070 Laptop GPU");
        assert_eq!(parsed.1, "Disconnected");
        assert_eq!(parsed.2, "oem16.inf");
    }

    #[test]
    fn cuda_tuning_decision_is_not_authoritative_before_full_model_qualification() {
        let decision = cuda_tuning_decision(None);
        assert_eq!(decision.backend, BackendKind::Cuda);
        assert!(!decision.authoritative);
        assert!(!decision.automatic);
        assert!(decision.reason.contains("full-model correctness"));
    }

    #[test]
    fn benchmark_corpus_parser_enforces_version_bounds_and_unique_prompts() {
        let corpus = parse_benchmark_corpus(
            br#"{
                "format":"lightbridge-benchmark-corpus",
                "version":1,
                "prompts":["Hello","Explain one thing."]
            }"#,
        )
        .unwrap();
        assert_eq!(corpus.prompts.len(), 2);
        assert!(parse_benchmark_corpus(
            br#"{
                "format":"lightbridge-benchmark-corpus",
                "version":1,
                "prompts":["same","same"]
            }"#,
        )
        .unwrap_err()
        .to_string()
        .contains("duplicates"));
        assert!(parse_benchmark_corpus(
            br#"{
                "format":"lightbridge-benchmark-corpus",
                "version":2,
                "prompts":["Hello"]
            }"#,
        )
        .is_err());
    }
}
