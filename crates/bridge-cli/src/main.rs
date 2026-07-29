use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use bridge_cache::{CacheConfig, CompressedCache};
use bridge_cli::{build_report, render_json, render_text};
use bridge_core::sys::{memory_status, CpuTopology};
use bridge_format::{ExpertKey, ExpertLayout};
use bridge_io_windows::{file_storage, FileStorage, ReadCancellation};
use bridge_kernels_cpu::{recommended_thread_count, CpuCapabilities};
use bridge_model_hy3::validate_selected_model;
use bridge_prepare::{prepare_sidecar, DirectExpertIndex, PrepareOptions};
use bridge_runtime::{
    validate_selected_payload, CancellationToken, ExpertSourceOptions, Hy3ChatEngine, Hy3ChatSession,
    Hy3ScalarOptions, SamplingConfig,
};
use bridge_tokenizer::{ChatMessage, ChatTemplateOptions, Hy3Tokenizer, ReasoningEffort};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Serialize;

const MAX_JSON_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_CACHE_HEAT_ENTRIES: usize = 65_536;

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
        /// Measure cold, admission, and repeated warm-state runs in one authenticated engine.
        #[arg(long)]
        cold_warm: bool,
        #[arg(long)]
        json: bool,
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
            cold_warm,
            json,
        } => bench(runtime, sampling, prompt, cold_warm, json),
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
    capabilities: CapabilityReport,
}

#[derive(Debug, Serialize)]
struct NvidiaStatus {
    available: bool,
    detail: String,
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
    grouped_prefill: bool,
    persistent_kv: bool,
    mtp_acceleration: bool,
    experimental_igpu: bool,
    server: bool,
    selected_model_required_for_chat: bool,
}

fn doctor(json: bool) -> Result<()> {
    let nvidia = probe_nvidia();
    let cpu = CpuTopology::detect();
    let cpu_capabilities = CpuCapabilities::detect();
    let report = DoctorReport {
        engine_version: env!("CARGO_PKG_VERSION"),
        operating_system: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        cpu,
        memory: memory_status(),
        nvidia,
        capabilities: CapabilityReport {
            scalar_reference: true,
            q8_k_activation_path: true,
            cpu_parallel_backend: true,
            cpu_parallel_default_threads: recommended_thread_count(),
            cpu_simd_backend: cpu_capabilities.avx2_dot_kernel_available(),
            parallel_expert_prefetch: true,
            persistent_expert_heat: true,
            cuda_backend: false,
            grouped_prefill: false,
            persistent_kv: true,
            mtp_acceleration: false,
            experimental_igpu: false,
            server: true,
            selected_model_required_for_chat: true,
        },
    };
    if json {
        write_json(&report)
    } else {
        let text = format!(
            "LightBridge {}\nOS: {} {}\nCPU: {}\nPhysical cores: {}\nLogical processors: {}\nISA: {}\nRAM: {} total, {} available\nNVIDIA: {}\nExecution: {} with {} bounded threads (AVX2 detected: {}, AVX-512 VNNI detected: {})\nExpert prefetch: parallel\nExpert heat persistence: available\nCUDA backend: unavailable\nGrouped prefill: unavailable\nPersistent KV: model-bound checksummed sessions available\nMTP: not applicable to selected model\nExperimental iGPU: unavailable\nServer: available\n",
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
            cpu_capabilities.backend_name(),
            report.capabilities.cpu_parallel_default_threads,
            cpu_capabilities.avx2,
            cpu_capabilities.avx512_vnni,
        );
        write_stdout(text.as_bytes())
    }
}

fn probe_nvidia() -> NvidiaStatus {
    let output = ProcessCommand::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(output) if output.status.success() => NvidiaStatus {
            available: true,
            detail: String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        },
        Ok(output) => NvidiaStatus {
            available: false,
            detail: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        },
        Err(error) => NvidiaStatus {
            available: false,
            detail: error.to_string(),
        },
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

#[allow(clippy::too_many_arguments)]
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

fn bench(
    runtime: RuntimeArgs,
    sampling: SamplingArgs,
    prompt: String,
    cold_warm: bool,
    json: bool,
) -> Result<()> {
    let engine = open_engine(&runtime)?;
    if cold_warm {
        let mut runs = Vec::new();
        let mut expected_tokens = None;
        for phase in ["cold", "admission", "warm"] {
            let cache_before = engine.model().cache_stats()?;
            let completion = engine.complete(
                &[ChatMessage::user(prompt.clone())],
                &ChatTemplateOptions::default(),
                sampling_config(&sampling),
                &CancellationToken::new(),
                |_| ControlFlow::Continue(()),
            )?;
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
                "stop_reason": completion.generation.stop_reason,
            }));
        }
        let backend = engine.model().backend_name();
        let cpu_threads = engine.model().cpu_threads();
        persist_cache_heat(&runtime, &engine)?;
        let report = serde_json::json!({
            "model": runtime.model,
            "backend": backend,
            "cpu_threads": cpu_threads,
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

    let completion = engine.complete(
        &[ChatMessage::user(prompt)],
        &ChatTemplateOptions::default(),
        sampling_config(&sampling),
        &CancellationToken::new(),
        |_| ControlFlow::Continue(()),
    )?;
    let stats = completion.generation.stats;
    let backend = engine.model().backend_name();
    let cpu_threads = engine.model().cpu_threads();
    persist_cache_heat(&runtime, &engine)?;
    let report = serde_json::json!({
        "model": runtime.model,
        "backend": backend,
        "cpu_threads": cpu_threads,
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": stats.generated_tokens,
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
                "Backend: {}{}\nPrompt tokens: {}\nGenerated tokens: {}\nPrefill: {} ms\nDecode: {} ms\nDecode throughput: {:.3} tokens/s\n",
                backend,
                cpu_threads.map(|threads| format!(" ({threads} threads)")).unwrap_or_default(),
                stats.prompt_tokens,
                stats.generated_tokens,
                stats.prefill_duration.as_millis(),
                stats.decode_duration.as_millis(),
                tokens_per_second(stats.generated_tokens, stats.decode_duration),
            )
            .as_bytes(),
        )
    }
}

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
            ExecutionModeArg::ScalarQ8K => bridge_kernels_reference::ReferenceExecutionMode::LlamaQ8K,
            ExecutionModeArg::DequantF32 => bridge_kernels_reference::ReferenceExecutionMode::DequantF32,
        },
        cpu_threads: if args.cpu_threads == 0 {
            recommended_thread_count()
        } else {
            args.cpu_threads
        },
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
}
