use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bridge_cli::InspectionReport;
use bridge_core::tensor::TensorDesc;
use bridge_gguf::GgufValueType;
use bridge_model_hy3::{generate_selected_iq2_m_schema, Hy3Profile};

const SELECTED_FILE_LEN: u64 = 96_019_311_104;
const SELECTED_ENCODED_TENSOR_BYTES: u64 = 96_014_150_912;
const SELECTED_TENSOR_COUNT: u64 = 1_278;
const SELECTED_METADATA_COUNT: u64 = 45;
const MAX_FIXTURE_HEADER_BYTES: u64 = 8 * 1024 * 1024;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_or_else(
            |error| u128::MAX - error.duration().as_nanos(),
            |duration| duration.as_nanos(),
        );
        let root = std::env::temp_dir();
        Self::create_in(&root, nonce, &NEXT_TEMP_ID)
            .unwrap_or_else(|error| panic!("failed to create test directory in {}: {error}", root.display()))
    }

    fn create_in(root: &Path, nonce: u128, counter: &AtomicU64) -> io::Result<Self> {
        loop {
            let sequence = counter.fetch_add(1, Ordering::Relaxed);
            let path = Self::candidate_path(root, nonce, sequence);
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }

    fn candidate_path(root: &Path, nonce: u128, sequence: u64) -> PathBuf {
        root.join(format!(
            "lightbridge-cli-{}-{nonce:032x}-{sequence:016x}",
            std::process::id()
        ))
    }

    fn path(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            if !std::thread::panicking() {
                panic!("failed to remove test directory {}: {error}", self.path.display());
            }
        }
    }
}

struct FixtureEncoder<W> {
    writer: W,
    metadata_entries: u64,
}

impl<W: Write> FixtureEncoder<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            metadata_entries: 0,
        }
    }

    fn bytes(&mut self, value: &[u8]) -> io::Result<()> {
        self.writer.write_all(value)
    }

    fn u8(&mut self, value: u8) -> io::Result<()> {
        self.bytes(&[value])
    }

    fn u32(&mut self, value: u32) -> io::Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn i32(&mut self, value: i32) -> io::Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> io::Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn f32(&mut self, value: f32) -> io::Result<()> {
        self.u32(value.to_bits())
    }

    fn string(&mut self, value: &str) -> io::Result<()> {
        self.u64(value.len() as u64)?;
        self.bytes(value.as_bytes())
    }

    fn metadata_prefix(&mut self, key: &str, ty: GgufValueType) -> io::Result<()> {
        self.metadata_entries += 1;
        self.string(key)?;
        self.u32(ty as u32)
    }

    fn metadata_string(&mut self, key: &str, value: &str) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::String)?;
        self.string(value)
    }

    fn metadata_u32(&mut self, key: &str, value: u32) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::U32)?;
        self.u32(value)
    }

    fn metadata_i32(&mut self, key: &str, value: i32) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::I32)?;
        self.i32(value)
    }

    fn metadata_f32(&mut self, key: &str, value: f32) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::F32)?;
        self.f32(value)
    }

    fn metadata_bool(&mut self, key: &str, value: bool) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::Bool)?;
        self.u8(u8::from(value))
    }

    fn metadata_string_array<'a>(
        &mut self,
        key: &str,
        values: impl ExactSizeIterator<Item = &'a str>,
    ) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::Array)?;
        self.u32(GgufValueType::String as u32)?;
        self.u64(values.len() as u64)?;
        for value in values {
            self.string(value)?;
        }
        Ok(())
    }

    fn metadata_repeated_empty_strings(&mut self, key: &str, count: u64) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::Array)?;
        self.u32(GgufValueType::String as u32)?;
        self.u64(count)?;
        for _ in 0..count {
            self.u64(0)?;
        }
        Ok(())
    }

    fn metadata_repeated_i32(&mut self, key: &str, value: i32, count: u64) -> io::Result<()> {
        self.metadata_prefix(key, GgufValueType::Array)?;
        self.u32(GgufValueType::I32 as u32)?;
        self.u64(count)?;
        for _ in 0..count {
            self.i32(value)?;
        }
        Ok(())
    }
}

fn write_selected_fixture(path: &Path) -> io::Result<u64> {
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?;
    mark_sparse(&file)?;
    file.set_len(SELECTED_FILE_LEN)?;

    let mut encoder = FixtureEncoder::new(BufWriter::new(file));
    encoder.bytes(b"GGUF")?;
    encoder.u32(3)?;
    encoder.u64(SELECTED_TENSOR_COUNT)?;
    encoder.u64(SELECTED_METADATA_COUNT)?;
    write_selected_metadata(&mut encoder)?;
    assert_eq!(encoder.metadata_entries, SELECTED_METADATA_COUNT);

    let profile = Hy3Profile::selected_iq2_m();
    let schema = generate_selected_iq2_m_schema(profile.config()).unwrap();
    assert_eq!(schema.len() as u64, SELECTED_TENSOR_COUNT);

    let mut next_offset = 0_u64;
    for spec in schema {
        let tensor = TensorDesc::new(spec.name(), spec.shape(), spec.ty(), next_offset).unwrap();
        encoder.string(tensor.name())?;
        encoder.u32(tensor.n_dims())?;
        for &dimension in tensor.shape() {
            encoder.u64(dimension)?;
        }
        encoder.u32(tensor.ty().discriminant())?;
        encoder.u64(tensor.relative_offset())?;

        next_offset = align_32(next_offset.checked_add(tensor.encoded_bytes().unwrap()).unwrap());
    }
    assert_eq!(next_offset, SELECTED_ENCODED_TENSOR_BYTES);

    encoder.writer.flush()?;
    let header_bytes = encoder.writer.stream_position()?;
    assert!(
        header_bytes < MAX_FIXTURE_HEADER_BYTES,
        "fixture wrote {header_bytes} bytes before its sparse payload hole"
    );
    assert_eq!(encoder.writer.get_ref().metadata()?.len(), SELECTED_FILE_LEN);
    drop(encoder);
    let allocated_bytes = allocated_size(path)?;
    assert!(
        allocated_bytes < MAX_FIXTURE_HEADER_BYTES * 2,
        "sparse fixture allocated {allocated_bytes} physical bytes"
    );
    Ok(header_bytes)
}

fn write_selected_metadata<W: Write>(encoder: &mut FixtureEncoder<W>) -> io::Result<()> {
    encoder.metadata_string("general.architecture", "hy_v3")?;
    encoder.metadata_string("general.type", "model")?;
    encoder.metadata_i32("general.sampling.top_k", -1)?;
    encoder.metadata_f32("general.sampling.top_p", 1.0)?;
    encoder.metadata_f32("general.sampling.temp", 0.9)?;
    encoder.metadata_string("general.name", "Hy3 Src")?;
    encoder.metadata_string("general.size_label", "192x10B")?;
    encoder.metadata_string("general.license", "apache-2.0")?;
    encoder.metadata_string_array(
        "general.tags",
        ["hunyuan", "hy3", "moe", "text-generation", "text-generation"].into_iter(),
    )?;
    encoder.metadata_u32("general.quantization_version", 2)?;
    encoder.metadata_u32("general.file_type", 29)?;
    encoder.metadata_u32("hy_v3.block_count", 80)?;
    encoder.metadata_u32("hy_v3.context_length", 1_048_576)?;
    encoder.metadata_u32("hy_v3.embedding_length", 4_096)?;
    encoder.metadata_u32("hy_v3.feed_forward_length", 13_312)?;
    encoder.metadata_u32("hy_v3.attention.head_count", 64)?;
    encoder.metadata_u32("hy_v3.attention.head_count_kv", 8)?;
    encoder.metadata_u32("hy_v3.attention.key_length", 128)?;
    encoder.metadata_u32("hy_v3.attention.value_length", 128)?;
    encoder.metadata_f32("hy_v3.attention.layer_norm_rms_epsilon", 0.000_01)?;
    encoder.metadata_f32("hy_v3.rope.freq_base", 11_158_840.0)?;
    encoder.metadata_string("hy_v3.rope.scaling.type", "yarn")?;
    encoder.metadata_f32("hy_v3.rope.scaling.factor", 4.0)?;
    encoder.metadata_u32("hy_v3.rope.scaling.original_context_length", 262_144)?;
    encoder.metadata_u32("hy_v3.expert_count", 192)?;
    encoder.metadata_u32("hy_v3.expert_used_count", 8)?;
    encoder.metadata_u32("hy_v3.expert_feed_forward_length", 1_536)?;
    encoder.metadata_u32("hy_v3.expert_shared_feed_forward_length", 1_536)?;
    encoder.metadata_bool("hy_v3.expert_weights_norm", true)?;
    encoder.metadata_f32("hy_v3.expert_weights_scale", 2.826)?;
    encoder.metadata_u32("hy_v3.expert_gating_func", 2)?;
    encoder.metadata_string("tokenizer.ggml.model", "gpt2")?;
    encoder.metadata_string("tokenizer.ggml.pre", "hunyuan-dense")?;
    encoder.metadata_u32("tokenizer.ggml.bos_token_id", 120_000)?;
    encoder.metadata_u32("tokenizer.ggml.eos_token_id", 120_025)?;
    encoder.metadata_u32("tokenizer.ggml.padding_token_id", 120_002)?;
    encoder.metadata_u32("tokenizer.ggml.seperator_token_id", 120_007)?;
    encoder.metadata_repeated_empty_strings("tokenizer.ggml.tokens", 120_832)?;
    encoder.metadata_repeated_i32("tokenizer.ggml.token_type", 1, 120_832)?;
    encoder.metadata_repeated_empty_strings("tokenizer.ggml.merges", 119_758)?;
    encoder.metadata_string("tokenizer.chat_template", "fixture template")?;
    encoder.metadata_string("quantize.imatrix.file", "/fixture/hy3.imatrix")?;
    encoder.metadata_string("quantize.imatrix.dataset", "/fixture/calibration.txt")?;
    encoder.metadata_u32("quantize.imatrix.entries_count", 876)?;
    encoder.metadata_u32("quantize.imatrix.chunks_count", 40)
}

fn write_wrong_architecture_fixture(path: &Path) -> io::Result<()> {
    let mut encoder = FixtureEncoder::new(BufWriter::new(File::create(path)?));
    encoder.bytes(b"GGUF")?;
    encoder.u32(3)?;
    encoder.u64(0)?;
    encoder.u64(1)?;
    encoder.metadata_string("general.architecture", "wrong_arch")?;
    encoder.writer.flush()?;
    let position = encoder.writer.stream_position()?;
    encoder.writer.get_ref().set_len(align_32(position))
}

fn align_32(value: u64) -> u64 {
    value.checked_add(31).unwrap() & !31
}

#[cfg(windows)]
fn mark_sparse(file: &File) -> io::Result<()> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;
    use std::ptr::{null, null_mut};

    const FSCTL_SET_SPARSE: u32 = 590_020;

    #[link(name = "kernel32")]
    extern "system" {
        fn DeviceIoControl(
            device: *mut c_void,
            control_code: u32,
            input: *const c_void,
            input_size: u32,
            output: *mut c_void,
            output_size: u32,
            bytes_returned: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
    }

    let mut bytes_returned = 0_u32;
    // SAFETY: the file handle remains valid for the call; this no-buffer control code accepts
    // null input/output buffers, and `bytes_returned` points to writable storage.
    let succeeded = unsafe {
        DeviceIoControl(
            file.as_raw_handle(),
            FSCTL_SET_SPARSE,
            null(),
            0,
            null_mut(),
            0,
            &mut bytes_returned,
            null_mut(),
        )
    };
    if succeeded == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn mark_sparse(_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(windows)]
fn allocated_size(path: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCompressedFileSizeW(path: *const u16, high: *mut u32) -> u32;
        fn GetLastError() -> u32;
        fn SetLastError(error: u32);
    }

    let wide_path = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let mut high = 0_u32;
    // SAFETY: `wide_path` is NUL-terminated and remains alive during the call; `high` is writable.
    let low = unsafe {
        SetLastError(0);
        GetCompressedFileSizeW(wide_path.as_ptr(), &mut high)
    };
    if low == u32::MAX {
        // SAFETY: GetLastError has no preconditions and is read immediately after the size call.
        let error = unsafe { GetLastError() };
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error as i32));
        }
    }
    Ok((u64::from(high) << 32) | u64::from(low))
}

#[cfg(unix)]
fn allocated_size(path: &Path) -> io::Result<u64> {
    use std::os::unix::fs::MetadataExt;

    Ok(path.metadata()?.blocks().saturating_mul(512))
}

#[cfg(not(any(unix, windows)))]
fn allocated_size(path: &Path) -> io::Result<u64> {
    Ok(path.metadata()?.len())
}

fn bridge(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bridge"))
        .args(args)
        .output()
        .unwrap()
}

fn utf8(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).unwrap()
}

fn assert_application_error(output: &Output) -> &str {
    assert!(!output.status.success());
    assert!(output.stdout.is_empty(), "stdout: {}", utf8(&output.stdout));
    let stderr = utf8(&output.stderr);
    assert!(stderr.starts_with("error:"), "stderr: {stderr}");
    assert_eq!(
        stderr.lines().count(),
        1,
        "application error must be one concise chain: {stderr}"
    );
    assert!(!stderr.contains("panicked at"), "stderr: {stderr}");
    assert!(!stderr.contains("stack backtrace"), "stderr: {stderr}");
    stderr
}

#[test]
fn temp_directories_retry_collisions_and_clean_only_their_owned_paths() {
    let parent = TestDirectory::new();
    let counter = AtomicU64::new(0);
    let nonce = 0x1234_u128;
    let stale = TestDirectory::candidate_path(&parent.path, nonce, 0);
    fs::create_dir(&stale).unwrap();
    fs::write(stale.join("sentinel"), b"stale").unwrap();

    let first = TestDirectory::create_in(&parent.path, nonce, &counter).unwrap();
    let second = TestDirectory::create_in(&parent.path, nonce, &counter).unwrap();
    let first_path = first.path.clone();
    let second_path = second.path.clone();
    fs::write(first.path("owner"), b"first").unwrap();
    fs::write(second.path("owner"), b"second").unwrap();

    assert_ne!(first_path, stale);
    assert_ne!(first_path, second_path);
    drop(first);
    assert!(!first_path.exists());
    assert!(second_path.exists());
    assert_eq!(fs::read(second_path.join("owner")).unwrap(), b"second");
    assert_eq!(fs::read(stale.join("sentinel")).unwrap(), b"stale");

    drop(second);
    assert!(!second_path.exists());
    assert!(stale.exists());
    assert!(parent.path.exists());
}

#[test]
fn valid_selected_profile_prints_stable_text_sections() {
    let directory = TestDirectory::new();
    let model = directory.path("selected.gguf");
    let header_bytes = write_selected_fixture(&model).unwrap();
    assert!(header_bytes < MAX_FIXTURE_HEADER_BYTES);

    let output = bridge(&["inspect-gguf", "--model", model.to_str().unwrap()]);

    assert!(output.status.success(), "stderr: {}", utf8(&output.stderr));
    assert!(output.stderr.is_empty(), "stderr: {}", utf8(&output.stderr));
    let stdout = utf8(&output.stdout);
    for heading in [
        "Model\n",
        "\nFiles\n",
        "\nGGUF\n",
        "\nHy3\n",
        "\nTokenizer\n",
        "\nTensor types\n",
        "\nTensor roles\n",
        "\nLayers\n",
        "\nExpert storage\n",
        "\nExecution status\n",
        "\nWarnings\n",
    ] {
        assert!(stdout.contains(heading), "missing {heading:?} in:\n{stdout}");
    }
}

#[test]
fn json_mode_emits_exactly_one_deserializable_report() {
    let directory = TestDirectory::new();
    let model = directory.path("selected.gguf");
    write_selected_fixture(&model).unwrap();

    let output = bridge(&["inspect-gguf", "--model", model.to_str().unwrap(), "--json"]);

    assert!(output.status.success(), "stderr: {}", utf8(&output.stderr));
    assert!(output.stderr.is_empty(), "stderr: {}", utf8(&output.stderr));
    assert_eq!(output.stdout.first(), Some(&b'{'));
    assert_eq!(output.stdout.last(), Some(&b'\n'));
    let report: InspectionReport = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report.hy3.block_count, 80);
    assert_eq!(report.gguf.tensor_count, SELECTED_TENSOR_COUNT);
    assert_eq!(report.files[0].logical_size, SELECTED_FILE_LEN);
}

#[test]
fn missing_model_path_is_a_concise_application_error() {
    let directory = TestDirectory::new();
    let missing = directory.path("does-not-exist.gguf");

    let output = bridge(&["inspect-gguf", "--model", missing.to_str().unwrap()]);

    let stderr = assert_application_error(&output);
    assert!(stderr.contains(missing.to_str().unwrap()), "stderr: {stderr}");
}

#[test]
fn malformed_gguf_fails_without_panic_or_backtrace() {
    let directory = TestDirectory::new();
    let malformed = directory.path("malformed.gguf");
    fs::write(&malformed, b"definitely not GGUF").unwrap();

    let output = bridge(&["inspect-gguf", "--model", malformed.to_str().unwrap()]);

    assert_application_error(&output);
}

#[test]
fn wrong_architecture_reports_the_key_and_values() {
    let directory = TestDirectory::new();
    let model = directory.path("wrong-architecture.gguf");
    write_wrong_architecture_fixture(&model).unwrap();

    let output = bridge(&["inspect-gguf", "--model", model.to_str().unwrap()]);

    let stderr = assert_application_error(&output);
    assert!(stderr.contains("general.architecture"), "stderr: {stderr}");
    assert!(stderr.contains("wrong_arch"), "stderr: {stderr}");
    assert!(stderr.contains("hy_v3"), "stderr: {stderr}");
}

#[test]
fn missing_model_flag_is_rejected_by_clap() {
    let output = bridge(&["inspect-gguf"]);

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = utf8(&output.stderr);
    assert!(stderr.contains("--model <PATH>"), "stderr: {stderr}");
    assert!(!stderr.contains("panicked at"), "stderr: {stderr}");
}

#[test]
fn unsupported_subcommand_is_rejected_by_clap() {
    let output = bridge(&["run"]);

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = utf8(&output.stderr);
    assert!(
        stderr.contains("unrecognized subcommand 'run'"),
        "stderr: {stderr}"
    );
}

#[test]
fn root_help_advertises_only_the_implemented_command() {
    let output = bridge(&["--help"]);

    assert!(output.status.success(), "stderr: {}", utf8(&output.stderr));
    assert!(output.stderr.is_empty());
    let stdout = utf8(&output.stdout);
    assert!(stdout.contains("inspect-gguf"), "stdout: {stdout}");
    for command in ["run", "serve", "chat", "prepare"] {
        assert!(
            !stdout.lines().any(|line| line.trim_start().starts_with(command)),
            "root help advertises unimplemented {command:?} command:\n{stdout}"
        );
    }
}

#[test]
fn inspect_help_documents_model_and_json_flags() {
    let output = bridge(&["inspect-gguf", "--help"]);

    assert!(output.status.success(), "stderr: {}", utf8(&output.stderr));
    assert!(output.stderr.is_empty());
    let stdout = utf8(&output.stdout);
    assert!(stdout.contains("--model <PATH>"), "stdout: {stdout}");
    assert!(stdout.contains("--json"), "stdout: {stdout}");
}
