use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bridge_format::{ExpertKey, ExpertLayout, Sidecar};
use bridge_gguf_split::open_set;
use bridge_io_windows::ReadCancellation;
use bridge_model_hy3::validate_model_with_profile;
use bridge_prepare::{
    prepare_sidecar, tensor_directory_sha256, verify_source_bindings, DirectExpertIndex, DirectExpertStore,
    PrepareOptions,
};
use bridge_test_model::{ReducedHy3Model, EXPERT_COUNT};

struct TempDirectory {
    path: PathBuf,
}

impl TempDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("lightbridge-prepare-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self { path }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn fixture(
    directory: &TempDirectory,
) -> (
    ReducedHy3Model,
    bridge_gguf_split::GgufSet,
    bridge_model_hy3::ValidatedHy3Model,
) {
    let model = ReducedHy3Model::new().unwrap();
    let source = directory.path("reduced.gguf");
    fs::write(&source, model.gguf_bytes().unwrap()).unwrap();
    let set = open_set(&source).unwrap();
    let validated = validate_model_with_profile(&set, &model.profile().unwrap()).unwrap();
    (model, set, validated)
}

fn assert_expert_equal(direct: &bridge_prepare::DirectExpertBytes, sidecar: &bridge_format::ExpertBytes) {
    assert_eq!(sidecar.gate(), direct.gate);
    assert_eq!(sidecar.up(), direct.up);
    assert_eq!(sidecar.down(), direct.down);
}

#[test]
fn direct_index_and_reads_use_exact_validated_expert_slabs() {
    let directory = TempDirectory::new();
    let (_model, set, validated) = fixture(&directory);
    let index = DirectExpertIndex::build(&validated).unwrap();
    assert_eq!(index.records().len(), EXPERT_COUNT);
    let store = DirectExpertStore::open(&set, &validated).unwrap();
    let key = ExpertKey { layer: 1, expert: 2 };
    let bytes = store.read_expert(key, &ReadCancellation::new()).unwrap();
    assert!(!bytes.gate.is_empty());
    assert!(!bytes.up.is_empty());
    assert!(!bytes.down.is_empty());
    let record = store.index().get(key).unwrap();
    assert_eq!(bytes.gate.len() as u64, record.gate.length());
    assert_eq!(bytes.up.len() as u64, record.up.length());
    assert_eq!(bytes.down.len() as u64, record.down.length());
}

#[test]
fn both_sidecar_layouts_are_lossless_and_source_bound() {
    for layout in [ExpertLayout::Sequential, ExpertLayout::FusedGateUp] {
        let directory = TempDirectory::new();
        let (_model, set, validated) = fixture(&directory);
        let data = directory.path("experts.bin");
        let manifest_path = directory.path("experts.json");
        let report = prepare_sidecar(
            &set,
            &validated,
            &data,
            &manifest_path,
            PrepareOptions {
                layout,
                alignment: 64,
                hash_chunk_bytes: 4096,
                ..PrepareOptions::default()
            },
            &ReadCancellation::new(),
        )
        .unwrap();
        assert_eq!(report.record_count, EXPERT_COUNT);
        assert_eq!(
            report.tensor_directory_sha256,
            tensor_directory_sha256(&set).unwrap()
        );

        let sidecar = Sidecar::open(&data, &manifest_path).unwrap();
        sidecar.verify_data_hash(&ReadCancellation::new()).unwrap();
        verify_source_bindings(&set, sidecar.manifest(), 4096, &ReadCancellation::new()).unwrap();

        let direct = DirectExpertStore::open(&set, &validated).unwrap();
        for expert in 0..EXPERT_COUNT as u32 {
            let key = ExpertKey { layer: 1, expert };
            assert_expert_equal(
                &direct.read_expert(key, &ReadCancellation::new()).unwrap(),
                &sidecar.read_expert(key, &ReadCancellation::new()).unwrap(),
            );
        }
    }
}

#[test]
fn cancellation_and_existing_outputs_fail_without_partial_replacement() {
    let directory = TempDirectory::new();
    let (_model, set, validated) = fixture(&directory);
    let data = directory.path("experts.bin");
    let manifest = directory.path("experts.json");
    fs::write(&data, b"keep").unwrap();
    let error = prepare_sidecar(
        &set,
        &validated,
        &data,
        &manifest,
        PrepareOptions::default(),
        &ReadCancellation::new(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("already exists"));
    assert_eq!(fs::read(&data).unwrap(), b"keep");
    assert!(!manifest.exists());

    let data = directory.path("cancelled.bin");
    let manifest = directory.path("cancelled.json");
    let cancellation = ReadCancellation::new();
    cancellation.cancel();
    let error = prepare_sidecar(
        &set,
        &validated,
        &data,
        &manifest,
        PrepareOptions::default(),
        &cancellation,
    )
    .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    assert!(!data.exists());
    assert!(!manifest.exists());
}

#[test]
fn overwrite_replaces_both_outputs_only_after_verification() {
    let directory = TempDirectory::new();
    let (_model, set, validated) = fixture(&directory);
    let data = directory.path("experts.bin");
    let manifest = directory.path("experts.json");
    fs::write(&data, b"old-data").unwrap();
    fs::write(&manifest, b"old-manifest").unwrap();
    prepare_sidecar(
        &set,
        &validated,
        &data,
        &manifest,
        PrepareOptions {
            overwrite: true,
            alignment: 64,
            hash_chunk_bytes: 4096,
            ..PrepareOptions::default()
        },
        &ReadCancellation::new(),
    )
    .unwrap();
    Sidecar::open(&data, &manifest).unwrap();
    assert_ne!(fs::read(&data).unwrap(), b"old-data");
}

#[test]
fn source_binding_detects_same_length_tampering() {
    let directory = TempDirectory::new();
    let (_model, set, validated) = fixture(&directory);
    let data = directory.path("experts.bin");
    let manifest_path = directory.path("experts.json");
    prepare_sidecar(
        &set,
        &validated,
        &data,
        &manifest_path,
        PrepareOptions {
            alignment: 64,
            hash_chunk_bytes: 4096,
            ..PrepareOptions::default()
        },
        &ReadCancellation::new(),
    )
    .unwrap();
    let sidecar = Sidecar::open(&data, &manifest_path).unwrap();

    let source = set.files()[0].path();
    overwrite_one_byte(source);
    let error = verify_source_bindings(&set, sidecar.manifest(), 4096, &ReadCancellation::new()).unwrap_err();
    assert!(error.to_string().contains("identity mismatch"));
}

fn overwrite_one_byte(path: &Path) {
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut file = fs::OpenOptions::new().read(true).write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 1;
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();
}
