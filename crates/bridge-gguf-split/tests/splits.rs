use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use bridge_gguf::{GgufError, GgufValueType, MetadataError};
use bridge_gguf_split::{open_set, SplitError};

static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let number = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("bridge-gguf-split-{}-{number}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct TensorFixture {
    name: &'static str,
    offset: u64,
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    const fn magic(self) -> [u8; 4] {
        match self {
            Self::Little => *b"GGUF",
            Self::Big => *b"FUGG",
        }
    }

    const fn u16(self, value: u16) -> [u8; 2] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }

    const fn u32(self, value: u32) -> [u8; 4] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }

    const fn i32(self, value: i32) -> [u8; 4] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }

    const fn u64(self, value: u64) -> [u8; 8] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }
}

struct Fixture {
    order: ByteOrder,
    version: u32,
    metadata: Vec<(&'static str, u32, Vec<u8>)>,
    tensors: Vec<TensorFixture>,
    payload_bytes: usize,
    alignment: usize,
}

impl Fixture {
    fn new() -> Self {
        Self {
            order: ByteOrder::Little,
            version: 3,
            metadata: Vec::new(),
            tensors: Vec::new(),
            payload_bytes: 0,
            alignment: 32,
        }
    }

    fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    fn big_endian(mut self) -> Self {
        self.order = ByteOrder::Big;
        self
    }

    fn alignment(mut self, alignment: u32) -> Self {
        self.metadata
            .push(("general.alignment", 4, self.order.u32(alignment).to_vec()));
        self.alignment = usize::try_from(alignment).unwrap();
        self
    }

    fn split(mut self, ordinal: u16, count: u16) -> Self {
        self.metadata
            .push(("split.no", 2, self.order.u16(ordinal).to_vec()));
        self.metadata
            .push(("split.count", 2, self.order.u16(count).to_vec()));
        self
    }

    fn split_count(mut self, count: i32) -> Self {
        self.metadata
            .push(("split.tensors.count", 5, self.order.i32(count).to_vec()));
        self
    }

    fn metadata(mut self, key: &'static str, ty: u32, bytes: Vec<u8>) -> Self {
        self.metadata.push((key, ty, bytes));
        self
    }

    fn tensor(mut self, name: &'static str, offset: u64) -> Self {
        self.tensors.push(TensorFixture { name, offset });
        self.payload_bytes = self.payload_bytes.max(
            usize::try_from(offset)
                .unwrap()
                .checked_add(self.alignment)
                .unwrap(),
        );
        self
    }

    fn payload_bytes(mut self, bytes: usize) -> Self {
        self.payload_bytes = bytes;
        self
    }

    fn write_to(&self, path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend(self.order.magic());
        bytes.extend(self.order.u32(self.version));
        bytes.extend(self.order.u64(u64::try_from(self.tensors.len()).unwrap()));
        bytes.extend(self.order.u64(u64::try_from(self.metadata.len()).unwrap()));
        for (key, ty, value) in &self.metadata {
            write_string(&mut bytes, key, self.order);
            bytes.extend(self.order.u32(*ty));
            bytes.extend(value);
        }
        for tensor in &self.tensors {
            write_string(&mut bytes, tensor.name, self.order);
            bytes.extend(self.order.u32(1));
            bytes.extend(self.order.u64(1));
            bytes.extend(self.order.u32(0));
            bytes.extend(self.order.u64(tensor.offset));
        }
        let data_offset = bytes
            .len()
            .checked_add(self.alignment.checked_sub(1).unwrap())
            .unwrap()
            & !self.alignment.checked_sub(1).unwrap();
        bytes.resize(data_offset, 0);
        bytes.resize(data_offset.checked_add(self.payload_bytes).unwrap(), 0);
        fs::write(path, bytes).unwrap();
    }
}

fn write_string(bytes: &mut Vec<u8>, value: &str, order: ByteOrder) {
    bytes.extend(order.u64(u64::try_from(value.len()).unwrap()));
    bytes.extend(value.as_bytes());
}

fn write_split(directory: &TempDir, name: &str, fixture: Fixture) -> PathBuf {
    let path = directory.file(name);
    fixture.write_to(&path);
    path
}

fn create_file_symlink(target: &Path, link: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link)
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
}

fn symlinks_available(target: &Path, link: &Path) -> bool {
    match create_file_symlink(target, link) {
        Ok(()) => true,
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
            eprintln!(
                "skipping symlink assertion because this Windows account lacks symlink permission: {error}"
            );
            false
        }
        Err(error) => panic!("failed to create test file symlink: {error}"),
    }
}

#[test]
fn opens_an_ordinary_non_numbered_single_file() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "model.gguf", Fixture::new().tensor("only", 0));

    let set = open_set(&entry).unwrap();

    assert_eq!(set.files().len(), 1);
    assert_eq!(set.files()[0].ordinal(), 0);
    assert_eq!(set.files()[0].count(), 1);
    assert_eq!(set.tensors().ordered()[0].descriptor().name(), "only");
}

#[test]
fn accepts_any_numbered_member_and_orders_shards_by_ordinal() {
    let directory = TempDir::new();
    write_split(
        &directory,
        "name-00003-of-00003.gguf",
        Fixture::new().split(2, 3).tensor("third", 0),
    );
    let entry = write_split(
        &directory,
        "name-00001-of-00003.gguf",
        Fixture::new().split(0, 3).tensor("first", 0),
    );
    write_split(
        &directory,
        "name-00002-of-00003.gguf",
        Fixture::new().split(1, 3).tensor("second", 0),
    );

    let set = open_set(directory.file("name-00003-of-00003.gguf")).unwrap();

    assert_eq!(
        set.files()
            .iter()
            .map(|shard| shard.ordinal())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        set.tensors()
            .ordered()
            .iter()
            .map(|location| location.descriptor().name())
            .collect::<Vec<_>>(),
        vec!["first", "second", "third"]
    );
    assert_eq!(set.files()[0].path(), entry.canonicalize().unwrap());
}

#[test]
fn rejects_missing_expected_shard() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00001-of-00003.gguf", Fixture::new().split(0, 3));
    write_split(&directory, "name-00003-of-00003.gguf", Fixture::new().split(2, 3));
    assert!(matches!(open_set(entry), Err(SplitError::MissingShard(_))));
}

#[test]
fn rejects_zero_filename_index() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00000-of-00003.gguf", Fixture::new().split(0, 3));
    assert!(matches!(
        open_set(entry),
        Err(SplitError::InvalidNumberedFilename(_))
    ));
}

#[test]
fn rejects_filename_index_greater_than_total() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00004-of-00003.gguf", Fixture::new().split(3, 3));
    assert!(matches!(
        open_set(entry),
        Err(SplitError::InvalidNumberedFilename(_))
    ));
}

#[test]
fn rejects_inconsistent_total_count_in_sibling_filename() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00001-of-00002.gguf", Fixture::new().split(0, 2));
    write_split(&directory, "name-00002-of-00003.gguf", Fixture::new().split(1, 3));
    assert!(matches!(open_set(entry), Err(SplitError::MissingShard(_))));
}

#[test]
fn rejects_duplicate_effective_ordinal_from_metadata() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00001-of-00002.gguf", Fixture::new().split(0, 2));
    write_split(&directory, "name-00002-of-00002.gguf", Fixture::new().split(0, 2));
    assert!(matches!(
        open_set(entry),
        Err(SplitError::SplitMetadataDisagreement { key: "split.no", .. })
    ));
}

#[test]
fn rejects_filename_and_split_number_disagreement() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00001-of-00002.gguf", Fixture::new().split(1, 2));
    write_split(&directory, "name-00002-of-00002.gguf", Fixture::new().split(1, 2));
    assert!(matches!(
        open_set(entry),
        Err(SplitError::SplitMetadataDisagreement { key: "split.no", .. })
    ));
}

#[test]
fn rejects_filename_and_split_count_disagreement() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00001-of-00002.gguf", Fixture::new().split(0, 3));
    write_split(&directory, "name-00002-of-00002.gguf", Fixture::new().split(1, 2));
    assert!(matches!(
        open_set(entry),
        Err(SplitError::SplitMetadataDisagreement {
            key: "split.count",
            ..
        })
    ));
}

#[test]
fn rejects_aggregate_tensor_count_disagreement() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "name-00001-of-00002.gguf",
        Fixture::new().split(0, 2).split_count(3).tensor("one", 0),
    );
    write_split(
        &directory,
        "name-00002-of-00002.gguf",
        Fixture::new().split(1, 2).split_count(3).tensor("two", 0),
    );
    assert!(matches!(
        open_set(entry),
        Err(SplitError::AggregateTensorDirectoryDisagreement)
    ));
}

#[test]
fn rejects_duplicate_tensor_names_across_shards() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "name-00001-of-00002.gguf",
        Fixture::new().split(0, 2).tensor("same", 0),
    );
    write_split(
        &directory,
        "name-00002-of-00002.gguf",
        Fixture::new().split(1, 2).tensor("same", 0),
    );
    assert!(matches!(open_set(entry), Err(SplitError::DuplicateTensorName(name)) if name == "same"));
}

#[test]
fn rejects_a_nonzero_first_tensor_offset_at_the_parser_boundary() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "model.gguf",
        Fixture::new().tensor("past-end", 8).payload_bytes(0),
    );
    assert!(matches!(
        open_set(entry),
        Err(SplitError::Gguf(GgufError::TensorOffsetMismatch {
            name,
            actual_offset: 8,
            expected_offset: 0,
            alignment: 32,
        })) if name == "past-end"
    ));
}

#[test]
fn rejects_non_file_input() {
    let directory = TempDir::new();
    assert!(matches!(open_set(&directory.0), Err(SplitError::NotAFile(_))));
}

#[test]
fn ignores_similarly_named_unrelated_siblings() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00001-of-00002.gguf", Fixture::new().split(0, 2));
    write_split(&directory, "name-00002-of-00002.gguf", Fixture::new().split(1, 2));
    write_split(&directory, "name-00003-of-00003.gguf", Fixture::new().split(2, 3));

    let set = open_set(entry).unwrap();

    assert_eq!(set.files().len(), 2);
}

#[test]
fn rejects_missing_required_split_metadata() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "name-00001-of-00002.gguf", Fixture::new());
    write_split(&directory, "name-00002-of-00002.gguf", Fixture::new().split(1, 2));
    assert!(matches!(
        open_set(entry),
        Err(SplitError::MissingSplitMetadata { key: "split.no", .. })
    ));
}

#[test]
fn rejects_wrongly_typed_required_split_metadata() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "typed-00001-of-00002.gguf",
        Fixture::new()
            .metadata("split.no", 4, 0_u32.to_le_bytes().to_vec())
            .metadata("split.count", 2, 2_u16.to_le_bytes().to_vec()),
    );
    write_split(
        &directory,
        "typed-00002-of-00002.gguf",
        Fixture::new().split(1, 2),
    );
    assert!(matches!(
        open_set(entry),
        Err(SplitError::Metadata(MetadataError::WrongType {
            key,
            expected: GgufValueType::U16,
            actual: GgufValueType::U32,
        })) if key == "split.no"
    ));
}

#[test]
fn rejects_an_ordinary_entry_symlink_that_escapes_its_caller_parent() {
    let input = TempDir::new();
    let outside = TempDir::new();
    let target = write_split(&outside, "model.gguf", Fixture::new());
    let entry = input.file("model.gguf");
    if !symlinks_available(&target, &entry) {
        return;
    }

    assert!(matches!(open_set(entry), Err(SplitError::ShardEscapesParent(_))));
}

#[test]
fn rejects_a_numbered_entry_symlink_that_escapes_its_caller_parent() {
    let input = TempDir::new();
    let outside = TempDir::new();
    let target = write_split(&outside, "model-00001-of-00002.gguf", Fixture::new().split(0, 2));
    write_split(&outside, "model-00002-of-00002.gguf", Fixture::new().split(1, 2));
    let entry = input.file("model-00001-of-00002.gguf");
    if !symlinks_available(&target, &entry) {
        return;
    }

    assert!(matches!(open_set(entry), Err(SplitError::ShardEscapesParent(_))));
}

#[test]
fn rejects_an_expected_sibling_symlink_that_escapes_the_entry_parent() {
    let directory = TempDir::new();
    let outside = TempDir::new();
    let entry = write_split(
        &directory,
        "model-00001-of-00002.gguf",
        Fixture::new().split(0, 2),
    );
    let target = write_split(&outside, "model-00002-of-00002.gguf", Fixture::new().split(1, 2));
    let sibling = directory.file("model-00002-of-00002.gguf");
    if !symlinks_available(&target, &sibling) {
        return;
    }

    assert!(matches!(open_set(entry), Err(SplitError::ShardEscapesParent(_))));
}

#[test]
fn treats_non_five_digit_split_like_names_as_ordinary_files() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "model-1-of-2.gguf", Fixture::new().tensor("only", 0));

    let set = open_set(entry).unwrap();

    assert_eq!(set.files().len(), 1);
    assert_eq!(set.files()[0].count(), 1);
}

#[test]
fn rejects_five_digit_totals_that_do_not_fit_split_count_u16() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "model-00001-of-65536.gguf", Fixture::new());

    assert!(
        matches!(open_set(entry), Err(SplitError::InvalidNumberedFilename(name)) if name == "model-00001-of-65536.gguf")
    );
}

#[test]
fn rejects_single_file_optional_split_metadata_that_contradicts_ordinal_or_count() {
    let directory = TempDir::new();
    let ordinal = write_split(&directory, "ordinal.gguf", Fixture::new().split(1, 1));
    let count = write_split(&directory, "count.gguf", Fixture::new().split(0, 2));

    assert!(matches!(
        open_set(ordinal),
        Err(SplitError::SplitMetadataDisagreement { key: "split.no", .. })
    ));
    assert!(matches!(
        open_set(count),
        Err(SplitError::SplitMetadataDisagreement {
            key: "split.count",
            ..
        })
    ));
}

#[test]
fn rejects_negative_or_wrongly_typed_aggregate_tensor_counts() {
    let directory = TempDir::new();
    let negative = write_split(&directory, "negative.gguf", Fixture::new().split_count(-1));
    let wrong_type = write_split(
        &directory,
        "wrong-type.gguf",
        Fixture::new().metadata("split.tensors.count", 4, 1_u32.to_le_bytes().to_vec()),
    );

    assert!(matches!(
        open_set(negative),
        Err(SplitError::NegativeAggregateTensorCount(_))
    ));
    assert!(matches!(
        open_set(wrong_type),
        Err(SplitError::Metadata(MetadataError::WrongType {
            key,
            expected: GgufValueType::I32,
            actual: GgufValueType::U32,
        })) if key == "split.tensors.count"
    ));
}

#[test]
fn rejects_different_declared_aggregate_counts_between_shards() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "model-00001-of-00002.gguf",
        Fixture::new().split(0, 2).split_count(2).tensor("one", 0),
    );
    write_split(
        &directory,
        "model-00002-of-00002.gguf",
        Fixture::new().split(1, 2).split_count(3).tensor("two", 0),
    );

    assert!(matches!(
        open_set(entry),
        Err(SplitError::AggregateTensorCountDisagreement)
    ));
}

#[test]
fn rejects_an_unaligned_first_tensor_offset_at_the_parser_boundary() {
    let directory = TempDir::new();
    let entry = write_split(&directory, "unaligned.gguf", Fixture::new().tensor("bad", 4));

    assert!(matches!(
        open_set(entry),
        Err(SplitError::Gguf(GgufError::TensorOffsetMismatch {
            name,
            actual_offset: 4,
            expected_offset: 0,
            alignment: 32,
        })) if name == "bad"
    ));
}

#[test]
fn rejects_shards_with_different_gguf_versions() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "model-00001-of-00002.gguf",
        Fixture::new().split(0, 2),
    );
    let second = write_split(
        &directory,
        "model-00002-of-00002.gguf",
        Fixture::new().version(2).split(1, 2),
    );

    let error = open_set(entry).expect_err("heterogeneous GGUF versions must be rejected");

    assert_eq!(
        error.to_string(),
        format!(
            "GGUF shard {:?} has version 2, expected common version 3",
            second.canonicalize().unwrap()
        )
    );
}

#[test]
fn rejects_shards_with_different_endianness() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "model-00001-of-00002.gguf",
        Fixture::new().split(0, 2),
    );
    let second = write_split(
        &directory,
        "model-00002-of-00002.gguf",
        Fixture::new().big_endian().split(1, 2),
    );

    let error = open_set(entry).expect_err("heterogeneous GGUF endianness must be rejected");

    assert_eq!(
        error.to_string(),
        format!(
            "GGUF shard {:?} has endianness Big, expected common endianness Little",
            second.canonicalize().unwrap()
        )
    );
}

#[test]
fn rejects_shards_with_different_alignment() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "model-00001-of-00002.gguf",
        Fixture::new().split(0, 2),
    );
    let second = write_split(
        &directory,
        "model-00002-of-00002.gguf",
        Fixture::new().alignment(64).split(1, 2),
    );

    let error = open_set(entry).expect_err("heterogeneous GGUF alignments must be rejected");

    assert_eq!(
        error.to_string(),
        format!(
            "GGUF shard {:?} has alignment 64, expected common alignment 32",
            second.canonicalize().unwrap()
        )
    );
}

#[test]
fn accepts_shards_with_different_metadata_counts() {
    let directory = TempDir::new();
    let entry = write_split(
        &directory,
        "model-00001-of-00002.gguf",
        Fixture::new()
            .split(0, 2)
            .metadata("extra", 4, 7_u32.to_le_bytes().to_vec()),
    );
    write_split(
        &directory,
        "model-00002-of-00002.gguf",
        Fixture::new().split(1, 2),
    );

    let set = open_set(entry).unwrap();

    assert_eq!(set.files()[0].parsed().metadata.len(), 3);
    assert_eq!(set.files()[1].parsed().metadata.len(), 2);
}
