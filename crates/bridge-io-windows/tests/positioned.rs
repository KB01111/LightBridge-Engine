use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bridge_io_windows::{file_storage, PositionedFile, ReadCancellation, ReadError, ReadLimits};

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(bytes: &[u8]) -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lightbridge-positioned-{}-{nonce}.bin",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();
        Self { path }
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn reads_exact_bounded_ranges_without_moving_a_shared_cursor() {
    let fixture = TempFile::new(b"0123456789abcdef");
    let file = PositionedFile::open(&fixture.path, ReadLimits { max_request_bytes: 8 }).unwrap();
    let cancellation = ReadCancellation::new();
    assert_eq!(file.read_exact_at(4..10, &cancellation).unwrap(), b"456789");
    assert_eq!(file.read_exact_at(0..4, &cancellation).unwrap(), b"0123");
}

#[test]
fn rejects_oversized_and_out_of_bounds_requests_before_allocation() {
    let fixture = TempFile::new(b"0123456789abcdef");
    let file = PositionedFile::open(&fixture.path, ReadLimits { max_request_bytes: 4 }).unwrap();
    let cancellation = ReadCancellation::new();
    assert!(matches!(
        file.read_exact_at(0..5, &cancellation),
        Err(ReadError::RequestTooLarge {
            requested: 5,
            maximum: 4
        })
    ));
    assert!(matches!(
        file.read_exact_at(15..17, &cancellation),
        Err(ReadError::RangeOutOfBounds { .. })
    ));
    assert!(matches!(
        file.read_exact_at(std::ops::Range { start: 8, end: 7 }, &cancellation),
        Err(ReadError::InvertedRange { .. })
    ));
}

#[test]
fn cancellation_is_observed_before_reading() {
    let fixture = TempFile::new(b"0123456789abcdef");
    let file = PositionedFile::open(&fixture.path, ReadLimits::default()).unwrap();
    let cancellation = ReadCancellation::new();
    cancellation.cancel();
    assert!(matches!(
        file.read_exact_at(0..4, &cancellation),
        Err(ReadError::Cancelled)
    ));
}

#[test]
fn concurrent_reads_are_independent() {
    let fixture = TempFile::new(b"0123456789abcdef");
    let file = Arc::new(PositionedFile::open(&fixture.path, ReadLimits::default()).unwrap());
    let left = {
        let file = Arc::clone(&file);
        std::thread::spawn(move || file.read_exact_at(0..8, &ReadCancellation::new()).unwrap())
    };
    let right = {
        let file = Arc::clone(&file);
        std::thread::spawn(move || file.read_exact_at(8..16, &ReadCancellation::new()).unwrap())
    };
    assert_eq!(left.join().unwrap(), b"01234567");
    assert_eq!(right.join().unwrap(), b"89abcdef");
}

#[test]
fn reports_logical_and_allocated_storage_without_reading_payload() {
    let fixture = TempFile::new(b"0123456789abcdef");
    let storage = file_storage(&fixture.path).unwrap();
    assert_eq!(storage.logical_bytes, 16);
    assert!(storage.allocated_bytes >= storage.logical_bytes);
}
