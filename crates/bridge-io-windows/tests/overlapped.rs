#![cfg(windows)]

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bridge_io_windows::{FileBuffering, OverlappedFile, OverlappedRead, ReadCancellation, ReadSlotPool};

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lightbridge-overlapped-{}-{nonce}.bin",
            std::process::id()
        ));
        let bytes = (0..16_384).map(|index| (index % 251) as u8).collect::<Vec<_>>();
        fs::write(&path, bytes).unwrap();
        Self { path }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn buffered_iocp_submits_and_completes_a_batch() {
    let fixture = Fixture::new();
    let file = OverlappedFile::open(&fixture.path, FileBuffering::Buffered).unwrap();
    let pool = ReadSlotPool::new(2, 4096, 4096).unwrap();
    let mut first = pool.try_acquire().unwrap().unwrap();
    let mut second = pool.try_acquire().unwrap().unwrap();
    let mut reads = [
        OverlappedRead {
            offset: 0,
            buffer: &mut first.as_mut_slice()[..4096],
        },
        OverlappedRead {
            offset: 4096,
            buffer: &mut second.as_mut_slice()[..4096],
        },
    ];
    file.read_many(&mut reads, &ReadCancellation::new()).unwrap();
    assert_eq!(first.as_slice()[0], 0);
    assert_eq!(second.as_slice()[0], (4096 % 251) as u8);
}

#[test]
fn unbuffered_iocp_enforces_and_satisfies_device_alignment() {
    let fixture = Fixture::new();
    let file = OverlappedFile::open(&fixture.path, FileBuffering::Unbuffered).unwrap();
    let alignment = file.alignment();
    let slot_bytes = alignment.max(4096);
    let pool = ReadSlotPool::new(1, slot_bytes, alignment).unwrap();
    let mut lease = pool.try_acquire().unwrap().unwrap();
    let mut reads = [OverlappedRead {
        offset: 0,
        buffer: &mut lease.as_mut_slice()[..slot_bytes],
    }];
    file.read_many(&mut reads, &ReadCancellation::new()).unwrap();
    assert_eq!(lease.as_slice()[0], 0);

    let mut ordinary = vec![0_u8; slot_bytes];
    let mut invalid = [OverlappedRead {
        offset: 1,
        buffer: &mut ordinary,
    }];
    assert!(file.read_many(&mut invalid, &ReadCancellation::new()).is_err());
}
