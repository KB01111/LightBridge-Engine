//! Bounded positioned reads with cancellation and no shared file cursor.

use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStorage {
    pub logical_bytes: u64,
    pub allocated_bytes: u64,
    pub is_sparse: bool,
    pub is_compressed: bool,
}

/// Reports logical and physically allocated bytes without reading the file.
///
/// On Windows this uses `GetCompressedFileSizeW`, whose result is the actual
/// disk allocation for sparse and compressed files.
pub fn file_storage(path: impl AsRef<Path>) -> std::io::Result<FileStorage> {
    file_storage_platform(path.as_ref())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadLimits {
    pub max_request_bytes: usize,
}

impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReadCancellation {
    cancelled: Arc<AtomicBool>,
}

impl ReadCancellation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub struct PositionedFile {
    file: File,
    path: PathBuf,
    length: u64,
    limits: ReadLimits,
}

impl PositionedFile {
    pub fn open(path: impl AsRef<Path>, limits: ReadLimits) -> Result<Self, ReadError> {
        if limits.max_request_bytes == 0 {
            return Err(ReadError::ZeroRequestLimit);
        }
        let path = path.as_ref().to_owned();
        let file = File::open(&path).map_err(|source| ReadError::Open {
            path: path.clone(),
            source,
        })?;
        let metadata = file.metadata().map_err(|source| ReadError::Metadata {
            path: path.clone(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(ReadError::NotAFile(path));
        }
        Ok(Self {
            file,
            path,
            length: metadata.len(),
            limits,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn length(&self) -> u64 {
        self.length
    }

    pub const fn limits(&self) -> ReadLimits {
        self.limits
    }

    pub fn read_exact_at(
        &self,
        range: Range<u64>,
        cancellation: &ReadCancellation,
    ) -> Result<Vec<u8>, ReadError> {
        let length = self.validate_range(&range)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| ReadError::AllocationFailed { requested: length })?;
        output.resize(length, 0);
        self.read_exact_at_into(range.start, &mut output, cancellation)?;
        Ok(output)
    }

    pub fn read_exact_at_into(
        &self,
        offset: u64,
        output: &mut [u8],
        cancellation: &ReadCancellation,
    ) -> Result<(), ReadError> {
        let length = u64::try_from(output.len()).map_err(|_| ReadError::ArithmeticOverflow)?;
        let end = offset.checked_add(length).ok_or(ReadError::ArithmeticOverflow)?;
        self.validate_range(&(offset..end))?;
        if cancellation.is_cancelled() {
            return Err(ReadError::Cancelled);
        }

        let mut completed = 0_usize;
        while completed < output.len() {
            if cancellation.is_cancelled() {
                return Err(ReadError::Cancelled);
            }
            let completed_u64 = u64::try_from(completed).map_err(|_| ReadError::ArithmeticOverflow)?;
            let current_offset = offset
                .checked_add(completed_u64)
                .ok_or(ReadError::ArithmeticOverflow)?;
            let read =
                read_at_once(&self.file, &mut output[completed..], current_offset).map_err(|source| {
                    ReadError::Read {
                        path: self.path.clone(),
                        offset: current_offset,
                        source,
                    }
                })?;
            if read == 0 {
                return Err(ReadError::UnexpectedEof {
                    path: self.path.clone(),
                    offset,
                    expected: output.len(),
                    actual: completed,
                });
            }
            completed = completed.checked_add(read).ok_or(ReadError::ArithmeticOverflow)?;
        }
        Ok(())
    }

    fn validate_range(&self, range: &Range<u64>) -> Result<usize, ReadError> {
        if range.end < range.start {
            return Err(ReadError::InvertedRange {
                start: range.start,
                end: range.end,
            });
        }
        if range.end > self.length {
            return Err(ReadError::RangeOutOfBounds {
                start: range.start,
                end: range.end,
                file_length: self.length,
            });
        }
        let length_u64 = range.end - range.start;
        let length = usize::try_from(length_u64).map_err(|_| ReadError::ArithmeticOverflow)?;
        if length > self.limits.max_request_bytes {
            return Err(ReadError::RequestTooLarge {
                requested: length,
                maximum: self.limits.max_request_bytes,
            });
        }
        Ok(length)
    }
}

#[cfg(windows)]
fn read_at_once(file: &File, output: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;

    file.seek_read(output, offset)
}

#[cfg(unix)]
fn read_at_once(file: &File, output: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;

    file.read_at(output, offset)
}

#[cfg(not(any(windows, unix)))]
compile_error!("bridge-io-windows currently supports Windows and Unix positioned-read APIs");

#[cfg(windows)]
fn file_storage_platform(path: &Path) -> std::io::Result<FileStorage> {
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::MetadataExt;
    use windows_sys::Win32::Foundation::{GetLastError, SetLastError, NO_ERROR};
    use windows_sys::Win32::Storage::FileSystem::{
        GetCompressedFileSizeW, FILE_ATTRIBUTE_COMPRESSED, FILE_ATTRIBUTE_SPARSE_FILE, INVALID_FILE_SIZE,
    };

    let metadata = path.metadata()?;
    let mut wide = path.as_os_str().encode_wide().collect::<Vec<_>>();
    wide.push(0);
    let mut high = 0_u32;
    // SAFETY: `wide` is a live NUL-terminated UTF-16 path and `high` is a
    // writable DWORD. Resetting last-error is required to distinguish a valid
    // low word of `INVALID_FILE_SIZE` from failure.
    let low = unsafe {
        SetLastError(NO_ERROR);
        GetCompressedFileSizeW(wide.as_ptr(), &mut high)
    };
    if low == INVALID_FILE_SIZE {
        // SAFETY: `GetLastError` has no preconditions and immediately follows
        // the size call whose status it disambiguates.
        let error = unsafe { GetLastError() };
        if error != NO_ERROR {
            return Err(std::io::Error::from_raw_os_error(error as i32));
        }
    }
    let attributes = metadata.file_attributes();
    Ok(FileStorage {
        logical_bytes: metadata.file_size(),
        allocated_bytes: (u64::from(high) << 32) | u64::from(low),
        is_sparse: attributes & FILE_ATTRIBUTE_SPARSE_FILE != 0,
        is_compressed: attributes & FILE_ATTRIBUTE_COMPRESSED != 0,
    })
}

#[cfg(unix)]
fn file_storage_platform(path: &Path) -> std::io::Result<FileStorage> {
    use std::os::unix::fs::MetadataExt;

    let metadata = path.metadata()?;
    let allocated_bytes = metadata.blocks().saturating_mul(512);
    Ok(FileStorage {
        logical_bytes: metadata.len(),
        allocated_bytes,
        is_sparse: allocated_bytes < metadata.len(),
        is_compressed: false,
    })
}

#[cfg(not(any(windows, unix)))]
fn file_storage_platform(path: &Path) -> std::io::Result<FileStorage> {
    let metadata = path.metadata()?;
    Ok(FileStorage {
        logical_bytes: metadata.len(),
        allocated_bytes: metadata.len(),
        is_sparse: false,
        is_compressed: false,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("positioned-read request limit must be non-zero")]
    ZeroRequestLimit,
    #[error("failed to open {path:?}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to inspect {path:?}: {source}")]
    Metadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("path is not a regular file: {0:?}")]
    NotAFile(PathBuf),
    #[error("read range is inverted: {start}..{end}")]
    InvertedRange { start: u64, end: u64 },
    #[error("read range {start}..{end} exceeds file length {file_length}")]
    RangeOutOfBounds { start: u64, end: u64, file_length: u64 },
    #[error("read request is {requested} bytes, maximum is {maximum}")]
    RequestTooLarge { requested: usize, maximum: usize },
    #[error("positioned read from {path:?} at offset {offset} failed: {source}")]
    Read {
        path: PathBuf,
        offset: u64,
        #[source]
        source: std::io::Error,
    },
    #[error("unexpected end of {path:?} while reading at {offset}: expected {expected} bytes, got {actual}")]
    UnexpectedEof {
        path: PathBuf,
        offset: u64,
        expected: usize,
        actual: usize,
    },
    #[error("positioned read was cancelled")]
    Cancelled,
    #[error("checked arithmetic overflow while sizing positioned read")]
    ArithmeticOverflow,
    #[error("allocation failed while reserving {requested} positioned-read bytes")]
    AllocationFailed { requested: usize },
}
