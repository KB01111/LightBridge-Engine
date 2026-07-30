//! Bounded positioned reads with cancellation and no shared file cursor.

use std::alloc::{alloc, dealloc, Layout};
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::time::Duration;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadSlotToken {
    pub index: usize,
    pub generation: u64,
}

#[derive(Debug, Clone)]
pub struct ReadSlotPool {
    inner: Arc<ReadSlotPoolInner>,
}

#[derive(Debug)]
struct ReadSlotPoolInner {
    state: Mutex<ReadSlotPoolState>,
    changed: Condvar,
    slot_bytes: usize,
    alignment: usize,
    poison: u8,
}

#[derive(Debug)]
struct ReadSlotPoolState {
    available: Vec<(usize, AlignedBuffer)>,
    allocated: usize,
    generations: Vec<u64>,
    leased: Vec<bool>,
}

impl ReadSlotPool {
    pub fn new(slot_count: usize, slot_bytes: usize, alignment: usize) -> Result<Self, SlotPoolError> {
        if slot_count == 0 {
            return Err(SlotPoolError::ZeroSlots);
        }
        if slot_bytes == 0 {
            return Err(SlotPoolError::ZeroSlotBytes);
        }
        if !alignment.is_power_of_two() {
            return Err(SlotPoolError::InvalidAlignment(alignment));
        }
        let mut available = Vec::new();
        available
            .try_reserve_exact(slot_count.min(64))
            .map_err(|_| SlotPoolError::AllocationFailed)?;
        let mut generations = Vec::new();
        generations
            .try_reserve_exact(slot_count)
            .map_err(|_| SlotPoolError::AllocationFailed)?;
        generations.resize(slot_count, 0);
        let mut leased = Vec::new();
        leased
            .try_reserve_exact(slot_count)
            .map_err(|_| SlotPoolError::AllocationFailed)?;
        leased.resize(slot_count, false);
        Ok(Self {
            inner: Arc::new(ReadSlotPoolInner {
                state: Mutex::new(ReadSlotPoolState {
                    available,
                    allocated: 0,
                    generations,
                    leased,
                }),
                changed: Condvar::new(),
                slot_bytes,
                alignment,
                poison: 0xdd,
            }),
        })
    }

    pub fn slot_bytes(&self) -> usize {
        self.inner.slot_bytes
    }

    pub fn alignment(&self) -> usize {
        self.inner.alignment
    }

    pub fn try_acquire(&self) -> Result<Option<ReadSlotLease>, SlotPoolError> {
        let mut state = self.inner.state.lock().map_err(|_| SlotPoolError::Poisoned)?;
        acquire_locked(&self.inner, &mut state)
    }

    pub fn acquire(&self, cancellation: &ReadCancellation) -> Result<ReadSlotLease, SlotPoolError> {
        let mut state = self.inner.state.lock().map_err(|_| SlotPoolError::Poisoned)?;
        loop {
            if cancellation.is_cancelled() {
                return Err(SlotPoolError::Cancelled);
            }
            if let Some(lease) = acquire_locked(&self.inner, &mut state)? {
                return Ok(lease);
            }
            let (next, _) = self
                .inner
                .changed
                .wait_timeout(state, Duration::from_millis(10))
                .map_err(|_| SlotPoolError::Poisoned)?;
            state = next;
        }
    }

    pub fn is_current(&self, token: ReadSlotToken) -> Result<bool, SlotPoolError> {
        let state = self.inner.state.lock().map_err(|_| SlotPoolError::Poisoned)?;
        Ok(state
            .generations
            .get(token.index)
            .zip(state.leased.get(token.index))
            .is_some_and(|(&generation, &leased)| leased && generation == token.generation))
    }
}

fn acquire_locked(
    inner: &Arc<ReadSlotPoolInner>,
    state: &mut ReadSlotPoolState,
) -> Result<Option<ReadSlotLease>, SlotPoolError> {
    let (index, buffer) = if let Some(available) = state.available.pop() {
        available
    } else {
        if state.allocated == state.generations.len() {
            return Ok(None);
        }
        let index = state.allocated;
        let buffer = AlignedBuffer::new(inner.slot_bytes, inner.alignment)?;
        state.allocated += 1;
        (index, buffer)
    };
    let generation = state.generations[index]
        .checked_add(1)
        .ok_or(SlotPoolError::GenerationOverflow)?;
    state.generations[index] = generation;
    state.leased[index] = true;
    Ok(Some(ReadSlotLease {
        inner: Arc::clone(inner),
        token: ReadSlotToken { index, generation },
        buffer: Some(buffer),
    }))
}

#[derive(Debug)]
pub struct ReadSlotLease {
    inner: Arc<ReadSlotPoolInner>,
    token: ReadSlotToken,
    buffer: Option<AlignedBuffer>,
}

impl ReadSlotLease {
    pub const fn token(&self) -> ReadSlotToken {
        self.token
    }

    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().expect("live lease owns its slot").as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer
            .as_mut()
            .expect("live lease owns its slot")
            .as_mut_slice()
    }

    pub fn address(&self) -> usize {
        self.buffer.as_ref().expect("live lease owns its slot").address()
    }
}

impl Drop for ReadSlotLease {
    fn drop(&mut self) {
        let Some(mut buffer) = self.buffer.take() else {
            return;
        };
        buffer.as_mut_slice().fill(self.inner.poison);
        let Ok(mut state) = self.inner.state.lock() else {
            return;
        };
        state.leased[self.token.index] = false;
        state.available.push((self.token.index, buffer));
        self.inner.changed.notify_one();
    }
}

struct AlignedBuffer {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl AlignedBuffer {
    fn new(length: usize, alignment: usize) -> Result<Self, SlotPoolError> {
        let layout = Layout::from_size_align(length, alignment)
            .map_err(|_| SlotPoolError::InvalidAlignment(alignment))?;
        // SAFETY: `layout` is non-zero and valid. The allocation is owned by
        // this value and released with the same layout in `Drop`.
        let pointer = unsafe { alloc(layout) };
        let pointer = NonNull::new(pointer).ok_or(SlotPoolError::AllocationFailed)?;
        // SAFETY: the allocation covers exactly `layout.size()` writable
        // bytes and is exclusively owned.
        unsafe { std::ptr::write_bytes(pointer.as_ptr(), 0, layout.size()) };
        Ok(Self { pointer, layout })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is live for `self`, initialized, and no
        // mutable borrow exists while `&self` is held.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` guarantees exclusive access to the live
        // allocation.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }

    fn address(&self) -> usize {
        self.pointer.as_ptr() as usize
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AlignedBuffer")
            .field("length", &self.layout.size())
            .field("alignment", &self.layout.align())
            .finish_non_exhaustive()
    }
}

// SAFETY: the buffer has unique ownership and contains only bytes.
unsafe impl Send for AlignedBuffer {}
// SAFETY: shared access exposes only immutable bytes; mutation requires the
// unique `&mut AlignedBuffer` held by a live slot lease.
unsafe impl Sync for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: this is the same live pointer and layout returned by
        // `alloc`, and `Drop` runs exactly once.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

#[derive(Debug)]
pub struct PositionedFile {
    file: File,
    path: PathBuf,
    length: u64,
    limits: ReadLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileBuffering {
    Buffered,
    Unbuffered,
}

#[cfg(windows)]
#[derive(Debug)]
pub struct OverlappedFile {
    file: File,
    completion_port: windows_sys::Win32::Foundation::HANDLE,
    completion_lock: Mutex<()>,
    path: PathBuf,
    length: u64,
    buffering: FileBuffering,
    alignment: usize,
}

#[cfg(windows)]
pub struct OverlappedRead<'a> {
    pub offset: u64,
    pub buffer: &'a mut [u8],
}

#[cfg(windows)]
impl std::fmt::Debug for OverlappedRead<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OverlappedRead")
            .field("offset", &self.offset)
            .field("length", &self.buffer.len())
            .finish()
    }
}

#[cfg(windows)]
impl OverlappedFile {
    pub fn open(path: impl AsRef<Path>, buffering: FileBuffering) -> Result<Self, ReadError> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::Storage::FileSystem::{FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED};
        use windows_sys::Win32::System::IO::CreateIoCompletionPort;

        let path = path.as_ref().to_owned();
        let flags = FILE_FLAG_OVERLAPPED
            | if buffering == FileBuffering::Unbuffered {
                FILE_FLAG_NO_BUFFERING
            } else {
                0
            };
        let file = fs_open_overlapped(&path, flags)?;
        let metadata = file.metadata().map_err(|source| ReadError::Metadata {
            path: path.clone(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(ReadError::NotAFile(path));
        }
        let alignment = file_alignment(&file, buffering == FileBuffering::Unbuffered)?;
        // SAFETY: `INVALID_HANDLE_VALUE` requests a new completion port and
        // all other arguments follow the Win32 IOCP creation contract.
        let completion_port =
            unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 1) };
        if completion_port.is_null() {
            return Err(ReadError::CompletionPort {
                path,
                operation: "create",
                source: std::io::Error::last_os_error(),
            });
        }
        // SAFETY: `file` is an overlapped handle and `completion_port` is a
        // live IOCP created above.
        let associated =
            unsafe { CreateIoCompletionPort(file.as_raw_handle().cast(), completion_port, 0, 0) };
        if associated.is_null() {
            let source = std::io::Error::last_os_error();
            // SAFETY: the port was created successfully and has not been
            // transferred into the returned owner.
            unsafe { CloseHandle(completion_port) };
            return Err(ReadError::CompletionPort {
                path,
                operation: "associate file",
                source,
            });
        }
        Ok(Self {
            file,
            completion_port,
            completion_lock: Mutex::new(()),
            path,
            length: metadata.len(),
            buffering,
            alignment,
        })
    }

    pub const fn buffering(&self) -> FileBuffering {
        self.buffering
    }

    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    pub const fn length(&self) -> u64 {
        self.length
    }

    /// Submits every request before waiting for IOCP completions. A call is
    /// serialized per file so completion ownership remains local, while all
    /// reads inside the batch are genuinely overlapped.
    pub fn read_many(
        &self,
        requests: &mut [OverlappedRead<'_>],
        cancellation: &ReadCancellation,
    ) -> Result<(), ReadError> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{GetLastError, ERROR_IO_PENDING, WAIT_TIMEOUT};
        use windows_sys::Win32::Storage::FileSystem::ReadFile;
        use windows_sys::Win32::System::IO::{CancelIoEx, GetQueuedCompletionStatus};

        if cancellation.is_cancelled() {
            return Err(ReadError::Cancelled);
        }
        for request in requests.iter() {
            self.validate_overlapped_request(request)?;
        }
        let _guard = self
            .completion_lock
            .lock()
            .map_err(|_| ReadError::CompletionLockPoisoned)?;
        let handle = self.file.as_raw_handle().cast();
        let mut pending = requests
            .iter()
            .map(|request| PendingRead::new(request.offset, request.buffer.len()))
            .collect::<Vec<_>>();
        let mut submitted = 0_usize;
        for (request, operation) in requests.iter_mut().zip(&mut pending) {
            let length =
                u32::try_from(request.buffer.len()).map_err(|_| ReadError::OverlappedReadTooLarge {
                    requested: request.buffer.len(),
                })?;
            // SAFETY: the file was opened for overlapped reads; the output
            // slice and boxed OVERLAPPED remain live until every completion is
            // drained below.
            let started = unsafe {
                ReadFile(
                    handle,
                    request.buffer.as_mut_ptr(),
                    length,
                    std::ptr::null_mut(),
                    operation.overlapped.as_mut(),
                )
            };
            if started == 0 {
                // SAFETY: immediately reads the calling thread's last error.
                let error = unsafe { GetLastError() };
                if error != ERROR_IO_PENDING {
                    cancel_and_drain(handle, self.completion_port, &mut pending[..submitted]);
                    return Err(ReadError::OverlappedSubmit {
                        path: self.path.clone(),
                        offset: request.offset,
                        source: std::io::Error::from_raw_os_error(error as i32),
                    });
                }
            }
            submitted += 1;
        }

        let mut completed = 0_usize;
        let mut first_error = None;
        let mut cancellation_sent = false;
        while completed < submitted {
            if cancellation.is_cancelled() && !cancellation_sent {
                for operation in pending.iter().filter(|operation| !operation.completed) {
                    // SAFETY: every pointer belongs to a submitted operation
                    // and remains live until its completion is drained.
                    unsafe { CancelIoEx(handle, operation.overlapped.as_ref()) };
                }
                cancellation_sent = true;
            }
            let mut bytes = 0_u32;
            let mut key = 0_usize;
            let mut overlapped = std::ptr::null_mut();
            // SAFETY: all output pointers are valid and the completion port is
            // live. A short timeout permits cancellation polling.
            let status = unsafe {
                GetQueuedCompletionStatus(self.completion_port, &mut bytes, &mut key, &mut overlapped, 10)
            };
            if overlapped.is_null() {
                if status == 0 {
                    // SAFETY: immediately follows the failed wait.
                    let error = unsafe { GetLastError() };
                    if error == WAIT_TIMEOUT {
                        continue;
                    }
                    if first_error.is_none() {
                        first_error = Some(ReadError::CompletionWait {
                            source: std::io::Error::from_raw_os_error(error as i32),
                        });
                    }
                    break;
                }
                if first_error.is_none() {
                    first_error = Some(ReadError::UnknownCompletion);
                }
                break;
            }
            let Some(operation) = pending
                .iter_mut()
                .find(|operation| std::ptr::eq(operation.overlapped.as_ref(), overlapped))
            else {
                if first_error.is_none() {
                    first_error = Some(ReadError::UnknownCompletion);
                }
                break;
            };
            if operation.completed {
                if first_error.is_none() {
                    first_error = Some(ReadError::DuplicateCompletion);
                }
                break;
            }
            operation.completed = true;
            completed += 1;
            if status == 0 && first_error.is_none() {
                first_error = Some(ReadError::OverlappedCompletion {
                    path: self.path.clone(),
                    offset: operation.offset,
                    source: std::io::Error::last_os_error(),
                });
            } else if bytes as usize != operation.expected && first_error.is_none() {
                let actual_bytes = bytes as usize;
                let actual_end = operation.offset.saturating_add(actual_bytes as u64);
                let expected_end = operation.offset.saturating_add(operation.expected as u64);
                let is_final_partial = actual_bytes < operation.expected
                    && actual_end <= self.length
                    && expected_end > self.length;
                if !is_final_partial {
                    first_error = Some(ReadError::UnexpectedEof {
                        path: self.path.clone(),
                        offset: operation.offset,
                        expected: operation.expected,
                        actual: actual_bytes,
                    });
                }
            }
        }
        if first_error.is_some() && completed < submitted {
            cancel_and_drain(handle, self.completion_port, pending);
        }
        if cancellation_sent {
            return Err(ReadError::Cancelled);
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
    }

    fn validate_overlapped_request(&self, request: &OverlappedRead<'_>) -> Result<(), ReadError> {
        let length = u64::try_from(request.buffer.len()).map_err(|_| ReadError::ArithmeticOverflow)?;
        let end = request
            .offset
            .checked_add(length)
            .ok_or(ReadError::ArithmeticOverflow)?;
        if self.buffering == FileBuffering::Unbuffered {
            let alignment_u64 = self.alignment as u64;
            let padded_length = self.length.checked_add(alignment_u64 - 1)
                .and_then(|value| value.checked_div(alignment_u64))
                .and_then(|blocks| blocks.checked_mul(alignment_u64))
                .ok_or(ReadError::ArithmeticOverflow)?;
            if end > padded_length {
                return Err(ReadError::RangeOutOfBounds {
                    start: request.offset,
                    end,
                    file_length: self.length,
                });
            }
            if request.offset % alignment_u64 != 0 {
                return Err(ReadError::UnbufferedOffsetAlignment {
                    offset: request.offset,
                    alignment: self.alignment,
                });
            }
            if request.buffer.len() % self.alignment != 0 {
                return Err(ReadError::UnbufferedLengthAlignment {
                    length: request.buffer.len(),
                    alignment: self.alignment,
                });
            }
            if request.buffer.as_ptr() as usize % self.alignment != 0 {
                return Err(ReadError::UnbufferedBufferAlignment {
                    address: request.buffer.as_ptr() as usize,
                    alignment: self.alignment,
                });
            }
        } else {
            if end > self.length {
                return Err(ReadError::RangeOutOfBounds {
                    start: request.offset,
                    end,
                    file_length: self.length,
                });
            }
        }
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for OverlappedFile {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        // SAFETY: this value uniquely owns the completion port.
        unsafe { CloseHandle(self.completion_port) };
    }
}

#[cfg(windows)]
struct PendingRead {
    overlapped: Box<windows_sys::Win32::System::IO::OVERLAPPED>,
    offset: u64,
    expected: usize,
    completed: bool,
}

#[cfg(windows)]
impl PendingRead {
    fn new(offset: u64, expected: usize) -> Self {
        use windows_sys::Win32::System::IO::OVERLAPPED;
        // SAFETY: an all-zero OVERLAPPED is the documented initialization.
        let mut overlapped = Box::new(unsafe { std::mem::zeroed::<OVERLAPPED>() });
        overlapped.Anonymous.Anonymous.Offset = offset as u32;
        overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        Self {
            overlapped,
            offset,
            expected,
            completed: false,
        }
    }
}

#[cfg(windows)]
fn fs_open_overlapped(path: &Path, flags: u32) -> Result<File, ReadError> {
    use std::os::windows::fs::OpenOptionsExt;
    let mut options = std::fs::OpenOptions::new();
    options.read(true).custom_flags(flags);
    options.open(path).map_err(|source| ReadError::Open {
        path: path.to_owned(),
        source,
    })
}

#[cfg(windows)]
fn file_alignment(file: &File, require_physical_sector: bool) -> Result<usize, ReadError> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileAlignmentInfo, FileStorageInfo, GetFileInformationByHandleEx, FILE_ALIGNMENT_INFO,
        FILE_STORAGE_INFO,
    };
    // SAFETY: an all-zero output structure is valid for this query.
    let mut information = unsafe { std::mem::zeroed::<FILE_ALIGNMENT_INFO>() };
    // SAFETY: the handle is live and the output buffer exactly matches the
    // requested information class.
    let status = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle().cast(),
            FileAlignmentInfo,
            (&mut information as *mut FILE_ALIGNMENT_INFO).cast(),
            std::mem::size_of::<FILE_ALIGNMENT_INFO>() as u32,
        )
    };
    if status == 0 {
        return Err(ReadError::AlignmentQuery {
            source: std::io::Error::last_os_error(),
        });
    }
    let device_alignment = usize::try_from(information.AlignmentRequirement)
        .ok()
        .and_then(|requirement| requirement.checked_add(1))
        .filter(|alignment| alignment.is_power_of_two())
        .ok_or(ReadError::InvalidDeviceAlignment(
            information.AlignmentRequirement,
        ))?;

    if !require_physical_sector {
        return Ok(device_alignment);
    }

    // `FILE_FLAG_NO_BUFFERING` also requires transfer sizes and addresses to
    // satisfy the volume's physical-sector contract. FILE_STORAGE_INFO is the
    // handle-local query for both logical and physical sector sizes.
    // SAFETY: an all-zero FILE_STORAGE_INFO is valid for this query.
    let mut storage = unsafe { std::mem::zeroed::<FILE_STORAGE_INFO>() };
    // SAFETY: the handle is live and the output buffer exactly matches the
    // requested information class.
    let storage_status = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle().cast(),
            FileStorageInfo,
            (&mut storage as *mut FILE_STORAGE_INFO).cast(),
            std::mem::size_of::<FILE_STORAGE_INFO>() as u32,
        )
    };
    if storage_status == 0 {
        return Err(ReadError::PhysicalSectorQuery {
            source: std::io::Error::last_os_error(),
        });
    }

    [
        storage.LogicalBytesPerSector,
        storage.PhysicalBytesPerSectorForAtomicity,
        storage.PhysicalBytesPerSectorForPerformance,
        storage.FileSystemEffectivePhysicalBytesPerSectorForAtomicity,
    ]
    .into_iter()
    .filter_map(|bytes| usize::try_from(bytes).ok())
    .filter(|bytes| *bytes != 0)
    .try_fold(device_alignment, |current, bytes| {
        if bytes.is_power_of_two() {
            Ok(current.max(bytes))
        } else {
            Err(ReadError::InvalidPhysicalSectorAlignment(bytes))
        }
    })
}

#[cfg(windows)]
fn cancel_and_drain(
    handle: windows_sys::Win32::Foundation::HANDLE,
    completion_port: windows_sys::Win32::Foundation::HANDLE,
    pending: &mut [PendingRead],
) {
    use windows_sys::Win32::System::IO::{CancelIoEx, GetQueuedCompletionStatus};
    for operation in pending.iter() {
        // SAFETY: the operation was submitted and remains live.
        unsafe { CancelIoEx(handle, operation.overlapped.as_ref()) };
    }
    for _ in 0..pending.len() {
        let mut bytes = 0;
        let mut key = 0;
        let mut overlapped = std::ptr::null_mut();
        // SAFETY: drains one completion before any OVERLAPPED is dropped.
        unsafe {
            GetQueuedCompletionStatus(completion_port, &mut bytes, &mut key, &mut overlapped, u32::MAX)
        };
    }
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
    #[error("failed to {operation} I/O completion port for {path:?}: {source}")]
    CompletionPort {
        path: PathBuf,
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to query file alignment: {source}")]
    AlignmentQuery {
        #[source]
        source: std::io::Error,
    },
    #[error("failed to query physical-sector alignment for unbuffered I/O: {source}")]
    PhysicalSectorQuery {
        #[source]
        source: std::io::Error,
    },
    #[error("device reported invalid alignment requirement {0}")]
    InvalidDeviceAlignment(u32),
    #[error("device reported invalid physical-sector alignment {0}")]
    InvalidPhysicalSectorAlignment(usize),
    #[error("overlapped read is {requested} bytes, exceeding the Win32 DWORD limit")]
    OverlappedReadTooLarge { requested: usize },
    #[error("unbuffered read offset {offset} is not aligned to {alignment} bytes")]
    UnbufferedOffsetAlignment { offset: u64, alignment: usize },
    #[error("unbuffered read length {length} is not aligned to {alignment} bytes")]
    UnbufferedLengthAlignment { length: usize, alignment: usize },
    #[error("unbuffered read buffer address {address:#x} is not aligned to {alignment} bytes")]
    UnbufferedBufferAlignment { address: usize, alignment: usize },
    #[error("failed to submit overlapped read at {offset} for {path:?}: {source}")]
    OverlappedSubmit {
        path: PathBuf,
        offset: u64,
        #[source]
        source: std::io::Error,
    },
    #[error("overlapped read at {offset} for {path:?} failed: {source}")]
    OverlappedCompletion {
        path: PathBuf,
        offset: u64,
        #[source]
        source: std::io::Error,
    },
    #[error("I/O completion wait failed: {source}")]
    CompletionWait {
        #[source]
        source: std::io::Error,
    },
    #[error("I/O completion port returned an unknown OVERLAPPED pointer")]
    UnknownCompletion,
    #[error("I/O completion port returned the same OVERLAPPED pointer twice")]
    DuplicateCompletion,
    #[error("I/O completion lock is poisoned")]
    CompletionLockPoisoned,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SlotPoolError {
    #[error("read slot pool must contain at least one slot")]
    ZeroSlots,
    #[error("read slot size must be greater than zero")]
    ZeroSlotBytes,
    #[error("read slot alignment {0} is not a power of two")]
    InvalidAlignment(usize),
    #[error("failed to allocate an aligned read slot")]
    AllocationFailed,
    #[error("read slot pool lock is poisoned")]
    Poisoned,
    #[error("read slot acquisition was cancelled")]
    Cancelled,
    #[error("read slot generation overflow")]
    GenerationOverflow,
}
