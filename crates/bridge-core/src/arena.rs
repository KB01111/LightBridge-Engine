//! Fixed, aligned scratch allocation for ingestion work.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::mem::size_of;
use std::ptr::NonNull;

use crate::error::{CoreError, Result};

pub const CACHE_LINE: usize = 64;

/// An owned zeroed allocation with an explicitly checked alignment.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    pub fn new(len: usize, align: usize) -> Result<Self> {
        let layout = checked_layout(len, align)?;
        // SAFETY: `layout` was created by `Layout::from_size_align` and is non-zero in size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(CoreError::AllocationFailed {
            size: layout.size(),
            align,
        })?;
        Ok(Self { ptr, layout, len })
    }

    pub const fn len(&self) -> usize {
        self.len
    }
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub const fn capacity(&self) -> usize {
        self.layout.size()
    }
    pub const fn alignment(&self) -> usize {
        self.layout.align()
    }
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is live and initialized by `alloc_zeroed` for at least `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` uniquely borrows the live initialized allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_capacity(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` uniquely borrows the whole initialized allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: this is exactly the allocation and layout created by `new`.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

fn checked_layout(len: usize, align: usize) -> Result<Layout> {
    if align == 0 || !align.is_power_of_two() {
        return Err(CoreError::InvalidAllocationLayout { size: len, align });
    }
    let requested = len.max(1);
    let padded = requested
        .checked_add(align - 1)
        .map(|value| value & !(align - 1))
        .ok_or(CoreError::InvalidAllocationLayout { size: len, align })?;
    Layout::from_size_align(padded, align)
        .map_err(|_| CoreError::InvalidAllocationLayout { size: len, align })
}

/// A bump allocator over a fixed 64-byte-aligned backing allocation.
pub struct Arena {
    buffer: AlignedBuffer,
    base_alignment: usize,
    logical_capacity: usize,
    cursor: usize,
    high_water: usize,
}

impl Arena {
    pub fn with_capacity_bytes(bytes: usize) -> Result<Self> {
        let buffer = AlignedBuffer::new(bytes, CACHE_LINE)?;
        Ok(Self {
            base_alignment: buffer.alignment(),
            buffer,
            logical_capacity: bytes,
            cursor: 0,
            high_water: 0,
        })
    }

    pub fn with_capacity_f32(n: usize) -> Result<Self> {
        let bytes = n
            .checked_mul(size_of::<f32>())
            .ok_or(CoreError::ArithmeticOverflow("arena f32 capacity"))?;
        Self::with_capacity_bytes(bytes)
    }

    pub fn alloc_f32(&mut self, n: usize) -> Option<&mut [f32]> {
        let bytes = n.checked_mul(size_of::<f32>())?;
        let bytes = self.alloc_bytes(bytes, CACHE_LINE)?;
        // SAFETY: the allocation start is 64-byte aligned and `bytes` is an exact multiple of f32.
        Some(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f32>(), n) })
    }

    /// Returns an exclusive, zeroed region or `None` if the request is invalid or does not fit.
    pub fn alloc_bytes(&mut self, len: usize, align: usize) -> Option<&mut [u8]> {
        if align == 0 || !align.is_power_of_two() || align > self.base_alignment {
            return None;
        }
        let start = align_up(self.cursor, align)?;
        let end = start.checked_add(len)?;
        if end > self.logical_capacity {
            return None;
        }
        self.cursor = end;
        self.high_water = self.high_water.max(end);
        let region = &mut self.buffer.as_mut_capacity()[start..end];
        region.fill(0);
        Some(region)
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
    pub const fn used(&self) -> usize {
        self.cursor
    }
    pub const fn high_water(&self) -> usize {
        self.high_water
    }
    pub const fn capacity(&self) -> usize {
        self.logical_capacity
    }
    pub const fn base_alignment(&self) -> usize {
        self.base_alignment
    }
}

fn align_up(value: usize, align: usize) -> Option<usize> {
    value.checked_add(align - 1).map(|aligned| aligned & !(align - 1))
}

/// Padded to a cache line to prevent adjacent worker state from sharing one.
#[repr(align(64))]
#[derive(Debug, Default, Clone, Copy)]
pub struct CachePadded<T>(pub T);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_rejects_overflow_and_excess_alignment() {
        let mut arena = Arena::with_capacity_bytes(256).unwrap();
        assert!(arena.alloc_f32(usize::MAX).is_none());
        assert!(arena.alloc_bytes(1, CACHE_LINE * 2).is_none());
    }

    #[test]
    fn default_arena_allocations_are_cache_line_aligned() {
        let mut arena = Arena::with_capacity_bytes(256).unwrap();
        assert_eq!(arena.base_alignment(), CACHE_LINE);
        let bytes = arena.alloc_bytes(3, CACHE_LINE).unwrap();
        assert_eq!(bytes.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[test]
    fn arena_reports_requested_capacity_without_exposing_padding() {
        for &(requested, can_allocate_byte) in &[(0, false), (1, true), (64, true)] {
            let mut arena = Arena::with_capacity_bytes(requested).unwrap();
            assert_eq!(arena.capacity(), requested);
            assert_eq!(arena.alloc_bytes(1, 1).is_some(), can_allocate_byte);
        }
    }

    #[test]
    fn arena_exact_boundary_exhausts_requested_capacity() {
        let mut arena = Arena::with_capacity_bytes(4).unwrap();
        assert_eq!(arena.alloc_bytes(4, 1).unwrap().len(), 4);
        assert!(arena.alloc_bytes(1, 1).is_none());
    }

    #[test]
    fn one_byte_arena_is_exhausted_after_one_byte_allocation() {
        let mut arena = Arena::with_capacity_bytes(1).unwrap();
        assert_eq!(arena.alloc_bytes(1, 1).unwrap().len(), 1);
        assert!(arena.alloc_bytes(1, 1).is_none());
    }

    #[test]
    fn arena_bump_alignment_survives_disturbed_cursor_for_each_supported_alignment() {
        for &align in &[1, 2, 4, 8, 16, 32, 64] {
            let mut arena = Arena::with_capacity_bytes(128).unwrap();
            assert_eq!(arena.alloc_bytes(1, 1).unwrap().len(), 1);

            let bytes = arena.alloc_bytes(1, align).unwrap();
            assert_eq!(bytes.as_ptr() as usize % align, 0, "alignment {align}");
            assert_eq!(arena.used(), if align == 1 { 2 } else { align + 1 });
        }
    }

    #[test]
    fn aligned_buffer_is_fallible_and_honours_page_alignment() {
        let buffer = AlignedBuffer::new(1, 4096).unwrap();
        assert_eq!(buffer.as_ptr() as usize % 4096, 0);
        assert!(AlignedBuffer::new(1, 3).is_err());
        assert!(AlignedBuffer::new(usize::MAX, 4096).is_err());
    }
}
