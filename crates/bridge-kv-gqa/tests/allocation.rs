use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bridge_kv_gqa::PagedKvCache;

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn appending_across_pages_allocates_nothing_after_construction() {
    let mut cache = PagedKvCache::new(1, 1, 2, 2, 2, 5).unwrap();
    let keys = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let values = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let result = cache.append_tokens(0, 3, &keys, &values);
    COUNTING.store(false, Ordering::SeqCst);

    result.unwrap();
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
