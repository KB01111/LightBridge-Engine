use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bridge_core::ggml_type::GgmlType;
use bridge_kernels_reference::{
    moe_selected_into, PackedMatrix, PayloadEndian, ReferenceExecutionMode, SelectedExpert, SwiGluExpert,
    SwiGluScratch,
};

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

fn matrix(bytes: &[u8]) -> PackedMatrix<'_> {
    PackedMatrix::from_parts(GgmlType::F32, PayloadEndian::Little, 1, 1, bytes).unwrap()
}

#[test]
fn moe_forward_allocates_nothing_after_scratch_construction() {
    let one = 1.0_f32.to_le_bytes();
    let expert = SwiGluExpert::new(matrix(&one), matrix(&one), matrix(&one)).unwrap();
    let routed = [SelectedExpert {
        expert_id: 0,
        coefficient: 1.0,
        expert,
    }];
    let input = [1.0_f32];
    let mut output = [0.0_f32];
    let mut activation = [0.0_f32; 2];
    let mut preflight = [0.0_f32];
    let mut decoded = [0.0_f32; 256];
    let mut q8 = [];
    let mut scratch = SwiGluScratch::new(&mut activation, &mut decoded, &mut q8);

    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let result = moe_selected_into(
        ReferenceExecutionMode::DequantF32,
        &routed,
        expert,
        &input,
        &mut output,
        &mut preflight,
        &mut scratch,
    );
    COUNTING.store(false, Ordering::SeqCst);

    result.unwrap();
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
