use std::io;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Barrier,
};

use bridge_cache::{CacheConfig, CacheError, CompressedCache, LoadError};
use bridge_io_windows::ReadSlotPool;

fn cache(capacity_bytes: usize, admit_after_requests: u64) -> CompressedCache<u32> {
    CompressedCache::new(CacheConfig {
        capacity_bytes,
        admit_after_requests,
    })
    .unwrap()
}

#[test]
fn lru_eviction_stays_within_the_fixed_ceiling() {
    let cache = cache(8, 1);
    drop(
        cache
            .get_or_try_insert(1, 4, || Ok::<_, io::Error>(vec![1; 4]))
            .unwrap(),
    );
    drop(
        cache
            .get_or_try_insert(2, 4, || Ok::<_, io::Error>(vec![2; 4]))
            .unwrap(),
    );
    assert!(cache.get(&1).unwrap().is_some());
    drop(
        cache
            .get_or_try_insert(3, 4, || Ok::<_, io::Error>(vec![3; 4]))
            .unwrap(),
    );
    let stats = cache.stats().unwrap();
    assert_eq!(stats.used_bytes, 8);
    assert_eq!(stats.resident_entries, 2);
    assert_eq!(stats.evictions, 1);
    assert!(cache.get(&1).unwrap().is_some());
    assert!(cache.get(&2).unwrap().is_none());
    assert!(cache.get(&3).unwrap().is_some());
}

#[test]
fn pinned_entries_block_reservation_before_loader_runs() {
    let cache = cache(8, 1);
    let lease = cache
        .get_or_try_insert(1, 8, || Ok::<_, io::Error>(vec![1; 8]))
        .unwrap();
    let loads = AtomicUsize::new(0);
    let error = cache
        .get_or_try_insert(2, 1, || {
            loads.fetch_add(1, Ordering::Relaxed);
            Ok::<_, io::Error>(vec![2])
        })
        .unwrap_err();
    assert!(matches!(
        error,
        LoadError::Cache(CacheError::CapacityPinned { requested: 1 })
    ));
    assert_eq!(loads.load(Ordering::Relaxed), 0);
    assert_eq!(lease.bytes(), &[1; 8]);
}

#[test]
fn concurrent_misses_are_deduplicated() {
    let cache = Arc::new(cache(64, 1));
    let barrier = Arc::new(Barrier::new(2));
    let loads = Arc::new(AtomicUsize::new(0));
    let first = {
        let cache = Arc::clone(&cache);
        let barrier = Arc::clone(&barrier);
        let loads = Arc::clone(&loads);
        std::thread::spawn(move || {
            cache
                .get_or_try_insert(7, 16, || {
                    loads.fetch_add(1, Ordering::Relaxed);
                    barrier.wait();
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Ok::<_, io::Error>(vec![7; 16])
                })
                .unwrap()
        })
    };
    barrier.wait();
    let second = {
        let cache = Arc::clone(&cache);
        let loads = Arc::clone(&loads);
        std::thread::spawn(move || {
            cache
                .get_or_try_insert(7, 16, || {
                    loads.fetch_add(1, Ordering::Relaxed);
                    Ok::<_, io::Error>(vec![9; 16])
                })
                .unwrap()
        })
    };
    let first = first.join().unwrap();
    let second = second.join().unwrap();
    assert_eq!(loads.load(Ordering::Relaxed), 1);
    assert_eq!(first.bytes(), &[7; 16]);
    assert_eq!(second.bytes(), &[7; 16]);
    assert!(cache.stats().unwrap().waits >= 1);
}

#[test]
fn hysteretic_admission_requires_reuse() {
    let cache = cache(16, 2);
    let loads = AtomicUsize::new(0);
    drop(
        cache
            .get_or_try_insert(1, 4, || {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok::<_, io::Error>(vec![1; 4])
            })
            .unwrap(),
    );
    assert_eq!(cache.stats().unwrap().resident_entries, 0);
    drop(
        cache
            .get_or_try_insert(1, 4, || {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok::<_, io::Error>(vec![1; 4])
            })
            .unwrap(),
    );
    assert_eq!(cache.stats().unwrap().resident_entries, 1);
    let hit = cache
        .get_or_try_insert(1, 4, || {
            loads.fetch_add(1, Ordering::Relaxed);
            Ok::<_, io::Error>(vec![9; 4])
        })
        .unwrap();
    assert_eq!(hit.bytes(), &[1; 4]);
    assert_eq!(loads.load(Ordering::Relaxed), 2);
}

#[test]
fn bad_loader_size_releases_the_reservation() {
    let cache = cache(8, 1);
    let error = cache
        .get_or_try_insert(1, 4, || Ok::<_, io::Error>(vec![1; 3]))
        .unwrap_err();
    assert!(matches!(
        error,
        LoadError::Cache(CacheError::LoadedSizeMismatch {
            expected: 4,
            actual: 3
        })
    ));
    assert_eq!(cache.stats().unwrap().reserved_bytes, 0);
    let lease = cache
        .get_or_try_insert(1, 4, || Ok::<_, io::Error>(vec![2; 4]))
        .unwrap();
    assert_eq!(lease.bytes(), &[2; 4]);
}

#[test]
fn heat_round_trips_with_input_bounds() {
    let source = cache(16, 2);
    drop(
        source
            .get_or_try_insert(3, 4, || Ok::<_, io::Error>(vec![3; 4]))
            .unwrap(),
    );
    let json = source.export_heat_json(8).unwrap();
    let destination = cache(16, 2);
    assert_eq!(destination.import_heat_json(&json, 4096, 8).unwrap(), 1);
    drop(
        destination
            .get_or_try_insert(3, 4, || Ok::<_, io::Error>(vec![3; 4]))
            .unwrap(),
    );
    assert_eq!(destination.stats().unwrap().resident_entries, 1);
}

#[test]
fn aligned_read_slot_is_retained_by_leases_and_recycled_after_eviction() {
    let cache = cache(64, 1);
    let slots = ReadSlotPool::new(1, 64, 64).unwrap();
    let mut original = None;
    let lease = cache
        .get_or_try_insert_read_slot(9, 16, || {
            let mut slot = slots.try_acquire().unwrap().unwrap();
            original = Some(slot.token());
            slot.as_mut_slice()[..16].fill(9);
            Ok::<_, io::Error>(slot)
        })
        .unwrap();
    assert_eq!(lease.bytes(), &[9; 16]);
    assert!(slots.try_acquire().unwrap().is_none());
    drop(lease);
    assert!(slots.try_acquire().unwrap().is_none());

    assert_eq!(cache.clear_unpinned().unwrap(), 1);
    let recycled = slots.try_acquire().unwrap().unwrap();
    assert_ne!(Some(recycled.token()), original);
    assert!(recycled.as_slice().iter().all(|&byte| byte == 0xdd));
}

#[test]
fn charged_slots_evict_before_the_physical_pool_can_be_exhausted() {
    let cache = cache(16, 1);
    let slots = ReadSlotPool::new(2, 8, 8).unwrap();
    for key in [1_u32, 2] {
        let lease = cache
            .get_or_try_insert_read_slot_charged(key, 4, 8, || {
                let mut slot = slots.try_acquire().unwrap().unwrap();
                slot.as_mut_slice()[..4].fill(key as u8);
                Ok::<_, io::Error>(slot)
            })
            .unwrap();
        assert_eq!(lease.bytes(), &[key as u8; 4]);
    }
    assert!(slots.try_acquire().unwrap().is_none());

    let third = cache
        .get_or_try_insert_read_slot_charged(3, 4, 8, || {
            let mut slot = slots.try_acquire().unwrap().unwrap();
            slot.as_mut_slice()[..4].fill(3);
            Ok::<_, io::Error>(slot)
        })
        .unwrap();
    assert_eq!(third.bytes(), &[3; 4]);
    drop(third);

    let stats = cache.stats().unwrap();
    assert_eq!(stats.used_bytes, 16);
    assert_eq!(stats.resident_entries, 2);
    assert_eq!(stats.evictions, 1);
    assert!(slots.try_acquire().unwrap().is_none());
}
