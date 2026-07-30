use bridge_io_windows::{ReadCancellation, ReadSlotPool, SlotPoolError};

#[test]
fn aligned_slots_are_bounded_poisoned_and_generation_stamped() {
    let pool = ReadSlotPool::new(1, 4096, 4096).unwrap();
    let mut first = pool.try_acquire().unwrap().unwrap();
    assert_eq!(first.address() % 4096, 0);
    assert!(pool.try_acquire().unwrap().is_none());
    first.as_mut_slice()[0] = 7;
    let stale = first.token();
    assert!(pool.is_current(stale).unwrap());
    drop(first);
    assert!(!pool.is_current(stale).unwrap());

    let second = pool.try_acquire().unwrap().unwrap();
    assert_ne!(second.token().generation, stale.generation);
    assert!(second.as_slice().iter().all(|&byte| byte == 0xdd));
    assert!(!pool.is_current(stale).unwrap());
}

#[test]
fn acquisition_observes_cancellation_and_configuration_errors() {
    assert!(matches!(
        ReadSlotPool::new(0, 4096, 4096),
        Err(SlotPoolError::ZeroSlots)
    ));
    assert!(matches!(
        ReadSlotPool::new(1, 4096, 3),
        Err(SlotPoolError::InvalidAlignment(3))
    ));

    let pool = ReadSlotPool::new(1, 64, 64).unwrap();
    let _lease = pool.try_acquire().unwrap().unwrap();
    let cancellation = ReadCancellation::new();
    cancellation.cancel();
    assert!(matches!(
        pool.acquire(&cancellation),
        Err(SlotPoolError::Cancelled)
    ));
}
