use bridge_kv_gqa::{KvError, PagedKvCache};

#[test]
fn page_addressing_preserves_token_head_and_layer_order() {
    let mut cache = PagedKvCache::new(2, 2, 2, 1, 2, 5).unwrap();
    assert_eq!(cache.page_count(), 3);
    assert_eq!(cache.page_tokens(), 2);

    for token in 0..5 {
        let key = [
            token as f32,
            token as f32 + 0.25,
            token as f32 + 10.0,
            token as f32 + 10.25,
        ];
        let value = [token as f32 + 20.0, token as f32 + 30.0];
        assert_eq!(cache.append(0, &key, &value).unwrap(), token);
    }
    assert_eq!(cache.stored_tokens(0).unwrap(), 5);
    assert_eq!(cache.stored_tokens(1).unwrap(), 0);

    for token in 0..5 {
        assert_eq!(
            cache.key(0, token, 0).unwrap(),
            [token as f32, token as f32 + 0.25]
        );
        assert_eq!(
            cache.key(0, token, 1).unwrap(),
            [token as f32 + 10.0, token as f32 + 10.25]
        );
        assert_eq!(cache.value(0, token, 0).unwrap(), [token as f32 + 20.0]);
        assert_eq!(cache.value(0, token, 1).unwrap(), [token as f32 + 30.0]);
    }
}

#[test]
fn batch_append_is_atomic_across_length_capacity_and_finite_errors() {
    let mut cache = PagedKvCache::new(1, 1, 2, 1, 2, 3).unwrap();
    cache.append(0, &[1.0, 2.0], &[3.0]).unwrap();

    assert!(matches!(
        cache.append_tokens(0, 2, &[1.0, 2.0], &[3.0, 4.0]),
        Err(KvError::LengthMismatch {
            field: "append keys",
            expected: 4,
            actual: 2,
        })
    ));
    assert_eq!(cache.stored_tokens(0).unwrap(), 1);

    assert!(matches!(
        cache.append_tokens(0, 1, &[f32::NAN, 2.0], &[3.0]),
        Err(KvError::NonFiniteValue {
            field: "append keys",
            index: 0,
            ..
        })
    ));
    assert_eq!(cache.stored_tokens(0).unwrap(), 1);
    assert_eq!(cache.key(0, 0, 0).unwrap(), [1.0, 2.0]);

    cache
        .append_tokens(0, 2, &[4.0, 5.0, 6.0, 7.0], &[8.0, 9.0])
        .unwrap();
    assert!(matches!(
        cache.append(0, &[10.0, 11.0], &[12.0]),
        Err(KvError::CapacityExhausted {
            layer: 0,
            stored: 3,
            additional: 1,
            capacity: 3,
        })
    ));
    assert_eq!(cache.stored_tokens(0).unwrap(), 3);
}

#[test]
fn reset_reuses_capacity_and_range_errors_are_typed() {
    let mut cache = PagedKvCache::new(1, 2, 1, 1, 1, 2).unwrap();
    cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
    assert!(matches!(
        cache.key(1, 0, 0),
        Err(KvError::LayerOutOfRange {
            layer: 1,
            layer_count: 1,
        })
    ));
    assert!(matches!(
        cache.key(0, 1, 0),
        Err(KvError::TokenOutOfRange {
            token: 1,
            stored_tokens: 1,
        })
    ));
    assert!(matches!(
        cache.key(0, 0, 2),
        Err(KvError::HeadOutOfRange {
            head: 2,
            head_count: 2,
        })
    ));

    cache.reset();
    assert_eq!(cache.stored_tokens(0).unwrap(), 0);
    assert_eq!(cache.remaining_tokens(0).unwrap(), 2);
    cache.append(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
    assert_eq!(cache.key(0, 0, 1).unwrap(), [6.0]);
}

#[test]
fn rewind_all_is_atomic_and_reuses_committed_pages() {
    let mut cache = PagedKvCache::new_lazy(2, 1, 1, 1, 2, 6).unwrap();
    for layer in 0..2 {
        cache
            .append_tokens(layer, 3, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .unwrap();
    }
    let allocated = cache.allocated_page_count();
    cache.rewind_all(1).unwrap();
    assert_eq!(cache.stored_tokens(0).unwrap(), 1);
    assert_eq!(cache.stored_tokens(1).unwrap(), 1);
    assert_eq!(cache.allocated_page_count(), allocated);

    cache.append(0, &[9.0], &[10.0]).unwrap();
    assert_eq!(cache.key(0, 1, 0).unwrap(), [9.0]);
    assert!(matches!(
        cache.rewind_all(2),
        Err(KvError::RewindBeyondStored {
            layer: 1,
            requested: 2,
            stored: 1,
        })
    ));
    assert_eq!(cache.stored_tokens(0).unwrap(), 2);
    assert_eq!(cache.stored_tokens(1).unwrap(), 1);
}

#[test]
fn lazy_cache_exposes_large_logical_capacity_and_commits_only_touched_pages() {
    let mut cache = PagedKvCache::new_lazy(2, 2, 2, 1, 4, 1_000_000).unwrap();
    assert_eq!(cache.page_count(), 250_000);
    assert_eq!(cache.allocated_page_count(), 0);
    assert_eq!(cache.allocated_bytes().unwrap(), 0);

    cache
        .append_tokens(
            1,
            5,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                18.0, 19.0, 20.0,
            ],
            &[21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
        )
        .unwrap();
    assert_eq!(cache.allocated_page_count(), 2);
    assert_eq!(cache.allocated_bytes().unwrap(), 2 * 4 * 2 * (2 + 1) * 4);
    assert_eq!(cache.key(1, 4, 1).unwrap(), [19.0, 20.0]);
    assert_eq!(cache.value(1, 4, 1).unwrap(), [30.0]);

    cache.reset();
    assert_eq!(cache.allocated_page_count(), 2);
    assert_eq!(cache.stored_tokens(1).unwrap(), 0);
}

#[test]
fn model_bound_snapshot_round_trips_lazy_pages_and_layer_lengths() {
    let binding = [0x5a_u8; 32];
    let mut source = PagedKvCache::new_lazy(2, 2, 2, 1, 2, 8).unwrap();
    source
        .append_tokens(
            0,
            2,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
    source
        .append(1, &[13.0, 14.0, 15.0, 16.0], &[17.0, 18.0])
        .unwrap();
    let snapshot = source.export_snapshot(binding, 64 * 1024).unwrap();

    let mut restored = PagedKvCache::new_lazy(2, 2, 2, 1, 2, 8).unwrap();
    restored.restore_snapshot(binding, &snapshot, 64 * 1024).unwrap();
    assert_eq!(restored.stored_tokens(0).unwrap(), 2);
    assert_eq!(restored.stored_tokens(1).unwrap(), 1);
    assert_eq!(restored.key(0, 1, 1).unwrap(), [7.0, 8.0]);
    assert_eq!(restored.value(0, 1, 0).unwrap(), [11.0]);
    assert_eq!(restored.key(1, 0, 0).unwrap(), [13.0, 14.0]);
    assert_eq!(restored.allocated_page_count(), 2);
}

#[test]
fn snapshot_rejects_wrong_binding_corruption_bounds_and_configuration_atomically() {
    let binding = [0x11_u8; 32];
    let mut source = PagedKvCache::new_lazy(1, 1, 1, 1, 2, 4).unwrap();
    source.append(0, &[1.0], &[2.0]).unwrap();
    let snapshot = source.export_snapshot(binding, 4096).unwrap();
    assert!(matches!(
        source.export_snapshot(binding, snapshot.len() - 1),
        Err(KvError::SnapshotTooLarge { .. })
    ));

    let mut destination = PagedKvCache::new_lazy(1, 1, 1, 1, 2, 4).unwrap();
    destination.append(0, &[9.0], &[10.0]).unwrap();
    assert!(matches!(
        destination.restore_snapshot([0x22; 32], &snapshot, 4096),
        Err(KvError::SnapshotBinding)
    ));
    assert_eq!(destination.key(0, 0, 0).unwrap(), [9.0]);

    let mut corrupted = snapshot.clone();
    corrupted[20] ^= 0x80;
    assert!(matches!(
        destination.restore_snapshot(binding, &corrupted, 4096),
        Err(KvError::SnapshotChecksum)
    ));
    assert_eq!(destination.key(0, 0, 0).unwrap(), [9.0]);

    let mut wrong_shape = PagedKvCache::new_lazy(1, 1, 2, 1, 2, 4).unwrap();
    assert!(matches!(
        wrong_shape.restore_snapshot(binding, &snapshot, 4096),
        Err(KvError::SnapshotConfiguration {
            field: "key_dimension",
            expected: 2,
            actual: 1,
        })
    ));
    assert_eq!(wrong_shape.stored_tokens(0).unwrap(), 0);
}
