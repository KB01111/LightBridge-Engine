//! Fixed-ceiling, pin-aware, deduplicating compressed-byte cache.

use std::collections::HashMap;
use std::error::Error;
use std::hash::Hash;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use bridge_io_windows::ReadSlotLease;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const HEAT_FORMAT: &str = "lightbridge-cache-heat";
const HEAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheConfig {
    pub capacity_bytes: usize,
    pub admit_after_requests: u64,
}

impl CacheConfig {
    pub fn validate(self) -> Result<(), CacheError> {
        if self.capacity_bytes == 0 {
            return Err(CacheError::ZeroCapacity);
        }
        if self.admit_after_requests == 0 {
            return Err(CacheError::ZeroAdmissionThreshold);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CompressedCache<K> {
    inner: Arc<Inner<K>>,
}

#[derive(Debug)]
struct Inner<K> {
    config: CacheConfig,
    state: Mutex<State<K>>,
    changed: Condvar,
}

#[derive(Debug)]
struct State<K> {
    entries: HashMap<K, Entry>,
    heat: HashMap<K, u64>,
    used_bytes: usize,
    reserved_bytes: usize,
    tick: u64,
    hits: u64,
    misses: u64,
    loads: u64,
    waits: u64,
    evictions: u64,
}

#[derive(Debug)]
enum Entry {
    Loading {
        expected_bytes: usize,
        charge_bytes: usize,
        resident: bool,
    },
    Ready {
        bytes: Arc<CachePayload>,
        charge_bytes: usize,
        pins: usize,
        last_used: u64,
        resident: bool,
    },
}

#[derive(Debug)]
enum CachePayload {
    Owned(Vec<u8>),
    ReadSlot { lease: ReadSlotLease, length: usize },
}

impl CachePayload {
    /// Reports the logical number of bytes represented by the payload.
    ///
    /// # Examples
    ///
    /// ```
    /// let payload = CachePayload::Owned(vec![1, 2, 3]);
    /// assert_eq!(payload.len(), 3);
    /// ```
    ///
    /// # Returns
    ///
    /// The logical payload length in bytes.
    fn len(&self) -> usize {
        match self {
            Self::Owned(bytes) => bytes.len(),
            Self::ReadSlot { length, .. } => *length,
        }
    }

    /// Returns the logical payload bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let payload = CachePayload::Owned(vec![1, 2, 3]);
    /// assert_eq!(payload.bytes(), &[1, 2, 3]);
    /// ```
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::ReadSlot { lease, length } => &lease.as_slice()[..*length],
        }
    }

    /// Reports the size of the storage backing the payload.
    ///
    /// # Examples
    ///
    /// ```
    /// let payload = CachePayload::Owned(vec![1, 2, 3]);
    /// assert_eq!(payload.backing_len(), 3);
    /// ```
    ///
    /// # Returns
    ///
    /// The number of bytes available in the payload's backing storage.
    fn backing_len(&self) -> usize {
        match self {
            Self::Owned(bytes) => bytes.len(),
            Self::ReadSlot { lease, .. } => lease.as_slice().len(),
        }
    }

    /// Validates that the payload has sufficient backing storage for its logical length.
    ///
    /// # Examples
    ///
    /// ```
    /// let payload = CachePayload::Owned(vec![1, 2, 3]);
    /// assert!(payload.validate_backing().is_ok());
    /// ```
    fn validate_backing(&self) -> Result<(), CacheError>
    fn validate_backing(&self) -> Result<(), CacheError> {
        match self {
            Self::ReadSlot { lease, length } if lease.as_slice().len() < *length => {
                Err(CacheError::BackingTooSmall {
                    required: *length,
                    actual: lease.as_slice().len(),
                })
            }
            Self::Owned(_) | Self::ReadSlot { .. } => Ok(()),
        }
    }
}

impl<K> CompressedCache<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(config: CacheConfig) -> Result<Self, CacheError> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                state: Mutex::new(State {
                    entries: HashMap::new(),
                    heat: HashMap::new(),
                    used_bytes: 0,
                    reserved_bytes: 0,
                    tick: 0,
                    hits: 0,
                    misses: 0,
                    loads: 0,
                    waits: 0,
                    evictions: 0,
                }),
                changed: Condvar::new(),
            }),
        })
    }

    pub fn config(&self) -> CacheConfig {
        self.inner.config
    }

    pub fn get(&self, key: &K) -> Result<Option<CacheLease<K>>, CacheError> {
        let mut state = self.lock()?;
        let tick = next_tick(&mut state);
        let Some(entry) = state.entries.get_mut(key) else {
            state.misses = state.misses.saturating_add(1);
            return Ok(None);
        };
        match entry {
            Entry::Loading { .. } => Ok(None),
            Entry::Ready {
                bytes,
                pins,
                last_used,
                ..
            } => {
                *pins = pins.checked_add(1).ok_or(CacheError::ArithmeticOverflow)?;
                *last_used = tick;
                let bytes = Arc::clone(bytes);
                state.hits = state.hits.saturating_add(1);
                Ok(Some(CacheLease {
                    inner: Arc::clone(&self.inner),
                    key: key.clone(),
                    bytes,
                }))
            }
        }
    }

    /// Loads and caches an owned byte vector for a key.
    ///
    /// The loader runs when the key is not already ready in the cache. Its output
    /// must contain exactly `expected_bytes` bytes.
    ///
    /// # Errors
    ///
    /// Returns the loader error or a cache error if loading cannot be admitted or
    /// the loaded data has an invalid size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bridge_cache::{CacheConfig, CompressedCache};
    /// let cache = CompressedCache::new(CacheConfig {
    ///     capacity_bytes: 1024,
    ///     admit_after_requests: 1,
    /// }).unwrap();
    ///
    /// let lease = cache
    ///     .get_or_try_insert("item", 4, || {
    ///         Ok::<_, std::convert::Infallible>(b"data".to_vec())
    ///     })
    ///     .unwrap();
    ///
    /// assert_eq!(lease.bytes(), b"data");
    /// ```
    pub fn get_or_try_insert<E, F>(
        &self,
        key: K,
        expected_bytes: usize,
        loader: F,
    ) -> Result<CacheLease<K>, LoadError<E>>
    where
        E: Error + Send + Sync + 'static,
        F: FnOnce() -> Result<Vec<u8>, E>,
    {
        self.get_or_try_insert_payload(key, expected_bytes, expected_bytes, || {
            loader().map(CachePayload::Owned)
        })
    }

    /// Loads a cache entry into a reusable aligned read slot.
    ///
    /// The loader must provide a slot whose logical length is `expected_bytes` and whose backing storage is at least that large.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use bridge_cache::{CacheConfig, CompressedCache, ReadSlotLease};
    /// # let cache = CompressedCache::new(CacheConfig {
    /// #     capacity_bytes: 4096,
    /// #     admit_after_requests: 1,
    /// # }).unwrap();
    /// let lease = cache.get_or_try_insert_read_slot(
    ///     "block",
    ///     1024,
    ///     || -> Result<ReadSlotLease, std::io::Error> {
    ///         unimplemented!()
    ///     },
    /// )?;
    /// # let _ = lease;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Returns
    ///
    /// The loaded cache lease.
    pub fn get_or_try_insert_read_slot<E, F>(
        &self,
        key: K,
        expected_bytes: usize,
        loader: F,
    ) -> Result<CacheLease<K>, LoadError<E>>
    where
        E: Error + Send + Sync + 'static,
        F: FnOnce() -> Result<ReadSlotLease, E>,
    {
        self.get_or_try_insert_read_slot_charged(key, expected_bytes, expected_bytes, loader)
    }

    /// Loads and caches data in a reusable read slot with explicit capacity charging.
    ///
    /// `expected_bytes` specifies the logical payload length, while `charge_bytes`
    /// specifies the amount reserved against the cache capacity. The loaded read
    /// slot must contain at least `charge_bytes` bytes of backing storage.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use bridge_cache::{CacheConfig, CompressedCache};
    /// # use std::convert::Infallible;
    /// let cache = CompressedCache::new(CacheConfig {
    ///     capacity_bytes: 1024,
    ///     admit_after_requests: 1,
    /// }).unwrap();
    ///
    /// let _lease = cache.get_or_try_insert_read_slot_charged::<Infallible, _>(
    ///     "item".to_owned(),
    ///     128,
    ///     256,
    ///     || panic!("provide a ReadSlotLease"),
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a cache error when the requested sizes are invalid, the loaded
    /// payload has an unexpected length, or its backing storage is too small.
    pub fn get_or_try_insert_read_slot_charged<E, F>(
        &self,
        key: K,
        expected_bytes: usize,
        charge_bytes: usize,
        loader: F,
    ) -> Result<CacheLease<K>, LoadError<E>>
    where
        E: Error + Send + Sync + 'static,
        F: FnOnce() -> Result<ReadSlotLease, E>,
    {
        self.get_or_try_insert_payload(key, expected_bytes, charge_bytes, || {
            loader().map(|lease| CachePayload::ReadSlot {
                lease,
                length: expected_bytes,
            })
        })
    }

    /// Loads and caches a payload for a key, reusing an existing entry or coordinating concurrent loads.
    ///
    /// The payload must have the expected logical length, and its backing storage must cover
    /// `charge_bytes`. Capacity reservations and cache admission are tracked using the supplied
    /// charge.
    ///
    /// # Examples
    ///
    /// ```
    /// let cache = CompressedCache::new(CacheConfig {
    ///     capacity_bytes: 3,
    ///     admit_after_requests: 1,
    /// }).unwrap();
    ///
    /// let lease = cache
    ///     .get_or_try_insert_payload("item".to_string(), 3, 3, || {
    ///         Ok::<_, std::convert::Infallible>(CachePayload::Owned(vec![1, 2, 3]))
    ///     })
    ///     .unwrap();
    ///
    /// assert_eq!(lease.bytes(), &[1, 2, 3]);
    /// ```
    fn get_or_try_insert_payload<E, F>(
        &self,
        key: K,
        expected_bytes: usize,
        charge_bytes: usize,
        loader: F,
    ) -> Result<CacheLease<K>, LoadError<E>>
    where
        E: Error + Send + Sync + 'static,
        F: FnOnce() -> Result<CachePayload, E>,
    {
        if expected_bytes == 0 {
            return Err(LoadError::Cache(CacheError::EmptyEntry));
        }
        if charge_bytes < expected_bytes {
            return Err(LoadError::Cache(CacheError::ChargeTooSmall {
                payload: expected_bytes,
                charge: charge_bytes,
            }));
        }
        if charge_bytes > self.inner.config.capacity_bytes {
            return Err(LoadError::Cache(CacheError::EntryTooLarge {
                requested: charge_bytes,
                capacity: self.inner.config.capacity_bytes,
            }));
        }

        let mut loader = Some(loader);
        loop {
            let mut state = self.lock().map_err(LoadError::Cache)?;
            let tick = next_tick(&mut state);
            match state.entries.get_mut(&key) {
                Some(Entry::Ready {
                    bytes,
                    pins,
                    last_used,
                    ..
                }) => {
                    *pins = pins
                        .checked_add(1)
                        .ok_or(LoadError::Cache(CacheError::ArithmeticOverflow))?;
                    *last_used = tick;
                    let bytes = Arc::clone(bytes);
                    state.hits = state.hits.saturating_add(1);
                    return Ok(CacheLease {
                        inner: Arc::clone(&self.inner),
                        key,
                        bytes,
                    });
                }
                Some(Entry::Loading { .. }) => {
                    state.waits = state.waits.saturating_add(1);
                    state = self
                        .inner
                        .changed
                        .wait(state)
                        .map_err(|_| LoadError::Cache(CacheError::Poisoned))?;
                    drop(state);
                    continue;
                }
                None => {
                    state.misses = state.misses.saturating_add(1);
                    let requests = {
                        let requests = state.heat.entry(key.clone()).or_insert(0);
                        *requests = requests.saturating_add(1);
                        *requests
                    };
                    let resident = requests >= self.inner.config.admit_after_requests;
                    reserve_capacity(&mut state, self.inner.config.capacity_bytes, charge_bytes)
                        .map_err(LoadError::Cache)?;
                    state.reserved_bytes = state
                        .reserved_bytes
                        .checked_add(charge_bytes)
                        .ok_or(LoadError::Cache(CacheError::ArithmeticOverflow))?;
                    state.loads = state.loads.saturating_add(1);
                    state.entries.insert(
                        key.clone(),
                        Entry::Loading {
                            expected_bytes,
                            charge_bytes,
                            resident,
                        },
                    );
                    drop(state);

                    let mut reservation = LoadReservation {
                        inner: Arc::clone(&self.inner),
                        key: key.clone(),
                        expected_bytes,
                        charge_bytes,
                        armed: true,
                    };
                    let bytes = loader
                        .take()
                        .expect("loader is consumed only by the reserving caller")(
                    )
                    .map_err(LoadError::Loader)?;
                    bytes.validate_backing().map_err(LoadError::Cache)?;
                    if bytes.len() != expected_bytes {
                        return Err(LoadError::Cache(CacheError::LoadedSizeMismatch {
                            expected: expected_bytes,
                            actual: bytes.len(),
                        }));
                    }
                    if bytes.backing_len() < charge_bytes {
                        return Err(LoadError::Cache(CacheError::BackingTooSmall {
                            required: charge_bytes,
                            actual: bytes.backing_len(),
                        }));
                    }
                    // Preserve either the loader-owned allocation or the
                    // aligned slot without a slice conversion or byte copy.
                    let bytes = Arc::new(bytes);
                    let mut state = self.lock().map_err(LoadError::Cache)?;
                    let resident = match state.entries.get(&key) {
                        Some(Entry::Loading {
                            expected_bytes: reserved,
                            charge_bytes: reserved_charge,
                            resident,
                        }) if *reserved == expected_bytes && *reserved_charge == charge_bytes => *resident,
                        _ => {
                            drop(state);
                            return Err(LoadError::Cache(CacheError::ReservationLost));
                        }
                    };
                    let Some(reserved_bytes) = state.reserved_bytes.checked_sub(charge_bytes) else {
                        drop(state);
                        return Err(LoadError::Cache(CacheError::ArithmeticOverflow));
                    };
                    let Some(used_bytes) = state.used_bytes.checked_add(charge_bytes) else {
                        drop(state);
                        return Err(LoadError::Cache(CacheError::ArithmeticOverflow));
                    };
                    state.reserved_bytes = reserved_bytes;
                    state.used_bytes = used_bytes;
                    let tick = next_tick(&mut state);
                    state.entries.insert(
                        key.clone(),
                        Entry::Ready {
                            bytes: Arc::clone(&bytes),
                            charge_bytes,
                            pins: 1,
                            last_used: tick,
                            resident,
                        },
                    );
                    reservation.armed = false;
                    self.inner.changed.notify_all();
                    return Ok(CacheLease {
                        inner: Arc::clone(&self.inner),
                        key,
                        bytes,
                    });
                }
            }
        }
    }

    pub fn stats(&self) -> Result<CacheStats, CacheError> {
        let state = self.lock()?;
        let mut pinned_entries = 0;
        let mut resident_entries = 0;
        let mut loading_entries = 0;
        for entry in state.entries.values() {
            match entry {
                Entry::Loading { .. } => loading_entries += 1,
                Entry::Ready { pins, resident, .. } => {
                    if *pins > 0 {
                        pinned_entries += 1;
                    }
                    if *resident {
                        resident_entries += 1;
                    }
                }
            }
        }
        Ok(CacheStats {
            capacity_bytes: self.inner.config.capacity_bytes,
            used_bytes: state.used_bytes,
            reserved_bytes: state.reserved_bytes,
            resident_entries,
            pinned_entries,
            loading_entries,
            heat_entries: state.heat.len(),
            hits: state.hits,
            misses: state.misses,
            loads: state.loads,
            waits: state.waits,
            evictions: state.evictions,
        })
    }

    /// Removes all ready cache entries that have no active leases.
    ///
    /// # Examples
    ///
    /// ```
    /// let cache = CompressedCache::<String>::new(CacheConfig {
    ///     capacity_bytes: 16,
    ///     admit_after_requests: 1,
    /// }).unwrap();
    ///
    /// let lease = cache
    ///     .get_or_try_insert("key".to_owned(), 3, || Ok::<_, ()>(vec![1, 2, 3]))
    ///     .unwrap();
    /// drop(lease);
    ///
    /// assert_eq!(cache.clear_unpinned().unwrap(), 1);
    /// ```
    ///
    /// Returns the number of entries removed.
    pub fn clear_unpinned(&self) -> Result<usize, CacheError> {
        let mut state = self.lock()?;
        let keys = state
            .entries
            .iter()
            .filter_map(|(key, entry)| match entry {
                Entry::Ready { pins: 0, .. } => Some(key.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut removed = 0;
        for key in keys {
            if let Some(Entry::Ready { charge_bytes, .. }) = state.entries.remove(&key) {
                state.used_bytes = state
                    .used_bytes
                    .checked_sub(charge_bytes)
                    .ok_or(CacheError::ArithmeticOverflow)?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    pub fn export_heat_json(&self, maximum_entries: usize) -> Result<Vec<u8>, HeatError>
    where
        K: Ord + Serialize,
    {
        let state = self.lock().map_err(HeatError::Cache)?;
        let mut entries = state
            .heat
            .iter()
            .map(|(key, &requests)| HeatEntry {
                key: key.clone(),
                requests,
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| {
            right
                .requests
                .cmp(&left.requests)
                .then_with(|| left.key.cmp(&right.key))
        });
        entries.truncate(maximum_entries);
        serde_json::to_vec_pretty(&HeatSnapshot {
            format: HEAT_FORMAT.into(),
            version: HEAT_VERSION,
            entries,
        })
        .map_err(HeatError::Json)
    }

    pub fn import_heat_json(
        &self,
        bytes: &[u8],
        maximum_json_bytes: usize,
        maximum_entries: usize,
    ) -> Result<usize, HeatError>
    where
        K: DeserializeOwned,
    {
        if bytes.len() > maximum_json_bytes {
            return Err(HeatError::InputTooLarge {
                actual: bytes.len(),
                maximum: maximum_json_bytes,
            });
        }
        let snapshot: HeatSnapshot<K> = serde_json::from_slice(bytes).map_err(HeatError::Json)?;
        if snapshot.format != HEAT_FORMAT || snapshot.version != HEAT_VERSION {
            return Err(HeatError::WrongFormat {
                format: snapshot.format,
                version: snapshot.version,
            });
        }
        if snapshot.entries.len() > maximum_entries {
            return Err(HeatError::TooManyEntries {
                actual: snapshot.entries.len(),
                maximum: maximum_entries,
            });
        }
        let mut state = self.lock().map_err(HeatError::Cache)?;
        for entry in snapshot.entries {
            if entry.requests == 0 {
                return Err(HeatError::ZeroRequests);
            }
            let requests = state.heat.entry(entry.key).or_insert(0);
            *requests = (*requests).max(entry.requests);
        }
        Ok(state.heat.len())
    }

    fn lock(&self) -> Result<MutexGuard<'_, State<K>>, CacheError> {
        self.inner.state.lock().map_err(|_| CacheError::Poisoned)
    }
}

#[derive(Debug)]
pub struct CacheLease<K>
where
    K: Clone + Eq + Hash,
{
    inner: Arc<Inner<K>>,
    key: K,
    bytes: Arc<CachePayload>,
}

impl<K> CacheLease<K>
where
    K: Clone + Eq + Hash,
{
    /// Provides the cached payload as a byte slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let cache = CompressedCache::new(CacheConfig {
    ///     capacity_bytes: 1024,
    ///     admit_after_requests: 1,
    /// })
    /// .unwrap();
    /// let lease = cache
    ///     .get_or_try_insert::<(), _>(1u64, 3, || Ok(vec![1, 2, 3]))
    ///     .unwrap();
    ///
    /// assert_eq!(lease.bytes(), &[1, 2, 3]);
    /// ```
    ///
    /// # Returns
    ///
    /// A byte slice containing the cached payload.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.bytes()
    }

    /// Returns the key associated with this lease.
    ///
    /// # Examples
    ///
    /// ```
    /// let cache = CompressedCache::new(CacheConfig {
    ///     capacity_bytes: 3,
    ///     admit_after_requests: 1,
    /// }).unwrap();
    /// let lease = cache
    ///     .get_or_try_insert("key", 3, || Ok::<_, ()>(vec![1, 2, 3]))
    ///     .unwrap();
    ///
    /// assert_eq!(lease.key(), &"key");
    /// ```
    pub fn key(&self) -> &K {
        &self.key
    }
}

impl<K> Drop for CacheLease<K>
where
    K: Clone + Eq + Hash,
{
    /// Releases the lease's pin and removes the entry when it is unresident and no longer pinned.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// drop(lease);
    /// ```
    fn drop(&mut self) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let remove_ephemeral = match state.entries.get_mut(&self.key) {
            Some(Entry::Ready { pins, resident, .. }) => {
                *pins = pins.saturating_sub(1);
                *pins == 0 && !*resident
            }
            _ => false,
        };
        if remove_ephemeral {
            if let Some(Entry::Ready { charge_bytes, .. }) = state.entries.remove(&self.key) {
                state.used_bytes = state.used_bytes.saturating_sub(charge_bytes);
            }
        }
        self.inner.changed.notify_all();
    }
}

struct LoadReservation<K>
where
    K: Clone + Eq + Hash,
{
    inner: Arc<Inner<K>>,
    key: K,
    expected_bytes: usize,
    charge_bytes: usize,
    armed: bool,
}

impl<K> Drop for LoadReservation<K>
where
    K: Clone + Eq + Hash,
{
    /// Releases an active loading reservation when it is dropped.
    ///
    /// Removes the matching loading entry, releases its reserved capacity, and
    /// wakes threads waiting for the cache state to change.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn example() {
    /// // A loading reservation is released automatically when it leaves scope.
    /// # }
    /// ```
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let matches = matches!(
            state.entries.get(&self.key),
            Some(Entry::Loading {
                expected_bytes,
                charge_bytes,
                ..
            }) if *expected_bytes == self.expected_bytes && *charge_bytes == self.charge_bytes
        );
        if matches {
            state.entries.remove(&self.key);
            state.reserved_bytes = state.reserved_bytes.saturating_sub(self.charge_bytes);
        }
        self.inner.changed.notify_all();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CacheStats {
    pub capacity_bytes: usize,
    pub used_bytes: usize,
    pub reserved_bytes: usize,
    pub resident_entries: usize,
    pub pinned_entries: usize,
    pub loading_entries: usize,
    pub heat_entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub loads: u64,
    pub waits: u64,
    pub evictions: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CacheError {
    #[error("cache capacity must be non-zero")]
    ZeroCapacity,
    #[error("cache admission threshold must be non-zero")]
    ZeroAdmissionThreshold,
    #[error("cache entries must be non-empty")]
    EmptyEntry,
    #[error("entry needs {requested} bytes but cache capacity is {capacity}")]
    EntryTooLarge { requested: usize, capacity: usize },
    #[error("cache charge {charge} bytes is smaller than payload {payload} bytes")]
    ChargeTooSmall { payload: usize, charge: usize },
    #[error("cache cannot reserve {requested} bytes because resident or in-flight entries are pinned")]
    CapacityPinned { requested: usize },
    #[error("loader returned {actual} bytes, expected exactly {expected}")]
    LoadedSizeMismatch { expected: usize, actual: usize },
    #[error("cache payload backing is {actual} bytes, required {required}")]
    BackingTooSmall { required: usize, actual: usize },
    #[error("cache load reservation disappeared before completion")]
    ReservationLost,
    #[error("cache synchronization state is poisoned")]
    Poisoned,
    #[error("checked arithmetic overflow in cache accounting")]
    ArithmeticOverflow,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError<E: Error + 'static> {
    #[error("cache loader failed: {0}")]
    Loader(#[source] E),
    #[error(transparent)]
    Cache(#[from] CacheError),
}

#[derive(Debug, thiserror::Error)]
pub enum HeatError {
    #[error(transparent)]
    Cache(#[from] CacheError),
    #[error("cache heat JSON failed: {0}")]
    Json(serde_json::Error),
    #[error("cache heat input is {actual} bytes, maximum is {maximum}")]
    InputTooLarge { actual: usize, maximum: usize },
    #[error("cache heat snapshot has format {format:?} version {version}")]
    WrongFormat { format: String, version: u32 },
    #[error("cache heat snapshot has {actual} entries, maximum is {maximum}")]
    TooManyEntries { actual: usize, maximum: usize },
    #[error("cache heat snapshot contains a zero request count")]
    ZeroRequests,
}

#[derive(Debug, Serialize, Deserialize)]
struct HeatSnapshot<K> {
    format: String,
    version: u32,
    entries: Vec<HeatEntry<K>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HeatEntry<K> {
    key: K,
    requests: u64,
}

fn next_tick<K>(state: &mut State<K>) -> u64 {
    state.tick = state.tick.saturating_add(1);
    state.tick
}

/// Ensures a requested reservation fits within the cache capacity by evicting the least recently used unpinned entries when necessary.
///
/// # Errors
///
/// Returns [`CacheError::CapacityPinned`] if capacity cannot be made available because all eligible entries are pinned. Returns [`CacheError::ArithmeticOverflow`] if accounting arithmetic overflows or becomes inconsistent.
///
/// # Examples
///
/// ```
/// let mut state = State::default();
/// assert!(reserve_capacity(&mut state, 1024, 512).is_ok());
/// ```
fn reserve_capacity<K>(state: &mut State<K>, capacity: usize, requested: usize) -> Result<(), CacheError>
where
    K: Clone + Eq + Hash,
{
    loop {
        let committed = state
            .used_bytes
            .checked_add(state.reserved_bytes)
            .and_then(|bytes| bytes.checked_add(requested))
            .ok_or(CacheError::ArithmeticOverflow)?;
        if committed <= capacity {
            return Ok(());
        }
        let candidate = state
            .entries
            .iter()
            .filter_map(|(key, entry)| match entry {
                Entry::Ready {
                    pins: 0, last_used, ..
                } => Some((key.clone(), *last_used)),
                _ => None,
            })
            .min_by_key(|(_, last_used)| *last_used)
            .map(|(key, _)| key);
        let Some(candidate) = candidate else {
            return Err(CacheError::CapacityPinned { requested });
        };
        if let Some(Entry::Ready { charge_bytes, .. }) = state.entries.remove(&candidate) {
            state.used_bytes = state
                .used_bytes
                .checked_sub(charge_bytes)
                .ok_or(CacheError::ArithmeticOverflow)?;
            state.evictions = state.evictions.saturating_add(1);
        }
    }
}
