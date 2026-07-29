use crate::error::Result;
use crate::KvError;
use sha2::{Digest, Sha256};

pub const KV_SNAPSHOT_FORMAT: &str = "lightbridge-kv";
pub const KV_SNAPSHOT_VERSION: u32 = 1;
const KV_SNAPSHOT_MAGIC: &[u8; 8] = b"LBKV0001";
const KV_SNAPSHOT_DIGEST_BYTES: usize = 32;

#[derive(Debug)]
pub struct PagedKvCache {
    layer_count: usize,
    kv_head_count: usize,
    key_dimension: usize,
    value_dimension: usize,
    page_tokens: usize,
    page_count: usize,
    token_capacity: usize,
    lengths: Vec<usize>,
    pages: Vec<Vec<Option<KvPage>>>,
}

#[derive(Debug)]
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl PagedKvCache {
    /// Constructs an eagerly backed cache.
    ///
    /// This preserves allocation-free token appends for bounded contexts.
    pub fn new(
        layer_count: usize,
        kv_head_count: usize,
        key_dimension: usize,
        value_dimension: usize,
        page_tokens: usize,
        token_capacity: usize,
    ) -> Result<Self> {
        Self::new_with_allocation(
            layer_count,
            kv_head_count,
            key_dimension,
            value_dimension,
            page_tokens,
            token_capacity,
            true,
        )
    }

    /// Constructs a virtually large cache whose payload pages are allocated
    /// only when their token range is first used.
    pub fn new_lazy(
        layer_count: usize,
        kv_head_count: usize,
        key_dimension: usize,
        value_dimension: usize,
        page_tokens: usize,
        token_capacity: usize,
    ) -> Result<Self> {
        Self::new_with_allocation(
            layer_count,
            kv_head_count,
            key_dimension,
            value_dimension,
            page_tokens,
            token_capacity,
            false,
        )
    }

    fn new_with_allocation(
        layer_count: usize,
        kv_head_count: usize,
        key_dimension: usize,
        value_dimension: usize,
        page_tokens: usize,
        token_capacity: usize,
        eager: bool,
    ) -> Result<Self> {
        for (field, value) in [
            ("layer_count", layer_count),
            ("kv_head_count", kv_head_count),
            ("key_dimension", key_dimension),
            ("value_dimension", value_dimension),
            ("page_tokens", page_tokens),
            ("token_capacity", token_capacity),
        ] {
            if value == 0 {
                return Err(KvError::ZeroParameter { field });
            }
        }
        let page_count = token_capacity
            .checked_add(page_tokens - 1)
            .ok_or(KvError::ArithmeticOverflow {
                operation: "KV page count rounding",
            })?
            / page_tokens;
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(layer_count)
            .map_err(|_| KvError::AllocationFailed {
                field: "layer page tables",
                elements: layer_count,
            })?;
        for _ in 0..layer_count {
            let mut layer_pages = Vec::new();
            layer_pages
                .try_reserve_exact(page_count)
                .map_err(|_| KvError::AllocationFailed {
                    field: "page table",
                    elements: page_count,
                })?;
            layer_pages.resize_with(page_count, || None);
            if eager {
                for page in &mut layer_pages {
                    *page = Some(KvPage::new(
                        page_tokens,
                        kv_head_count,
                        key_dimension,
                        value_dimension,
                    )?);
                }
            }
            pages.push(layer_pages);
        }

        Ok(Self {
            layer_count,
            kv_head_count,
            key_dimension,
            value_dimension,
            page_tokens,
            page_count,
            token_capacity,
            lengths: zeroed_vec(layer_count, "layer lengths")?,
            pages,
        })
    }

    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub const fn kv_head_count(&self) -> usize {
        self.kv_head_count
    }

    pub const fn key_dimension(&self) -> usize {
        self.key_dimension
    }

    pub const fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    pub const fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    pub const fn page_count(&self) -> usize {
        self.page_count
    }

    pub const fn token_capacity(&self) -> usize {
        self.token_capacity
    }

    pub fn allocated_page_count(&self) -> usize {
        self.pages
            .iter()
            .map(|pages| pages.iter().filter(|page| page.is_some()).count())
            .sum()
    }

    pub fn allocated_bytes(&self) -> Result<usize> {
        self.pages
            .iter()
            .flat_map(|pages| pages.iter().flatten())
            .try_fold(0_usize, |total, page| {
                let page_bytes = page
                    .keys
                    .len()
                    .checked_add(page.values.len())
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or(KvError::ArithmeticOverflow {
                        operation: "allocated KV bytes",
                    })?;
                total.checked_add(page_bytes).ok_or(KvError::ArithmeticOverflow {
                    operation: "allocated KV bytes",
                })
            })
    }

    pub fn stored_tokens(&self, layer: usize) -> Result<usize> {
        self.validate_layer(layer)?;
        Ok(self.lengths[layer])
    }

    pub fn remaining_tokens(&self, layer: usize) -> Result<usize> {
        Ok(self.token_capacity - self.stored_tokens(layer)?)
    }

    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<usize> {
        self.append_tokens(layer, 1, key, value)?;
        Ok(self.lengths[layer] - 1)
    }

    /// Atomically appends token-major `[token, kv_head, dimension]` K/V rows.
    pub fn append_tokens(
        &mut self,
        layer: usize,
        token_count: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        self.validate_layer(layer)?;
        if token_count == 0 {
            return Err(KvError::ZeroParameter {
                field: "append token_count",
            });
        }
        let keys_per_token =
            self.kv_head_count
                .checked_mul(self.key_dimension)
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "keys per token",
                })?;
        let values_per_token =
            self.kv_head_count
                .checked_mul(self.value_dimension)
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "values per token",
                })?;
        let expected_keys = token_count
            .checked_mul(keys_per_token)
            .ok_or(KvError::ArithmeticOverflow {
                operation: "appended key length",
            })?;
        let expected_values =
            token_count
                .checked_mul(values_per_token)
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "appended value length",
                })?;
        require_length("append keys", expected_keys, keys.len())?;
        require_length("append values", expected_values, values.len())?;
        validate_finite("append keys", keys)?;
        validate_finite("append values", values)?;

        let stored = self.lengths[layer];
        let new_length = stored
            .checked_add(token_count)
            .ok_or(KvError::ArithmeticOverflow {
                operation: "appended token count",
            })?;
        if new_length > self.token_capacity {
            return Err(KvError::CapacityExhausted {
                layer,
                stored,
                additional: token_count,
                capacity: self.token_capacity,
            });
        }
        self.ensure_pages(layer, stored, new_length)?;
        let key_dimension = self.key_dimension;
        let value_dimension = self.value_dimension;

        for token_offset in 0..token_count {
            let token = stored + token_offset;
            for head in 0..self.kv_head_count {
                let source_key = (token_offset * self.kv_head_count + head) * key_dimension;
                let (page, target_key) = self.page_and_key_offset(layer, token, head)?;
                page.keys[target_key..target_key + key_dimension]
                    .copy_from_slice(&keys[source_key..source_key + key_dimension]);

                let source_value = (token_offset * self.kv_head_count + head) * value_dimension;
                let (page, target_value) = self.page_and_value_offset(layer, token, head)?;
                page.values[target_value..target_value + value_dimension]
                    .copy_from_slice(&values[source_value..source_value + value_dimension]);
            }
        }
        self.lengths[layer] = new_length;
        Ok(())
    }

    pub fn key(&self, layer: usize, token: usize, head: usize) -> Result<&[f32]> {
        self.validate_indices(layer, token, head)?;
        let page = self.page(layer, token)?;
        let offset = self.key_offset(token, head);
        Ok(&page.keys[offset..offset + self.key_dimension])
    }

    pub fn value(&self, layer: usize, token: usize, head: usize) -> Result<&[f32]> {
        self.validate_indices(layer, token, head)?;
        let page = self.page(layer, token)?;
        let offset = self.value_offset(token, head);
        Ok(&page.values[offset..offset + self.value_dimension])
    }

    pub fn reset(&mut self) {
        self.lengths.fill(0);
    }

    /// Atomically rewinds every layer to a previously committed token count.
    ///
    /// Backing pages remain allocated and are overwritten by later appends.
    pub fn rewind_all(&mut self, token_count: usize) -> Result<()> {
        for (layer, &stored) in self.lengths.iter().enumerate() {
            if token_count > stored {
                return Err(KvError::RewindBeyondStored {
                    layer,
                    requested: token_count,
                    stored,
                });
            }
        }
        self.lengths.fill(token_count);
        Ok(())
    }

    /// Serializes committed KV rows into a bounded, checksummed snapshot.
    ///
    /// `model_binding` is an opaque caller-provided fingerprint that prevents
    /// restoring state into a different checkpoint with matching dimensions.
    pub fn export_snapshot(&self, model_binding: [u8; 32], maximum_bytes: usize) -> Result<Vec<u8>> {
        if maximum_bytes == 0 {
            return Err(KvError::ZeroSnapshotLimit);
        }
        let header_bytes = KV_SNAPSHOT_MAGIC
            .len()
            .checked_add(std::mem::size_of::<u32>())
            .and_then(|value| value.checked_add(model_binding.len()))
            .and_then(|value| value.checked_add(6 * std::mem::size_of::<u64>()))
            .and_then(|value| value.checked_add(self.layer_count.checked_mul(8)?))
            .and_then(|value| value.checked_add(KV_SNAPSHOT_DIGEST_BYTES))
            .ok_or(KvError::ArithmeticOverflow {
                operation: "KV snapshot header bytes",
            })?;
        let values_per_token = self
            .kv_head_count
            .checked_mul(self.key_dimension.checked_add(self.value_dimension).ok_or(
                KvError::ArithmeticOverflow {
                    operation: "KV snapshot dimensions",
                },
            )?)
            .ok_or(KvError::ArithmeticOverflow {
                operation: "KV snapshot values per token",
            })?;
        let payload_bytes = self.lengths.iter().try_fold(0_usize, |total, &tokens| {
            let bytes = tokens
                .checked_mul(values_per_token)
                .and_then(|value| value.checked_mul(4))
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "KV snapshot payload bytes",
                })?;
            total.checked_add(bytes).ok_or(KvError::ArithmeticOverflow {
                operation: "KV snapshot payload bytes",
            })
        })?;
        let snapshot_bytes = header_bytes
            .checked_add(payload_bytes)
            .ok_or(KvError::ArithmeticOverflow {
                operation: "KV snapshot total bytes",
            })?;
        if snapshot_bytes > maximum_bytes {
            return Err(KvError::SnapshotTooLarge {
                actual: snapshot_bytes,
                maximum: maximum_bytes,
            });
        }

        let mut output = Vec::new();
        output
            .try_reserve_exact(snapshot_bytes)
            .map_err(|_| KvError::AllocationFailed {
                field: "KV snapshot bytes",
                elements: snapshot_bytes,
            })?;
        output.extend_from_slice(KV_SNAPSHOT_MAGIC);
        output.extend_from_slice(&KV_SNAPSHOT_VERSION.to_le_bytes());
        output.extend_from_slice(&model_binding);
        for value in [
            self.layer_count,
            self.kv_head_count,
            self.key_dimension,
            self.value_dimension,
            self.page_tokens,
            self.token_capacity,
        ] {
            output.extend_from_slice(&usize_to_u64(value)?.to_le_bytes());
        }
        for &length in &self.lengths {
            output.extend_from_slice(&usize_to_u64(length)?.to_le_bytes());
        }
        for layer in 0..self.layer_count {
            for token in 0..self.lengths[layer] {
                for head in 0..self.kv_head_count {
                    for &value in self.key(layer, token, head)? {
                        output.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                    for &value in self.value(layer, token, head)? {
                        output.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                }
            }
        }
        let digest = Sha256::digest(&output);
        output.extend_from_slice(&digest);
        debug_assert_eq!(output.len(), snapshot_bytes);
        Ok(output)
    }

    /// Atomically replaces this cache from a validated snapshot.
    pub fn restore_snapshot(
        &mut self,
        expected_model_binding: [u8; 32],
        bytes: &[u8],
        maximum_bytes: usize,
    ) -> Result<()> {
        self.restore_snapshot_internal(expected_model_binding, bytes, maximum_bytes, false)
    }

    /// Restores a causal-model snapshot only if every layer has the same
    /// committed token count.
    pub fn restore_uniform_snapshot(
        &mut self,
        expected_model_binding: [u8; 32],
        bytes: &[u8],
        maximum_bytes: usize,
    ) -> Result<()> {
        self.restore_snapshot_internal(expected_model_binding, bytes, maximum_bytes, true)
    }

    fn restore_snapshot_internal(
        &mut self,
        expected_model_binding: [u8; 32],
        bytes: &[u8],
        maximum_bytes: usize,
        require_uniform_lengths: bool,
    ) -> Result<()> {
        if maximum_bytes == 0 {
            return Err(KvError::ZeroSnapshotLimit);
        }
        if bytes.len() > maximum_bytes {
            return Err(KvError::SnapshotTooLarge {
                actual: bytes.len(),
                maximum: maximum_bytes,
            });
        }
        if bytes.len() < KV_SNAPSHOT_DIGEST_BYTES {
            return Err(KvError::SnapshotTruncated {
                offset: bytes.len(),
                needed: KV_SNAPSHOT_DIGEST_BYTES - bytes.len(),
            });
        }
        let payload_end = bytes.len() - KV_SNAPSHOT_DIGEST_BYTES;
        let expected_digest = Sha256::digest(&bytes[..payload_end]);
        if expected_digest.as_slice() != &bytes[payload_end..] {
            return Err(KvError::SnapshotChecksum);
        }

        let mut cursor = SnapshotCursor::new(&bytes[..payload_end]);
        if cursor.take(KV_SNAPSHOT_MAGIC.len())? != KV_SNAPSHOT_MAGIC {
            return Err(KvError::SnapshotMagic);
        }
        let version = cursor.u32()?;
        if version != KV_SNAPSHOT_VERSION {
            return Err(KvError::SnapshotVersion {
                expected: KV_SNAPSHOT_VERSION,
                actual: version,
            });
        }
        if cursor.take(expected_model_binding.len())? != expected_model_binding {
            return Err(KvError::SnapshotBinding);
        }
        for (field, expected) in [
            ("layer_count", self.layer_count),
            ("kv_head_count", self.kv_head_count),
            ("key_dimension", self.key_dimension),
            ("value_dimension", self.value_dimension),
            ("page_tokens", self.page_tokens),
            ("token_capacity", self.token_capacity),
        ] {
            let actual = u64_to_usize(cursor.u64()?)?;
            if actual != expected {
                return Err(KvError::SnapshotConfiguration {
                    field,
                    expected,
                    actual,
                });
            }
        }
        let mut lengths = Vec::new();
        lengths
            .try_reserve_exact(self.layer_count)
            .map_err(|_| KvError::AllocationFailed {
                field: "KV snapshot layer lengths",
                elements: self.layer_count,
            })?;
        for layer in 0..self.layer_count {
            let length = u64_to_usize(cursor.u64()?)?;
            if length > self.token_capacity {
                return Err(KvError::CapacityExhausted {
                    layer,
                    stored: 0,
                    additional: length,
                    capacity: self.token_capacity,
                });
            }
            lengths.push(length);
        }
        if require_uniform_lengths {
            let expected = lengths.first().copied().unwrap_or(0);
            if let Some((layer, &actual)) = lengths
                .iter()
                .enumerate()
                .find(|(_, length)| **length != expected)
            {
                return Err(KvError::SnapshotLayerLength {
                    layer,
                    expected,
                    actual,
                });
            }
        }

        let mut restored = Self::new_lazy(
            self.layer_count,
            self.kv_head_count,
            self.key_dimension,
            self.value_dimension,
            self.page_tokens,
            self.token_capacity,
        )?;
        let keys_per_token =
            self.kv_head_count
                .checked_mul(self.key_dimension)
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "KV snapshot keys per token",
                })?;
        let values_per_token =
            self.kv_head_count
                .checked_mul(self.value_dimension)
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "KV snapshot values per token",
                })?;
        let mut keys = zeroed_vec(keys_per_token, "KV snapshot token keys")?;
        let mut values = zeroed_vec(values_per_token, "KV snapshot token values")?;
        for (layer, &length) in lengths.iter().enumerate() {
            for _ in 0..length {
                for head in 0..self.kv_head_count {
                    let key_start = head * self.key_dimension;
                    for lane in 0..self.key_dimension {
                        keys[key_start + lane] = cursor.f32()?;
                    }
                    let value_start = head * self.value_dimension;
                    for lane in 0..self.value_dimension {
                        values[value_start + lane] = cursor.f32()?;
                    }
                }
                restored.append(layer, &keys, &values)?;
            }
        }
        if cursor.remaining() != 0 {
            return Err(KvError::SnapshotTrailingBytes {
                actual: cursor.remaining(),
            });
        }
        *self = restored;
        Ok(())
    }

    fn validate_layer(&self, layer: usize) -> Result<()> {
        if layer < self.layer_count {
            Ok(())
        } else {
            Err(KvError::LayerOutOfRange {
                layer,
                layer_count: self.layer_count,
            })
        }
    }

    fn validate_indices(&self, layer: usize, token: usize, head: usize) -> Result<()> {
        self.validate_layer(layer)?;
        if token >= self.lengths[layer] {
            return Err(KvError::TokenOutOfRange {
                token,
                stored_tokens: self.lengths[layer],
            });
        }
        if head >= self.kv_head_count {
            return Err(KvError::HeadOutOfRange {
                head,
                head_count: self.kv_head_count,
            });
        }
        Ok(())
    }

    fn ensure_pages(&mut self, layer: usize, start: usize, end: usize) -> Result<()> {
        let first = start / self.page_tokens;
        let last = (end - 1) / self.page_tokens;
        let missing = self.pages[layer][first..=last]
            .iter()
            .filter(|page| page.is_none())
            .count();
        if missing == 0 {
            return Ok(());
        }
        let mut allocated = Vec::new();
        allocated
            .try_reserve_exact(missing)
            .map_err(|_| KvError::AllocationFailed {
                field: "new KV pages",
                elements: missing,
            })?;
        for page_index in first..=last {
            if self.pages[layer][page_index].is_none() {
                allocated.push((
                    page_index,
                    KvPage::new(
                        self.page_tokens,
                        self.kv_head_count,
                        self.key_dimension,
                        self.value_dimension,
                    )?,
                ));
            }
        }
        for (page_index, page) in allocated {
            self.pages[layer][page_index] = Some(page);
        }
        Ok(())
    }

    fn page(&self, layer: usize, token: usize) -> Result<&KvPage> {
        self.pages[layer][token / self.page_tokens]
            .as_ref()
            .ok_or(KvError::ArithmeticOverflow {
                operation: "missing allocated KV page",
            })
    }

    fn page_and_key_offset(
        &mut self,
        layer: usize,
        token: usize,
        head: usize,
    ) -> Result<(&mut KvPage, usize)> {
        let offset = self.key_offset(token, head);
        let page =
            self.pages[layer][token / self.page_tokens]
                .as_mut()
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "missing allocated KV page",
                })?;
        Ok((page, offset))
    }

    fn page_and_value_offset(
        &mut self,
        layer: usize,
        token: usize,
        head: usize,
    ) -> Result<(&mut KvPage, usize)> {
        let offset = self.value_offset(token, head);
        let page =
            self.pages[layer][token / self.page_tokens]
                .as_mut()
                .ok_or(KvError::ArithmeticOverflow {
                    operation: "missing allocated KV page",
                })?;
        Ok((page, offset))
    }

    fn key_offset(&self, token: usize, head: usize) -> usize {
        ((token % self.page_tokens) * self.kv_head_count + head) * self.key_dimension
    }

    fn value_offset(&self, token: usize, head: usize) -> usize {
        ((token % self.page_tokens) * self.kv_head_count + head) * self.value_dimension
    }
}

struct SnapshotCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SnapshotCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(KvError::ArithmeticOverflow {
                operation: "KV snapshot cursor",
            })?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(KvError::SnapshotTruncated {
                offset: self.offset,
                needed: end.saturating_sub(self.bytes.len()),
            })?;
        self.offset = end;
        Ok(value)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| {
            KvError::ArithmeticOverflow {
                operation: "KV snapshot U32",
            }
        })?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| {
            KvError::ArithmeticOverflow {
                operation: "KV snapshot U64",
            }
        })?))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn usize_to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| KvError::ArithmeticOverflow {
        operation: "KV snapshot integer encoding",
    })
}

fn u64_to_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| KvError::ArithmeticOverflow {
        operation: "KV snapshot integer decoding",
    })
}

impl KvPage {
    fn new(
        page_tokens: usize,
        kv_head_count: usize,
        key_dimension: usize,
        value_dimension: usize,
    ) -> Result<Self> {
        let key_elements = page_tokens
            .checked_mul(kv_head_count)
            .and_then(|value| value.checked_mul(key_dimension))
            .ok_or(KvError::ArithmeticOverflow {
                operation: "key page storage",
            })?;
        let value_elements = page_tokens
            .checked_mul(kv_head_count)
            .and_then(|value| value.checked_mul(value_dimension))
            .ok_or(KvError::ArithmeticOverflow {
                operation: "value page storage",
            })?;
        Ok(Self {
            keys: zeroed_vec(key_elements, "key page storage")?,
            values: zeroed_vec(value_elements, "value page storage")?,
        })
    }
}

fn zeroed_vec<T>(elements: usize, field: &'static str) -> Result<Vec<T>>
where
    T: Clone + Default,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| KvError::AllocationFailed { field, elements })?;
    values.resize(elements, T::default());
    Ok(values)
}

fn require_length(field: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(KvError::LengthMismatch {
            field,
            expected,
            actual,
        })
    }
}

fn validate_finite(field: &'static str, values: &[f32]) -> Result<()> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(KvError::NonFiniteValue {
                field,
                index,
                bits: value.to_bits(),
            });
        }
    }
    Ok(())
}
