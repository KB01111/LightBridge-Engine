//! Host inspection for GGUF ingestion: CPU topology, instruction sets, and memory.
//!
//! The Ryzen AI 9 HX 370 is a *heterogeneous* part (Zen 5 + Zen 5c) with SMT. The engine must not
//! assume that all logical processors perform identically, so this module reports the real
//! topology — physical core count, SMT grouping, and Windows' efficiency class per core — and
//! `bridge doctor` then *measures* each group rather than trusting the labels.
//!
//! # Unsafe invariants
//!
//! The Win32 calls below are declared locally instead of pulled from a binding crate so the FFI
//! signatures are visible at the call site. For each:
//!
//! * `GlobalMemoryStatusEx` requires `dwLength` to be set to `size_of::<MEMORYSTATUSEX>()`
//!   before the call; we do that and the struct is `#[repr(C)]`.
//! * `GetLogicalProcessorInformationEx` is called twice: once with a null buffer to learn the
//!   required length, then with a `Vec<u8>` of exactly that length. We only read inside
//!   `returned_length` bytes and advance by each record's self-reported `size`, which is how the
//!   API defines iteration.
//! * `SetThreadAffinityMask` takes the current pseudo-handle, which needs no closing.

/// Instruction-set features relevant to the CPU kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct IsaSupport {
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub bmi2: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    pub avx_vnni: bool,
}

impl IsaSupport {
    pub fn detect() -> IsaSupport {
        #[cfg(target_arch = "x86_64")]
        {
            IsaSupport {
                sse2: is_x86_feature_detected!("sse2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                f16c: is_x86_feature_detected!("f16c"),
                bmi2: is_x86_feature_detected!("bmi2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                avx512vnni: is_x86_feature_detected!("avx512vnni"),
                avx512bf16: is_x86_feature_detected!("avx512bf16"),
                avx_vnni: is_x86_feature_detected!("avxvnni"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            IsaSupport::default()
        }
    }

    /// Compact, stable string used in autotune fingerprints.
    pub fn tag(&self) -> String {
        let mut v: Vec<&str> = Vec::new();
        if self.avx2 {
            v.push("avx2");
        }
        if self.fma {
            v.push("fma");
        }
        if self.f16c {
            v.push("f16c");
        }
        if self.avx512f {
            v.push("avx512f");
        }
        if self.avx512bw {
            v.push("avx512bw");
        }
        if self.avx512vl {
            v.push("avx512vl");
        }
        if self.avx512vnni {
            v.push("avx512vnni");
        }
        if self.avx512bf16 {
            v.push("avx512bf16");
        }
        if self.avx_vnni {
            v.push("avxvnni");
        }
        if v.is_empty() {
            "baseline".to_string()
        } else {
            v.join("+")
        }
    }
}

/// One physical core and the logical processors that share it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PhysicalCore {
    /// Logical processor indices (global, across groups) belonging to this core.
    pub logical: Vec<u32>,
    /// Windows processor group.
    pub group: u16,
    /// Windows affinity mask within the group.
    pub mask: u64,
    /// Windows `EfficiencyClass`: higher is a higher-performance core. On the HX 370 the Zen 5
    /// cores and the denser Zen 5c cores report different classes.
    pub efficiency_class: u8,
}

impl PhysicalCore {
    pub fn is_smt(&self) -> bool {
        self.logical.len() > 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CpuTopology {
    pub brand: String,
    pub logical_processors: u32,
    pub physical_cores: Vec<PhysicalCore>,
    pub isa: IsaSupport,
}

impl CpuTopology {
    pub fn n_physical(&self) -> usize {
        self.physical_cores.len()
    }

    pub fn smt_enabled(&self) -> bool {
        self.physical_cores.iter().any(|c| c.is_smt())
    }

    /// Distinct efficiency classes present, descending (fastest first).
    pub fn efficiency_classes(&self) -> Vec<u8> {
        let mut v: Vec<u8> = self.physical_cores.iter().map(|c| c.efficiency_class).collect();
        v.sort_unstable();
        v.dedup();
        v.reverse();
        v
    }

    pub fn is_heterogeneous(&self) -> bool {
        self.efficiency_classes().len() > 1
    }

    /// One logical processor per physical core, fastest class first. This is the default worker
    /// set for memory-bound decode; `bridge doctor` measures whether adding the SMT siblings
    /// helps before the runtime uses them.
    pub fn one_thread_per_core(&self) -> Vec<u32> {
        let mut cores: Vec<&PhysicalCore> = self.physical_cores.iter().collect();
        cores.sort_by_key(|c| std::cmp::Reverse(c.efficiency_class));
        cores.iter().filter_map(|c| c.logical.first().copied()).collect()
    }

    pub fn cores_in_class(&self, class: u8) -> Vec<&PhysicalCore> {
        self.physical_cores
            .iter()
            .filter(|c| c.efficiency_class == class)
            .collect()
    }

    pub fn detect() -> CpuTopology {
        CpuTopology {
            brand: cpu_brand(),
            logical_processors: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            physical_cores: detect_physical_cores(),
            isa: IsaSupport::detect(),
        }
    }
}

fn cpu_brand() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        if let Some(b) = cpuid.get_processor_brand_string() {
            return b.as_str().trim().to_string();
        }
        if let Some(v) = cpuid.get_vendor_info() {
            return v.as_str().to_string();
        }
    }
    "unknown".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryStatus {
    pub total_physical: u64,
    pub available_physical: u64,
    pub total_pagefile: u64,
    pub available_pagefile: u64,
}

// ---------------------------------------------------------------------------------------------
// Windows implementation
// ---------------------------------------------------------------------------------------------

#[cfg(windows)]
mod win {
    use super::{MemoryStatus, PhysicalCore};

    #[repr(C)]
    #[derive(Default)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    const RELATION_PROCESSOR_CORE: u32 = 0;
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
    const PROCESSOR_INFO_HEADER_BYTES: usize = 8;
    const PROCESSOR_RELATIONSHIP_BYTES: usize = 24;
    const GROUP_AFFINITY_BYTES: usize = std::mem::size_of::<usize>() + 8;

    extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MemoryStatusEx) -> i32;
        fn GetLogicalProcessorInformationEx(
            RelationshipType: u32,
            Buffer: *mut u8,
            ReturnedLength: *mut u32,
        ) -> i32;
        fn GetLastError() -> u32;
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(hThread: isize, dwThreadAffinityMask: usize) -> usize;
    }

    pub fn memory_status() -> MemoryStatus {
        let mut ms = MemoryStatusEx {
            dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
            ..Default::default()
        };
        // SAFETY: `dw_length` is initialized to the exact struct size as the API requires, and
        // the pointer refers to a live, correctly aligned `#[repr(C)]` value.
        let ok = unsafe { GlobalMemoryStatusEx(&mut ms) };
        if ok == 0 {
            return MemoryStatus {
                total_physical: 0,
                available_physical: 0,
                total_pagefile: 0,
                available_pagefile: 0,
            };
        }
        MemoryStatus {
            total_physical: ms.ull_total_phys,
            available_physical: ms.ull_avail_phys,
            total_pagefile: ms.ull_total_page_file,
            available_pagefile: ms.ull_avail_page_file,
        }
    }

    pub fn physical_cores() -> Vec<PhysicalCore> {
        let mut len: u32 = 0;
        // SAFETY: the documented "query required size" form: null buffer, out-param length.
        let rc = unsafe {
            GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len)
        };
        // SAFETY: no preconditions.
        if rc != 0 || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER || len == 0 {
            return Vec::new();
        }
        let requested_len = match usize::try_from(len) {
            Ok(value) => value,
            Err(_) => return Vec::new(),
        };
        let mut buf = vec![0u8; requested_len];
        // SAFETY: `buf` is exactly `len` bytes as reported by the previous call.
        let rc =
            unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len) };
        if rc == 0 {
            return Vec::new();
        }

        let returned_len = match usize::try_from(len) {
            Ok(value) if value <= buf.len() => value,
            _ => return Vec::new(),
        };
        parse_physical_cores(&buf[..returned_len])
    }

    pub(super) fn parse_physical_cores(bytes: &[u8]) -> Vec<PhysicalCore> {
        let mut cores = Vec::new();
        let mut offset = 0usize;
        while offset < bytes.len() {
            let header_end = match offset.checked_add(PROCESSOR_INFO_HEADER_BYTES) {
                Some(value) if value <= bytes.len() => value,
                _ => return Vec::new(),
            };
            let relationship = match read_u32(bytes, offset) {
                Some(value) => value,
                None => return Vec::new(),
            };
            let record_size = match read_u32(bytes, offset + 4).and_then(|value| usize::try_from(value).ok())
            {
                Some(value) if value >= PROCESSOR_INFO_HEADER_BYTES => value,
                _ => return Vec::new(),
            };
            let record_end = match offset.checked_add(record_size) {
                Some(value) if value <= bytes.len() => value,
                _ => return Vec::new(),
            };

            if relationship == RELATION_PROCESSOR_CORE {
                let relationship_end = match header_end.checked_add(PROCESSOR_RELATIONSHIP_BYTES) {
                    Some(value) if value <= record_end => value,
                    _ => return Vec::new(),
                };
                let efficiency_class = bytes[header_end + 1];
                let group_count = match read_u16(bytes, header_end + 22) {
                    Some(0) | None => return Vec::new(),
                    Some(value) => usize::from(value),
                };
                let group_bytes = match group_count.checked_mul(GROUP_AFFINITY_BYTES) {
                    Some(value) => value,
                    None => return Vec::new(),
                };
                let groups_end = match relationship_end.checked_add(group_bytes) {
                    Some(value) if value <= record_end => value,
                    _ => return Vec::new(),
                };
                for group_offset in (relationship_end..groups_end).step_by(GROUP_AFFINITY_BYTES) {
                    let mask = match read_usize(bytes, group_offset) {
                        Some(value) => value,
                        None => return Vec::new(),
                    };
                    let group = match read_u16(bytes, group_offset + std::mem::size_of::<usize>()) {
                        Some(value) => value,
                        None => return Vec::new(),
                    };
                    let logical = (0..usize::BITS)
                        .filter(|bit| mask & (1usize << bit) != 0)
                        .map(|bit| u32::from(group) * 64 + bit)
                        .collect::<Vec<_>>();
                    if !logical.is_empty() {
                        cores.push(PhysicalCore {
                            logical,
                            group,
                            mask: mask as u64,
                            efficiency_class,
                        });
                    }
                }
            }
            offset = record_end;
        }
        cores
    }

    fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
        let end = offset.checked_add(2)?;
        Some(u16::from_ne_bytes(bytes.get(offset..end)?.try_into().ok()?))
    }

    fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
        let end = offset.checked_add(4)?;
        Some(u32::from_ne_bytes(bytes.get(offset..end)?.try_into().ok()?))
    }

    fn read_usize(bytes: &[u8], offset: usize) -> Option<usize> {
        let end = offset.checked_add(std::mem::size_of::<usize>())?;
        let raw = bytes.get(offset..end)?;
        if std::mem::size_of::<usize>() == 8 {
            Some(u64::from_ne_bytes(raw.try_into().ok()?) as usize)
        } else {
            Some(u32::from_ne_bytes(raw.try_into().ok()?) as usize)
        }
    }

    /// Pin the calling thread to a single logical processor. Returns the previous mask.
    pub fn pin_current_thread(logical: u32) -> Option<u64> {
        if logical >= usize::BITS {
            return None;
        }
        let mask = 1usize << logical;
        // SAFETY: `GetCurrentThread` returns a pseudo-handle that never needs closing, and the
        // mask is a single valid bit within the thread's processor group.
        let prev = unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) };
        if prev == 0 {
            None
        } else {
            Some(prev as u64)
        }
    }
}

#[cfg(not(windows))]
mod win {
    use super::{MemoryStatus, PhysicalCore};

    pub fn memory_status() -> MemoryStatus {
        MemoryStatus {
            total_physical: 0,
            available_physical: 0,
            total_pagefile: 0,
            available_pagefile: 0,
        }
    }

    pub fn physical_cores() -> Vec<PhysicalCore> {
        Vec::new()
    }

    pub fn pin_current_thread(_logical: u32) -> Option<u64> {
        None
    }
}

pub fn memory_status() -> MemoryStatus {
    win::memory_status()
}

/// Pin the calling thread to one logical processor. No-op outside Windows.
pub fn pin_current_thread(logical: u32) -> Option<u64> {
    win::pin_current_thread(logical)
}

fn detect_physical_cores() -> Vec<PhysicalCore> {
    let cores = win::physical_cores();
    if !cores.is_empty() {
        return cores;
    }
    // Fallback: assume no SMT and no heterogeneity rather than inventing a topology.
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    (0..n as u32)
        .map(|i| PhysicalCore {
            logical: vec![i],
            group: 0,
            mask: 1u64 << (i % 64),
            efficiency_class: 0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    fn core_record(mask: usize, group: u16, efficiency_class: u8) -> Vec<u8> {
        let record_size = 48_u32;
        let mut record = vec![0_u8; record_size as usize];
        record[0..4].copy_from_slice(&0_u32.to_ne_bytes());
        record[4..8].copy_from_slice(&record_size.to_ne_bytes());
        record[9] = efficiency_class;
        record[30..32].copy_from_slice(&1_u16.to_ne_bytes());
        record[32..32 + std::mem::size_of::<usize>()].copy_from_slice(&mask.to_ne_bytes());
        record[40..42].copy_from_slice(&group.to_ne_bytes());
        record
    }

    #[test]
    fn topology_is_self_consistent() {
        let t = CpuTopology::detect();
        assert!(t.logical_processors >= 1);
        assert!(!t.physical_cores.is_empty());
        let total_logical: usize = t.physical_cores.iter().map(|c| c.logical.len()).sum();
        assert!(
            total_logical <= t.logical_processors as usize * 2,
            "core enumeration produced {total_logical} logical processors for {} reported",
            t.logical_processors
        );
        assert!(!t.one_thread_per_core().is_empty());
        // Every physical core must expose at least one logical processor.
        assert!(t.physical_cores.iter().all(|c| !c.logical.is_empty()));
    }

    #[test]
    fn memory_is_reported() {
        let m = memory_status();
        if cfg!(windows) {
            assert!(m.total_physical > 0, "GlobalMemoryStatusEx returned no RAM size");
            assert!(m.available_physical <= m.total_physical);
        }
    }

    #[test]
    fn isa_tag_is_stable() {
        let isa = IsaSupport::detect();
        let tag = isa.tag();
        assert_eq!(tag, IsaSupport::detect().tag());
        if isa.avx2 {
            assert!(tag.contains("avx2"));
        }
    }

    #[cfg(windows)]
    #[test]
    fn topology_parser_rejects_truncated_header_record_and_group_array() {
        assert!(win::parse_physical_cores(&[0; 7]).is_empty());

        let mut truncated_record = core_record(0b1, 0, 7);
        truncated_record.pop();
        assert!(win::parse_physical_cores(&truncated_record).is_empty());

        let mut truncated_group_array = core_record(0b1, 0, 7);
        truncated_group_array[4..8].copy_from_slice(&32_u32.to_ne_bytes());
        truncated_group_array.truncate(32);
        assert!(win::parse_physical_cores(&truncated_group_array).is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn topology_parser_copies_a_complete_core_record_without_alignment_assumptions() {
        let cores = win::parse_physical_cores(&core_record(0b101, 2, 9));
        assert_eq!(cores.len(), 1);
        assert_eq!(cores[0].logical, vec![128, 130]);
        assert_eq!(cores[0].group, 2);
        assert_eq!(cores[0].mask, 0b101);
        assert_eq!(cores[0].efficiency_class, 9);
    }
}
