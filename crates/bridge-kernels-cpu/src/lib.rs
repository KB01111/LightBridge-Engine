//! Bounded host-CPU execution for the exact Q8_K kernel path.

use std::collections::BTreeSet;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use bridge_core::sys::pin_current_thread;
use bridge_kernels_reference::ReferenceExecutionMode;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackendConfig {
    pub threads: usize,
}

impl Default for CpuBackendConfig {
    fn default() -> Self {
        Self {
            threads: recommended_thread_count(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuCapabilities {
    pub avx2: bool,
    pub avx_vnni: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512_vnni: bool,
}

impl CpuCapabilities {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = std::is_x86_feature_detected!("avx2");
            let extended = std::arch::x86_64::__cpuid_count(7, 1);
            Self {
                avx2,
                avx_vnni: avx2 && extended.eax & (1 << 4) != 0,
                avx512f: std::is_x86_feature_detected!("avx512f"),
                avx512bw: std::is_x86_feature_detected!("avx512bw"),
                avx512vl: std::is_x86_feature_detected!("avx512vl"),
                avx512_vnni: std::is_x86_feature_detected!("avx512vnni"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2: false,
                avx_vnni: false,
                avx512f: false,
                avx512bw: false,
                avx512vl: false,
                avx512_vnni: false,
            }
        }
    }

    pub const fn avx2_dot_kernel_available(self) -> bool {
        cfg!(target_arch = "x86_64") && self.avx2
    }

    pub const fn avx_vnni_dot_kernel_available(self) -> bool {
        cfg!(target_arch = "x86_64") && self.avx2 && self.avx_vnni
    }

    pub const fn avx512_dot_kernel_available(self) -> bool {
        cfg!(target_arch = "x86_64")
            && self.avx2
            && self.avx512f
            && self.avx512bw
            && self.avx512vl
            && self.avx512_vnni
    }

    pub const fn backend_name(self) -> &'static str {
        if self.avx2_dot_kernel_available() {
            "cpu_parallel_avx2_q8_k"
        } else {
            "cpu_parallel_scalar_q8_k"
        }
    }
}

pub struct CpuBackend {
    config: CpuBackendConfig,
    capabilities: CpuCapabilities,
    cpu_set_ids: Vec<u32>,
    pool: ThreadPool,
}

impl std::fmt::Debug for CpuBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CpuBackend")
            .field("config", &self.config)
            .field("capabilities", &self.capabilities)
            .field("cpu_set_ids", &self.cpu_set_ids)
            .finish_non_exhaustive()
    }
}

impl CpuBackend {
    pub fn new(config: CpuBackendConfig) -> Result<Self, CpuBackendError> {
        Self::new_with_cpu_set(config, &[])
    }

    pub fn new_with_cpu_set(config: CpuBackendConfig, cpu_set_ids: &[u32]) -> Result<Self, CpuBackendError> {
        if config.threads == 0 {
            return Err(CpuBackendError::ZeroThreads);
        }
        if !cpu_set_ids.is_empty() && cpu_set_ids.len() != config.threads {
            return Err(CpuBackendError::CpuSetCount {
                threads: config.threads,
                cpu_set_ids: cpu_set_ids.len(),
            });
        }
        if cpu_set_ids.iter().copied().collect::<BTreeSet<_>>().len() != cpu_set_ids.len() {
            return Err(CpuBackendError::DuplicateCpuSet);
        }

        let mut builder = ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .thread_name(|index| format!("lightbridge-cpu-{index}"));
        let affinity_failed = Arc::new(AtomicBool::new(false));
        if !cpu_set_ids.is_empty() {
            let selected = Arc::new(cpu_set_ids.to_vec());
            let failed = Arc::clone(&affinity_failed);
            builder = builder.start_handler(move |index| {
                if pin_current_thread(selected[index]).is_none() {
                    failed.store(true, Ordering::Release);
                }
            });
        }
        let pool = builder
            .build()
            .map_err(|error| CpuBackendError::Build(error.to_string()))?;
        if affinity_failed.load(Ordering::Acquire) {
            return Err(CpuBackendError::Affinity);
        }
        Ok(Self {
            config,
            capabilities: CpuCapabilities::detect(),
            cpu_set_ids: cpu_set_ids.to_vec(),
            pool,
        })
    }

    pub const fn config(&self) -> CpuBackendConfig {
        self.config
    }

    pub const fn capabilities(&self) -> CpuCapabilities {
        self.capabilities
    }

    pub fn cpu_set_ids(&self) -> &[u32] {
        &self.cpu_set_ids
    }

    pub const fn execution_mode(&self) -> ReferenceExecutionMode {
        ReferenceExecutionMode::CpuParallelQ8K
    }

    pub const fn backend_name(&self) -> &'static str {
        self.capabilities.backend_name()
    }

    pub const fn simd_active(&self) -> bool {
        self.capabilities.avx2_dot_kernel_available()
    }

    pub fn install<OP, R>(&self, operation: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(operation)
    }
}

pub fn recommended_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().div_ceil(2))
        .unwrap_or(1)
        .max(1)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CpuBackendError {
    #[error("CPU backend thread count must be greater than zero")]
    ZeroThreads,
    #[error(
        "CPU affinity requires exactly one logical CPU ID per worker; got {cpu_set_ids} IDs for {threads} workers"
    )]
    CpuSetCount { threads: usize, cpu_set_ids: usize },
    #[error("CPU affinity IDs must be unique")]
    DuplicateCpuSet,
    #[error("Windows rejected at least one persistent worker affinity assignment")]
    Affinity,
    #[error("failed to build bounded CPU thread pool: {0}")]
    Build(String),
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use rayon::prelude::*;

    use super::*;

    #[test]
    fn bounded_pool_executes_parallel_work_and_reports_exact_configuration() {
        let backend = CpuBackend::new(CpuBackendConfig { threads: 2 }).unwrap();
        let names = Mutex::new(BTreeSet::new());
        backend.install(|| {
            (0..64_usize).into_par_iter().for_each(|_| {
                names
                    .lock()
                    .unwrap()
                    .insert(std::thread::current().name().unwrap_or_default().to_owned());
            });
        });
        assert_eq!(backend.config().threads, 2);
        assert_eq!(
            backend.simd_active(),
            CpuCapabilities::detect().avx2_dot_kernel_available()
        );
        assert_eq!(backend.backend_name(), CpuCapabilities::detect().backend_name());
        assert!(!names.lock().unwrap().is_empty());
        assert!(names
            .lock()
            .unwrap()
            .iter()
            .all(|name| name.starts_with("lightbridge-cpu-")));
    }

    #[test]
    fn rejects_an_unbounded_zero_thread_configuration() {
        assert_eq!(
            CpuBackend::new(CpuBackendConfig { threads: 0 }).unwrap_err(),
            CpuBackendError::ZeroThreads
        );
        assert_eq!(
            CpuBackend::new_with_cpu_set(CpuBackendConfig { threads: 2 }, &[0]).unwrap_err(),
            CpuBackendError::CpuSetCount {
                threads: 2,
                cpu_set_ids: 1
            }
        );
        assert_eq!(
            CpuBackend::new_with_cpu_set(CpuBackendConfig { threads: 2 }, &[0, 0]).unwrap_err(),
            CpuBackendError::DuplicateCpuSet
        );
        assert!(recommended_thread_count() > 0);
    }

    #[cfg(windows)]
    #[test]
    fn persistent_worker_accepts_a_detected_windows_cpu_assignment() {
        let selected = bridge_core::sys::CpuTopology::detect().one_thread_per_core();
        let backend = CpuBackend::new_with_cpu_set(CpuBackendConfig { threads: 1 }, &selected[..1]).unwrap();
        backend.install(|| assert!(std::thread::current().name().is_some()));
        assert_eq!(backend.cpu_set_ids(), &selected[..1]);
    }
}
