//! Bounded host-CPU execution for the exact Q8_K kernel path.

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
    pub avx512f: bool,
    pub avx512_vnni: bool,
}

impl CpuCapabilities {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                avx512_vnni: std::is_x86_feature_detected!("avx512vnni"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2: false,
                avx512f: false,
                avx512_vnni: false,
            }
        }
    }

    pub const fn avx2_dot_kernel_available(self) -> bool {
        cfg!(target_arch = "x86_64") && self.avx2
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
    pool: ThreadPool,
}

impl std::fmt::Debug for CpuBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CpuBackend")
            .field("config", &self.config)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl CpuBackend {
    pub fn new(config: CpuBackendConfig) -> Result<Self, CpuBackendError> {
        if config.threads == 0 {
            return Err(CpuBackendError::ZeroThreads);
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .thread_name(|index| format!("lightbridge-cpu-{index}"))
            .build()
            .map_err(|error| CpuBackendError::Build(error.to_string()))?;
        Ok(Self {
            config,
            capabilities: CpuCapabilities::detect(),
            pool,
        })
    }

    pub const fn config(&self) -> CpuBackendConfig {
        self.config
    }

    pub const fn capabilities(&self) -> CpuCapabilities {
        self.capabilities
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
        assert!(recommended_thread_count() > 0);
    }
}
