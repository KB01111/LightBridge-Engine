use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaNvrtcCanary {
    pub nvrtc_major: i32,
    pub nvrtc_minor: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub ptx_bytes: usize,
    pub pinned_async_transfers: bool,
    pub elapsed_milliseconds: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaPackedQ8KFormatOracle {
    pub weight_type: String,
    pub rows: usize,
    pub logical_elements: usize,
    pub elapsed_milliseconds: f32,
    pub bit_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaPackedQ8KOracle {
    pub nvrtc_major: i32,
    pub nvrtc_minor: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub ptx_bytes: usize,
    pub pinned_async_transfers: bool,
    pub formats: Vec<CudaPackedQ8KFormatOracle>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaPackedQ8KExecution {
    pub weight_type: String,
    pub rows: usize,
    pub logical_elements: usize,
    pub weight_bytes: usize,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub staging_arena: usize,
    pub host_staging_milliseconds: f32,
    pub device_elapsed_milliseconds: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaPackedQ8KPairExecution {
    pub weight_types: [String; 2],
    pub rows: [usize; 2],
    pub logical_elements: usize,
    pub weight_bytes: [usize; 2],
    pub activation_bytes: usize,
    pub output_bytes: [usize; 2],
    pub staging_arenas: [usize; 2],
    pub host_staging_milliseconds: f32,
    pub device_elapsed_milliseconds: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaPackedQ8KBatchItem<'a> {
    pub weight_type: bridge_quant_layout::GgmlType,
    pub weights: &'a [u8],
    pub q8: &'a [u8],
    pub logical_elements: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaPackedQ8KBatchExecution {
    pub items: usize,
    pub rows: usize,
    pub weight_bytes: usize,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub staging_arena: usize,
    pub host_staging_milliseconds: f32,
    pub device_elapsed_milliseconds: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CudaReusablePackedQ8KCanary {
    pub passes: usize,
    pub formats: usize,
    pub bit_exact: bool,
    pub deterministic: bool,
    pub executions: Vec<CudaPackedQ8KExecution>,
}

#[derive(Debug, Error)]
pub enum CudaRuntimeError {
    #[error("NVRTC CUDA runtime loading is implemented only on Windows")]
    UnsupportedPlatform,
    #[error("no trusted {library} DLL could be loaded; tried {candidates:?}")]
    LibraryNotFound {
        library: &'static str,
        candidates: Vec<String>,
    },
    #[error("failed to load CUDA library {path}: {reason}")]
    LibraryLoad { path: String, reason: String },
    #[error("CUDA library {library} does not export required symbol {symbol}")]
    SymbolMissing { library: String, symbol: &'static str },
    #[error("unexpected function-pointer size for CUDA symbol {symbol}")]
    SymbolSize { symbol: &'static str },
    #[error("{operation} failed with NVRTC code {code} ({name}): {log}")]
    Nvrtc {
        operation: &'static str,
        code: i32,
        name: String,
        log: String,
    },
    #[error("{operation} failed with CUDA driver code {code} ({name})")]
    Driver {
        operation: &'static str,
        code: i32,
        name: String,
    },
    #[error("{field} reported {actual} bytes, exceeding the {maximum}-byte safety bound")]
    SizeBound {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("CUDA device compute capability {major}.{minor} is below required 8.9")]
    ComputeCapability { major: i32, minor: i32 },
    #[error(
        "CUDA runtime canary result {actual:#010x} at lane {lane} differs from expected {expected:#010x}"
    )]
    CanaryMismatch { lane: usize, expected: u32, actual: u32 },
    #[error("packed CUDA oracle host preparation failed during {operation}: {reason}")]
    PackedOracleHost { operation: &'static str, reason: String },
    #[error(
        "packed CUDA {weight_type} oracle result {actual:#010x} at row {row} differs from \
         scalar {expected:#010x}"
    )]
    PackedOracleMismatch {
        weight_type: &'static str,
        row: usize,
        expected: u32,
        actual: u32,
    },
    #[error("invalid packed CUDA request: {reason}")]
    InvalidPackedRequest { reason: &'static str },
    #[error("packed CUDA input validation failed: {reason}")]
    PackedValidation { reason: String },
    #[error("packed CUDA {arena} request is {requested} bytes, exceeding the {maximum}-byte bound")]
    PackedArenaBound {
        arena: &'static str,
        requested: usize,
        maximum: usize,
    },
    #[error(
        "packed CUDA allocation of {requested} bytes would violate the {reserve}-byte VRAM \
         reserve; driver reports {free} bytes free"
    )]
    VramReserve {
        requested: usize,
        free: usize,
        reserve: usize,
    },
    #[error("reusable packed CUDA executor initialization failed: {reason}")]
    PackedExecutorUnavailable { reason: String },
    #[error("reusable packed CUDA executor lock was poisoned")]
    PackedExecutorPoisoned,
}

#[cfg(not(windows))]
pub fn runtime_nvrtc_canary() -> Result<CudaNvrtcCanary, CudaRuntimeError> {
    Err(CudaRuntimeError::UnsupportedPlatform)
}

#[cfg(not(windows))]
pub fn runtime_packed_q8k_oracle() -> Result<CudaPackedQ8KOracle, CudaRuntimeError> {
    Err(CudaRuntimeError::UnsupportedPlatform)
}

#[cfg(not(windows))]
pub fn packed_q8k_gemv_into(
    _weight_type: bridge_quant_layout::GgmlType,
    _weights: &[u8],
    _q8: &[u8],
    _logical_elements: usize,
    _output: &mut [f32],
) -> Result<CudaPackedQ8KExecution, CudaRuntimeError> {
    Err(CudaRuntimeError::UnsupportedPlatform)
}

#[cfg(not(windows))]
pub fn packed_q8k_gemv_pair_into(
    _weight_types: [bridge_quant_layout::GgmlType; 2],
    _weights: [&[u8]; 2],
    _q8: &[u8],
    _logical_elements: usize,
    _outputs: [&mut [f32]; 2],
) -> Result<CudaPackedQ8KPairExecution, CudaRuntimeError> {
    Err(CudaRuntimeError::UnsupportedPlatform)
}

#[cfg(not(windows))]
pub fn packed_q8k_gemv_batch_into(
    _items: &[CudaPackedQ8KBatchItem<'_>],
    _output: &mut [f32],
) -> Result<CudaPackedQ8KBatchExecution, CudaRuntimeError> {
    Err(CudaRuntimeError::UnsupportedPlatform)
}

#[cfg(not(windows))]
pub fn runtime_reusable_packed_q8k_canary() -> Result<CudaReusablePackedQ8KCanary, CudaRuntimeError> {
    Err(CudaRuntimeError::UnsupportedPlatform)
}

#[cfg(windows)]
pub fn runtime_nvrtc_canary() -> Result<CudaNvrtcCanary, CudaRuntimeError> {
    windows::run()
}

#[cfg(windows)]
pub fn runtime_packed_q8k_oracle() -> Result<CudaPackedQ8KOracle, CudaRuntimeError> {
    windows::run_packed_q8k()
}

#[cfg(windows)]
pub fn packed_q8k_gemv_into(
    weight_type: bridge_quant_layout::GgmlType,
    weights: &[u8],
    q8: &[u8],
    logical_elements: usize,
    output: &mut [f32],
) -> Result<CudaPackedQ8KExecution, CudaRuntimeError> {
    windows::execute_packed_q8k(weight_type, weights, q8, logical_elements, output)
}

#[cfg(windows)]
pub fn packed_q8k_gemv_pair_into(
    weight_types: [bridge_quant_layout::GgmlType; 2],
    weights: [&[u8]; 2],
    q8: &[u8],
    logical_elements: usize,
    outputs: [&mut [f32]; 2],
) -> Result<CudaPackedQ8KPairExecution, CudaRuntimeError> {
    windows::execute_packed_q8k_pair(weight_types, weights, q8, logical_elements, outputs)
}

#[cfg(windows)]
pub fn packed_q8k_gemv_batch_into(
    items: &[CudaPackedQ8KBatchItem<'_>],
    output: &mut [f32],
) -> Result<CudaPackedQ8KBatchExecution, CudaRuntimeError> {
    windows::execute_packed_q8k_batch(items, output)
}

#[cfg(windows)]
pub fn runtime_reusable_packed_q8k_canary() -> Result<CudaReusablePackedQ8KCanary, CudaRuntimeError> {
    windows::run_reusable_packed_q8k_canary()
}

#[cfg(windows)]
mod windows {
    use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString, OsStr};
    use std::fs;
    use std::mem;
    use std::path::{Path, PathBuf};
    use std::ptr;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        LOAD_LIBRARY_SEARCH_SYSTEM32,
    };

    use bridge_quant_layout::{
        iq2_s_grid_table, iq3_s_grid_table, layout, quantize_row_q8_k_into, CpuDotBackend, GgmlType,
        ValidatedQ8KMatrix, Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMENTS,
    };

    use super::{
        CudaNvrtcCanary, CudaPackedQ8KBatchExecution, CudaPackedQ8KBatchItem, CudaPackedQ8KExecution,
        CudaPackedQ8KFormatOracle, CudaPackedQ8KOracle, CudaPackedQ8KPairExecution,
        CudaReusablePackedQ8KCanary, CudaRuntimeError,
    };

    const MAX_LOG_BYTES: usize = 1024 * 1024;
    const MAX_PTX_BYTES: usize = 16 * 1024 * 1024;
    const CANARY_LANES: usize = 1024;
    const PACKED_ORACLE_ROWS: usize = 7;
    const PACKED_ORACLE_ELEMENTS: usize = 4 * Q8_K_BLOCK_ELEMENTS;
    const PACKED_STAGING_ARENAS: usize = 2;
    const MAX_PACKED_BATCH_ITEMS: usize = 130;
    const MAX_PACKED_WEIGHT_BYTES: usize = 512 * 1024 * 1024;
    const MAX_PACKED_Q8_BYTES: usize = 16 * 1024 * 1024;
    const MAX_PACKED_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
    const GPU_MEMORY_RESERVE_BYTES: usize = 1_280 * 1024 * 1024;
    const REUSABLE_PACKED_THREADS: u32 = 32;
    const COMPUTE_CAPABILITY_MAJOR_ATTRIBUTE: i32 = 75;
    const COMPUTE_CAPABILITY_MINOR_ATTRIBUTE: i32 = 76;
    const STREAM_NON_BLOCKING: u32 = 1;

    type NvrtcProgram = *mut c_void;
    type NvrtcResult = c_int;
    type CuResult = c_int;
    type CuDevice = c_int;
    type CuContext = *mut c_void;
    type CuModule = *mut c_void;
    type CuFunction = *mut c_void;
    type CuStream = *mut c_void;
    type CuEvent = *mut c_void;
    type CuDevicePtr = u64;

    type NvrtcVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult;
    type NvrtcCreateProgram = unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult;
    type NvrtcCompileProgram = unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
    type NvrtcGetProgramLogSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
    type NvrtcGetProgramLog = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
    type NvrtcGetPtxSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
    type NvrtcGetPtx = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
    type NvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult;
    type NvrtcGetErrorString = unsafe extern "C" fn(NvrtcResult) -> *const c_char;

    type CuInit = unsafe extern "system" fn(c_uint) -> CuResult;
    type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, c_int) -> CuResult;
    type CuDeviceGetAttribute = unsafe extern "system" fn(*mut c_int, c_int, CuDevice) -> CuResult;
    type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
    type CuCtxSetCurrent = unsafe extern "system" fn(CuContext) -> CuResult;
    type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
    type CuModuleLoadDataEx = unsafe extern "system" fn(
        *mut CuModule,
        *const c_void,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> CuResult;
    type CuModuleGetFunction =
        unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
    type CuModuleUnload = unsafe extern "system" fn(CuModule) -> CuResult;
    type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
    type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;
    type CuMemGetInfo = unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult;
    type CuMemHostAlloc = unsafe extern "system" fn(*mut *mut c_void, usize, c_uint) -> CuResult;
    type CuMemFreeHost = unsafe extern "system" fn(*mut c_void) -> CuResult;
    type CuMemcpyHtoDAsync =
        unsafe extern "system" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult;
    type CuMemcpyDtoHAsync = unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult;
    type CuStreamCreate = unsafe extern "system" fn(*mut CuStream, c_uint) -> CuResult;
    type CuStreamDestroy = unsafe extern "system" fn(CuStream) -> CuResult;
    type CuEventCreate = unsafe extern "system" fn(*mut CuEvent, c_uint) -> CuResult;
    type CuEventRecord = unsafe extern "system" fn(CuEvent, CuStream) -> CuResult;
    type CuEventSynchronize = unsafe extern "system" fn(CuEvent) -> CuResult;
    type CuEventElapsedTime = unsafe extern "system" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
    type CuEventDestroy = unsafe extern "system" fn(CuEvent) -> CuResult;
    type CuLaunchKernel = unsafe extern "system" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult;
    type CuGetErrorName = unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult;

    struct DynamicLibrary {
        handle: HMODULE,
        display: String,
    }

    impl DynamicLibrary {
        fn open_absolute(path: &Path) -> Result<Self, CudaRuntimeError> {
            let wide = wide_null(path.as_os_str());
            let handle = unsafe {
                LoadLibraryExW(
                    wide.as_ptr(),
                    ptr::null_mut(),
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                )
            };
            if handle.is_null() {
                return Err(CudaRuntimeError::LibraryLoad {
                    path: path.display().to_string(),
                    reason: std::io::Error::last_os_error().to_string(),
                });
            }
            Ok(Self {
                handle,
                display: path.display().to_string(),
            })
        }

        fn open_system(name: &str) -> Result<Self, CudaRuntimeError> {
            let wide = wide_null(OsStr::new(name));
            let handle =
                unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
            if handle.is_null() {
                return Err(CudaRuntimeError::LibraryLoad {
                    path: name.to_owned(),
                    reason: std::io::Error::last_os_error().to_string(),
                });
            }
            Ok(Self {
                handle,
                display: name.to_owned(),
            })
        }

        unsafe fn symbol<T: Copy>(
            &self,
            name: &'static [u8],
            display_name: &'static str,
        ) -> Result<T, CudaRuntimeError> {
            let raw = unsafe { GetProcAddress(self.handle, name.as_ptr()) };
            if mem::size_of::<T>() != mem::size_of_val(&raw) {
                return Err(CudaRuntimeError::SymbolSize { symbol: display_name });
            }
            let raw = raw.ok_or_else(|| CudaRuntimeError::SymbolMissing {
                library: self.display.clone(),
                symbol: display_name,
            })?;
            Ok(unsafe { mem::transmute_copy(&raw) })
        }
    }

    impl Drop for DynamicLibrary {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.handle);
            }
        }
    }

    struct NvrtcApi {
        _library: DynamicLibrary,
        version: NvrtcVersion,
        create_program: NvrtcCreateProgram,
        compile_program: NvrtcCompileProgram,
        get_program_log_size: NvrtcGetProgramLogSize,
        get_program_log: NvrtcGetProgramLog,
        get_ptx_size: NvrtcGetPtxSize,
        get_ptx: NvrtcGetPtx,
        destroy_program: NvrtcDestroyProgram,
        get_error_string: NvrtcGetErrorString,
    }

    impl NvrtcApi {
        fn load() -> Result<Self, CudaRuntimeError> {
            let candidates = nvrtc_candidates();
            let mut attempted = Vec::new();
            let mut loaded = None;
            for path in &candidates {
                attempted.push(path.display().to_string());
                if let Ok(library) = DynamicLibrary::open_absolute(path) {
                    loaded = Some(library);
                    break;
                }
            }
            let library = loaded.ok_or(CudaRuntimeError::LibraryNotFound {
                library: "NVRTC",
                candidates: attempted,
            })?;
            unsafe {
                Ok(Self {
                    version: library.symbol(b"nvrtcVersion\0", "nvrtcVersion")?,
                    create_program: library.symbol(b"nvrtcCreateProgram\0", "nvrtcCreateProgram")?,
                    compile_program: library.symbol(b"nvrtcCompileProgram\0", "nvrtcCompileProgram")?,
                    get_program_log_size: library
                        .symbol(b"nvrtcGetProgramLogSize\0", "nvrtcGetProgramLogSize")?,
                    get_program_log: library.symbol(b"nvrtcGetProgramLog\0", "nvrtcGetProgramLog")?,
                    get_ptx_size: library.symbol(b"nvrtcGetPTXSize\0", "nvrtcGetPTXSize")?,
                    get_ptx: library.symbol(b"nvrtcGetPTX\0", "nvrtcGetPTX")?,
                    destroy_program: library.symbol(b"nvrtcDestroyProgram\0", "nvrtcDestroyProgram")?,
                    get_error_string: library.symbol(b"nvrtcGetErrorString\0", "nvrtcGetErrorString")?,
                    _library: library,
                })
            }
        }

        fn error_name(&self, code: NvrtcResult) -> String {
            let pointer = unsafe { (self.get_error_string)(code) };
            if pointer.is_null() {
                format!("NVRTC_ERROR_{code}")
            } else {
                unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned()
            }
        }

        fn check(&self, operation: &'static str, code: NvrtcResult) -> Result<(), CudaRuntimeError> {
            if code == 0 {
                Ok(())
            } else {
                Err(CudaRuntimeError::Nvrtc {
                    operation,
                    code,
                    name: self.error_name(code),
                    log: String::new(),
                })
            }
        }

        fn compile_source(
            &self,
            source: &str,
            source_name: &str,
        ) -> Result<(Vec<u8>, i32, i32), CudaRuntimeError> {
            let mut major = 0;
            let mut minor = 0;
            self.check("nvrtcVersion", unsafe { (self.version)(&mut major, &mut minor) })?;

            let source = CString::new(source).expect("trusted CUDA source contains no NUL");
            let name = CString::new(source_name).expect("trusted CUDA source name contains no NUL");
            let mut program = ptr::null_mut();
            self.check("nvrtcCreateProgram", unsafe {
                (self.create_program)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            })?;

            let compile_result = (|| {
                let options = [
                    CString::new("--gpu-architecture=compute_89").unwrap(),
                    CString::new("--std=c++14").unwrap(),
                    CString::new("--fmad=false").unwrap(),
                    CString::new("--ftz=false").unwrap(),
                    CString::new("--prec-div=true").unwrap(),
                    CString::new("--prec-sqrt=true").unwrap(),
                ];
                let option_pointers = options.iter().map(|value| value.as_ptr()).collect::<Vec<_>>();
                let compile_code = unsafe {
                    (self.compile_program)(
                        program,
                        c_int::try_from(option_pointers.len()).unwrap_or(c_int::MAX),
                        option_pointers.as_ptr(),
                    )
                };
                let log = self.program_log(program)?;
                if compile_code != 0 {
                    return Err(CudaRuntimeError::Nvrtc {
                        operation: "nvrtcCompileProgram",
                        code: compile_code,
                        name: self.error_name(compile_code),
                        log,
                    });
                }

                let mut ptx_size = 0;
                self.check("nvrtcGetPTXSize", unsafe {
                    (self.get_ptx_size)(program, &mut ptx_size)
                })?;
                if ptx_size > MAX_PTX_BYTES {
                    return Err(CudaRuntimeError::SizeBound {
                        field: "NVRTC PTX",
                        actual: ptx_size,
                        maximum: MAX_PTX_BYTES,
                    });
                }
                let mut ptx = vec![0_u8; ptx_size];
                self.check("nvrtcGetPTX", unsafe {
                    (self.get_ptx)(program, ptx.as_mut_ptr().cast())
                })?;
                Ok(ptx)
            })();

            let destroy_code = unsafe { (self.destroy_program)(&mut program) };
            if let Err(error) = self.check("nvrtcDestroyProgram", destroy_code) {
                if compile_result.is_ok() {
                    return Err(error);
                }
            }
            compile_result.map(|ptx| (ptx, major, minor))
        }

        fn compile_canary(&self) -> Result<(Vec<u8>, i32, i32), CudaRuntimeError> {
            self.compile_source(
                r#"
extern "C" __global__ void bridge_nvrtc_canary_v1(
    const float *input,
    float *output,
    unsigned int length
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        output[index] = input[index] + 1.0f;
    }
}
"#,
                "bridge_nvrtc_canary_v1.cu",
            )
        }

        fn compile_packed_q8k(&self) -> Result<(Vec<u8>, i32, i32), CudaRuntimeError> {
            self.compile_source(
                include_str!("native/packed_q8k.cu"),
                "bridge_packed_q8k_oracle_v1.cu",
            )
        }

        fn program_log(&self, program: NvrtcProgram) -> Result<String, CudaRuntimeError> {
            let mut log_size = 0;
            self.check("nvrtcGetProgramLogSize", unsafe {
                (self.get_program_log_size)(program, &mut log_size)
            })?;
            if log_size > MAX_LOG_BYTES {
                return Err(CudaRuntimeError::SizeBound {
                    field: "NVRTC compile log",
                    actual: log_size,
                    maximum: MAX_LOG_BYTES,
                });
            }
            if log_size == 0 {
                return Ok(String::new());
            }
            let mut log = vec![0_u8; log_size];
            self.check("nvrtcGetProgramLog", unsafe {
                (self.get_program_log)(program, log.as_mut_ptr().cast())
            })?;
            if log.last() == Some(&0) {
                log.pop();
            }
            Ok(String::from_utf8_lossy(&log).trim().to_owned())
        }
    }

    struct DriverApi {
        _library: DynamicLibrary,
        init: CuInit,
        device_get: CuDeviceGet,
        device_get_attribute: CuDeviceGetAttribute,
        ctx_create: CuCtxCreate,
        ctx_set_current: CuCtxSetCurrent,
        ctx_destroy: CuCtxDestroy,
        module_load_data_ex: CuModuleLoadDataEx,
        module_get_function: CuModuleGetFunction,
        module_unload: CuModuleUnload,
        mem_alloc: CuMemAlloc,
        mem_free: CuMemFree,
        mem_get_info: CuMemGetInfo,
        mem_host_alloc: CuMemHostAlloc,
        mem_free_host: CuMemFreeHost,
        memcpy_htod_async: CuMemcpyHtoDAsync,
        memcpy_dtoh_async: CuMemcpyDtoHAsync,
        stream_create: CuStreamCreate,
        stream_destroy: CuStreamDestroy,
        event_create: CuEventCreate,
        event_record: CuEventRecord,
        event_synchronize: CuEventSynchronize,
        event_elapsed_time: CuEventElapsedTime,
        event_destroy: CuEventDestroy,
        launch_kernel: CuLaunchKernel,
        get_error_name: CuGetErrorName,
    }

    impl DriverApi {
        fn load() -> Result<Self, CudaRuntimeError> {
            let library = DynamicLibrary::open_system("nvcuda.dll")?;
            unsafe {
                Ok(Self {
                    init: library.symbol(b"cuInit\0", "cuInit")?,
                    device_get: library.symbol(b"cuDeviceGet\0", "cuDeviceGet")?,
                    device_get_attribute: library
                        .symbol(b"cuDeviceGetAttribute\0", "cuDeviceGetAttribute")?,
                    ctx_create: library.symbol(b"cuCtxCreate_v2\0", "cuCtxCreate_v2")?,
                    ctx_set_current: library.symbol(b"cuCtxSetCurrent\0", "cuCtxSetCurrent")?,
                    ctx_destroy: library.symbol(b"cuCtxDestroy_v2\0", "cuCtxDestroy_v2")?,
                    module_load_data_ex: library.symbol(b"cuModuleLoadDataEx\0", "cuModuleLoadDataEx")?,
                    module_get_function: library.symbol(b"cuModuleGetFunction\0", "cuModuleGetFunction")?,
                    module_unload: library.symbol(b"cuModuleUnload\0", "cuModuleUnload")?,
                    mem_alloc: library.symbol(b"cuMemAlloc_v2\0", "cuMemAlloc_v2")?,
                    mem_free: library.symbol(b"cuMemFree_v2\0", "cuMemFree_v2")?,
                    mem_get_info: library.symbol(b"cuMemGetInfo_v2\0", "cuMemGetInfo_v2")?,
                    mem_host_alloc: library.symbol(b"cuMemHostAlloc\0", "cuMemHostAlloc")?,
                    mem_free_host: library.symbol(b"cuMemFreeHost\0", "cuMemFreeHost")?,
                    memcpy_htod_async: library.symbol(b"cuMemcpyHtoDAsync_v2\0", "cuMemcpyHtoDAsync_v2")?,
                    memcpy_dtoh_async: library.symbol(b"cuMemcpyDtoHAsync_v2\0", "cuMemcpyDtoHAsync_v2")?,
                    stream_create: library.symbol(b"cuStreamCreate\0", "cuStreamCreate")?,
                    stream_destroy: library.symbol(b"cuStreamDestroy_v2\0", "cuStreamDestroy_v2")?,
                    event_create: library.symbol(b"cuEventCreate\0", "cuEventCreate")?,
                    event_record: library.symbol(b"cuEventRecord\0", "cuEventRecord")?,
                    event_synchronize: library.symbol(b"cuEventSynchronize\0", "cuEventSynchronize")?,
                    event_elapsed_time: library.symbol(b"cuEventElapsedTime\0", "cuEventElapsedTime")?,
                    event_destroy: library.symbol(b"cuEventDestroy_v2\0", "cuEventDestroy_v2")?,
                    launch_kernel: library.symbol(b"cuLaunchKernel\0", "cuLaunchKernel")?,
                    get_error_name: library.symbol(b"cuGetErrorName\0", "cuGetErrorName")?,
                    _library: library,
                })
            }
        }

        fn error_name(&self, code: CuResult) -> String {
            let mut pointer = ptr::null();
            let result = unsafe { (self.get_error_name)(code, &mut pointer) };
            if result != 0 || pointer.is_null() {
                format!("CUDA_ERROR_{code}")
            } else {
                unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned()
            }
        }

        fn check(&self, operation: &'static str, code: CuResult) -> Result<(), CudaRuntimeError> {
            if code == 0 {
                Ok(())
            } else {
                Err(CudaRuntimeError::Driver {
                    operation,
                    code,
                    name: self.error_name(code),
                })
            }
        }

        fn ensure_vram_reserve(&self, requested: usize) -> Result<(), CudaRuntimeError> {
            let mut free = 0;
            let mut total = 0;
            self.check("cuMemGetInfo_v2", unsafe {
                (self.mem_get_info)(&mut free, &mut total)
            })?;
            let required = requested.saturating_add(GPU_MEMORY_RESERVE_BYTES);
            if free < required {
                return Err(CudaRuntimeError::VramReserve {
                    requested,
                    free,
                    reserve: GPU_MEMORY_RESERVE_BYTES,
                });
            }
            Ok(())
        }

        fn execute_canary(
            &self,
            ptx: &[u8],
            nvrtc_major: i32,
            nvrtc_minor: i32,
        ) -> Result<CudaNvrtcCanary, CudaRuntimeError> {
            self.check("cuInit", unsafe { (self.init)(0) })?;
            let mut device = 0;
            self.check("cuDeviceGet", unsafe { (self.device_get)(&mut device, 0) })?;
            let mut compute_major = 0;
            let mut compute_minor = 0;
            self.check("cuDeviceGetAttribute(compute major)", unsafe {
                (self.device_get_attribute)(&mut compute_major, COMPUTE_CAPABILITY_MAJOR_ATTRIBUTE, device)
            })?;
            self.check("cuDeviceGetAttribute(compute minor)", unsafe {
                (self.device_get_attribute)(&mut compute_minor, COMPUTE_CAPABILITY_MINOR_ATTRIBUTE, device)
            })?;
            if (compute_major, compute_minor) < (8, 9) {
                return Err(CudaRuntimeError::ComputeCapability {
                    major: compute_major,
                    minor: compute_minor,
                });
            }

            let mut context = ptr::null_mut();
            let mut module = ptr::null_mut();
            let mut function = ptr::null_mut();
            let mut stream = ptr::null_mut();
            let mut event_start = ptr::null_mut();
            let mut event_end = ptr::null_mut();
            let mut device_input = 0;
            let mut device_output = 0;
            let mut host_input = ptr::null_mut();
            let mut host_output = ptr::null_mut();
            let bytes = CANARY_LANES * mem::size_of::<f32>();

            let result = (|| {
                self.check("cuCtxCreate_v2", unsafe {
                    (self.ctx_create)(&mut context, 0, device)
                })?;
                self.check("cuModuleLoadDataEx", unsafe {
                    (self.module_load_data_ex)(
                        &mut module,
                        ptx.as_ptr().cast(),
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                })?;
                let kernel_name =
                    CString::new("bridge_nvrtc_canary_v1").expect("trusted kernel name has no NUL");
                self.check("cuModuleGetFunction", unsafe {
                    (self.module_get_function)(&mut function, module, kernel_name.as_ptr())
                })?;
                self.check("cuMemHostAlloc(input)", unsafe {
                    (self.mem_host_alloc)(&mut host_input, bytes, 0)
                })?;
                self.check("cuMemHostAlloc(output)", unsafe {
                    (self.mem_host_alloc)(&mut host_output, bytes, 0)
                })?;
                self.check("cuMemAlloc_v2(input)", unsafe {
                    (self.mem_alloc)(&mut device_input, bytes)
                })?;
                self.check("cuMemAlloc_v2(output)", unsafe {
                    (self.mem_alloc)(&mut device_output, bytes)
                })?;
                self.check("cuStreamCreate", unsafe {
                    (self.stream_create)(&mut stream, STREAM_NON_BLOCKING)
                })?;
                self.check("cuEventCreate(start)", unsafe {
                    (self.event_create)(&mut event_start, 0)
                })?;
                self.check("cuEventCreate(end)", unsafe {
                    (self.event_create)(&mut event_end, 0)
                })?;

                let input = unsafe { std::slice::from_raw_parts_mut(host_input.cast::<f32>(), CANARY_LANES) };
                let output =
                    unsafe { std::slice::from_raw_parts_mut(host_output.cast::<f32>(), CANARY_LANES) };
                for (index, value) in input.iter_mut().enumerate() {
                    *value = (index as f32 - 512.0) / 32.0;
                }
                output.fill(f32::NAN);

                self.check("cuEventRecord(start)", unsafe {
                    (self.event_record)(event_start, stream)
                })?;
                self.check("cuMemcpyHtoDAsync_v2", unsafe {
                    (self.memcpy_htod_async)(device_input, host_input.cast_const(), bytes, stream)
                })?;
                let mut input_argument = device_input;
                let mut output_argument = device_output;
                let mut length_argument = CANARY_LANES as u32;
                let mut arguments = [
                    (&mut input_argument as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut output_argument as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut length_argument as *mut u32).cast::<c_void>(),
                ];
                self.check("cuLaunchKernel", unsafe {
                    (self.launch_kernel)(
                        function,
                        (CANARY_LANES as u32).div_ceil(256),
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        arguments.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                })?;
                self.check("cuMemcpyDtoHAsync_v2", unsafe {
                    (self.memcpy_dtoh_async)(host_output, device_output, bytes, stream)
                })?;
                self.check("cuEventRecord(end)", unsafe {
                    (self.event_record)(event_end, stream)
                })?;
                self.check("cuEventSynchronize(end)", unsafe {
                    (self.event_synchronize)(event_end)
                })?;
                let mut elapsed_milliseconds = 0.0_f32;
                self.check("cuEventElapsedTime", unsafe {
                    (self.event_elapsed_time)(&mut elapsed_milliseconds, event_start, event_end)
                })?;
                for (lane, (&input, &actual)) in input.iter().zip(output.iter()).enumerate() {
                    let expected = input + 1.0;
                    if actual.to_bits() != expected.to_bits() {
                        return Err(CudaRuntimeError::CanaryMismatch {
                            lane,
                            expected: expected.to_bits(),
                            actual: actual.to_bits(),
                        });
                    }
                }
                Ok(CudaNvrtcCanary {
                    nvrtc_major,
                    nvrtc_minor,
                    compute_major,
                    compute_minor,
                    ptx_bytes: ptx.len(),
                    pinned_async_transfers: true,
                    elapsed_milliseconds,
                })
            })();

            let mut cleanup_error = None;
            let mut cleanup = |operation: &'static str, code: CuResult| {
                if code != 0 && cleanup_error.is_none() {
                    cleanup_error = Some(CudaRuntimeError::Driver {
                        operation,
                        code,
                        name: self.error_name(code),
                    });
                }
            };
            unsafe {
                if !event_end.is_null() {
                    cleanup("cuEventDestroy_v2(end)", (self.event_destroy)(event_end));
                }
                if !event_start.is_null() {
                    cleanup("cuEventDestroy_v2(start)", (self.event_destroy)(event_start));
                }
                if !stream.is_null() {
                    cleanup("cuStreamDestroy_v2", (self.stream_destroy)(stream));
                }
                if device_output != 0 {
                    cleanup("cuMemFree_v2(output)", (self.mem_free)(device_output));
                }
                if device_input != 0 {
                    cleanup("cuMemFree_v2(input)", (self.mem_free)(device_input));
                }
                if !host_output.is_null() {
                    cleanup("cuMemFreeHost(output)", (self.mem_free_host)(host_output));
                }
                if !host_input.is_null() {
                    cleanup("cuMemFreeHost(input)", (self.mem_free_host)(host_input));
                }
                if !module.is_null() {
                    cleanup("cuModuleUnload", (self.module_unload)(module));
                }
                if !context.is_null() {
                    cleanup("cuCtxDestroy_v2", (self.ctx_destroy)(context));
                }
            }
            match (result, cleanup_error) {
                (Ok(_), Some(error)) => Err(error),
                (result, _) => result,
            }
        }

        fn execute_packed_q8k(
            &self,
            ptx: &[u8],
            nvrtc_major: i32,
            nvrtc_minor: i32,
        ) -> Result<CudaPackedQ8KOracle, CudaRuntimeError> {
            let (q8, cases) = packed_oracle_inputs()?;
            self.check("cuInit", unsafe { (self.init)(0) })?;
            let mut device = 0;
            self.check("cuDeviceGet", unsafe { (self.device_get)(&mut device, 0) })?;
            let mut compute_major = 0;
            let mut compute_minor = 0;
            self.check("cuDeviceGetAttribute(compute major)", unsafe {
                (self.device_get_attribute)(&mut compute_major, COMPUTE_CAPABILITY_MAJOR_ATTRIBUTE, device)
            })?;
            self.check("cuDeviceGetAttribute(compute minor)", unsafe {
                (self.device_get_attribute)(&mut compute_minor, COMPUTE_CAPABILITY_MINOR_ATTRIBUTE, device)
            })?;
            if (compute_major, compute_minor) < (8, 9) {
                return Err(CudaRuntimeError::ComputeCapability {
                    major: compute_major,
                    minor: compute_minor,
                });
            }

            let iq2_grid = iq2_s_grid_table();
            let iq3_grid = iq3_s_grid_table();
            let max_weight_bytes = cases.iter().map(|case| case.weights.len()).max().unwrap_or(0);
            let output_bytes = PACKED_ORACLE_ROWS * mem::size_of::<f32>();
            let iq2_grid_bytes = mem::size_of_val(iq2_grid);
            let iq3_grid_bytes = mem::size_of_val(iq3_grid);

            let mut context = ptr::null_mut();
            let mut module = ptr::null_mut();
            let mut function = ptr::null_mut();
            let mut stream = ptr::null_mut();
            let mut event_start = ptr::null_mut();
            let mut event_end = ptr::null_mut();
            let mut device_weights = 0;
            let mut device_q8 = 0;
            let mut device_iq2_grid = 0;
            let mut device_iq3_grid = 0;
            let mut device_output = 0;
            let mut host_weights = ptr::null_mut();
            let mut host_q8 = ptr::null_mut();
            let mut host_iq2_grid = ptr::null_mut();
            let mut host_iq3_grid = ptr::null_mut();
            let mut host_output = ptr::null_mut();

            let result = (|| {
                self.check("cuCtxCreate_v2", unsafe {
                    (self.ctx_create)(&mut context, 0, device)
                })?;
                self.check("cuModuleLoadDataEx", unsafe {
                    (self.module_load_data_ex)(
                        &mut module,
                        ptx.as_ptr().cast(),
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                })?;
                let kernel_name = CString::new("bridge_q8k_gemv_v1").expect("trusted kernel name has no NUL");
                self.check("cuModuleGetFunction", unsafe {
                    (self.module_get_function)(&mut function, module, kernel_name.as_ptr())
                })?;

                self.check("cuMemHostAlloc(packed weights)", unsafe {
                    (self.mem_host_alloc)(&mut host_weights, max_weight_bytes, 0)
                })?;
                self.check("cuMemHostAlloc(Q8_K)", unsafe {
                    (self.mem_host_alloc)(&mut host_q8, q8.len(), 0)
                })?;
                self.check("cuMemHostAlloc(IQ2_S grid)", unsafe {
                    (self.mem_host_alloc)(&mut host_iq2_grid, iq2_grid_bytes, 0)
                })?;
                self.check("cuMemHostAlloc(IQ3_S grid)", unsafe {
                    (self.mem_host_alloc)(&mut host_iq3_grid, iq3_grid_bytes, 0)
                })?;
                self.check("cuMemHostAlloc(packed output)", unsafe {
                    (self.mem_host_alloc)(&mut host_output, output_bytes, 0)
                })?;

                self.check("cuMemAlloc_v2(packed weights)", unsafe {
                    (self.mem_alloc)(&mut device_weights, max_weight_bytes)
                })?;
                self.check("cuMemAlloc_v2(Q8_K)", unsafe {
                    (self.mem_alloc)(&mut device_q8, q8.len())
                })?;
                self.check("cuMemAlloc_v2(IQ2_S grid)", unsafe {
                    (self.mem_alloc)(&mut device_iq2_grid, iq2_grid_bytes)
                })?;
                self.check("cuMemAlloc_v2(IQ3_S grid)", unsafe {
                    (self.mem_alloc)(&mut device_iq3_grid, iq3_grid_bytes)
                })?;
                self.check("cuMemAlloc_v2(packed output)", unsafe {
                    (self.mem_alloc)(&mut device_output, output_bytes)
                })?;
                self.check("cuStreamCreate", unsafe {
                    (self.stream_create)(&mut stream, STREAM_NON_BLOCKING)
                })?;
                self.check("cuEventCreate(start)", unsafe {
                    (self.event_create)(&mut event_start, 0)
                })?;
                self.check("cuEventCreate(end)", unsafe {
                    (self.event_create)(&mut event_end, 0)
                })?;

                unsafe {
                    std::slice::from_raw_parts_mut(host_q8.cast::<u8>(), q8.len()).copy_from_slice(&q8);
                    std::slice::from_raw_parts_mut(host_iq2_grid.cast::<u64>(), iq2_grid.len())
                        .copy_from_slice(iq2_grid);
                    std::slice::from_raw_parts_mut(host_iq3_grid.cast::<u32>(), iq3_grid.len())
                        .copy_from_slice(iq3_grid);
                }
                self.check("cuMemcpyHtoDAsync_v2(Q8_K)", unsafe {
                    (self.memcpy_htod_async)(device_q8, host_q8.cast_const(), q8.len(), stream)
                })?;
                self.check("cuMemcpyHtoDAsync_v2(IQ2_S grid)", unsafe {
                    (self.memcpy_htod_async)(
                        device_iq2_grid,
                        host_iq2_grid.cast_const(),
                        iq2_grid_bytes,
                        stream,
                    )
                })?;
                self.check("cuMemcpyHtoDAsync_v2(IQ3_S grid)", unsafe {
                    (self.memcpy_htod_async)(
                        device_iq3_grid,
                        host_iq3_grid.cast_const(),
                        iq3_grid_bytes,
                        stream,
                    )
                })?;
                self.check("cuEventRecord(common upload)", unsafe {
                    (self.event_record)(event_end, stream)
                })?;
                self.check("cuEventSynchronize(common upload)", unsafe {
                    (self.event_synchronize)(event_end)
                })?;

                let pinned_weights =
                    unsafe { std::slice::from_raw_parts_mut(host_weights.cast::<u8>(), max_weight_bytes) };
                let output =
                    unsafe { std::slice::from_raw_parts_mut(host_output.cast::<f32>(), PACKED_ORACLE_ROWS) };
                let mut formats = Vec::with_capacity(cases.len());
                for case in &cases {
                    pinned_weights.fill(0xa5);
                    pinned_weights[..case.weights.len()].copy_from_slice(&case.weights);
                    output.fill(f32::NAN);

                    self.check("cuEventRecord(packed start)", unsafe {
                        (self.event_record)(event_start, stream)
                    })?;
                    self.check("cuMemcpyHtoDAsync_v2(packed weights)", unsafe {
                        (self.memcpy_htod_async)(
                            device_weights,
                            host_weights.cast_const(),
                            case.weights.len(),
                            stream,
                        )
                    })?;
                    let mut kind_argument = case.kind;
                    let mut weights_argument = device_weights;
                    let mut q8_argument = device_q8;
                    let mut iq2_grid_argument = device_iq2_grid;
                    let mut iq3_grid_argument = device_iq3_grid;
                    let mut elements_argument = c_int::try_from(PACKED_ORACLE_ELEMENTS).map_err(|_| {
                        CudaRuntimeError::PackedOracleHost {
                            operation: "logical element conversion",
                            reason: "oracle element count exceeds CUDA int".to_owned(),
                        }
                    })?;
                    let mut rows_argument = c_int::try_from(PACKED_ORACLE_ROWS).map_err(|_| {
                        CudaRuntimeError::PackedOracleHost {
                            operation: "row count conversion",
                            reason: "oracle row count exceeds CUDA int".to_owned(),
                        }
                    })?;
                    let mut output_argument = device_output;
                    let mut arguments = [
                        (&mut kind_argument as *mut c_int).cast::<c_void>(),
                        (&mut weights_argument as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut q8_argument as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut iq2_grid_argument as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut iq3_grid_argument as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut elements_argument as *mut c_int).cast::<c_void>(),
                        (&mut rows_argument as *mut c_int).cast::<c_void>(),
                        (&mut output_argument as *mut CuDevicePtr).cast::<c_void>(),
                    ];
                    self.check("cuLaunchKernel(packed Q8_K GEMV)", unsafe {
                        (self.launch_kernel)(
                            function,
                            (PACKED_ORACLE_ROWS as u32).div_ceil(64),
                            1,
                            1,
                            64,
                            1,
                            1,
                            0,
                            stream,
                            arguments.as_mut_ptr(),
                            ptr::null_mut(),
                        )
                    })?;
                    self.check("cuMemcpyDtoHAsync_v2(packed output)", unsafe {
                        (self.memcpy_dtoh_async)(host_output, device_output, output_bytes, stream)
                    })?;
                    self.check("cuEventRecord(packed end)", unsafe {
                        (self.event_record)(event_end, stream)
                    })?;
                    self.check("cuEventSynchronize(packed end)", unsafe {
                        (self.event_synchronize)(event_end)
                    })?;
                    let mut elapsed_milliseconds = 0.0_f32;
                    self.check("cuEventElapsedTime(packed)", unsafe {
                        (self.event_elapsed_time)(&mut elapsed_milliseconds, event_start, event_end)
                    })?;
                    for (row, (&expected, &actual)) in case.expected.iter().zip(output.iter()).enumerate() {
                        if actual.to_bits() != expected.to_bits() {
                            return Err(CudaRuntimeError::PackedOracleMismatch {
                                weight_type: case.name,
                                row,
                                expected: expected.to_bits(),
                                actual: actual.to_bits(),
                            });
                        }
                    }
                    formats.push(CudaPackedQ8KFormatOracle {
                        weight_type: case.name.to_owned(),
                        rows: PACKED_ORACLE_ROWS,
                        logical_elements: PACKED_ORACLE_ELEMENTS,
                        elapsed_milliseconds,
                        bit_exact: true,
                    });
                }

                Ok(CudaPackedQ8KOracle {
                    nvrtc_major,
                    nvrtc_minor,
                    compute_major,
                    compute_minor,
                    ptx_bytes: ptx.len(),
                    pinned_async_transfers: true,
                    formats,
                })
            })();

            let mut cleanup_error = None;
            let mut cleanup = |operation: &'static str, code: CuResult| {
                if code != 0 && cleanup_error.is_none() {
                    cleanup_error = Some(CudaRuntimeError::Driver {
                        operation,
                        code,
                        name: self.error_name(code),
                    });
                }
            };
            unsafe {
                if !event_end.is_null() {
                    cleanup("cuEventDestroy_v2(end)", (self.event_destroy)(event_end));
                }
                if !event_start.is_null() {
                    cleanup("cuEventDestroy_v2(start)", (self.event_destroy)(event_start));
                }
                if !stream.is_null() {
                    cleanup("cuStreamDestroy_v2", (self.stream_destroy)(stream));
                }
                if device_output != 0 {
                    cleanup("cuMemFree_v2(packed output)", (self.mem_free)(device_output));
                }
                if device_iq3_grid != 0 {
                    cleanup("cuMemFree_v2(IQ3_S grid)", (self.mem_free)(device_iq3_grid));
                }
                if device_iq2_grid != 0 {
                    cleanup("cuMemFree_v2(IQ2_S grid)", (self.mem_free)(device_iq2_grid));
                }
                if device_q8 != 0 {
                    cleanup("cuMemFree_v2(Q8_K)", (self.mem_free)(device_q8));
                }
                if device_weights != 0 {
                    cleanup("cuMemFree_v2(packed weights)", (self.mem_free)(device_weights));
                }
                if !host_output.is_null() {
                    cleanup("cuMemFreeHost(packed output)", (self.mem_free_host)(host_output));
                }
                if !host_iq3_grid.is_null() {
                    cleanup("cuMemFreeHost(IQ3_S grid)", (self.mem_free_host)(host_iq3_grid));
                }
                if !host_iq2_grid.is_null() {
                    cleanup("cuMemFreeHost(IQ2_S grid)", (self.mem_free_host)(host_iq2_grid));
                }
                if !host_q8.is_null() {
                    cleanup("cuMemFreeHost(Q8_K)", (self.mem_free_host)(host_q8));
                }
                if !host_weights.is_null() {
                    cleanup(
                        "cuMemFreeHost(packed weights)",
                        (self.mem_free_host)(host_weights),
                    );
                }
                if !module.is_null() {
                    cleanup("cuModuleUnload", (self.module_unload)(module));
                }
                if !context.is_null() {
                    cleanup("cuCtxDestroy_v2", (self.ctx_destroy)(context));
                }
            }
            match (result, cleanup_error) {
                (Ok(_), Some(error)) => Err(error),
                (result, _) => result,
            }
        }
    }

    struct TransferArena {
        host: *mut c_void,
        device: CuDevicePtr,
        capacity: usize,
    }

    impl TransferArena {
        const fn new() -> Self {
            Self {
                host: ptr::null_mut(),
                device: 0,
                capacity: 0,
            }
        }

        fn ensure(
            &mut self,
            driver: &DriverApi,
            name: &'static str,
            required: usize,
            maximum: usize,
        ) -> Result<(), CudaRuntimeError> {
            if required == 0 {
                return Err(CudaRuntimeError::InvalidPackedRequest {
                    reason: "transfer arenas require a non-zero byte count",
                });
            }
            if required > maximum {
                return Err(CudaRuntimeError::PackedArenaBound {
                    arena: name,
                    requested: required,
                    maximum,
                });
            }
            if required <= self.capacity {
                return Ok(());
            }
            let capacity = required.next_power_of_two().min(maximum);
            driver.ensure_vram_reserve(capacity)?;

            let mut new_host = ptr::null_mut();
            driver.check("cuMemHostAlloc(reusable arena)", unsafe {
                (driver.mem_host_alloc)(&mut new_host, capacity, 0)
            })?;
            let mut new_device = 0;
            if let Err(error) = driver.check("cuMemAlloc_v2(reusable arena)", unsafe {
                (driver.mem_alloc)(&mut new_device, capacity)
            }) {
                unsafe {
                    (driver.mem_free_host)(new_host);
                }
                return Err(error);
            }

            let old_host = mem::replace(&mut self.host, new_host);
            let old_device = mem::replace(&mut self.device, new_device);
            self.capacity = capacity;
            if old_device != 0 {
                driver.check("cuMemFree_v2(replaced arena)", unsafe {
                    (driver.mem_free)(old_device)
                })?;
            }
            if !old_host.is_null() {
                driver.check("cuMemFreeHost(replaced arena)", unsafe {
                    (driver.mem_free_host)(old_host)
                })?;
            }
            Ok(())
        }

        unsafe fn release(&mut self, driver: &DriverApi) {
            if self.device != 0 {
                unsafe {
                    (driver.mem_free)(self.device);
                }
                self.device = 0;
            }
            if !self.host.is_null() {
                unsafe {
                    (driver.mem_free_host)(self.host);
                }
                self.host = ptr::null_mut();
            }
            self.capacity = 0;
        }
    }

    struct PackedExecutor {
        driver: DriverApi,
        context: CuContext,
        module: CuModule,
        function: CuFunction,
        stream: CuStream,
        event_start: CuEvent,
        event_end: CuEvent,
        device_iq2_grid: CuDevicePtr,
        device_iq3_grid: CuDevicePtr,
        weight_arenas: [TransferArena; PACKED_STAGING_ARENAS],
        q8_arena: TransferArena,
        output_arena: TransferArena,
        next_staging_arena: usize,
    }

    // SAFETY: the CUDA context is explicitly made current for every operation,
    // and the only shared instance is protected by a mutex.
    unsafe impl Send for PackedExecutor {}

    impl PackedExecutor {
        fn new() -> Result<Self, CudaRuntimeError> {
            let nvrtc = NvrtcApi::load()?;
            let (ptx, _, _) = nvrtc.compile_packed_q8k()?;
            drop(nvrtc);

            let driver = DriverApi::load()?;
            driver.check("cuInit", unsafe { (driver.init)(0) })?;
            let mut device = 0;
            driver.check("cuDeviceGet", unsafe { (driver.device_get)(&mut device, 0) })?;
            let mut compute_major = 0;
            let mut compute_minor = 0;
            driver.check("cuDeviceGetAttribute(compute major)", unsafe {
                (driver.device_get_attribute)(&mut compute_major, COMPUTE_CAPABILITY_MAJOR_ATTRIBUTE, device)
            })?;
            driver.check("cuDeviceGetAttribute(compute minor)", unsafe {
                (driver.device_get_attribute)(&mut compute_minor, COMPUTE_CAPABILITY_MINOR_ATTRIBUTE, device)
            })?;
            if (compute_major, compute_minor) < (8, 9) {
                return Err(CudaRuntimeError::ComputeCapability {
                    major: compute_major,
                    minor: compute_minor,
                });
            }

            let mut executor = Self {
                driver,
                context: ptr::null_mut(),
                module: ptr::null_mut(),
                function: ptr::null_mut(),
                stream: ptr::null_mut(),
                event_start: ptr::null_mut(),
                event_end: ptr::null_mut(),
                device_iq2_grid: 0,
                device_iq3_grid: 0,
                weight_arenas: [TransferArena::new(), TransferArena::new()],
                q8_arena: TransferArena::new(),
                output_arena: TransferArena::new(),
                next_staging_arena: 0,
            };
            executor.initialize(device, &ptx)?;
            Ok(executor)
        }

        fn initialize(&mut self, device: CuDevice, ptx: &[u8]) -> Result<(), CudaRuntimeError> {
            self.driver
                .check("cuCtxCreate_v2(reusable packed executor)", unsafe {
                    (self.driver.ctx_create)(&mut self.context, 0, device)
                })?;
            self.driver
                .check("cuModuleLoadDataEx(reusable packed executor)", unsafe {
                    (self.driver.module_load_data_ex)(
                        &mut self.module,
                        ptx.as_ptr().cast(),
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                })?;
            let kernel_name = CString::new("bridge_q8k_gemv_v1").expect("trusted kernel name has no NUL");
            self.driver
                .check("cuModuleGetFunction(reusable packed executor)", unsafe {
                    (self.driver.module_get_function)(&mut self.function, self.module, kernel_name.as_ptr())
                })?;
            self.driver
                .check("cuStreamCreate(reusable packed executor)", unsafe {
                    (self.driver.stream_create)(&mut self.stream, STREAM_NON_BLOCKING)
                })?;
            self.driver.check("cuEventCreate(reusable start)", unsafe {
                (self.driver.event_create)(&mut self.event_start, 0)
            })?;
            self.driver.check("cuEventCreate(reusable end)", unsafe {
                (self.driver.event_create)(&mut self.event_end, 0)
            })?;

            let iq2_grid = iq2_s_grid_table();
            let iq3_grid = iq3_s_grid_table();
            let iq2_bytes = mem::size_of_val(iq2_grid);
            let iq3_bytes = mem::size_of_val(iq3_grid);
            self.driver
                .ensure_vram_reserve(iq2_bytes.saturating_add(iq3_bytes))?;
            self.driver.check("cuMemAlloc_v2(reusable IQ2_S grid)", unsafe {
                (self.driver.mem_alloc)(&mut self.device_iq2_grid, iq2_bytes)
            })?;
            self.driver.check("cuMemAlloc_v2(reusable IQ3_S grid)", unsafe {
                (self.driver.mem_alloc)(&mut self.device_iq3_grid, iq3_bytes)
            })?;

            let mut host_iq2 = ptr::null_mut();
            let mut host_iq3 = ptr::null_mut();
            let upload_result = (|| {
                self.driver.check("cuMemHostAlloc(reusable IQ2_S grid)", unsafe {
                    (self.driver.mem_host_alloc)(&mut host_iq2, iq2_bytes, 0)
                })?;
                self.driver.check("cuMemHostAlloc(reusable IQ3_S grid)", unsafe {
                    (self.driver.mem_host_alloc)(&mut host_iq3, iq3_bytes, 0)
                })?;
                unsafe {
                    std::slice::from_raw_parts_mut(host_iq2.cast::<u64>(), iq2_grid.len())
                        .copy_from_slice(iq2_grid);
                    std::slice::from_raw_parts_mut(host_iq3.cast::<u32>(), iq3_grid.len())
                        .copy_from_slice(iq3_grid);
                }
                self.driver
                    .check("cuMemcpyHtoDAsync_v2(reusable IQ2_S grid)", unsafe {
                        (self.driver.memcpy_htod_async)(
                            self.device_iq2_grid,
                            host_iq2.cast_const(),
                            iq2_bytes,
                            self.stream,
                        )
                    })?;
                self.driver
                    .check("cuMemcpyHtoDAsync_v2(reusable IQ3_S grid)", unsafe {
                        (self.driver.memcpy_htod_async)(
                            self.device_iq3_grid,
                            host_iq3.cast_const(),
                            iq3_bytes,
                            self.stream,
                        )
                    })?;
                self.driver.check("cuEventRecord(reusable grid upload)", unsafe {
                    (self.driver.event_record)(self.event_end, self.stream)
                })?;
                self.driver
                    .check("cuEventSynchronize(reusable grid upload)", unsafe {
                        (self.driver.event_synchronize)(self.event_end)
                    })
            })();
            let mut cleanup_error = None;
            if !host_iq3.is_null() {
                if let Err(error) = self.driver.check("cuMemFreeHost(reusable IQ3_S grid)", unsafe {
                    (self.driver.mem_free_host)(host_iq3)
                }) {
                    cleanup_error = Some(error);
                }
            }
            if !host_iq2.is_null() {
                if let Err(error) = self.driver.check("cuMemFreeHost(reusable IQ2_S grid)", unsafe {
                    (self.driver.mem_free_host)(host_iq2)
                }) {
                    cleanup_error.get_or_insert(error);
                }
            }
            match (upload_result, cleanup_error) {
                (Ok(()), Some(error)) => Err(error),
                (result, _) => result,
            }
        }

        fn launch_packed(
            &self,
            weight_type: GgmlType,
            weights: CuDevicePtr,
            q8: CuDevicePtr,
            logical_elements: usize,
            rows: usize,
            output: CuDevicePtr,
        ) -> Result<(), CudaRuntimeError> {
            let mut kind_argument = packed_kind(weight_type)?;
            let mut weights_argument = weights;
            let mut q8_argument = q8;
            let mut iq2_grid_argument = self.device_iq2_grid;
            let mut iq3_grid_argument = self.device_iq3_grid;
            let mut elements_argument =
                c_int::try_from(logical_elements).map_err(|_| CudaRuntimeError::InvalidPackedRequest {
                    reason: "logical element count exceeds CUDA int",
                })?;
            let mut rows_argument =
                c_int::try_from(rows).map_err(|_| CudaRuntimeError::InvalidPackedRequest {
                    reason: "row count exceeds CUDA int",
                })?;
            let mut output_argument = output;
            let mut arguments = [
                (&mut kind_argument as *mut c_int).cast::<c_void>(),
                (&mut weights_argument as *mut CuDevicePtr).cast::<c_void>(),
                (&mut q8_argument as *mut CuDevicePtr).cast::<c_void>(),
                (&mut iq2_grid_argument as *mut CuDevicePtr).cast::<c_void>(),
                (&mut iq3_grid_argument as *mut CuDevicePtr).cast::<c_void>(),
                (&mut elements_argument as *mut c_int).cast::<c_void>(),
                (&mut rows_argument as *mut c_int).cast::<c_void>(),
                (&mut output_argument as *mut CuDevicePtr).cast::<c_void>(),
            ];
            let rows = u32::try_from(rows).map_err(|_| CudaRuntimeError::InvalidPackedRequest {
                reason: "row count exceeds CUDA grid range",
            })?;
            self.driver
                .check("cuLaunchKernel(reusable packed Q8_K GEMV)", unsafe {
                    (self.driver.launch_kernel)(
                        self.function,
                        rows.div_ceil(REUSABLE_PACKED_THREADS),
                        1,
                        1,
                        REUSABLE_PACKED_THREADS,
                        1,
                        1,
                        0,
                        self.stream,
                        arguments.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                })
        }

        fn execute(
            &mut self,
            weight_type: GgmlType,
            weights: &[u8],
            q8: &[u8],
            logical_elements: usize,
            output: &mut [f32],
        ) -> Result<CudaPackedQ8KExecution, CudaRuntimeError> {
            self.driver
                .check("cuCtxSetCurrent(reusable packed executor)", unsafe {
                    (self.driver.ctx_set_current)(self.context)
                })?;
            let output_bytes = output.len().checked_mul(mem::size_of::<f32>()).ok_or(
                CudaRuntimeError::InvalidPackedRequest {
                    reason: "packed output byte count overflowed",
                },
            )?;
            let staging_arena = self.next_staging_arena;
            self.weight_arenas[staging_arena].ensure(
                &self.driver,
                "weight staging arena",
                weights.len(),
                MAX_PACKED_WEIGHT_BYTES,
            )?;
            self.q8_arena.ensure(
                &self.driver,
                "Q8_K activation arena",
                q8.len(),
                MAX_PACKED_Q8_BYTES,
            )?;
            self.output_arena.ensure(
                &self.driver,
                "output arena",
                output_bytes,
                MAX_PACKED_OUTPUT_BYTES,
            )?;

            let host_staging_started = Instant::now();
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.weight_arenas[staging_arena].host.cast::<u8>(),
                    weights.len(),
                )
                .copy_from_slice(weights);
                std::slice::from_raw_parts_mut(self.q8_arena.host.cast::<u8>(), q8.len()).copy_from_slice(q8);
                std::slice::from_raw_parts_mut(self.output_arena.host.cast::<f32>(), output.len())
                    .fill(f32::NAN);
            }
            let host_staging_milliseconds = host_staging_started.elapsed().as_secs_f32() * 1_000.0;

            self.driver
                .check("cuEventRecord(reusable packed start)", unsafe {
                    (self.driver.event_record)(self.event_start, self.stream)
                })?;
            self.driver
                .check("cuMemcpyHtoDAsync_v2(reusable packed weights)", unsafe {
                    (self.driver.memcpy_htod_async)(
                        self.weight_arenas[staging_arena].device,
                        self.weight_arenas[staging_arena].host.cast_const(),
                        weights.len(),
                        self.stream,
                    )
                })?;
            self.driver.check("cuMemcpyHtoDAsync_v2(reusable Q8_K)", unsafe {
                (self.driver.memcpy_htod_async)(
                    self.q8_arena.device,
                    self.q8_arena.host.cast_const(),
                    q8.len(),
                    self.stream,
                )
            })?;

            self.launch_packed(
                weight_type,
                self.weight_arenas[staging_arena].device,
                self.q8_arena.device,
                logical_elements,
                output.len(),
                self.output_arena.device,
            )?;
            self.driver
                .check("cuMemcpyDtoHAsync_v2(reusable packed output)", unsafe {
                    (self.driver.memcpy_dtoh_async)(
                        self.output_arena.host,
                        self.output_arena.device,
                        output_bytes,
                        self.stream,
                    )
                })?;
            self.driver.check("cuEventRecord(reusable packed end)", unsafe {
                (self.driver.event_record)(self.event_end, self.stream)
            })?;
            self.driver
                .check("cuEventSynchronize(reusable packed end)", unsafe {
                    (self.driver.event_synchronize)(self.event_end)
                })?;
            let mut device_elapsed_milliseconds = 0.0;
            self.driver.check("cuEventElapsedTime(reusable packed)", unsafe {
                (self.driver.event_elapsed_time)(
                    &mut device_elapsed_milliseconds,
                    self.event_start,
                    self.event_end,
                )
            })?;

            unsafe {
                output.copy_from_slice(std::slice::from_raw_parts(
                    self.output_arena.host.cast::<f32>(),
                    output.len(),
                ));
            }
            self.next_staging_arena = (staging_arena + 1) % PACKED_STAGING_ARENAS;
            Ok(CudaPackedQ8KExecution {
                weight_type: format!("{weight_type:?}"),
                rows: output.len(),
                logical_elements,
                weight_bytes: weights.len(),
                activation_bytes: q8.len(),
                output_bytes,
                staging_arena,
                host_staging_milliseconds,
                device_elapsed_milliseconds,
            })
        }

        fn execute_pair(
            &mut self,
            weight_types: [GgmlType; 2],
            weights: [&[u8]; 2],
            q8: &[u8],
            logical_elements: usize,
            outputs: [&mut [f32]; 2],
        ) -> Result<CudaPackedQ8KPairExecution, CudaRuntimeError> {
            let [first_weights, second_weights] = weights;
            let [first_output, second_output] = outputs;
            self.driver
                .check("cuCtxSetCurrent(reusable packed pair executor)", unsafe {
                    (self.driver.ctx_set_current)(self.context)
                })?;
            let first_output_bytes = first_output.len().checked_mul(mem::size_of::<f32>()).ok_or(
                CudaRuntimeError::InvalidPackedRequest {
                    reason: "first packed pair output byte count overflowed",
                },
            )?;
            let second_output_bytes = second_output.len().checked_mul(mem::size_of::<f32>()).ok_or(
                CudaRuntimeError::InvalidPackedRequest {
                    reason: "second packed pair output byte count overflowed",
                },
            )?;
            let total_output_bytes = first_output_bytes.checked_add(second_output_bytes).ok_or(
                CudaRuntimeError::InvalidPackedRequest {
                    reason: "combined packed pair output byte count overflowed",
                },
            )?;
            self.weight_arenas[0].ensure(
                &self.driver,
                "first weight staging arena",
                first_weights.len(),
                MAX_PACKED_WEIGHT_BYTES,
            )?;
            self.weight_arenas[1].ensure(
                &self.driver,
                "second weight staging arena",
                second_weights.len(),
                MAX_PACKED_WEIGHT_BYTES,
            )?;
            self.q8_arena.ensure(
                &self.driver,
                "Q8_K activation arena",
                q8.len(),
                MAX_PACKED_Q8_BYTES,
            )?;
            self.output_arena.ensure(
                &self.driver,
                "paired output arena",
                total_output_bytes,
                MAX_PACKED_OUTPUT_BYTES,
            )?;

            let host_staging_started = Instant::now();
            unsafe {
                std::slice::from_raw_parts_mut(self.weight_arenas[0].host.cast::<u8>(), first_weights.len())
                    .copy_from_slice(first_weights);
                std::slice::from_raw_parts_mut(self.weight_arenas[1].host.cast::<u8>(), second_weights.len())
                    .copy_from_slice(second_weights);
                std::slice::from_raw_parts_mut(self.q8_arena.host.cast::<u8>(), q8.len()).copy_from_slice(q8);
                std::slice::from_raw_parts_mut(
                    self.output_arena.host.cast::<f32>(),
                    first_output.len().saturating_add(second_output.len()),
                )
                .fill(f32::NAN);
            }
            let host_staging_milliseconds = host_staging_started.elapsed().as_secs_f32() * 1_000.0;

            self.driver
                .check("cuEventRecord(reusable packed pair start)", unsafe {
                    (self.driver.event_record)(self.event_start, self.stream)
                })?;
            self.driver
                .check("cuMemcpyHtoDAsync_v2(reusable first packed weights)", unsafe {
                    (self.driver.memcpy_htod_async)(
                        self.weight_arenas[0].device,
                        self.weight_arenas[0].host.cast_const(),
                        first_weights.len(),
                        self.stream,
                    )
                })?;
            self.driver
                .check("cuMemcpyHtoDAsync_v2(reusable second packed weights)", unsafe {
                    (self.driver.memcpy_htod_async)(
                        self.weight_arenas[1].device,
                        self.weight_arenas[1].host.cast_const(),
                        second_weights.len(),
                        self.stream,
                    )
                })?;
            self.driver
                .check("cuMemcpyHtoDAsync_v2(reusable paired Q8_K)", unsafe {
                    (self.driver.memcpy_htod_async)(
                        self.q8_arena.device,
                        self.q8_arena.host.cast_const(),
                        q8.len(),
                        self.stream,
                    )
                })?;
            let second_output_device = self
                .output_arena
                .device
                .checked_add(u64::try_from(first_output_bytes).map_err(|_| {
                    CudaRuntimeError::InvalidPackedRequest {
                        reason: "first packed pair output offset exceeds CUDA pointer range",
                    }
                })?)
                .ok_or(CudaRuntimeError::InvalidPackedRequest {
                    reason: "second packed pair output pointer overflowed",
                })?;
            self.launch_packed(
                weight_types[0],
                self.weight_arenas[0].device,
                self.q8_arena.device,
                logical_elements,
                first_output.len(),
                self.output_arena.device,
            )?;
            self.launch_packed(
                weight_types[1],
                self.weight_arenas[1].device,
                self.q8_arena.device,
                logical_elements,
                second_output.len(),
                second_output_device,
            )?;
            self.driver
                .check("cuMemcpyDtoHAsync_v2(reusable packed pair output)", unsafe {
                    (self.driver.memcpy_dtoh_async)(
                        self.output_arena.host,
                        self.output_arena.device,
                        total_output_bytes,
                        self.stream,
                    )
                })?;
            self.driver
                .check("cuEventRecord(reusable packed pair end)", unsafe {
                    (self.driver.event_record)(self.event_end, self.stream)
                })?;
            self.driver
                .check("cuEventSynchronize(reusable packed pair end)", unsafe {
                    (self.driver.event_synchronize)(self.event_end)
                })?;
            let mut device_elapsed_milliseconds = 0.0;
            self.driver
                .check("cuEventElapsedTime(reusable packed pair)", unsafe {
                    (self.driver.event_elapsed_time)(
                        &mut device_elapsed_milliseconds,
                        self.event_start,
                        self.event_end,
                    )
                })?;

            unsafe {
                let host = std::slice::from_raw_parts(
                    self.output_arena.host.cast::<f32>(),
                    first_output.len() + second_output.len(),
                );
                let (first, second) = host.split_at(first_output.len());
                first_output.copy_from_slice(first);
                second_output.copy_from_slice(second);
            }
            self.next_staging_arena = 0;
            Ok(CudaPackedQ8KPairExecution {
                weight_types: [format!("{:?}", weight_types[0]), format!("{:?}", weight_types[1])],
                rows: [first_output.len(), second_output.len()],
                logical_elements,
                weight_bytes: [first_weights.len(), second_weights.len()],
                activation_bytes: q8.len(),
                output_bytes: [first_output_bytes, second_output_bytes],
                staging_arenas: [0, 1],
                host_staging_milliseconds,
                device_elapsed_milliseconds,
            })
        }

        fn execute_batch(
            &mut self,
            items: &[CudaPackedQ8KBatchItem<'_>],
            output: &mut [f32],
        ) -> Result<CudaPackedQ8KBatchExecution, CudaRuntimeError> {
            self.driver
                .check("cuCtxSetCurrent(reusable packed batch executor)", unsafe {
                    (self.driver.ctx_set_current)(self.context)
                })?;
            let mut weight_offsets = [0_usize; MAX_PACKED_BATCH_ITEMS];
            let mut q8_offsets = [0_usize; MAX_PACKED_BATCH_ITEMS];
            let mut output_offsets = [0_usize; MAX_PACKED_BATCH_ITEMS];
            let mut weight_bytes = 0_usize;
            let mut activation_bytes = 0_usize;
            let mut output_rows = 0_usize;
            for (index, item) in items.iter().enumerate() {
                weight_offsets[index] = weight_bytes;
                q8_offsets[index] = activation_bytes;
                output_offsets[index] = output_rows;
                weight_bytes = weight_bytes.checked_add(item.weights.len()).ok_or(
                    CudaRuntimeError::InvalidPackedRequest {
                        reason: "packed batch weight byte count overflowed",
                    },
                )?;
                activation_bytes = activation_bytes.checked_add(item.q8.len()).ok_or(
                    CudaRuntimeError::InvalidPackedRequest {
                        reason: "packed batch activation byte count overflowed",
                    },
                )?;
                output_rows =
                    output_rows
                        .checked_add(item.rows)
                        .ok_or(CudaRuntimeError::InvalidPackedRequest {
                            reason: "packed batch output row count overflowed",
                        })?;
            }
            let output_bytes = output_rows.checked_mul(mem::size_of::<f32>()).ok_or(
                CudaRuntimeError::InvalidPackedRequest {
                    reason: "packed batch output byte count overflowed",
                },
            )?;
            let staging_arena = self.next_staging_arena;
            self.weight_arenas[staging_arena].ensure(
                &self.driver,
                "batch weight staging arena",
                weight_bytes,
                MAX_PACKED_WEIGHT_BYTES,
            )?;
            self.q8_arena.ensure(
                &self.driver,
                "batch Q8_K activation arena",
                activation_bytes,
                MAX_PACKED_Q8_BYTES,
            )?;
            self.output_arena.ensure(
                &self.driver,
                "batch output arena",
                output_bytes,
                MAX_PACKED_OUTPUT_BYTES,
            )?;

            let host_staging_started = Instant::now();
            unsafe {
                let host_weights = std::slice::from_raw_parts_mut(
                    self.weight_arenas[staging_arena].host.cast::<u8>(),
                    weight_bytes,
                );
                let host_q8 =
                    std::slice::from_raw_parts_mut(self.q8_arena.host.cast::<u8>(), activation_bytes);
                for (index, item) in items.iter().enumerate() {
                    host_weights[weight_offsets[index]..weight_offsets[index] + item.weights.len()]
                        .copy_from_slice(item.weights);
                    host_q8[q8_offsets[index]..q8_offsets[index] + item.q8.len()].copy_from_slice(item.q8);
                }
                std::slice::from_raw_parts_mut(self.output_arena.host.cast::<f32>(), output_rows)
                    .fill(f32::NAN);
            }
            let host_staging_milliseconds = host_staging_started.elapsed().as_secs_f32() * 1_000.0;

            self.driver
                .check("cuEventRecord(reusable packed batch start)", unsafe {
                    (self.driver.event_record)(self.event_start, self.stream)
                })?;
            self.driver
                .check("cuMemcpyHtoDAsync_v2(reusable packed batch weights)", unsafe {
                    (self.driver.memcpy_htod_async)(
                        self.weight_arenas[staging_arena].device,
                        self.weight_arenas[staging_arena].host.cast_const(),
                        weight_bytes,
                        self.stream,
                    )
                })?;
            self.driver
                .check("cuMemcpyHtoDAsync_v2(reusable packed batch Q8_K)", unsafe {
                    (self.driver.memcpy_htod_async)(
                        self.q8_arena.device,
                        self.q8_arena.host.cast_const(),
                        activation_bytes,
                        self.stream,
                    )
                })?;
            let device_offset = |base: CuDevicePtr,
                                 offset: usize,
                                 reason: &'static str|
             -> Result<CuDevicePtr, CudaRuntimeError> {
                base.checked_add(
                    u64::try_from(offset).map_err(|_| CudaRuntimeError::InvalidPackedRequest { reason })?,
                )
                .ok_or(CudaRuntimeError::InvalidPackedRequest { reason })
            };
            for (index, item) in items.iter().enumerate() {
                let output_offset_bytes = output_offsets[index].checked_mul(mem::size_of::<f32>()).ok_or(
                    CudaRuntimeError::InvalidPackedRequest {
                        reason: "packed batch output offset overflowed",
                    },
                )?;
                self.launch_packed(
                    item.weight_type,
                    device_offset(
                        self.weight_arenas[staging_arena].device,
                        weight_offsets[index],
                        "packed batch weight pointer overflowed",
                    )?,
                    device_offset(
                        self.q8_arena.device,
                        q8_offsets[index],
                        "packed batch activation pointer overflowed",
                    )?,
                    item.logical_elements,
                    item.rows,
                    device_offset(
                        self.output_arena.device,
                        output_offset_bytes,
                        "packed batch output pointer overflowed",
                    )?,
                )?;
            }
            self.driver
                .check("cuMemcpyDtoHAsync_v2(reusable packed batch output)", unsafe {
                    (self.driver.memcpy_dtoh_async)(
                        self.output_arena.host,
                        self.output_arena.device,
                        output_bytes,
                        self.stream,
                    )
                })?;
            self.driver
                .check("cuEventRecord(reusable packed batch end)", unsafe {
                    (self.driver.event_record)(self.event_end, self.stream)
                })?;
            self.driver
                .check("cuEventSynchronize(reusable packed batch end)", unsafe {
                    (self.driver.event_synchronize)(self.event_end)
                })?;
            let mut device_elapsed_milliseconds = 0.0;
            self.driver
                .check("cuEventElapsedTime(reusable packed batch)", unsafe {
                    (self.driver.event_elapsed_time)(
                        &mut device_elapsed_milliseconds,
                        self.event_start,
                        self.event_end,
                    )
                })?;
            unsafe {
                output.copy_from_slice(std::slice::from_raw_parts(
                    self.output_arena.host.cast::<f32>(),
                    output_rows,
                ));
            }
            self.next_staging_arena = (staging_arena + 1) % PACKED_STAGING_ARENAS;
            Ok(CudaPackedQ8KBatchExecution {
                items: items.len(),
                rows: output_rows,
                weight_bytes,
                activation_bytes,
                output_bytes,
                staging_arena,
                host_staging_milliseconds,
                device_elapsed_milliseconds,
            })
        }
    }

    impl Drop for PackedExecutor {
        fn drop(&mut self) {
            unsafe {
                if !self.context.is_null() {
                    (self.driver.ctx_set_current)(self.context);
                }
                self.output_arena.release(&self.driver);
                self.q8_arena.release(&self.driver);
                for arena in &mut self.weight_arenas {
                    arena.release(&self.driver);
                }
                if self.device_iq3_grid != 0 {
                    (self.driver.mem_free)(self.device_iq3_grid);
                }
                if self.device_iq2_grid != 0 {
                    (self.driver.mem_free)(self.device_iq2_grid);
                }
                if !self.event_end.is_null() {
                    (self.driver.event_destroy)(self.event_end);
                }
                if !self.event_start.is_null() {
                    (self.driver.event_destroy)(self.event_start);
                }
                if !self.stream.is_null() {
                    (self.driver.stream_destroy)(self.stream);
                }
                if !self.module.is_null() {
                    (self.driver.module_unload)(self.module);
                }
                if !self.context.is_null() {
                    (self.driver.ctx_destroy)(self.context);
                }
            }
        }
    }

    fn packed_kind(weight_type: GgmlType) -> Result<c_int, CudaRuntimeError> {
        match weight_type {
            GgmlType::Q4_K => Ok(0),
            GgmlType::Q5_K => Ok(1),
            GgmlType::IQ2_S => Ok(2),
            GgmlType::IQ3_S => Ok(3),
            _ => Err(CudaRuntimeError::InvalidPackedRequest {
                reason: "only Q4_K, Q5_K, IQ2_S, and IQ3_S weights are supported",
            }),
        }
    }

    static PACKED_EXECUTOR: OnceLock<Result<Mutex<PackedExecutor>, String>> = OnceLock::new();

    fn packed_executor() -> Result<&'static Mutex<PackedExecutor>, CudaRuntimeError> {
        PACKED_EXECUTOR
            .get_or_init(|| {
                PackedExecutor::new()
                    .map(Mutex::new)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|reason| CudaRuntimeError::PackedExecutorUnavailable {
                reason: reason.clone(),
            })
    }

    pub(super) fn execute_packed_q8k(
        weight_type: GgmlType,
        weights: &[u8],
        q8: &[u8],
        logical_elements: usize,
        output: &mut [f32],
    ) -> Result<CudaPackedQ8KExecution, CudaRuntimeError> {
        if output.is_empty() {
            return Err(CudaRuntimeError::InvalidPackedRequest {
                reason: "packed GEMV requires at least one output row",
            });
        }
        ValidatedQ8KMatrix::new(
            weight_type,
            weights,
            q8,
            logical_elements,
            output.len(),
            CpuDotBackend::Scalar,
        )
        .map_err(|error| CudaRuntimeError::PackedValidation {
            reason: error.to_string(),
        })?;

        let mut executor = packed_executor()?
            .lock()
            .map_err(|_| CudaRuntimeError::PackedExecutorPoisoned)?;
        executor.execute(weight_type, weights, q8, logical_elements, output)
    }

    pub(super) fn execute_packed_q8k_pair(
        weight_types: [GgmlType; 2],
        weights: [&[u8]; 2],
        q8: &[u8],
        logical_elements: usize,
        outputs: [&mut [f32]; 2],
    ) -> Result<CudaPackedQ8KPairExecution, CudaRuntimeError> {
        let [first_output, second_output] = outputs;
        if first_output.is_empty() || second_output.is_empty() {
            return Err(CudaRuntimeError::InvalidPackedRequest {
                reason: "packed paired GEMV requires at least one row in each output",
            });
        }
        ValidatedQ8KMatrix::new(
            weight_types[0],
            weights[0],
            q8,
            logical_elements,
            first_output.len(),
            CpuDotBackend::Scalar,
        )
        .map_err(|error| CudaRuntimeError::PackedValidation {
            reason: error.to_string(),
        })?;
        ValidatedQ8KMatrix::new(
            weight_types[1],
            weights[1],
            q8,
            logical_elements,
            second_output.len(),
            CpuDotBackend::Scalar,
        )
        .map_err(|error| CudaRuntimeError::PackedValidation {
            reason: error.to_string(),
        })?;
        let mut executor = packed_executor()?
            .lock()
            .map_err(|_| CudaRuntimeError::PackedExecutorPoisoned)?;
        executor.execute_pair(
            weight_types,
            weights,
            q8,
            logical_elements,
            [first_output, second_output],
        )
    }

    pub(super) fn execute_packed_q8k_batch(
        items: &[CudaPackedQ8KBatchItem<'_>],
        output: &mut [f32],
    ) -> Result<CudaPackedQ8KBatchExecution, CudaRuntimeError> {
        if items.is_empty() || items.len() > MAX_PACKED_BATCH_ITEMS {
            return Err(CudaRuntimeError::InvalidPackedRequest {
                reason: "packed batch item count must be within 1..=130",
            });
        }
        let mut expected_rows = 0_usize;
        for item in items {
            if item.rows == 0 {
                return Err(CudaRuntimeError::InvalidPackedRequest {
                    reason: "every packed batch item requires at least one output row",
                });
            }
            ValidatedQ8KMatrix::new(
                item.weight_type,
                item.weights,
                item.q8,
                item.logical_elements,
                item.rows,
                CpuDotBackend::Scalar,
            )
            .map_err(|error| CudaRuntimeError::PackedValidation {
                reason: error.to_string(),
            })?;
            expected_rows =
                expected_rows
                    .checked_add(item.rows)
                    .ok_or(CudaRuntimeError::InvalidPackedRequest {
                        reason: "packed batch output row count overflowed",
                    })?;
        }
        if output.len() != expected_rows {
            return Err(CudaRuntimeError::InvalidPackedRequest {
                reason: "packed batch output length must equal the sum of item rows",
            });
        }
        let mut executor = packed_executor()?
            .lock()
            .map_err(|_| CudaRuntimeError::PackedExecutorPoisoned)?;
        executor.execute_batch(items, output)
    }

    struct PackedOracleCase {
        weight_type: GgmlType,
        name: &'static str,
        kind: c_int,
        weights: Vec<u8>,
        expected: Vec<f32>,
    }

    fn packed_oracle_inputs() -> Result<(Vec<u8>, Vec<PackedOracleCase>), CudaRuntimeError> {
        let input = (0..PACKED_ORACLE_ELEMENTS)
            .map(|index| {
                let centered = ((index * 37 + 11) % 257) as f32 - 128.0;
                centered / 19.0
            })
            .collect::<Vec<_>>();
        let mut q8 = vec![0_u8; PACKED_ORACLE_ELEMENTS / Q8_K_BLOCK_ELEMENTS * Q8_K_BLOCK_BYTES];
        quantize_row_q8_k_into(&input, &mut q8).map_err(|error| CudaRuntimeError::PackedOracleHost {
            operation: "Q8_K activation quantization",
            reason: error.to_string(),
        })?;

        let formats = [
            (GgmlType::Q4_K, "Q4_K", 0),
            (GgmlType::Q5_K, "Q5_K", 1),
            (GgmlType::IQ2_S, "IQ2_S", 2),
            (GgmlType::IQ3_S, "IQ3_S", 3),
        ];
        let block_count = PACKED_ORACLE_ELEMENTS / Q8_K_BLOCK_ELEMENTS;
        let mut cases = Vec::with_capacity(formats.len());
        for (format_index, (weight_type, name, kind)) in formats.into_iter().enumerate() {
            let block_bytes = layout(weight_type)
                .map_err(|error| CudaRuntimeError::PackedOracleHost {
                    operation: "packed weight layout",
                    reason: error.to_string(),
                })?
                .block_bytes;
            let mut weights = vec![0_u8; PACKED_ORACLE_ROWS * block_count * block_bytes];
            for (index, byte) in weights.iter_mut().enumerate() {
                *byte = ((index * 73 + format_index * 41 + 19) % 251) as u8;
            }
            for block in 0..PACKED_ORACLE_ROWS * block_count {
                let offset = block * block_bytes;
                weights[offset..offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                if matches!(weight_type, GgmlType::Q4_K | GgmlType::Q5_K) {
                    weights[offset + 2..offset + 4].copy_from_slice(&0x3400_u16.to_le_bytes());
                }
            }
            let matrix = ValidatedQ8KMatrix::new(
                weight_type,
                &weights,
                &q8,
                PACKED_ORACLE_ELEMENTS,
                PACKED_ORACLE_ROWS,
                CpuDotBackend::Scalar,
            )
            .map_err(|error| CudaRuntimeError::PackedOracleHost {
                operation: "scalar packed matrix validation",
                reason: error.to_string(),
            })?;
            let expected = (0..PACKED_ORACLE_ROWS)
                .map(|row| {
                    matrix
                        .dot_row(row)
                        .map_err(|error| CudaRuntimeError::PackedOracleHost {
                            operation: "scalar packed row oracle",
                            reason: error.to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            cases.push(PackedOracleCase {
                weight_type,
                name,
                kind,
                weights,
                expected,
            });
        }
        Ok((q8, cases))
    }

    pub(super) fn run() -> Result<CudaNvrtcCanary, CudaRuntimeError> {
        let nvrtc = NvrtcApi::load()?;
        let (ptx, nvrtc_major, nvrtc_minor) = nvrtc.compile_canary()?;
        let driver = DriverApi::load()?;
        driver.execute_canary(&ptx, nvrtc_major, nvrtc_minor)
    }

    pub(super) fn run_packed_q8k() -> Result<CudaPackedQ8KOracle, CudaRuntimeError> {
        let nvrtc = NvrtcApi::load()?;
        let (ptx, nvrtc_major, nvrtc_minor) = nvrtc.compile_packed_q8k()?;
        let driver = DriverApi::load()?;
        driver.execute_packed_q8k(&ptx, nvrtc_major, nvrtc_minor)
    }

    pub(super) fn run_reusable_packed_q8k_canary() -> Result<CudaReusablePackedQ8KCanary, CudaRuntimeError> {
        const PASSES: usize = 2;
        let (q8, cases) = packed_oracle_inputs()?;
        let mut executions = Vec::with_capacity(PASSES * cases.len());
        let mut prior = Vec::<Vec<u32>>::with_capacity(cases.len());
        for pass in 0..PASSES {
            for (case_index, case) in cases.iter().enumerate() {
                let mut output = vec![f32::NAN; PACKED_ORACLE_ROWS];
                let execution = execute_packed_q8k(
                    case.weight_type,
                    &case.weights,
                    &q8,
                    PACKED_ORACLE_ELEMENTS,
                    &mut output,
                )?;
                let bits = output.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
                for (row, (&expected, &actual)) in case.expected.iter().zip(output.iter()).enumerate() {
                    if actual.to_bits() != expected.to_bits() {
                        return Err(CudaRuntimeError::PackedOracleMismatch {
                            weight_type: case.name,
                            row,
                            expected: expected.to_bits(),
                            actual: actual.to_bits(),
                        });
                    }
                }
                if pass == 0 {
                    prior.push(bits);
                } else if prior[case_index] != bits {
                    let row = prior[case_index]
                        .iter()
                        .zip(&bits)
                        .position(|(expected, actual)| expected != actual)
                        .unwrap_or(0);
                    return Err(CudaRuntimeError::PackedOracleMismatch {
                        weight_type: case.name,
                        row,
                        expected: prior[case_index][row],
                        actual: bits[row],
                    });
                }
                executions.push(execution);
            }
        }
        Ok(CudaReusablePackedQ8KCanary {
            passes: PASSES,
            formats: cases.len(),
            bit_exact: true,
            deterministic: true,
            executions,
        })
    }

    fn wide_null(value: &OsStr) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        value.encode_wide().chain(Some(0)).collect()
    }

    fn nvrtc_candidates() -> Vec<PathBuf> {
        let mut roots = Vec::new();
        for (name, value) in std::env::vars_os() {
            if name
                .to_string_lossy()
                .to_ascii_uppercase()
                .starts_with("CUDA_PATH")
            {
                roots.push(PathBuf::from(value));
            }
        }
        if let Some(program_files) = std::env::var_os("ProgramFiles") {
            let toolkit_root = PathBuf::from(program_files)
                .join("NVIDIA GPU Computing Toolkit")
                .join("CUDA");
            if let Ok(entries) = fs::read_dir(toolkit_root) {
                roots.extend(entries.filter_map(Result::ok).map(|entry| entry.path()));
            }
        }
        roots.sort();
        roots.dedup();

        let mut candidates = Vec::new();
        for root in roots {
            let bin = root.join("bin");
            let Ok(entries) = fs::read_dir(&bin) else {
                continue;
            };
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                let lowercase = name.to_ascii_lowercase();
                if lowercase.starts_with("nvrtc64_")
                    && lowercase.ends_with(".dll")
                    && !lowercase.contains("builtins")
                {
                    candidates.push(path);
                }
            }
        }
        candidates.sort();
        candidates.reverse();
        candidates.dedup();
        candidates
    }
}
