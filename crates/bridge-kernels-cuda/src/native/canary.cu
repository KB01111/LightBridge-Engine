#include <cuda_runtime_api.h>
#include <stdint.h>

extern "C" /**
 * @brief Reports compute capability and global memory for the first CUDA device.
 *
 * @param abi_version ABI version, which must be 1.
 * @param major Receives the device compute capability major version.
 * @param minor Receives the device compute capability minor version.
 * @param global_memory_bytes Receives the device's total global memory in bytes.
 * @return CUDA-style status code, including `cudaSuccess` on success,
 *         `cudaErrorInvalidValue` for invalid arguments, or
 *         `cudaErrorNoDevice` when no CUDA device is available.
 */
int bridge_cuda_canary_v1(
    uint32_t abi_version,
    int *major,
    int *minor,
    uint64_t *global_memory_bytes) {
    if (abi_version != 1 || major == nullptr || minor == nullptr ||
        global_memory_bytes == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (device_count == 0) {
        return static_cast<int>(cudaErrorNoDevice);
    }

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, 0);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    *major = properties.major;
    *minor = properties.minor;
    *global_memory_bytes = static_cast<uint64_t>(properties.totalGlobalMem);
    return static_cast<int>(cudaSuccess);
}
