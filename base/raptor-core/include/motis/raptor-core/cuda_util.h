#pragma once

#include <cuda_runtime.h>

#include <vector>
#include <string>

namespace motis {
namespace raptor {

#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
                                                  \
  }                                                         \
}

#define cc() {                  \
  cucheck_dev(cudaGetLastError());  \
}

template <typename T>
inline void cuda_free(T* ptr) {
  cudaFree(ptr);
  cc();
}

template <typename T>
inline void cuda_malloc_set(T** ptr, size_t const bytes, char const value) {
  cudaMalloc(ptr, bytes);           cc();
  cudaMemset(*ptr, value, bytes);   cc();
}

inline auto set_device(std::vector<std::string> const& device_prefs) {
  for (auto const& pick_device : device_prefs) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count); cc();

    for (auto device_id = 0; device_id < device_count; ++device_id) {
      cudaSetDevice(device_id); cc();
      cudaDeviceProp device_properties;
      cudaGetDeviceProperties(&device_properties, device_id); cc();
      std::string const device_name(device_properties.name);
      if (device_name == pick_device) { return device_id; }
    }
  }
  
  return -1;
}

} // namespace raptor
} // namespace motis