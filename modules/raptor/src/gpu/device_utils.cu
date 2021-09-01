#include "motis/raptor/gpu/device_utils.h"

#include "motis/raptor-core/cuda_util.h"

extern "C" {

void print_device_properties(cudaDeviceProp const& dp) {
  printf("Properties of device '%s':\n", dp.name);
  printf("\tCompute Capability:\t%i.%i\n", dp.major, dp.minor);
  printf("\tMultiprocessor Count:\t%i\n", dp.multiProcessorCount);
  printf("\tmaxThreadsPerBlock:\t%i\n", dp.maxThreadsPerBlock);
  printf("\tmaxThreadsPerDim:\t%i, %i, %i\n", dp.maxThreadsDim[0],
         dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
  printf("\tmaxGridSizePerDim:\t%i, %i, %i\n", dp.maxGridSize[0],
         dp.maxGridSize[1], dp.maxGridSize[2]);
}

int set_device(std::vector<std::string> const& device_prefs) {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  cc();

  if (device_count == 0) {
    return -1;
  }

  auto device_id = 0;
  for (; device_id < device_count; ++device_id) {
    cudaSetDevice(device_id);
    cc();
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, device_id);
    cc();
    std::string const device_name(device_properties.name);

    auto found = false;
    for (auto& pick_device : device_prefs) {
      if ((found = (device_name == pick_device))) {
        break;
      }
    }

    if (found) break;
  }

  if (device_id == device_count) {
    // preferred device was not found; defaulting to first one found
    device_id = 0;
  }

  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, device_id);
  cc();
  printf("Giving selected device properties:\n");
  print_device_properties(device_props);

  return device_id;
}
} // end extern "C"