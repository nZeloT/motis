#include "motis/raptor-core/raptor_query.h"
#include "motis/raptor-core/gpu_timetable.h"

#include "motis/kernel/copy_timetable.cuh"
#include "motis/kernel/gpu_raptor.cuh"
#include "motis/kernel/hybrid_raptor.cuh"
#include "motis/kernel/cluster_raptor.cuh"

namespace motis {

void fetch_result_from_device(d_query& dq) {
  auto& result = *dq.result_;

  #if SMALL_TIME

  // we do not need the last arrival array, it only exists because 
  // how we calculate the footpaths
  for (auto k = 0; k < dq.d_arrivals_.size() - 1; ++k) {
    cudaMemcpy(result[k],
               dq.d_arrivals_[k],
               dq.stop_count_ * sizeof(global_mem_time),
               cudaMemcpyDeviceToHost);                       cc();
  }

  #else

  auto const arrival_bytes = stop_count * sizeof(global_mem_time)
  auto const arrivals_bytes = arrival_bytes * max_round_k;
  global_mem_time *h_arrivals_ = (global_mem_time*) malloc(arrivals_bytes);

  std::vector<global_mem_time> tmp(stop_count);

  for (auto k = 0; k < d_arrivals.size() - 1; ++k) {
    cudaMemcpy(tmp.data(),
               dq.d_arrivals_[k],
               dq.stop_count_ * sizeof(global_mem_time),
               cudaMemcpyDeviceToHost);                       cc();
    for (auto s = 0; s < stop_count; ++s) {
      result[k][s] = (motis::time) tmp[s];
    }
  }

  #endif
}

__constant__ device_gpu_timetable GTT;
__device__ bool ANY_STATION_MARKED;

// RTX 2080
// constexpr unsigned int max_threads_per_sm = 1024;
// constexpr unsigned int num_sm = 46;

// GTX 1080
//constexpr unsigned int max_threads_per_sm = 2048;
//constexpr unsigned int num_sm = 20;

//GTX 1050 Ti
constexpr unsigned int max_threads_per_sm = 1024;
constexpr unsigned int num_sm = 6;

constexpr unsigned int block_dim_x = 32; // must always be 32!
constexpr unsigned int block_dim_y = 32; // range [1, ..., 32]
constexpr unsigned int block_size = block_dim_x * block_dim_y;
constexpr unsigned int min_blocks_per_sm = max_threads_per_sm / block_size;
constexpr unsigned int num_blocks = num_sm * min_blocks_per_sm;

const dim3 threads_per_block(block_dim_x, block_dim_y, 1);
const dim3 grid_dim(num_blocks, 1, 1);

// Shorthands
constexpr auto nb = num_blocks;
const auto tpb = threads_per_block;

typedef motis::time(*GetArrivalFun)
        (global_mem_time const * const, station_id const);
typedef bool(*UpdateArrivalFun)
        (global_mem_time* const, station_id const, motis::time const);
typedef station_id(*GetRouteStopFun)
        (route_stops_index const);
typedef motis::time(*GetStopArrivalFun)
        (stop_times_index const);
typedef motis::time(*GetStopDepartureFun)
        (stop_times_index const);

template<typename Kernel>
void inline launch_kernel(Kernel kernel, void** args, cudaStream_t stream) {
  cudaLaunchKernel((void*) kernel, grid_dim, tpb, args, 0, stream);
  cc();
}

template <typename Kernel>
void inline launch_coop_kernel(Kernel kernel, void** args,
                               cudaStream_t stream) {
  cudaLaunchCooperativeKernel((void*)kernel, grid_dim, tpb, args, 0, stream);
  cc();
}

void inline fetch_arrivals_async(d_query& dq, uint8_t const round_k,
                                 cudaStream_t s) {
  cudaMemcpyAsync((*dq.result_)[round_k], dq.d_arrivals_[round_k],
                  dq.stop_count_ * sizeof(global_mem_time),
                  cudaMemcpyDeviceToHost, s);
  cc();
}

} // namespace motis

#include "copy_timetable.cu"
#include "gpu_raptor.cu"
#include "hybrid_raptor.cu"
#include "cluster_raptor.cu"