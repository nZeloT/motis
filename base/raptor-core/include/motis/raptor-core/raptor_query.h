#pragma once

#include "motis/raptor-core/cuda_util.h"
#include "motis/raptor-core/raptor_result.h"
#include "motis/raptor-core/gpu_timetable.h"

namespace motis {
namespace raptor {

struct base_query {
  station_id source_{invalid<station_id>};
  station_id target_{invalid<station_id>};

  motis::time source_time_begin_{invalid<motis::time>};
  motis::time source_time_end_{invalid<motis::time>};
};

struct raptor_query : base_query {
  raptor_query() = delete;
  raptor_query(raptor_query const&) = delete;

  raptor_query(base_query const& bq, raptor_timetable const& tt) {
    static_cast<base_query&>(*this) = bq;
    result_ = std::make_unique<raptor_result>(tt.stop_count());
  }

  std::unique_ptr<raptor_result> result_;
};

struct d_query : base_query {
  d_query() = delete;

  d_query(base_query const& bq, raptor_timetable const& tt) {
    static_cast<base_query&>(*this) = bq;

    stop_count_ = tt.stop_count();

    // +1 due to scratchpad memory for GPU
    auto const arrival_bytes =
        stop_count_ * sizeof(global_mem_time) * (max_round_k + 1);

    cuda_malloc_set(&(d_arrivals_.front()), arrival_bytes, 0xFFu);
    for (auto k = 1u; k < d_arrivals_.size(); ++k) {
      d_arrivals_[k] = d_arrivals_[k - 1] + stop_count_;
    }

    footpaths_scratchpad_ =
        d_arrivals_.front() + (d_arrivals_.size() * stop_count_);

    size_t station_byte_count = ((tt.stop_count() / 32) + 1) * 4;
    size_t route_byte_count = ((tt.route_count() / 32) + 1) * 4;
    cuda_malloc_set(&station_marks_, station_byte_count, 0);
    cuda_malloc_set(&route_marks_, route_byte_count, 0);

    result_ = new raptor_result(stop_count_);

    cudaStreamCreate(&transfer_stream_);          cc();
    cudaStreamCreate(&proc_stream_);              cc();
  }

#if !defined(__CUDACC__)
  // Do not copy queries, else double free
  d_query(d_query const&) = delete;
#else
  // CUDA needs the copy constructor for the kernel call,
  // as we pass the query to the kernel, which must be a copy
  d_query(d_query const&) = default;
#endif

  __host__ __device__ ~d_query() { 
    // Only call free when destructor is called by host,
    // not when device kernel exits, as we pass the query to the kernel
    #if !defined(__CUDACC__) 
      cuda_free(d_arrivals_.front());
      cuda_free(station_marks_);
      cuda_free(route_marks_);
      delete result_;
    #endif
  }

  cudaStream_t proc_stream_;
  cudaStream_t transfer_stream_;

  arrival_ptrs d_arrivals_;
  global_mem_time* footpaths_scratchpad_;
  unsigned int* station_marks_;
  unsigned int* route_marks_;
  unsigned int finished_;
  raptor_result* result_;
  station_id stop_count_;
};

}  // namespace raptor
}  // namespace motis