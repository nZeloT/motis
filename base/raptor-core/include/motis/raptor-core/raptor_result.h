#pragma once

#include <array>
#include <cstring>

#include "motis/raptor-core/raptor_timetable.h"

namespace motis {
namespace raptor {

struct raptor_result {
  raptor_result() = delete;
  raptor_result(raptor_result const&) = delete;

  explicit raptor_result(int const arrival_times) /*: stop_count_(stop_count)*/ {
    size_t const bytes = arrival_times * sizeof(motis::time) * max_round_k;

    cudaMallocHost(&result_.front(), bytes);
    std::memset(result_.front(), 0xFF, bytes);

    for (auto k = 1; k < max_round_k; ++k) {
      result_[k] = result_[k - 1] + arrival_times;
    }
  }

  ~raptor_result() { cudaFreeHost(result_.front()); }

  arrivals const& operator[](raptor_round const index) const {
    return result_[index];
  }
  arrivals& operator[](raptor_round const index) {
    return result_[index];
  }

  //station_id stop_count_;
  std::array<arrivals, max_round_k> result_;
};

}  // namespace raptor
}  // namespace motis