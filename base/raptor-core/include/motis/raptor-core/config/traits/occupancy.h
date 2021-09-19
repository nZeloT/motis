#pragma once

#include <cstdint>
#include <optional>

namespace motis::raptor {

constexpr uint8_t max_occupancy = 2;

template<typename NestedTrait>
struct trait_occupancy {

  template <typename Timetable>
  static int get_arrival_time_idx(Timetable const& tt, int arr_time_idx,
                           int stop_time_idx) {
    int dimension_value = tt.stop_occupancy_[stop_time_idx].inbound_occupancy_;
    int dimension_idx   =  dimension_value * size();

    //innermost NestedTrait holds the final value and returns it up the chain
    return NestedTrait::get_arrival_time_idx(tt, arr_time_idx + dimension_idx,
                                             stop_time_idx);

  }

  inline static int size() {
    return (max_occupancy+1) * NestedTrait::size();
  }
};

}
