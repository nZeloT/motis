#pragma once

#include <cstdint>
#include <tuple>

namespace motis::raptor {

constexpr uint8_t max_occupancy = 2;

template <typename NestedTrait>
struct trait_max_occupancy {

  template <typename Timetable, typename StopTime, typename TimeVal>
  static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal*& arrivals, int arrivals_idx, Timetable const& tt,
      StopTime const& stop_time, int stop_time_idx) {

    auto const dimension_size = size();

    int current_occupancy_value =
        tt.stop_occupancies_[stop_time_idx].inbound_occupancy_;

    // as we are doing max occupancy we need to check
    // whether this trip with occupancy o gives better arrival times
    // for connections with max_occupancy >= o
    // this possibly leads to updates from min_occupancy up to max_occupancy
    // which can be costly
    auto r_value = std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    for (int occupancy = current_occupancy_value; occupancy <= max_occupancy;
         ++occupancy) {
      auto const r = NestedTrait::check_and_propagate(
          arrivals, arrivals_idx + (occupancy * dimension_size), tt, stop_time,
          stop_time_idx);

      //merge results into return value;
      // first time gives the minimal overall arrival time used during this update
      // and second gives an indication whether there was an update at all
      r_value = std::make_tuple(
          std::min(std::get<0>(r_value), std::get<0>(r)),
          std::get<1>(r_value) || std::get<1>(r));
    }

    return r_value;
  }

  inline static int size() { return (max_occupancy + 1) * NestedTrait::size(); }
};

}  // namespace motis::raptor
