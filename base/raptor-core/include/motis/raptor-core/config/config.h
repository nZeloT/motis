#pragma once

#include <tuple>

namespace motis::raptor {

template <typename Trait, typename Filter>
struct config {

  template <typename Timetable, typename StopTime, typename TimeVal>
  static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal*& arrivals, int arrivals_idx, Timetable const& tt,
      StopTime const& stop_time, int stop_time_idx) {
    return Trait::check_and_propagate(arrivals, arrivals_idx, tt, stop_time,
                                      stop_time_idx);
  }

  static int trait_size() { return Trait::size(); }

  template <typename Timetable>
  static bool is_filtered(Timetable const& tt, int stop_time_idx) {
    return Filter::is_filtered(tt, stop_time_idx);
  }
};

}  // namespace motis::raptor