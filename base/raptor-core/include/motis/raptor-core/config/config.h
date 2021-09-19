#pragma once

namespace motis::raptor {

template <typename Trait, typename Filter>
struct config {

  template <typename Timetable>
  static int get_arrival_time_idx(Timetable const& tt, int stop_idx,
                                  int stop_time_idx) {
    return Trait::get_arrival_time_idx(tt, stop_idx, stop_time_idx);
  }

  static int trait_size() { return Trait::size(); }

  template <typename Timetable>
  static bool is_filtered(Timetable const& tt, int stop_time_idx) {
    return Filter::is_filtered(tt, stop_time_idx);
  }
};

}