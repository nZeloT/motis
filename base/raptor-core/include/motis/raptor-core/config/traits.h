#pragma once

namespace motis::raptor {

template <typename Trait>
struct traits {

  template <typename Timetable>
  static int get_arrival_time_idx(Timetable const& tt, int stop_idx,
                           int stop_time_idx) {
    return Trait::get_arrival_time_idx(tt, stop_idx, stop_time_idx);
  }

  static int size() { return Trait::size(); }
};

struct trait_nop {

  template <typename Timetable>
  static int get_arrival_time_idx(Timetable const&, int stop_idx, int) {
    return stop_idx;
  }

  static int size() { return 1; }
};

}  // namespace motis::raptor