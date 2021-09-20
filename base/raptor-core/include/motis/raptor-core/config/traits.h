#pragma once

namespace motis::raptor {

template <typename Trait>
struct traits {

  template <typename Timetable, typename StopTime, typename TimeVal>
  static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal* arrivals, int arrivals_idx, Timetable const& tt,
      StopTime const& stop_time, int stop_time_idx) {
    return Trait::check_and_propagate(arrivals, arrivals_idx, tt, stop_time,
                                      stop_time_idx);
  }

  static int size() { return Trait::size(); }
};

struct trait_nop {

  template <typename Timetable>
  static int get_arrival_time_idx(Timetable const&, int stop_idx, int) {
    return stop_idx;
  }

  template <typename Timetable, typename StopTime, typename TimeVal>
  static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal*& arrivals, int arrivals_idx, Timetable const& _1,
      StopTime const& stop_time, int _2) {

    auto const current_arrival_time = arrivals[arrivals_idx];
    auto const current_stop_arrival = stop_time.arrival_;

    if(current_stop_arrival < current_arrival_time) {
      arrivals[arrivals_idx] = current_stop_arrival;

      return std::make_tuple(current_stop_arrival, true);
    }else{
      return std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    }

  }

  static int size() { return 1; }
};

}  // namespace motis::raptor