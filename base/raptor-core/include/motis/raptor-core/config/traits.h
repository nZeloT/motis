#pragma once

namespace motis::raptor {

template <typename Trait>
struct traits {

  inline static int size() { return Trait::size(); }

  template <typename Timetable, typename StopTime, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal* arrivals, int arrivals_idx, Timetable const& tt,
      StopTime const& stop_time, int stop_time_idx) {
    return Trait::check_and_propagate(arrivals, arrivals_idx, tt, stop_time,
                                      stop_time_idx);
  }

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& journey,
                               Candidate const& candidate) {
    return Trait::dominates(journey, candidate, 0);
  }

  // derive the trait values from the arrival time index
  template <typename ArrivalIndex>
  inline static std::vector<uint32_t> derive_trait_values(
      ArrivalIndex const idx) {
    return Trait::derive_trait_value(idx);
  }

  template <typename TimeVal>
  inline static void propagate_across_traits(TimeVal*& arrivals,
                                             int arrivals_idx,
                                             TimeVal const& propagate) {
    auto const size = traits<Trait>::size();
    for (int t_offset = 0; t_offset < size; ++t_offset) {
      arrivals[arrivals_idx + t_offset] = propagate;
    }
  }
};

struct trait_nop {

  // giving the neutral element for sizing
  inline static int size() { return 1; }

  // giving the neutral element of the konjunction
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& _1, Candidate const& _2, int _3) {
    return true;
  }

  // giving the neutral element of vectors
  template <typename ArrivalIndex>
  inline static std::vector<uint32_t> derive_trait_values(
      ArrivalIndex const idx) {
    return std::vector<uint32_t>{};
  }

  template <typename Timetable, typename StopTime, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal*& arrivals, int arrivals_idx, Timetable const& _1,
      StopTime const& stop_time, int _2) {

    auto const current_arrival_time = arrivals[arrivals_idx];
    auto const current_stop_arrival = stop_time.arrival_;

    if (current_stop_arrival < current_arrival_time) {
      arrivals[arrivals_idx] = current_stop_arrival;

      return std::make_tuple(current_stop_arrival, true);
    } else {
      return std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    }
  }
};

}  // namespace motis::raptor