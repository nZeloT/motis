#pragma once

#include <tuple>

namespace motis::raptor {

template <typename Trait, typename Filter>
struct config {

  inline static int trait_size() { return Trait::size(); }

  //used during route update to update a arrival time
  //and propagate it along the trait feature accordingly if desired
  template <typename Timetable, typename StopTime, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal*& arrivals, int arrivals_idx, Timetable const& tt,
      StopTime const& stop_time, int stop_time_idx) {
    return Trait::check_and_propagate(arrivals, arrivals_idx, tt, stop_time,
                                      stop_time_idx);
  }

  //helper function used during arrivals initialization; just distributes
  //an arrival time value across the traits without checking if there are
  //already better suitable values written
  template <typename TimeVal>
  inline static void propagate_across_traits(TimeVal*& arrivals, int arrivals_idx,
                                      TimeVal const& propagate) {
    Trait::propagate_across_traits(arrivals, arrivals_idx, propagate);
  }

  //check if a candidate journey dominates a given journey by checking on the
  //respective timetable values
  template<typename Journey, typename Candidate>
  inline static bool dominates(Journey const& journey,
                        Candidate const& candidate) {
    return Trait::dominates(journey, candidate);
  }

  //derive the trait values from the arrival time index
  //expecting that the stop_idx is already subtracted and the given index
  //only specifies the shifts within the traits
  template<typename ArrivalIndex>
  inline static std::vector<uint32_t> derive_trait_values(ArrivalIndex const idx) {
    return Trait::derive_trait_value(idx);
  }

  template <typename Timetable>
  inline static bool is_filtered(Timetable const& tt, int stop_time_idx) {
    return Filter::is_filtered(tt, stop_time_idx);
  }
};

}  // namespace motis::raptor