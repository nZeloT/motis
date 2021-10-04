#pragma once

#include <tuple>

namespace motis::raptor {

template <typename Trait, typename Filter>
struct config {
  using TraitData = typename Trait::TraitData;

  inline static int trait_size() { return Trait::size(); }

  inline static int get_arrival_idx(int const stop_idx,
                                    int const trait_offset = 0) {
    return stop_idx * trait_size() + trait_offset;
  }

  // used during route update to update a arrival time
  // and propagate it along the trait feature accordingly if desired
  template <typename Timetable, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal* const& prev_arrivals, TimeVal*& curr_arrivals,
      Timetable const& tt, int const r_id, int const t_id,
      int const departure_stop_id, int const current_stop_id,
      uint32_t const current_stop_time_idx,
      uint32_t const departure_stop_time_idx) {
    return Trait::check_and_propagate(
        prev_arrivals, curr_arrivals, tt, r_id, t_id, departure_stop_id,
        current_stop_id, get_arrival_idx(departure_stop_id),
        get_arrival_idx(current_stop_id), current_stop_time_idx,
        departure_stop_time_idx);
  }

  // helper function used during arrivals initialization; just distributes
  // an arrival time value across the traits without checking if there are
  // already better suitable values written
  template <typename TimeVal>
  inline static void propagate_across_traits(TimeVal*& arrivals,
                                             int arrivals_idx,
                                             TimeVal const& propagate) {
    Trait::propagate_across_traits(arrivals, arrivals_idx, propagate);
  }

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& journey,
                               Candidate const& candidate) {
    return Trait::dominates(journey, candidate);
  }

  // derive the trait values from the arrival time index
  // expecting that the stop_idx is already subtracted and the given index
  // only specifies the shifts within the traits
  template <typename ArrivalIndex>
  inline static TraitData derive_trait_values(ArrivalIndex const idx) {
    TraitData data{};
    Trait::derive_trait_values(data, idx);
    return data;
  }

  template <typename Timetable>
  inline static bool is_filtered(Timetable const& tt, int stop_time_idx) {
    return Filter::is_filtered(tt, stop_time_idx);
  }
};

}  // namespace motis::raptor