#pragma once

#include <tuple>

namespace motis::raptor {

template <typename Traits>
struct config {
  using TraitsData = typename Traits::TraitsData;

  inline static int trait_size() { return Traits::size(); }

  inline static int get_arrival_idx(uint32_t const stop_idx,
                                    uint32_t const trait_offset = 0) {
    return stop_idx * trait_size() + trait_offset;
  }

  inline static bool is_update_required(TraitsData const& td, uint32_t t_offset) {
    return Traits::is_update_required(trait_size(), td, t_offset);
  }

  inline static bool is_trait_satisfied(TraitsData const& td, uint32_t t_offset) {
    return Traits::is_trait_satisfied(trait_size(), td, t_offset);
  }

  // derive the trait values from the arrival time index
  // expecting that the stop_idx is already subtracted and the given index
  // only specifies the shifts within the traits
  inline static TraitsData get_traits_data(uint32_t const trait_offset) {
    TraitsData data{};
    Traits::get_trait_data(trait_size(), data, trait_offset);
    return data;
  }

  template <typename Timetable>
  inline static void update_traits_aggregate(TraitsData& aggregate_dt,
                                             Timetable const& tt, uint32_t r_id,
                                             uint32_t t_id, uint32_t s_offset,
                                             uint32_t sti) {
    Traits::update_aggregate(aggregate_dt, tt, r_id, t_id, s_offset, sti);
  }

  inline static void reset_traits_aggregate(TraitsData& dt) {
    Traits::reset_aggregate(dt);
  }

  template <typename Timetable>
  inline static bool trip_matches_traits(TraitsData const& dt,
                                         Timetable const& tt,
                                         uint32_t const route_id,
                                         uint32_t const trip_id,
                                         uint32_t const dep_stop_offset,
                                         uint32_t const arr_stop_offset) {
    return Traits::trip_matches_traits(dt, tt, route_id, trip_id, dep_stop_offset,
                                       arr_stop_offset);
  }

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& journey,
                               Candidate const& candidate) {
    return Traits::dominates(journey, candidate);
  }
};

}  // namespace motis::raptor