#pragma once

#include <tuple>

namespace motis::raptor {

template <typename Traits, typename Filter>
struct config {
  using TraitsData = typename Traits::TraitsData;

  inline static int trait_size() { return Traits::size(); }

  inline static int get_arrival_idx(uint32_t const stop_idx,
                                    uint32_t const trait_offset = 0) {
    return stop_idx * trait_size() + trait_offset;
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
  inline static bool trip_matches_traits(TraitsData const& dt,
                                         Timetable const& tt,
                                         uint32_t const route_id,
                                         uint32_t const trip_id,
                                         uint32_t const stop_offset,
                                         uint32_t const stop_time_idx) {
    return Traits::trip_matches_traits(dt, tt, route_id, trip_id, stop_offset,
                                       stop_time_idx);
  }

  // used during route update to update a arrival time
  // and propagate it along the trait feature accordingly if desired
  template <typename Timetable, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_update_arrivals_old(
      TimeVal* const& prev_arrivals, TimeVal*& curr_arrivals,
      Timetable const& tt,
      uint32_t const dep_stop_id, uint32_t const curr_stop_id,
      uint32_t const departure_stop_time_idx,
      uint32_t const current_stop_time_idx) {
    return Traits::check_and_update_arrivals_old(
        trait_size(), prev_arrivals, curr_arrivals, tt,
        get_arrival_idx(dep_stop_id), get_arrival_idx(curr_stop_id),
        departure_stop_time_idx, current_stop_time_idx);
  }

  // used during route update to update a arrival time
  // and propagate it along the trait feature accordingly if desired
  template <typename Timetable, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_update_arrivals(
      TimeVal* const& prev_arrivals, TimeVal*& curr_arrivals,
      Timetable const& tt, TraitsData const& aggregate_dt,
      uint32_t const dep_stop_id, uint32_t const curr_stop_id,
      uint32_t const departure_stop_time_idx,
      uint32_t const current_stop_time_idx) {
    return Traits::check_and_update_arrivals(
        trait_size(), prev_arrivals, curr_arrivals, tt, aggregate_dt,
        get_arrival_idx(dep_stop_id), get_arrival_idx(curr_stop_id),
        departure_stop_time_idx, current_stop_time_idx);
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

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& journey,
                               Candidate const& candidate) {
    return Traits::dominates(journey, candidate);
  }

  template <typename Timetable>
  inline static bool is_filtered(Timetable const& tt, int stop_time_idx) {
    return Filter::is_filtered(tt, stop_time_idx);
  }
};

}  // namespace motis::raptor