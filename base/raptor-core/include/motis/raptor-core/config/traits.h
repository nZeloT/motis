#pragma once

#include <tuple>

namespace motis::raptor {

template <typename... TraitData>
struct trait_data : public TraitData... {};

template <typename... Trait>
struct traits;

template <typename FirstTrait, typename... RestTraits>
struct traits<FirstTrait, RestTraits...> {
  using TraitsData = trait_data<FirstTrait, RestTraits...>;

  inline static uint32_t size() {
    auto size = FirstTrait::value_range_size();
    return size * traits<RestTraits...>::size();
  }

  inline static void get_trait_data(uint32_t const total_size, TraitsData& dt,
                                    uint32_t const trait_offset) {
    auto const [rest_trait_size, first_trait_idx, rest_trait_offset] =
        _trait_values(total_size, trait_offset);

    FirstTrait::fill_trait_data_from_idx(dt, first_trait_idx);

    traits<RestTraits...>::get_trait_data(rest_trait_size, dt,
                                          rest_trait_offset);
  }

  template <typename Timetable>
  inline static bool trip_matches_traits(
      TraitsData const& dt, Timetable const& tt, uint32_t const r_id,
      uint32_t const t_id, uint32_t const dep_offset, uint32_t const arr_offset) {
    return FirstTrait::trip_matches_trait(dt, tt, r_id, t_id, dep_offset, arr_offset) &&
           traits<RestTraits...>::trip_matches_traits(dt, tt, r_id, t_id,
                                                      dep_offset, arr_offset);
  }

  template <typename Timetable, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_update_arrivals_old(
      uint32_t total_size, TimeVal* const& prev_arrival, TimeVal*& curr_arrival,
      Timetable const& tt, uint32_t const dep_arrivals_idx,
      uint32_t const cur_arrivals_idx, uint32_t const dep_sti,
      uint32_t const curr_sti) {

    auto const first_value_size = FirstTrait::value_range_size();
    auto const nested_trait_size = total_size / first_value_size;

    auto result = std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    uint32_t it = FirstTrait::init_trait_iteration_old();
    while (it != std::numeric_limits<uint32_t>::max()) {

      auto const [min_arr_time, rest_satisfied] =
          traits<RestTraits...>::check_and_update_arrivals_old(
              nested_trait_size, prev_arrival, curr_arrival, tt,
              (dep_arrivals_idx + it * nested_trait_size),
              (cur_arrivals_idx + it * nested_trait_size), dep_sti, curr_sti);

      std::get<0>(result) = std::min(std::get<0>(result), min_arr_time);
      std::get<1>(result) =
          FirstTrait::is_satisfied(std::get<1>(result), it, rest_satisfied);

      it = FirstTrait::next_trait_iteration_old(tt, dep_sti, curr_sti, it);
    }

    return result;
  }

  inline static bool is_update_required(uint32_t total_size,
                                        TraitsData const& td,
                                        uint32_t t_offset) {

    auto const [rest_trait_size, first_trait_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_update_required(td, first_trait_idx) &&
           traits<RestTraits...>::is_update_required(rest_trait_offset, td,
                                                     rest_trait_offset);
  }

  inline static bool is_trait_satisfied(uint32_t total_size,
                                        TraitsData const& td,
                                        uint32_t t_offset) {
    auto const [rest_trait_size, first_trait_idx, rest_trait_offset] =
        _trait_values(total_size, t_offset);

    return FirstTrait::is_trait_satisfied(td, first_trait_idx) &&
           traits<RestTraits...>::is_trait_satisfied(rest_trait_size, td,
                                                     t_offset);
  }

  // helper to aggregate values while progressing through the route stop by stop
  template <typename Timetable>
  inline static void update_aggregate(TraitsData& aggregate_dt,
                                      Timetable const& tt, uint32_t const r_id,
                                      uint32_t const t_id,
                                      uint32_t const s_offset,
                                      uint32_t const sti) {
    FirstTrait::update_aggregate(aggregate_dt, tt, r_id, t_id, s_offset, sti);
    traits<RestTraits...>::update_aggregate(aggregate_dt, tt, r_id, t_id,
                                            s_offset, sti);
  }

  // reset the aggregate everytime the departure station changes
  inline static void reset_aggregate(TraitsData& aggregate_dt) {
    FirstTrait::reset_aggregate(aggregate_dt);
    traits<RestTraits...>::reset_aggregate(aggregate_dt);
  }

  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& j, Candidate const& c) {
    return FirstTrait::dominates(j, c) &&
           traits<RestTraits...>::dominates(j, c);
  }

  inline static std::tuple<uint32_t, uint32_t, uint32_t> _trait_values(
      uint32_t const total_size, uint32_t const t_offset) {
    auto const first_value_size = FirstTrait::value_range_size();
    auto const rest_trait_size = total_size / first_value_size;

    auto const first_trait_idx = t_offset / rest_trait_size;
    auto const rest_trait_offset = t_offset % rest_trait_size;

    return std::make_tuple(rest_trait_size, first_trait_idx, rest_trait_offset);
  }
};

template <>
struct traits<> {
  using TraitsData = trait_data<>;

  inline static uint32_t size() { return 1; }

  template <typename Data>
  inline static void get_trait_data(uint32_t const _1, Data& _2,
                                    uint32_t const _3) {}

  template <typename Data, typename Timetable>
  inline static bool trip_matches_traits(Data const& dt, Timetable const& tt,
                                         uint32_t const r_id,
                                         uint32_t const t_id,
                                         uint32_t const dep_offset,
                                         uint32_t const arr_offset) {
    return true;
  }

  // Return value gives the lowest written arrival time and an indication
  // whether
  //   the traits have been satisfied i.e. there is an arrival value written
  //   for all possible trait values and therefore no better arrival time can
  //   be archived with subsequent trips having later departure times
  template <typename Timetable, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_update_arrivals_old(
      uint32_t total_size, TimeVal* const& prev_arrival, TimeVal*& curr_arrival,
      Timetable const& tt, uint32_t const dep_arrivals_idx,
      uint32_t const cur_arrivals_idx, uint32_t const dep_sti,
      uint32_t const curr_sti) {

    auto const InvalidTime = std::numeric_limits<TimeVal>::max();

    // 1. check if the departure station has a valid arrival time on the
    //    previous round with the given trait offset (already added to the
    //    departure_arr_idx)
    auto const departure_arr_time = prev_arrival[dep_arrivals_idx];
    if (departure_arr_time == InvalidTime) {
      return std::make_tuple(InvalidTime, false);
    }

    // 2. the departure station has a valid departure time
    //    now check that the given trip departs after the arrival at
    //    the departure station
    auto const departure_stop_trip_departure_time =
        tt.stop_times_[dep_sti].departure_;
    if (departure_arr_time > departure_stop_trip_departure_time) {
      return std::make_tuple(InvalidTime, false);
    }

    // TODO: check if this is still needed; i think not

    // 3. there exists a valid arrival time on the departure station with the
    //    given trait offset; now check whether the known departure time is
    //    lower than the arrival time at the current stop
    auto const current_stop_arrival = tt.stop_times_[curr_sti].arrival_;
    if (current_stop_arrival <= departure_arr_time) {
      return std::make_tuple(InvalidTime, false);
    }

    // 4. there exists a departure station which has an arrival time
    //    less than the arrival time at the current stop; therefore now
    //    check whether the new arrival time is lower than the already known
    //    possible arrival time at this stop
    auto const current_arrival_time = curr_arrival[cur_arrivals_idx];
    if (current_stop_arrival < current_arrival_time) {
      curr_arrival[cur_arrivals_idx] = current_stop_arrival;

      return std::make_tuple(current_stop_arrival, true);
    } else {
      return std::make_tuple(InvalidTime, false);
    }
  }

  template <typename Data>
  inline static bool is_update_required(uint32_t _1, Data const& _2,
                                        uint32_t _3) {
    return true;  // return natural element of conjunction
  }

  template <typename Data>
  inline static bool is_trait_satisfied(uint32_t _1, Data const& _2,
                                        uint32_t _3) {
    return true;  // return natural element of conjunction
  }

  template <typename Data, typename Timetable>
  inline static void update_aggregate(Data& _1, Timetable const& _2,
                                      uint32_t const _3, uint32_t const _4,
                                      uint32_t const _5, uint32_t const _6) {}

  template <typename Data>
  inline static void reset_aggregate(Data& _1) {}

  // giving the neutral element of the conjunction
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& _1, Candidate const& _2) {
    return true;
  }
};

}  // namespace motis::raptor