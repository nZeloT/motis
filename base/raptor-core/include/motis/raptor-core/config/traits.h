#pragma once

namespace motis::raptor {

template <typename Trait>
struct traits {
  using TraitData = typename Trait::Data;

  inline static int size() { return Trait::size(); }

  template <typename Timetable, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal* const& prev_arrival, TimeVal* curr_arrival, Timetable const& tt,
      int const r_id, int const t_id, int const departure_stop_id,
      int const current_stop_id, int const departure_arr_idx,
      int const current_arr_idx, int const stop_time_idx) {
    return Trait::check_and_propagate(
        prev_arrival, curr_arrival, tt, r_id, t_id, departure_stop_id,
        current_stop_id, departure_arr_idx, current_arr_idx, stop_time_idx);
  }

  // check if a candidate journey dominates a given journey by checking on the
  // respective timetable values
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& journey,
                               Candidate const& candidate) {
    return Trait::dominates(journey, candidate);
  }

  // derive the trait values from the arrival time index
  template <typename ArrivalIdx>
  inline static void derive_trait_values(TraitData& data, ArrivalIdx idx) {
    Trait::derive_trait_values(data, idx);
  }

  template <typename TimeVal>
  inline static void propagate_across_traits(TimeVal*& arrivals,
                                             int arrivals_idx,
                                             TimeVal const& propagate) {
    auto const size = traits<Trait>::size();
    for (int t_offset = 0; t_offset < size; ++t_offset) {
      arrivals[(arrivals_idx * size) + t_offset] = propagate;
    }
  }
};

struct trait_data_nop {};
struct trait_nop {
  using Data = trait_data_nop;

  // giving the neutral element for sizing
  inline static int size() { return 1; }

  // giving the neutral element of the conjunction
  template <typename Journey, typename Candidate>
  inline static bool dominates(Journey const& _1, Candidate const& _2) {
    return true;
  }

  // giving the neutral element of vectors
  template <typename ArrivalIdx>
  inline static void derive_trait_values(Data& _1, ArrivalIdx const _2) {}

  template <typename Timetable, typename TimeVal>
  inline static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal* const& prev_arrival, TimeVal* curr_arrival, Timetable const& tt,
      int const r_id, int const t_id, int const departure_stop_id,
      int const current_stop_id, int const departure_arr_idx,
      int const current_arr_idx, int const stop_time_idx) {

    // 1. check if the departure station has a valid arrival time on the
    //    previous round with the given trait offset (already added to the
    //    departure_arr_idx)
    auto const departure_arr_time = prev_arrival[departure_arr_idx];
    if(departure_arr_time == std::numeric_limits<TimeVal>::max()){
      return std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    }

    // 2. there exists a valid arrival time on the departure station with the
    //    given trait offset; now check whether the known departure time is
    //    lower than the arrival time at the current stop
    auto const current_stop_arrival = tt.stop_times_[stop_time_idx].arrival_;
    if(current_stop_arrival <= departure_arr_time) {
      return std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    }

    // 3. there exists a departure station which has an arrival time
    //    less than the arrival time at the current stop; therefore now
    //    check whether the new arrival time is lower than the already known
    //    possible arrival time at this stop

    auto const current_arrival_time = curr_arrival[current_arr_idx];
    if (current_stop_arrival < current_arrival_time) {
      curr_arrival[current_arr_idx] = current_stop_arrival;

      return std::make_tuple(current_stop_arrival, true);
    } else {
      return std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    }
  }
};

}  // namespace motis::raptor