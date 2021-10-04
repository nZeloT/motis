#pragma once

#include <cstdint>
#include <tuple>

namespace motis::raptor {

constexpr uint8_t max_occupancy = 2;

template <typename NestedTrait>
struct trait_data_max_occupancy : public NestedTrait::Data {
  uint8_t max_occupancy_;
};

template <typename NestedTrait>
struct trait_max_occupancy {
  using Data = trait_data_max_occupancy<NestedTrait>;

  inline static int size() { return (max_occupancy + 1) * NestedTrait::size(); }

  template <typename Timetable, typename TimeVal>
  static TimeVal check_and_propagate(
      TimeVal* const& prev_arrival, TimeVal* curr_arrival, Timetable const& tt,
      int const r_id, int const t_id, int const departure_stop_id,
      int const current_stop_id, int const departure_arr_idx,
      int const current_arr_idx, uint32_t const current_stop_time_idx,
      uint32_t const departure_stop_time_idx) {

    auto const dimension_size = NestedTrait::size();

    int current_occupancy_value =
        tt.stop_occupancies_[current_stop_time_idx].inbound_occupancy_;

    // as we are doing max occupancy we need to check
    // whether this trip with occupancy o gives better arrival times
    // for connections with max_occupancy >= o
    // this possibly leads to updates from min_occupancy up to max_occupancy
    // which can be costly
    auto r_value = std::numeric_limits<TimeVal>::max();
    for (int occupancy = current_occupancy_value; occupancy <= max_occupancy;
         ++occupancy) {
      auto const occupancy_trait_shift = occupancy * dimension_size;
      auto const r = NestedTrait::check_and_propagate(
          prev_arrival, curr_arrival, tt, r_id, t_id, departure_stop_id,
          current_stop_id, departure_arr_idx + occupancy_trait_shift,
          current_arr_idx + occupancy_trait_shift, current_stop_time_idx,
          departure_stop_time_idx);

      r_value = std::min(r_value, r);
    }

    return r_value;
  }

  // check if journey dominates candidate in max_occupancy
  template <typename Journey, typename Candidate>
  static bool dominates(Journey const& journey, Candidate const& candidate) {
    // 1. determine candidate max_occupancy
    auto const candidate_max_occ = candidate.trait_data_.max_occupancy_;
    // 2. compare against journeys max_occupancy
    auto const dominates = journey.occupancy_max_ <= candidate_max_occ;
    // 3. return result conjunction
    return dominates &&
           NestedTrait::dominates(journey, candidate);
  }

  template <typename ArrivalIdx>
  inline static void derive_trait_values(Data& data, ArrivalIdx const idx) {
    auto const dimension_size = NestedTrait::size();
    // Trait value = Trait index in our case
    // because value and index align
    uint32_t const trait_value = idx / dimension_size;
    uint32_t const nested_idx = idx % dimension_size;

    data.max_occupancy_ = trait_value;
    NestedTrait::derive_trait_values(data, nested_idx);
  }
};

}  // namespace motis::raptor
