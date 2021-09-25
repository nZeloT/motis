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

  template <typename Timetable, typename StopTime, typename TimeVal>
  static std::tuple<TimeVal, bool> check_and_propagate(
      TimeVal*& arrivals, int arrivals_idx, Timetable const& tt,
      StopTime const& stop_time, int stop_time_idx) {

    auto const dimension_size = NestedTrait::size();

    int current_occupancy_value =
        tt.stop_occupancies_[stop_time_idx].inbound_occupancy_;

    // as we are doing max occupancy we need to check
    // whether this trip with occupancy o gives better arrival times
    // for connections with max_occupancy >= o
    // this possibly leads to updates from min_occupancy up to max_occupancy
    // which can be costly
    auto r_value = std::make_tuple(std::numeric_limits<TimeVal>::max(), false);
    for (int occupancy = current_occupancy_value; occupancy <= max_occupancy;
         ++occupancy) {
      auto const r = NestedTrait::check_and_propagate(
          arrivals, arrivals_idx + (occupancy * dimension_size), tt, stop_time,
          stop_time_idx);

      // merge results into return value;
      //  first time gives the minimal overall arrival time used during this
      //  update and second gives an indication whether there was an update at
      //  all
      r_value = std::make_tuple(std::min(std::get<0>(r_value), std::get<0>(r)),
                                std::get<1>(r_value) || std::get<1>(r));
    }

    return r_value;
  }

  // check if journey dominates candidate in max_occupancy
  template <typename Journey, typename Candidate>
  static bool dominates(Journey const& journey, Candidate const& candidate) {
    // 1. determine candidate max_occupancy
    auto const candidate_max_occ = candidate.trait_data_.max_occupancy_;
    // 2. compare against journeys max_occupancy
    auto const dominates = journey.occupancy_max_ < candidate_max_occ;
    // 3. return result konjunction
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
