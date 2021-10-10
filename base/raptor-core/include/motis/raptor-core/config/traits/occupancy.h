#pragma once

#include <cstdint>
#include <tuple>

namespace motis::raptor {

constexpr uint8_t max_occupancy = 2;

// linearly scale max_occupancy values to indices
constexpr uint32_t moc_value_range_size = max_occupancy + 1;

struct trait_max_occupancy {
  // Trait Data
  uint8_t max_occupancy_;

  inline static uint32_t value_range_size() { return moc_value_range_size; }

  template <typename TraitsData>
  inline static void fill_trait_data_from_idx(TraitsData& dt,
                                              uint32_t const dimension_idx) {
    // can be used as occupancy at idx 0
    //  maps to an occupancy value of 0
    dt.max_occupancy_ = dimension_idx;
  }

  template <typename TraitsData, typename Timetable>
  inline static bool trip_matches_trait(TraitsData const& dt,
                                        Timetable const& tt, uint32_t const _1,
                                        uint32_t const _2, uint32_t const _3,
                                        uint32_t const sti) {

    auto const trip_arrival_occupancy =
        tt.stop_occupancies_[sti].inbound_occupancy_;

    return trip_arrival_occupancy <= dt.max_occupancy_;
  }

  inline static uint32_t init_trait_iteration_old() {
    return 2;  // iterate downwards
  }

  template <typename TraitsData>
  inline static uint32_t init_trait_iteration(TraitsData const& _1) {
    return 2;  // iterate downwards
  }

  inline static bool is_satisfied(bool current_state, uint32_t trait_it,
                                  bool rest_satisfied) {
    // trait_it == 0 indicates that the possibly wrote a value for max_occupancy
    // = 0 this means we also have an upper bound for max_occuancies > 0
    return trait_it == 0 && rest_satisfied;
  }

  template <typename Timetable>
  inline static uint32_t next_trait_iteration_old(Timetable const& tt,
                                                  uint32_t dep_sti,
                                                  uint32_t curr_sti,
                                                  uint32_t current_trait_it) {
    if (current_trait_it == 0) {
      return std::numeric_limits<uint32_t>::max();
    }

    auto const next_it = current_trait_it - 1;

    //just check occupancy on arrival sti
    auto const max_occ = tt.stop_occupancies_[curr_sti].inbound_occupancy_;

    if (max_occ <= next_it) {
      return next_it;
    } else {
      return std::numeric_limits<uint32_t>::max();
    }
  }

  template <typename TraitsData>
  inline static uint32_t next_trait_iteration(TraitsData const& aggregate_dt,
                                              uint32_t current_trait_it) {
    if (current_trait_it == 0) {
      return std::numeric_limits<uint32_t>::max();
    }

    auto const next_it = current_trait_it - 1;
    if (aggregate_dt.max_occupancy_ <= next_it) {
      return next_it;
    } else {
      return std::numeric_limits<uint32_t>::max();
    }
  }

  template <typename TraitsData, typename Timetable>
  inline static void update_aggregate(TraitsData& aggregate_dt,
                                      Timetable const& tt, uint32_t const _1,
                                      uint32_t const _2, uint32_t const _3,
                                      uint32_t const sti) {
    auto const stop_occupancy = tt.stop_occupancies_[sti].inbound_occupancy_;
    aggregate_dt.max_occupancy_ =
        std::max(aggregate_dt.max_occupancy_, stop_occupancy);
  }

  template <typename TraitsData>
  inline static void reset_aggregate(TraitsData& aggregate_dt) {
    aggregate_dt.max_occupancy_ = 0;
  }

  // check if journey dominates candidate in max_occupancy
  template <typename Journey, typename Candidate>
  static bool dominates(Journey const& journey, Candidate const& candidate) {
    // 1. determine candidate max_occupancy
    auto const candidate_max_occ = candidate.trait_data_.max_occupancy_;
    // 2. compare against journeys max_occupancy
    return journey.occupancy_max_ <= candidate_max_occ;
  }
};

}  // namespace motis::raptor
