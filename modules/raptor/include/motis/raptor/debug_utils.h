#pragma once

#include <sstream>
#include <unordered_map>
#include <vector>

#include "motis/core/schedule/trip.h"
#include "motis/raptor-core/raptor_timetable.h"
#include "utl/get_or_create.h"

namespace motis::raptor {

struct schedule_debug {
  std::unordered_map<std::string,
                     std::unordered_map<route_id, std::vector<trip_id>>>
      trip_dbg_to_route_trips_;
  std::vector<std::string> raptor_route_trip_to_trip_debug_;

  void insert_dbg(std::string const& dbg, route_id r_id, trip_id t_id) {
    raptor_route_trip_to_trip_debug_.emplace_back(std::string{dbg});

    auto& route_map = utl::get_or_create(
        trip_dbg_to_route_trips_, std::string{dbg},
        []() { return std::unordered_map<route_id, std::vector<trip_id>>{}; });

    auto& trip_vec = utl::get_or_create(
        route_map, r_id, []() { return std::vector<trip_id>{}; });

    trip_vec.emplace_back(t_id);
  }

  [[nodiscard]] std::string str(std::string const& dbg) const {
    auto const& route_map = trip_dbg_to_route_trips_.at(dbg);
    std::stringstream str{};
    for(auto const& [r_id, trips] : route_map) {
      str << ";\tr_id: " << +r_id;

      if (!trips.empty()) {
        str << ";\tt_ids: " << trips[0];
        for (int idx = 1, size = trips.size(); idx < size; ++idx) {
          str << ", " << trips[idx];
        }
      }
    }

    return str.str();
  }
};

}  // namespace motis::raptor