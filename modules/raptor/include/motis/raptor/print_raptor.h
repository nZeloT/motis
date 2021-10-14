#pragma once

#include <iomanip>
#include <iostream>

#include "motis/raptor-core/raptor_query.h"
#include "motis/raptor-core/raptor_timetable.h"

namespace motis {
namespace raptor {

inline std::string get_string(station_id const s_id,
                              raptor_schedule const& sched) {
  return "SID: " + std::to_string(s_id) +
         " -- EVA: " + sched.raptor_id_to_eva_[s_id];
}

inline void print_station(station_id const s_id, raptor_schedule const& sched) {
  std::cout << "SID: " << s_id << " -- EVA: " << sched.raptor_id_to_eva_[s_id]
            << '\n';
}

inline void print_stations(raptor_schedule const& sched) {
  for (int i = 0; i < sched.raptor_id_to_eva_.size(); ++i) {
    std::cout << "SID: " << std::setw(7) << +i
              << ";\tEVA: " << sched.raptor_id_to_eva_[i] << std::endl;
  }
}

inline void print_route(route_id r_id, raptor_timetable const& tt) {
  auto const& route = tt.routes_[r_id];

  auto stop_count = route.stop_count_;
  auto index_into_route_stops = route.index_to_route_stops_;
  auto index_into_stop_times = route.index_to_stop_times_;

  std::cout << r_id << "\t{ ";
  for (station_id stop_offset = 0; stop_offset < stop_count; ++stop_offset) {
    std::cout << stop_offset << ": "
              << tt.route_stops_[index_into_route_stops + stop_offset] << " ";
  }
  std::cout << "} " << stop_count << "\n";

  for (trip_count trip_offset = 0; trip_offset < route.trip_count_;
       ++trip_offset) {
    std::cout << trip_offset << " \t[ ";
    for (station_id stop_offset = 0; stop_offset < stop_count; ++stop_offset) {
      auto st_idx =
          index_into_stop_times + (trip_offset * stop_count) + stop_offset;
      std::cout << stop_offset << ": "
                << "(" << tt.stop_times_[st_idx].arrival_ << ","
                << tt.stop_times_[st_idx].departure_ << ")"
                << "(" << +tt.stop_occupancies_[st_idx].inbound_occupancy_ << ")"
                << " ; ";
    }
    std::cout << "]\n";
  }
}

inline void print_routes(std::vector<route_id> const& r_ids,
                         raptor_timetable const& tt) {
  for_each(std::begin(r_ids), std::end(r_ids),
           std::bind(&print_route, std::placeholders::_1, tt));
}

inline void print_station_arrivals(station_id const s_id,
                                   raptor_result const& raptor_result) {
  std::cout << s_id << "(station) Arrivals: [ ";
  for (auto k = 0; k < max_round_k; ++k) {
    std::cout << raptor_result[k][s_id] << " ";
  }
  std::cout << "]\n";
}

template <class Container>
inline void print_js(Container const& js) {
  std::cout << "Journeys: " << js.size() << '\n';
  for (auto const& j : js) {
    std::cout << "Departure: " << get_departure(j)
              << " Arrival: " << get_arrival(j)
              << " Interchanges: " << get_transfers(j) << '\n';
  }
};

inline std::vector<route_id> routes_from_station(station_id const s_id,
                                                 raptor_timetable const& tt) {
  std::vector<route_id> routes;

  auto const& station = tt.stops_[s_id];
  auto const& next_station = tt.stops_[s_id + 1];
  for (auto stop_routes_idx = station.index_to_stop_routes_;
       stop_routes_idx < next_station.index_to_stop_routes_;
       ++stop_routes_idx) {
    routes.push_back(tt.stop_routes_[stop_routes_idx]);
  }

  return routes;
}

inline std::vector<route_id> get_routes_containing(
    std::vector<station_id> const& stations, raptor_timetable const& tt) {
  std::vector<route_id> routes;

  for (route_id r_id = 0; r_id < tt.route_count(); ++r_id) {
    auto const& route = tt.routes_[r_id];

    auto const route_begin =
        std::begin(tt.route_stops_) + route.index_to_route_stops_;
    auto const route_end = route_begin + route.stop_count_;

    bool contains_all = true;
    for (auto const s_id : stations) {
      auto found = std::find(route_begin, route_end, s_id);
      if (found == route_end) {
        contains_all = false;
        break;
      }
    }

    if (contains_all) {
      routes.emplace_back(r_id);
    }
  }

  return routes;
}

inline void print_route_arrivals(route_id const r_id,
                                 raptor_timetable const& tt,
                                 arrivals const& arrs) {
  auto const& route = tt.routes_[r_id];
  auto const base_rsi = route.index_to_route_stops_;
  std::cout << "[ ";
  for (stop_offset so = 0; so < route.stop_count_; ++so) {
    auto const rs = tt.route_stops_[base_rsi + so];
    std::cout << rs << ":" << arrs[rs] << " ";
  }
  std::cout << "]\n";
}

inline void print_route_trip_debug_strings(raptor_schedule const& rp_sched) {
  auto const& tt = rp_sched.timetable_;
  auto const routes = tt.routes_;
  for (int i = 0; i < routes.size(); ++i) {
    auto const route = routes[i];
    for (int j = 0; j < routes[i].trip_count_; ++j) {
      auto const dbg =
          rp_sched
              .raptor_route_trip_to_trip_debug_[route.index_to_trip_dbg_ + j];
      std::cout << "Route: " << std::setw(4) << +i
                << ";\tTrip: " << std::setw(3) << j << ";\tDbg: " << dbg
                << std::endl;
    }
  }
}

inline void print_query(raptor_query const& query) {
  std::cout << "Received Query: " << std::endl;
  std::cout << "Start Station:  " << std::setw(7) << +query.source_ << " -> "
            << std::setw(6) << +query.source_time_begin_ << std::endl;
  std::cout << "End Station:    " << std::setw(7) << +query.target_
            << std::endl;
}

template <typename Config>
inline void print_results(raptor_result const& result,
                          raptor_timetable const& tt) {
  auto const trait_size = Config::trait_size();
  for (int round_k = 0; round_k < max_round_k; ++round_k) {
    std::cout << "Results Round " << +round_k << std::endl;
    for (int i = 0; i < tt.stop_count(); ++i) {
      for (int j = 0; j < trait_size; ++j) {
        if (valid(result[round_k][(i * trait_size) + j]))
          std::cout << "Stop Id: " << std::setw(7) << +i << " -> "
                    << std::setw(6) << +result[round_k][(i * trait_size) + j]
                    << "; Arrival Idx: " << std::setw(6)
                    << +((i * trait_size) + j)
                    << "; Trait Offset: " << std::setw(4) << +j << std::endl;
      }
    }
  }
}

}  // namespace raptor
}  // namespace motis