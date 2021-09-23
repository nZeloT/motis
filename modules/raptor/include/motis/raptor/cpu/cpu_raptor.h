#pragma once

#include <cstdio>
#include <iomanip>
#include <iostream>

#include "motis/core/common/logging.h"

#include "motis/raptor-core/raptor_query.h"

namespace motis::raptor {

struct mark_store {
  mark_store() = delete;
  explicit mark_store(size_t const size) : marks_(size, false) {}

  void mark(int const idx) { marks_[idx] = true; }
  [[nodiscard]] bool marked(int const idx) const { return marks_[idx]; }
  void reset() { std::fill(std::begin(marks_), std::end(marks_), false); }

private:
  std::vector<bool> marks_;
};

inline void set_upper_bounds(std::vector<std::vector<motis::time>>& arrivals,
                             uint8_t round_k) {
  std::memcpy(arrivals[round_k].data(), arrivals[round_k - 1].data(),
              arrivals[round_k].size() * sizeof(motis::time));
}

template <typename Config>
inline trip_count get_earliest_trip(raptor_timetable const& tt,
                                    raptor_route const& route,
                                    earliest_arrivals const& prev_ea,
                                    stop_times_index const r_stop_offset) {

  // TODO: check where to place trip filter

  station_id const stop_id =
      tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];

  // station was never visited, there can't be a earliest trip
  if (!valid(prev_ea[stop_id])) {
    return invalid<trip_count>;
  }

  // get first defined earliest trip for the stop in the route
  auto const first_trip_stop_idx = route.index_to_stop_times_ + r_stop_offset;
  auto const last_trip_stop_idx =
      first_trip_stop_idx + ((route.trip_count_ - 1) * route.stop_count_);

  trip_count current_trip = 0;
  for (auto stop_time_idx = first_trip_stop_idx;
       stop_time_idx <= last_trip_stop_idx;
       stop_time_idx += route.stop_count_) {

    auto const stop_time = tt.stop_times_[stop_time_idx];
    if (prev_ea[stop_id] <= stop_time.departure_) {
      return current_trip;
    }

    ++current_trip;
  }

  return invalid<trip_count>;
}

inline trip_count get_next_trip_id(raptor_route const& route,
                                   trip_count const current_trip) {

  // get first defined earliest trip for the stop in the route
  auto const first_trip_stop_idx = route.index_to_stop_times_;
  auto const last_trip_stop_idx =
      first_trip_stop_idx + ((route.trip_count_ - 1) * route.stop_count_);

  auto const next_trip_id = current_trip + 1;
  auto const next_trip_stop_idx =
      first_trip_stop_idx + (next_trip_id * route.stop_count_);

  // assuming that every route has more than one stop
  if (next_trip_stop_idx < last_trip_stop_idx) {
    return next_trip_id;
  } else {
    return invalid<trip_count>;
  }
}

inline void init_arrivals(raptor_result& result, earliest_arrivals& prev_ea,
                          raptor_query const& q,
                          raptor_schedule const& raptor_sched,
                          mark_store& station_marks) {

  result[0][q.source_] = q.source_time_begin_;
  prev_ea[q.source_] = q.source_time_begin_;
  station_marks.mark(q.source_);

  for (auto const& f : raptor_sched.initialization_footpaths_[q.source_]) {
    result[0][f.to_] = q.source_time_begin_ + f.duration_;
    prev_ea[f.to_] = result[0][f.to_];
    station_marks.mark(f.to_);
  }
}

inline void print_route_stop_ids(raptor_timetable const& tt,
                                 route_id route_id,
                                 raptor_route const& route) {
  std::cout << "Route " << +route_id << " | ";
  for(int r_stop = 0; r_stop < route.stop_count_; ++r_stop) {
    if(r_stop > 0)
      std::cout << " -> ";
    std::cout <<  +tt.route_stops_[route.index_to_route_stops_ + r_stop];
  }
  std::cout << std::endl;
}

template <typename Config>
inline void update_route(raptor_timetable const& tt, route_id const r_id,
                         earliest_arrivals const& prev_ea,
                         arrivals& current_round, earliest_arrivals& ea,
                         mark_store& station_marks) {

  auto const& route = tt.routes_[r_id];
  //print_route_stop_ids(tt, r_id, route);

  trip_count earliest_trip_id = invalid<trip_count>;
  for (station_id r_stop_offset = 0; r_stop_offset < route.stop_count_;
       ++r_stop_offset) {

    if (!valid(earliest_trip_id)) {
      earliest_trip_id =
          get_earliest_trip<Config>(tt, route, prev_ea, r_stop_offset);
      continue;
    }

    auto const stop_id =
        tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];

    // we need to iterate through all possible trips starting from the earliest
    //  catchable as the criteria from the configuration possibly favor
    //  a later trip because it benefits from other properties than arrival time
    // TODO: maybe it would be useful to implement a time cap here
    //        e.g. a trip departing 6 hours after the earliest possible seems
    //        an unlikely alternative for any traveler independent of the
    //        criteria applied in the current configuration
    for (trip_count trip_id = earliest_trip_id; trip_id < invalid<trip_count>;
         trip_id = get_next_trip_id(route, trip_id)) {

      auto const current_stop_time_idx = route.index_to_stop_times_ +
                                         (trip_id * route.stop_count_) +
                                         r_stop_offset;

      auto const& stop_time = tt.stop_times_[current_stop_time_idx];

      auto [min_arrival_update, mark_station] = Config::check_and_propagate(
          current_round, stop_id, tt, stop_time, current_stop_time_idx);

      if (mark_station) {
        station_marks.mark(stop_id);
      }

      if (min_arrival_update < ea[stop_id]) {
        // write the earliest arrival time for this stop after this round
        //  as this is a lower bound for the trip search
        ea[stop_id] = min_arrival_update;
      }
    }

    // check if we could catch an earlier trip
    // (only relevant for processing the next stop)
    auto const& stop_time =
        tt.stop_times_[route.index_to_stop_times_ +
                       (earliest_trip_id * route.stop_count_) + r_stop_offset];
    auto const previous_k_arrival = prev_ea[stop_id];
    if (previous_k_arrival <= stop_time.departure_) {
      earliest_trip_id =
          get_earliest_trip<Config>(tt, route, prev_ea, r_stop_offset);
    }
  }
}

template <typename Config>
inline void update_footpaths(raptor_timetable const& tt,
                             arrivals& current_round,
                             arrivals const& current_round_c,
                             earliest_arrivals& ea, mark_store& station_marks) {

  // How far do we need to skip until the next stop is reached?
  auto const config_trait_size = Config::trait_size();

  for (station_id stop_id = 0; stop_id < tt.stop_count(); ++stop_id) {

    auto index_into_transfers = tt.stops_[stop_id].index_to_transfers_;
    auto next_index_into_transfers = tt.stops_[stop_id + 1].index_to_transfers_;

    for (auto current_index = index_into_transfers;
         current_index < next_index_into_transfers; ++current_index) {

      auto const& footpath = tt.footpaths_[current_index];

      for (int s_trait_offset = 0; s_trait_offset < config_trait_size;
           ++s_trait_offset) {

        // TODO: how to determine domination of certain journeys
        //       and therewith skip certain updates
        //       is this even possible?

        // if (arrivals[round_k][stop_id] == invalid_time) { continue; }
        // if (earliest_arrivals[stop_id] == invalid_time) { continue; }
        if (!valid(current_round_c[stop_id + s_trait_offset])) {
          continue;
        }

        // there is no triangle inequality in the footpath graph!
        // we cannot use the normal arrival values,
        // but need to use the earliest arrival values as read
        // and write to the normal arrivals,
        // otherwise it is possible that two footpaths
        // are chained together
        motis::time const new_arrival =
            current_round_c[stop_id + s_trait_offset] + footpath.duration_;

        motis::time to_arrival = current_round[footpath.to_];

        if (new_arrival < to_arrival) {
          station_marks.mark(footpath.to_);
          current_round[footpath.to_] = new_arrival;

          if (current_round[footpath.to_] < ea[footpath.to_]) {
            ea[footpath.to_] = new_arrival;
          }
        }
      }
    }
  }
}

template <typename Config>
inline void invoke_cpu_raptor(const raptor_query& query, raptor_statistics&,
                              const raptor_schedule& raptor_sched) {
  auto const& tt = raptor_sched.timetable_;

  auto& result = *query.result_;
  // std::cout << "Received Query: " << std::endl;
  // std::cout << "Start Station:  " << std::setw(7) << +query.source_ << " -> "
  //           << std::setw(6) << +query.source_time_begin_ << std::endl;
  // std::cout << "End Station:    " << std::setw(7) << +query.target_ <<
  // std::endl;

  // TODO: check whether the ea array should also be initialized at
  //        ea[q.source_]
  // TODO: also check whether one of prev_ea or ea can be eliminated
  earliest_arrivals prev_ea(tt.stop_count(), invalid<motis::time>);
  earliest_arrivals ea(tt.stop_count(), invalid<motis::time>);

  std::vector<motis::time> current_round_arrivals(tt.stop_count() *
                                                  Config::trait_size());

  mark_store station_marks(tt.stop_count());
  mark_store route_marks(tt.route_count());

  init_arrivals(result, prev_ea, query, raptor_sched, station_marks);

  for (raptor_round round_k = 1; round_k < max_round_k; ++round_k) {
    bool any_marked = false;

    for (auto s_id = 0; s_id < tt.stop_count(); ++s_id) {
      if (!station_marks.marked(s_id)) {
        continue;
      }
      if (!any_marked) any_marked = true;
      auto const& stop = tt.stops_[s_id];
      for (auto sri = stop.index_to_stop_routes_;
           sri < stop.index_to_stop_routes_ + stop.route_count_; ++sri) {
        route_marks.mark(tt.stop_routes_[sri]);
      }
    }
    if (!any_marked) {
      break;
    }

    station_marks.reset();

    for (route_id r_id = 0; r_id < tt.route_count(); ++r_id) {
      if (!route_marks.marked(r_id)) {
        continue;
      }

      update_route<Config>(tt, r_id, prev_ea, result[round_k], ea,
                           station_marks);
    }

    route_marks.reset();

    std::memcpy(current_round_arrivals.data(), result[round_k],
                current_round_arrivals.size() * sizeof(motis::time));

    update_footpaths<Config>(tt, result[round_k], current_round_arrivals.data(),
                             ea, station_marks);

    // copy earliest arrival times
    std::memcpy(prev_ea.data(), ea.data(),
                prev_ea.size() * sizeof(motis::time));
  }

  //for (int round_k = 0; round_k < max_round_k; ++round_k) {
  //  std::cout << "Results Round " << +round_k << std::endl;
  //  for (int i = 0; i < tt.stop_count(); ++i) {
  //    if (valid(result[round_k][i]))
  //      std::cout << "Stop Id: " << std::setw(7) << +i << " -> " << std::setw(6)
  //                << +result[round_k][i] << std::endl;
  //  }
  //}
}

}  // namespace motis::raptor