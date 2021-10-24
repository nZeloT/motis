#pragma once

#include <cstdio>
#include <iomanip>
#include <iostream>

#include "motis/core/common/logging.h"
#include "motis/raptor-core/raptor_query.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_schedule.h"
#include "motis/raptor/raptor_statistics.h"

namespace motis::raptor {
using departure_station = std::vector<station_id>;

struct mark_store {
  mark_store() = delete;
  explicit mark_store(size_t const size) : marks_(size, false) {}

  void mark(int const idx) { marks_[idx] = true; }
  [[nodiscard]] bool marked(int const idx) const { return marks_[idx]; }
  void reset() { std::fill(std::begin(marks_), std::end(marks_), false); }

private:
  std::vector<bool> marks_;
};

template <typename Config>
struct departure_info {
  inline void reset() {
    dep_station_ = invalid<station_id>;
    dep_sti_ = invalid<stop_times_index>;
    Config::reset_traits_aggregate(aggregate_);
  }

  station_id dep_station_ = invalid<station_id>;
  stop_times_index dep_sti_ = invalid<stop_times_index>;
  typename Config::TraitData aggregate_{};
};

template <typename Config>
inline trip_count get_earliest_trip(raptor_timetable const& tt,
                                    raptor_route const& route,
                                    earliest_arrivals const& prev_ea,
                                    stop_times_index const r_stop_offset) {

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

template <typename Config>
inline void init_arrivals(raptor_result& result, earliest_arrivals& prev_ea,
                          raptor_query const& q,
                          raptor_schedule const& raptor_sched,
                          mark_store& station_marks) {

  auto const traits_size = Config::trait_size();
  auto propagate_across_traits = [traits_size](arrivals& arrivals,
                                               station_id stop_id,
                                               motis::time arrival_val) {
    auto const last_arr_idx = (stop_id * traits_size) + traits_size;
    for (int arr_idx = (stop_id * traits_size); arr_idx < last_arr_idx;
         ++arr_idx) {
      arrivals[arr_idx] = std::min(arrival_val, arrivals[arr_idx]);
    }
  };

  propagate_across_traits(result[0], q.source_, q.source_time_begin_);
  prev_ea[q.source_] = q.source_time_begin_;
  station_marks.mark(q.source_);

  for (auto const& f : raptor_sched.initialization_footpaths_[q.source_]) {
    motis::time const arr = q.source_time_begin_ + f.duration_;
    propagate_across_traits(result[0], f.to_, arr);
    prev_ea[f.to_] = std::min(arr, prev_ea[f.to_]);
    station_marks.mark(f.to_);
  }
}

template <typename Config>
inline void update_route(raptor_timetable const& tt, route_id const r_id,
                         arrivals& previous_round, arrivals& current_round,
                         earliest_arrivals& prev_ea, earliest_arrivals& ea,
                         mark_store& station_marks) {

  auto const& route = tt.routes_[r_id];

  trip_count earliest_trip_id = invalid<trip_count>;
  station_id departure_id = invalid<station_id>;
  station_id departure_stop_offset = invalid<station_id>;
  for (station_id r_stop_offset = 0; r_stop_offset < route.stop_count_;
       ++r_stop_offset) {

    if (!valid(earliest_trip_id)) {
      earliest_trip_id =
          get_earliest_trip<Config>(tt, route, prev_ea, r_stop_offset);
      departure_id =
          tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];
      departure_stop_offset = r_stop_offset;
      continue;
    }

    auto const stop_id =
        tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];

    // we need to iterate through all possible trips starting from the
    // earliest
    //  catchable as the criteria from the configuration possibly favor
    //  a later trip because it benefits from other properties than arrival time

    for (trip_count trip_id = earliest_trip_id; trip_id < route.trip_count_;
         ++trip_id) {

      auto const current_stop_time_idx = route.index_to_stop_times_ +
                                         (trip_id * route.stop_count_) +
                                         r_stop_offset;

      auto const departure_stop_time_idx = route.index_to_stop_times_ +
                                           (trip_id * route.stop_count_) +
                                           departure_stop_offset;

      auto const& stop_time = tt.stop_times_[current_stop_time_idx];

      auto [min_arrival_update, traits_satisfied] =
          Config::check_and_update_arrivals_old(
              previous_round, current_round, tt, departure_id, stop_id,
              departure_stop_time_idx, current_stop_time_idx);

      if (valid(min_arrival_update)) {
        station_marks.mark(stop_id);
      }

      if (min_arrival_update < ea[stop_id]) {
        // write the earliest arrival time for this stop after this round
        //  as this is a lower bound for the trip search
        ea[stop_id] = min_arrival_update;
      }

      if (traits_satisfied) {
        break;  // we can skip further looping through the trips on this stop
      }
    }

    // check if we could catch an earlier trip
    // (only relevant for processing the next stop)
    auto const& stop_time =
        tt.stop_times_[route.index_to_stop_times_ +
                       (earliest_trip_id * route.stop_count_) + r_stop_offset];
    auto const previous_k_arrival = prev_ea[stop_id];
    if (previous_k_arrival <= stop_time.departure_) {
      auto old_earliest_trip_id = earliest_trip_id;
      earliest_trip_id =
          get_earliest_trip<Config>(tt, route, prev_ea, r_stop_offset);
      if (old_earliest_trip_id != earliest_trip_id) {
        departure_id =
            tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];
        departure_stop_offset = r_stop_offset;
      }
    }
  }
}

template<typename Config>
inline bool update_stop_if_required(
    typename Config::TraitsData const& trait_data, uint32_t trait_offset,
    arrivals& current_round, mark_store& station_marks, earliest_arrivals& ea,
    stop_times_index const arrival_idx, station_id stop_id,
    stop_time const& current_stop_time) {

  if (Config::is_update_required(trait_data, trait_offset)) {
    if(current_stop_time.arrival_ < current_round[arrival_idx]) {
      current_round[arrival_idx] = current_stop_time.arrival_;
      station_marks.mark(stop_id);

      if (current_stop_time.arrival_ < ea[stop_id]) {
        // write the earliest arrival time for this stop after this round
        //  as this is a lower bound for the trip search
        ea[stop_id] = current_stop_time.arrival_;
      }

      if (Config::is_trait_satisfied(trait_data, trait_offset)) {
        return true;
      }
    }
  }

  return false;
}

template <typename Config>
inline void update_route2(raptor_timetable const& tt, route_id const r_id,
                          arrivals& previous_round, arrivals& current_round,
                          earliest_arrivals& prev_ea, earliest_arrivals& ea,
                          mark_store& station_marks) {

  auto const& route = tt.routes_[r_id];

  auto const trait_size = Config::trait_size();
  uint32_t satisfied_stop_cnt = 0;
  typename Config::TraitsData trait_data{};

  for (trip_count trip_id = 0; trip_id < route.trip_count_; ++trip_id) {

    auto const trip_first_stop_sti =
        route.index_to_stop_times_ + (trip_id * route.stop_count_);

    for (uint32_t trait_offset = 0; trait_offset < trait_size; ++trait_offset) {
      station_id departure_station = invalid<station_id>;

      for (station_id r_stop_offset = 0; r_stop_offset < route.stop_count_;
           ++r_stop_offset) {

        station_id const stop_id =
            tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];

        auto const current_sti = trip_first_stop_sti + r_stop_offset;

        auto const current_stop_time = tt.stop_times_[current_sti];
        auto const arrival_idx = Config::get_arrival_idx(stop_id, trait_offset);

        // iff there is an invalid departure id
        //     => we can skip if there is no arrival known at this stop or if
        //     the trip can't be catched at this stop
        if (!valid(departure_station) &&
            (!valid(previous_round[arrival_idx]) ||
             previous_round[arrival_idx] > current_stop_time.departure_)) {
          continue;
        }

        if(valid(departure_station)) {
          Config::update_traits_aggregate(trait_data, tt, r_id, trip_id,
                                          r_stop_offset, current_sti);

          //even though the current station could soon serve as departure station
          // it may still be that the arrival time improves
          // for connections with the same trait offset but more transfers
          // therefore we also want to store an arrival time for this station
          // before it becomes the new departure station
          update_stop_if_required<Config>(trait_data, trait_offset,
                                          current_round, station_marks,
                                          ea, arrival_idx, stop_id,
                                          current_stop_time);
        }

        // if we could reach this stop in the previous round
        //   and the stop arrival time is earlier than the trip departure time
        //   this stop can serve as new departure stop
        //   as a departure later in the route can't worsen but just improve
        //   the result at the following stations it is preferred to reset
        //   the departure stop in these cases
        if (previous_round[arrival_idx] <= current_stop_time.departure_) {

          departure_station = stop_id;
          Config::reset_traits_aggregate(trait_data);

          // we can't improve the arrival time on the station the trip was
          // boarded
          continue;
        }

        update_stop_if_required<Config>(trait_data, trait_offset, current_round,
                                        station_marks, ea, arrival_idx,
                                        stop_id, current_stop_time);
      }

      Config::reset_traits_aggregate(trait_data);
    }

    if (satisfied_stop_cnt == route.stop_count_ - 1) {
      // we can't reach satisfaction for the first stop
      break;
    }
  }
}

template <typename Config>
inline void update_footpaths(raptor_timetable const& tt,
                             arrivals& current_round,
                             arrivals const& current_round_c,
                             earliest_arrivals& ea, mark_store& station_marks) {

  // How far do we need to skip until the next stop is reached?
  auto const trait_size = Config::trait_size();

  for (station_id stop_id = 0; stop_id < tt.stop_count(); ++stop_id) {

    auto index_into_transfers = tt.stops_[stop_id].index_to_transfers_;
    auto next_index_into_transfers = tt.stops_[stop_id + 1].index_to_transfers_;

    for (auto current_index = index_into_transfers;
         current_index < next_index_into_transfers; ++current_index) {

      auto const& footpath = tt.footpaths_[current_index];

      for (int s_trait_offset = 0; s_trait_offset < trait_size;
           ++s_trait_offset) {
        auto const from_arr_idx =
            Config::get_arrival_idx(stop_id, s_trait_offset);
        auto const to_arr_idx =
            Config::get_arrival_idx(footpath.to_, s_trait_offset);

        // TODO: how to determine domination of certain journeys
        //       and therewith skip certain updates
        //       is this even possible?

        // if (arrivals[round_k][stop_id] == invalid_time) { continue; }
        // if (earliest_arrivals[stop_id] == invalid_time) { continue; }
        if (!valid(current_round_c[from_arr_idx])) {
          continue;
        }

        // there is no triangle inequality in the footpath graph!
        // we cannot use the normal arrival values,
        // but need to use the earliest arrival values as read
        // and write to the normal arrivals,
        // otherwise it is possible that two footpaths
        // are chained together
        motis::time const new_arrival =
            current_round_c[from_arr_idx] + footpath.duration_;

        motis::time to_arrival = current_round[to_arr_idx];

        if (new_arrival < to_arrival) {
          station_marks.mark(footpath.to_);
          current_round[to_arr_idx] = new_arrival;

          if (current_round[to_arr_idx] < ea[footpath.to_]) {
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

#ifdef _DEBUG
  print_query(query);
  //print_stations(raptor_sched);
  //print_route_trip_debug_strings(raptor_sched);
#endif

  // TODO: also check whether one of prev_ea or ea can be eliminated
  earliest_arrivals prev_ea(tt.stop_count(), invalid<motis::time>);
  earliest_arrivals ea(tt.stop_count(), invalid<motis::time>);

  std::vector<motis::time> current_round_arrivals(tt.stop_count() *
                                                  Config::trait_size());

  mark_store station_marks(tt.stop_count());
  mark_store route_marks(tt.route_count());

  init_arrivals<Config>(result, prev_ea, query, raptor_sched, station_marks);

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

      update_route2<Config>(tt, r_id, result[round_k - 1], result[round_k],
                            prev_ea, ea, station_marks);
    }

    route_marks.reset();

    std::memcpy(current_round_arrivals.data(), result[round_k],
                current_round_arrivals.size() * sizeof(motis::time));

    update_footpaths<Config>(tt, result[round_k], current_round_arrivals.data(),
                             ea, station_marks);

    std::memset(prev_ea.data(), invalid<motis::time>,
                prev_ea.size() * sizeof(motis::time));
    std::memcpy(prev_ea.data(), ea.data(),
                prev_ea.size() * sizeof(motis::time));
    std::memset(ea.data(), invalid<motis::time>,
                ea.size() * sizeof(motis::time));
  }

  // print_results<Config>(result, tt);
}

}  // namespace motis::raptor