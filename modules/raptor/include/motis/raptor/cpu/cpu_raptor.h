#pragma once

#include <motis/raptor-core/raptor_query.h>
namespace motis {
namespace raptor {

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
  std::memcpy(arrivals[round_k].data(), 
              arrivals[round_k - 1].data(), 
              arrivals[round_k].size() * sizeof(motis::time));
}

template<typename Config>
inline trip_count get_earliest_trip(raptor_timetable const& tt,
                                    raptor_route const& route,
                                    arrivals const& prev_arrivals,
                                    stop_times_index const r_stop_offset) {

  station_id const stop_id = tt.route_stops_[route.index_to_route_stops_
                                            + r_stop_offset];
  
  // station was never visited, there can't be a earliest trip
  if (!valid(prev_arrivals[stop_id])) { return invalid<trip_count>; }

  // get first defined earliest trip for the stop in the route
  auto const first_trip_stop_idx = route.index_to_stop_times_ + r_stop_offset;
  auto const last_trip_stop_idx = first_trip_stop_idx
                                + ((route.trip_count_ - 1) * route.stop_count_);
                              
  trip_count current_trip = 0;
  for (auto stop_time_idx = first_trip_stop_idx; 
            stop_time_idx <= last_trip_stop_idx;
            stop_time_idx += route.stop_count_) {

    auto const stop_time = tt.stop_times_[stop_time_idx];
    if (prev_arrivals[stop_id] <= stop_time.departure_) {
      return current_trip;
    }

    ++current_trip;
  }

  return invalid<trip_count>;
}

template<typename Config>
inline void init_arrivals(raptor_result& result,
                          raptor_query const& q, 
                          raptor_schedule const& raptor_sched,
                          mark_store& station_marks) {

  result[0][q.source_] = q.source_time_begin_;
  station_marks.mark(q.source_);

  for (auto const& f : raptor_sched.initialization_footpaths_[q.source_]) {
    result[0][f.to_] = q.source_time_begin_
                     + f.duration_;
    station_marks.mark(f.to_);
  }
}

template<typename Config>
inline void update_route(raptor_timetable const& tt,
                         route_id const r_id,
                         arrivals const& prev_arrivals,
                         arrivals& current_round,
                         earliest_arrivals& ea,
                         mark_store& station_marks) {
  auto const& route = tt.routes_[r_id];

  trip_count earliest_trip_id = invalid<trip_count>;
  for (station_id r_stop_offset = 0; 
                  r_stop_offset < route.stop_count_; 
                ++r_stop_offset) {

    if (!valid(earliest_trip_id)) {
      earliest_trip_id =
          get_earliest_trip<Config>(tt, route, prev_arrivals, r_stop_offset);
      continue;
    }

    auto const stop_id = tt.route_stops_[route.index_to_route_stops_
                                                + r_stop_offset];
    auto const current_stop_time_idx = route.index_to_stop_times_ 
                                     + (earliest_trip_id * route.stop_count_) 
                                     + r_stop_offset;

    auto const& stop_time = tt.stop_times_[current_stop_time_idx];

    // need the minimum due to footpaths updating arrivals 
    // and not earliest arrivals
    auto const min_stop_arrival = std::min(current_round[stop_id], ea[stop_id]);

    if (stop_time.arrival_ < min_stop_arrival) {
      station_marks.mark(stop_id);
      current_round[stop_id] = stop_time.arrival_;
      ea[stop_id] = stop_time.arrival_;
    }

    // check if we could catch an earlier trip
    auto const previous_k_arrival = prev_arrivals[stop_id];
    if (previous_k_arrival <= stop_time.departure_) {
      earliest_trip_id =
          get_earliest_trip<Config>(tt, route, prev_arrivals, r_stop_offset);
    }
  }
}

template<typename Config>
inline void update_footpaths(raptor_timetable const& tt,
                             arrivals& current_round,
                             earliest_arrivals const& ea,
                             mark_store& station_marks) {

  for (station_id stop_id = 0; stop_id < tt.stop_count(); ++stop_id) {

    auto index_into_transfers = tt.stops_[stop_id].index_to_transfers_;
    auto next_index_into_transfers = tt.stops_[stop_id + 1].index_to_transfers_;

    for (auto current_index = index_into_transfers; 
              current_index < next_index_into_transfers; 
            ++current_index) {

      auto const& footpath = tt.footpaths_[current_index];

      // if (arrivals[round_k][stop_id] == invalid_time) { continue; }
      // if (earliest_arrivals[stop_id] == invalid_time) { continue; }
      if (!valid(ea[stop_id])) { continue; }

      // there is no triangle inequality in the footpath graph!
      // we cannot use the normal arrival values, 
      // but need to use the earliest arrival values as read
      // and write to the normal arrivals, 
      // otherwise it is possible that two footpaths
      // are chained together
      motis::time const new_arrival = ea[stop_id] + footpath.duration_;

      motis::time to_earliest_arrival = ea[footpath.to_];
      motis::time to_arrival = current_round[footpath.to_];

      auto const min = std::min(to_arrival, to_earliest_arrival);
      if (new_arrival < min) {
        station_marks.mark(footpath.to_);
        current_round[footpath.to_] = new_arrival;
      }
    }
  }

}

template<typename Config>
inline void invoke_cpu_raptor(const raptor_query& query, raptor_statistics&,
                              const raptor_schedule& raptor_sched) {
  auto const& tt = raptor_sched.timetable_;

  auto& result = *query.result_;
  earliest_arrivals ea(tt.stop_count(), invalid<motis::time>);

  mark_store station_marks(tt.stop_count());
  mark_store route_marks(tt.route_count());

  init_arrivals<Config>(result, query, raptor_sched, station_marks);

  for (raptor_round round_k = 1; round_k < max_round_k; ++round_k) {
    bool any_marked = false;

    for (auto s_id = 0; s_id < tt.stop_count(); ++s_id) {
      if (!station_marks.marked(s_id)) { continue; }
      if (!any_marked) any_marked = true;
      auto const& stop = tt.stops_[s_id];
      for (auto sri = stop.index_to_stop_routes_;
                sri < stop.index_to_stop_routes_ + stop.route_count_;
              ++sri) {
        route_marks.mark(tt.stop_routes_[sri]);
      }
    }
    if (!any_marked) { break; }

    station_marks.reset();
    
    for (route_id r_id = 0; r_id < tt.route_count(); ++r_id) {
      if (!route_marks.marked(r_id)) { continue; }

      update_route<Config>(tt, r_id,
                   result[round_k - 1], 
                   result[round_k], 
                   ea,
                   station_marks);
    }

    route_marks.reset();

    update_footpaths<Config>(tt, result[round_k], ea, station_marks);
  }
}

}  // namespace raptor
}  // namespace motis