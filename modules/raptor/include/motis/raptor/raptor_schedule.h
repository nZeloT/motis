// #include <vector>
// #include <cstdint>

// #include "motis/core/schedule/time.h"

// namespace motis {
// namespace raptor {

// using station_id = int32_t;
// using route_id = uint32_t;
// using footpath_id = int32_t;

// using motis_id = int32_t;

// using footpath_count = uint16_t;
// using route_count = uint16_t;

// using trip_id = uint16_t;
// using trip_count = uint16_t;
// using stop_count = uint16_t;

// using stop_offset = uint8_t;

// // index types
// using stop_times_index = uint32_t;
// using route_stops_index = uint32_t;
// using stop_routes_index = uint32_t;
// using footpaths_index = uint32_t;

// template<typename T> constexpr T invalid = std::numeric_limits<T>::max();

// // overload invalid for station id, 
// // since we have 32b and 24b station ids, which must be comparable
// template<> constexpr station_id invalid<station_id> = -1;

// template <typename T>
// constexpr auto valid(T const value) { return value != invalid<T>; }

// using raptor_round = uint8_t;

// constexpr raptor_round max_transfers = 6;
// constexpr raptor_round max_trips = max_transfers + 1;
// constexpr raptor_round max_round_k = max_trips + 1;
// constexpr raptor_round arrival_array_count = max_round_k + 1; 
// // for GPU footpaths

// using earliest_arrivals = std::vector<motis::time>;
// using arrivals = motis::time*;

// struct raptor_stop {
//   raptor_stop() = delete;
//   raptor_stop(footpath_count const fc, route_count const rc,
//               footpaths_index const it, stop_routes_index const isr)
//       : footpath_count_(fc),
//         route_count_(rc),
//         index_to_transfers_(it),
//         index_to_stop_routes_(isr) {}

//   footpath_count footpath_count_{0};
//   route_count route_count_{0}; 
//   footpaths_index index_to_transfers_{invalid<footpaths_index>};
//   stop_routes_index index_to_stop_routes_{invalid<stop_routes_index>};
// };

// struct raptor_route {
//   raptor_route() = delete;
//   raptor_route(trip_count const tc, stop_count const sc,
//                stop_times_index const sti, route_stops_index const rsi,
//                motis::time const stand_time)
//       : trip_count_(tc),
//         stop_count_(sc),
//         index_to_stop_times_(sti),
//         index_to_route_stops_(rsi) {}

//   trip_count trip_count_{0};
//   stop_count stop_count_{0};
//   stop_times_index index_to_stop_times_{invalid<stop_times_index>};
//   route_stops_index index_to_route_stops_{invalid<route_stops_index>};
//   motis::time stand_time_{invalid<motis::time>};
// };

// struct raptor_footpath {
//   raptor_footpath() = delete;
//   // raptor_footpath()
//   //     : to_(invalid<decltype(to_)>), duration_(invalid<decltype(duration_)>) {}
//   raptor_footpath(station_id const to, motis::time const dur)
//       : to_(to), duration_(dur) {}
//   station_id to_ : 24;
//   motis::time8 duration_;
// };

// struct stop_routes_entry {
//   stop_routes_entry() = delete;
//   stop_routes_entry(route_id const r_id, stop_offset const offset)
//       : r_id_(r_id), offset_(offset) {}
//   route_id r_id_ : 24; 
//   stop_offset offset_; // the stop is the offsets-th stop in the route
// };

// struct stop_time {
//   stop_time() = delete;
//   stop_time(motis::time const a, motis::time const d)
//       : arrival_(a), departure_(d) {}
//   motis::time arrival_;
//   motis::time departure_;
// };

// struct raptor_timetable {
//   std::vector<raptor_stop> stops_;
//   std::vector<raptor_route> routes_;
//   std::vector<raptor_footpath> footpaths_;

//   std::vector<stop_time> stop_times_;
//   std::vector<station_id> route_stops_;
//   std::vector<stop_routes_entry> stop_routes_;

//   // for every real station footpaths to effective stations,
//   // footpaths with .to_ == invalid_station are removed,
//   // so there are no invalid station ids 
//   std::vector<footpaths_index> real_footpaths_indices_;
//   std::vector<raptor_footpath> real_footpaths_;

//   auto stop_count() const { // subtract the sentinel
//     return static_cast<station_id>(stops_.size() - 1); 
//   }

//   auto route_count() const { // subtract the sentinel
//     return static_cast<route_id>(routes_.size() - 1);
//   }

//   auto footpath_count() const {
//     return static_cast<footpath_id>(footpaths_.size());
//   }
// };


// } // namespace raptor
// } // namespace motis