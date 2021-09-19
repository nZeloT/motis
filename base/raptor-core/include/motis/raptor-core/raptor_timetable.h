#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>
#include <set>

#include "motis/core/schedule/time.h"

namespace motis {
namespace raptor {

using time8 = uint8_t;

using station_id = int32_t;
using route_id = uint32_t;
using footpath_id = int32_t;

using motis_id = int32_t;

using footpath_count = uint16_t;
using route_count = uint16_t;

using trip_id = uint16_t;
using trip_count = uint16_t;
using stop_count = uint16_t;

using stop_offset = uint16_t;

// index types
using stop_times_index = uint32_t;
using route_stops_index = uint32_t;
using stop_routes_index = uint32_t;
using footpaths_index = uint32_t;

// additional attributes
using occupancy = uint8_t;

template<typename T> constexpr T invalid = std::numeric_limits<T>::max();

// overload invalid for station id, 
// since we have 32b and 24b station ids, which must be comparable
template<> constexpr station_id invalid<station_id> = -1;

template <typename T>
constexpr auto valid(T const& value) { return value != invalid<T>; }

using raptor_round = uint8_t;
using transfers = uint8_t;

constexpr raptor_round max_transfers = 6;
constexpr raptor_round max_trips = max_transfers + 1;
constexpr raptor_round max_round_k = max_trips + 1;

using earliest_arrivals = std::vector<motis::time>;
using arrivals = motis::time*;

struct raptor_stop {
  raptor_stop() = delete;
  raptor_stop(footpath_count const fc, route_count const rc,
              footpaths_index const it, stop_routes_index const isr)
      : footpath_count_(fc),
        route_count_(rc),
        index_to_transfers_(it),
        index_to_stop_routes_(isr) {}

  footpath_count footpath_count_;
  route_count route_count_; 
  footpaths_index index_to_transfers_;
  stop_routes_index index_to_stop_routes_;
};

struct raptor_route {
  raptor_route() = delete;
  raptor_route(trip_count const tc, stop_count const sc,
               stop_times_index const sti, route_stops_index const rsi,
               motis::time const st)
      : trip_count_(tc),
        stop_count_(sc),
        index_to_stop_times_(sti),
        index_to_route_stops_(rsi),
        stand_time_(st) {}

  trip_count trip_count_;
  stop_count stop_count_;
  stop_times_index index_to_stop_times_;
  route_stops_index index_to_route_stops_;
  motis::time stand_time_;
};

struct stop_time {
  motis::time arrival_{invalid<decltype(arrival_)>};
  motis::time departure_{0};
};

//holds occupancy of stop inbound edge
struct stop_occupancy {
  occupancy inbound_occupancy_{0};
};

struct raptor_footpath {
  raptor_footpath() = delete;
  raptor_footpath(station_id const to, motis::time const dur)
      : to_(to), duration_(dur) {}
  station_id to_ : 24;
  time8 duration_;
};

struct raptor_incoming_footpath {
  raptor_incoming_footpath() = delete;
  raptor_incoming_footpath(station_id from, motis::time duration)
      : from_(from), duration_(duration) {}
  station_id from_ : 24;
  motis::time duration_ : 8;
};

using partition_id = uint16_t;
struct partitioning {
  partitioning() = default;

  bool empty() const { return partitions_.empty(); }
  partition_id size() const { return partitions_.size(); }

  std::vector<partition_id> route_to_partition_;
  std::vector<std::vector<route_id>> partitions_;
};

using cls_id = partition_id;
using cls_station_id = uint16_t;
using cls_station_id_glb = station_id;

struct raptor_timetable {
  std::vector<raptor_stop> stops_;
  std::vector<raptor_route> routes_;
  std::vector<raptor_footpath> footpaths_;

  std::vector<stop_time> stop_times_;
  std::vector<stop_occupancy> stop_occupancies_;

  std::vector<station_id> route_stops_;
  std::vector<route_id> stop_routes_;

  auto stop_count() const { // subtract the sentinel
    return static_cast<station_id>(stops_.size() - 1); 
  }

  auto route_count() const { // subtract the sentinel
    return static_cast<route_id>(routes_.size() - 1);
  }

  auto footpath_count() const {
    return static_cast<footpath_id>(footpaths_.size());
  }
};



}  // namespace raptor
}  // namespace motis
