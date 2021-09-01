#pragma once 

#include <algorithm>
#include <vector>

#include "motis/raptor-core/raptor_timetable.h"

namespace motis {

using namespace raptor;

#define SMALL_TIME true

#if SMALL_TIME
  using global_mem_time = motis::time;
#else
  using global_mem_time = motis::time32;
#endif

using arrival_ptrs = std::array<global_mem_time*, max_round_k>;

using gpu_route = raptor_route;
using gpu_stop = raptor_stop;
using gpu_stop_time = stop_time;

struct gpu_footpath {
  gpu_footpath() : from_(invalid<decltype(from_)>), 
                   to_(invalid<decltype(to_)>),
                   duration_(invalid<decltype(duration_)>) {}

  gpu_footpath(station_id const from, 
               station_id const to, 
               motis::time const dur)
    : from_(from), to_(to), duration_(dur) {}

  station_id from_;
  station_id to_ : 24;
  time8 duration_;
};

struct cluster {
  cls_station_id border_station_count_;
  cls_station_id inland_station_count_;

  // arrival offset where the inland stations start
  // so arrivals_start_index_ + inland_station_count_  arrival entries
  // are exclusive to the cluster
  cls_station_id_glb arrivals_start_index_;

  cls_station_id_glb border_mapping_index_;

  route_id route_start_index_;
  route_id route_count_;
};

struct host_gpu_timetable {
  host_gpu_timetable() = default;

  // subtract the sentinel 
  auto stop_count() const { return stops_.size() - 1; }
  auto route_count() const { return routes_.size() - 1; }

  std::vector<gpu_stop> stops_;
  std::vector<gpu_route> routes_;
  std::vector<gpu_footpath> footpaths_;

  std::vector<gpu_stop_time> stop_times_;

  std::vector<station_id> route_stops_;
  std::vector<route_id> stop_routes_;

  std::vector<motis::time> stop_departures_;
  std::vector<motis::time> stop_arrivals_;

  std::vector<global_mem_time> transfer_times_;

  std::vector<footpaths_index> initialization_footpaths_indices_;
  std::vector<raptor_footpath> initialization_footpaths_;

  std::vector<cluster> clusters_;
  std::vector<cls_station_id> clustered_route_stops_;
  std::vector<station_id> border_mappings_;
};

struct device_gpu_timetable {
  gpu_stop* stops_;
  gpu_route* routes_;

  gpu_footpath* footpaths_;

  global_mem_time* transfer_times_;

  gpu_stop_time* stop_times_;

  motis::time* stop_arrivals_;
  motis::time* stop_departures_;

  station_id* route_stops_;

  route_id* stop_routes_;

  station_id stop_count_;
  route_id route_count_;
  footpath_id footpath_count_;

  footpaths_index* initialization_footpaths_indices_;
  raptor_footpath* initialization_footpaths_;

  cluster* clusters_;
  cls_id cluster_count_;
  cls_station_id* clustered_route_stops_;
  station_id* border_mappings_;
};



} // namespace motis