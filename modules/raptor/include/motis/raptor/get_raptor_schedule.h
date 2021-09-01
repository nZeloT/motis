#pragma once

#include <unordered_set>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/footpath.h"

#include "motis/raptor-core/raptor_timetable.h"
#include "motis/raptor-core/raptor_schedule.h"

namespace motis {
namespace raptor {

struct transformable_footpath {
  transformable_footpath(station_id const from, station_id const to,
                         motis::time const duration)
      : from_(from), to_(to), duration_(duration) {}

  transformable_footpath(motis::footpath const& f)
      : from_(f.from_station_), to_(f.to_station_), duration_(f.duration_) {}
  station_id from_;
  station_id to_;
  motis::time duration_;
};

struct transformable_stop {
  std::vector<transformable_footpath> footpaths_;
  std::vector<transformable_footpath> incoming_footpaths_;
  std::vector<route_id> stop_routes_;
  motis::time transfer_time_;
  std::string eva_;
  unsigned motis_station_index_;
  std::unordered_set<cls_station_id_glb> cluster_ids_;
};

struct raptor_lcon {
  raptor_lcon(station_id const from, station_id const to, motis::time const dep,
              motis::time const arrival, bool const in_allowed,
              bool const out_allowed, light_connection const* lc)
      : from_(from),
        to_(to),
        departure_(dep),
        arrival_(arrival),
        in_allowed_(in_allowed),
        out_allowed_(out_allowed),
        lcon_(lc) {}
  station_id from_;
  station_id to_;
  motis::time departure_;
  motis::time arrival_;
  bool in_allowed_;
  bool out_allowed_;
  light_connection const* lcon_;
};

struct transformable_trip {
  std::vector<raptor_lcon> lcons_;
  std::vector<stop_time> stop_times_;
};

struct transformable_route {
  std::vector<transformable_trip> trips_;
  std::vector<station_id> route_stops_;
  motis::time stand_time_;
  partition_id partition_id_;
};

struct transformable_timetable {
  std::vector<transformable_stop> stations_;
  std::vector<transformable_route> routes_;

  partitioning partitioning_;
  cls_station_id_glb total_border_stations_;
  std::vector<std::vector<station_id>> cluster_border_;
  std::vector<std::vector<station_id>> cluster_inland_;
};

std::unique_ptr<raptor_schedule> get_raptor_schedule(schedule const& sched, bool const, std::string const&);

} // namespace raptor
} // namespace motis