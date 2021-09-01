#pragma once

#include "motis/core/schedule/connection.h"

#include "motis/raptor-core/raptor_timetable.h"

namespace motis {
namespace raptor {


struct raptor_schedule {
  raptor_schedule() = default;
  raptor_schedule(raptor_schedule const&) = delete;

  raptor_timetable timetable_;

  std::unordered_map<std::string, station_id> eva_to_raptor_id_;
  std::vector<std::string> raptor_id_to_eva_;
  std::vector<unsigned> station_id_to_index_;
  std::vector<motis::time> transfer_times_;

  // for every station the departure events of the station
  // and the stations reachable by foot
  std::vector<std::vector<motis::time>> departure_events_;

  // uses the same indexing scheme as the stop times vector in the timetable,
  // but the first entry for every trip is a nullptr!
  // since #stop_times(trip) = #lcons(trip) + 1
  std::vector<light_connection const*> lcon_ptr_;

  // duration of the footpaths INCLUDE transfer time from the departure station
  std::vector<std::vector<raptor_footpath>> initialization_footpaths_;

  // duration REDUCED by the transfer times from the departure station
  std::vector<std::vector<raptor_incoming_footpath>> incoming_footpaths_;

  partitioning partitioning_;
  cls_station_id_glb total_border_stations_;
  std::vector<std::vector<station_id>> cluster_border_;
  std::vector<std::vector<station_id>> cluster_inland_;
};

}
}
