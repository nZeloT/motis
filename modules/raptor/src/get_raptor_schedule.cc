#include "motis/raptor/get_raptor_schedule.h"

#include <fstream>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include "motis/core/common/logging.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/raptor/raptor_util.h"

#include "motis/raptor/debug_utils.h"
#include "motis/raptor/route_conflict_graph.h"

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"

namespace motis::raptor {

using namespace motis::logging;

std::vector<stop_time> get_stop_times_from_lcons(
    std::vector<raptor_lcon> const& lcons) {
  std::vector<stop_time> stop_times(lcons.size() + 1);
  for (auto idx = 0; idx < lcons.size(); ++idx) {
    auto const& lcon = lcons[idx];
    if (lcon.in_allowed_) {
      stop_times[idx].departure_ = lcon.departure_;
    }
    if (lcon.out_allowed_) {
      stop_times[idx + 1].arrival_ = lcon.arrival_;
    }
  }
  return stop_times;
}

std::vector<stop_occupancy> get_stop_occupancies_from_lcons(
    std::vector<raptor_lcon> const& lcons) {
  auto const edge_count = lcons.size();
  auto const stop_count = edge_count + 1;

  std::vector<stop_occupancy> stop_occupancies(stop_count);

  // first stop has no inbound edge; therefore also no occupancy
  stop_occupancies[0].inbound_occupancy_ = 0;

  for (auto idx = 1; idx < stop_count; ++idx) {
    stop_occupancies[idx].inbound_occupancy_ = lcons[idx - 1].lcon_->occupancy_;
  }

  return stop_occupancies;
}

std::vector<station_id> get_route_stops_from_lcons(
    std::vector<raptor_lcon> const& lcons) {
  std::vector<station_id> route_stops;
  route_stops.reserve(lcons.size() + 1);
  for (auto const& lcon : lcons) {
    route_stops.push_back(lcon.from_);
  }
  route_stops.push_back(lcons.back().to_);
  return route_stops;
}

motis::time get_stand_time(transformable_route const& route) {
  motis::time route_stand_time = invalid<motis::time>;
  for (auto const& trip : route.trips_) {
    for (auto const& st : trip.stop_times_) {
      auto const stand_time = st.departure_ - st.arrival_;
      if (!valid(route_stand_time)) {
        route_stand_time = stand_time;
        continue;
      }

      if (stand_time != route_stand_time) {
        return invalid<motis::time>;
      }
    }
  }
  return invalid<motis::time>;
  return route_stand_time;
}

void init_stops(schedule const& sched, transformable_timetable& ttt) {
  ttt.stations_.resize(sched.stations_.size());

  for (auto s_id = 0; s_id < sched.stations_.size(); ++s_id) {
    auto& s = ttt.stations_[s_id];
    auto const& s_ptr = sched.stations_[s_id];

    s.motis_station_index_ = s_ptr->index_;
    s.transfer_time_ = s_ptr->transfer_time_;
    s.eva_ = s_ptr->eva_nr_;
  }
}

void init_stop_routes(transformable_timetable& ttt) {
  for (auto r_id = 0; r_id < ttt.routes_.size(); ++r_id) {
    auto const& route = ttt.routes_[r_id];
    auto const& trip = route.trips_.front();

    for (auto const& lcon : trip.lcons_) {
      ttt.stations_[lcon.from_].stop_routes_.push_back(r_id);
    }

    ttt.stations_[trip.lcons_.back().to_].stop_routes_.push_back(r_id);
  }

  for (auto& station : ttt.stations_) {
    sort_and_unique(station.stop_routes_);
  }
}

void init_routes(schedule const& sched, transformable_timetable& ttt) {
  using namespace motis::access;

  ttt.routes_.resize(sched.expanded_trips_.index_size() - 1);

  route_id r_id = 0;
  for (auto const& route_trips : sched.expanded_trips_) {
    auto& t_route = ttt.routes_[r_id];

    t_route.trips_.resize(route_trips.size());

    auto const& first_trip = route_trips[0];
    auto const in_allowed =
        utl::to_vec(stops(first_trip), [](trip_stop const& ts) {
          return ts.get_route_node()->is_in_allowed();
        });

    auto const out_allowed =
        utl::to_vec(stops(first_trip), [](trip_stop const& ts) {
          return ts.get_route_node()->is_out_allowed();
        });

    trip_id t_id = 0;
    for (auto const& trip : route_trips) {
      auto& t_trip = t_route.trips_[t_id];

      for (auto const& section : sections(trip)) {
        auto const& lc = section.lcon();
        auto const from = section.from_station_id();
        auto const to = section.to_station_id();
        auto const from_in_allowed = in_allowed[section.index()];
        auto const to_out_allowed = out_allowed[section.index() + 1];

        t_trip.lcons_.emplace_back(from, to, lc.d_time_, lc.a_time_,
                                   from_in_allowed, to_out_allowed, &lc);
      }

      t_trip.stop_times_ = get_stop_times_from_lcons(t_trip.lcons_);
      t_trip.stop_occupancies_ = get_stop_occupancies_from_lcons(t_trip.lcons_);
      t_trip.dbg_ = std::string{trip->dbg_.str()};
      ++t_id;
    }

    t_route.route_stops_ =
        get_route_stops_from_lcons(t_route.trips_.front().lcons_);
    t_route.stand_time_ = get_stand_time(t_route);

    ++r_id;
  }
}

void add_footpaths(schedule const& sched, transformable_timetable& ttt) {
  for (auto s_id = 0; s_id < ttt.stations_.size(); ++s_id) {
    auto const& motis_station = sched.stations_[s_id];
    auto& raptor_station = ttt.stations_[s_id];

    std::copy(std::begin(motis_station->outgoing_footpaths_),
              std::end(motis_station->outgoing_footpaths_),
              std::back_inserter(raptor_station.footpaths_));

    std::copy(std::begin(motis_station->incoming_footpaths_),
              std::end(motis_station->incoming_footpaths_),
              std::back_inserter(raptor_station.incoming_footpaths_));
  }
}

std::tuple<raptor_timetable, schedule_debug> create_raptor_timetable(
    transformable_timetable const& ttt) {
  raptor_timetable tt;
  schedule_debug dbg{};

  tt.stops_.reserve(ttt.stations_.size() + 1);

  for (auto s_id = 0u; s_id < ttt.stations_.size(); ++s_id) {
    auto const& t_stop = ttt.stations_[s_id];

    auto footpaths_idx = static_cast<footpaths_index>(tt.footpaths_.size());
    auto sr_idx = static_cast<stop_routes_index>(tt.stop_routes_.size());
    auto fc = static_cast<footpath_count>(t_stop.footpaths_.size());
    auto rc = static_cast<route_count>(t_stop.stop_routes_.size());
    tt.stops_.emplace_back(fc, rc, footpaths_idx, sr_idx);

    for_each(t_stop.footpaths_, [&tt](auto const& f) {
      tt.footpaths_.emplace_back(f.to_, f.duration_);
    });

    append_vector(tt.stop_routes_, t_stop.stop_routes_);
  }

  auto footpaths_idx = static_cast<footpaths_index>(tt.footpaths_.size());
  auto sr_idx = static_cast<stop_routes_index>(tt.stop_routes_.size());
  tt.stops_.emplace_back(0, 0, footpaths_idx, sr_idx);

  tt.routes_.reserve(ttt.routes_.size() + 1);
  for (route_id r_id = 0; r_id < ttt.routes_.size(); ++r_id) {
    auto const& t_route = ttt.routes_[r_id];

    auto sc = static_cast<stop_count>(t_route.route_stops_.size());
    auto tc = static_cast<trip_count>(t_route.trips_.size());
    auto stop_times_idx = static_cast<stop_times_index>(tt.stop_times_.size());
    auto rs_idx = static_cast<route_stops_index>(tt.route_stops_.size());
    auto rt_dbg_idx = static_cast<route_debug_index>(
        dbg.raptor_route_trip_to_trip_debug_.size());

    std::for_each(std::begin(t_route.trips_), std::end(t_route.trips_),
                  [&dbg, r_id, rt_dbg_idx](transformable_trip const& el) {
                    trip_id t_id = dbg.raptor_route_trip_to_trip_debug_.size() -
                                   rt_dbg_idx;
                    dbg.insert_dbg(el.dbg_, r_id, t_id);
                  });

    tt.routes_.emplace_back(tc, sc, stop_times_idx, rs_idx, t_route.stand_time_,
                            rt_dbg_idx);

    for (auto const& trip : t_route.trips_) {
      append_vector(tt.stop_times_, trip.stop_times_);
      append_vector(tt.stop_occupancies_, trip.stop_occupancies_);
    }
    append_vector(tt.route_stops_, t_route.route_stops_);
  }

  auto stop_times_idx = static_cast<stop_times_index>(tt.stop_times_.size());
  auto rs_idx = static_cast<route_stops_index>(tt.route_stops_.size());

  tt.routes_.emplace_back(0, 0, stop_times_idx, rs_idx, invalid<motis::time>,
                          0);

  return std::make_tuple(tt, dbg);
}

partitioning get_partitioning(std::string const& filepath) {
  partitioning p;

  std::ifstream partitioning_file(filepath);
  std::string line;

  route_id r_id = 0;
  while (partitioning_file.peek() != EOF && !partitioning_file.eof()) {
    std::getline(partitioning_file, line);
    partition_id p_id = std::stoi(line);

    if (p_id >= p.partitions_.size()) {
      p.partitions_.resize(p_id + 1);
    }

    p.partitions_[p_id].push_back(r_id);
    p.route_to_partition_.push_back(p_id);

    ++r_id;
  }

  partitioning_file.close();

  LOG(logging::info) << "Read partitioning with " << p.size() << " partitions";

  return p;
}

void sort_partitioning(partitioning& p, transformable_timetable const& ttt) {
  auto ascending_routes = [&](route_id const r1, route_id const r2) -> bool {
    auto const d1 = ttt.routes_[r1].trips_.front().lcons_.front().departure_;
    auto const d2 = ttt.routes_[r2].trips_.front().lcons_.front().departure_;
    return d1 < d2;
  };

  for (auto& partition : p.partitions_) {
    std::sort(std::begin(partition), std::end(partition), ascending_routes);
  }
}

void introduce_clustering(transformable_timetable& ttt, std::string const& fp) {
  ttt.partitioning_ = get_partitioning(fp);
  sort_partitioning(ttt.partitioning_, ttt);

  std::vector<std::set<cls_id>> station_to_clusters(ttt.stations_.size());

  auto gather_clusters = [&](size_t const r_id) {
    auto const partition_id = ttt.partitioning_.route_to_partition_[r_id];
    for (auto const route_stop : ttt.routes_[r_id].route_stops_) {
      station_to_clusters[route_stop].insert(partition_id);
      ttt.stations_[route_stop].cluster_ids_.insert(partition_id);
    }
  };

  for (auto r_id = 0U; r_id < ttt.partitioning_.route_to_partition_.size();
       ++r_id) {
    gather_clusters(r_id);
  }

  ttt.cluster_border_.resize(ttt.partitioning_.size());
  ttt.cluster_inland_.resize(ttt.partitioning_.size());

  int total_clusterless_stations = 0;
  int total_inland_stations = 0;
  int total_border_stations = 0;
  for (auto s_id = 0U; s_id < station_to_clusters.size(); ++s_id) {
    auto const& station_clusters = station_to_clusters[s_id];

    // Station is in no cluster
    if (station_clusters.empty()) {
      ++total_clusterless_stations;
      continue;
    }

    // Inland station, only a single cluster
    if (station_clusters.size() == 1) {
      ttt.cluster_inland_[*station_clusters.begin()].push_back(s_id);
      ++total_inland_stations;
    } else {  // Border station, multiple clusters
      for (auto const c_id : station_clusters) {
        ttt.cluster_border_[c_id].push_back(s_id);
      }
      ++total_border_stations;
    }
  }

  ttt.total_border_stations_ = total_border_stations;

  LOG(logging::info) << "Clustering had " << total_border_stations
                     << " border stations";
  LOG(logging::info) << "Clustering had " << total_inland_stations
                     << " inland stations";
  LOG(logging::info) << "Stations without cluster: "
                     << total_clusterless_stations;
  LOG(logging::info) << "Border + Inland + Clusterless stations: "
                     << total_border_stations + total_inland_stations +
                            total_clusterless_stations;

  size_t largest_cluster = 0;
  for (auto const& inland : ttt.cluster_inland_) {
    largest_cluster = std::max(largest_cluster, inland.size());
  }

  LOG(logging::info) << "Largest cluster w.r.t inland stations: "
                     << largest_cluster;

  // Create mapping from station id to global cluster station id
  // for the new memory layout for the arrivals:
  // [Borderstations | InlandStations C1 | C2 | .. | Stations w/o cluster]
  std::vector<cls_station_id_glb> station_id_to_cls_station_id(
      ttt.stations_.size(), invalid<cls_station_id_glb>);

  std::vector<size_t> permutation;
  permutation.reserve(ttt.stations_.size());

  cls_station_id_glb next_cls_station = 0;
  for (auto const& border : ttt.cluster_border_) {
    for (auto const s_id : border) {
      if (!valid(station_id_to_cls_station_id[s_id])) {
        station_id_to_cls_station_id[s_id] = next_cls_station;
        permutation.push_back(s_id);
        ++next_cls_station;
      }
    }
  }

  for (auto const& inland : ttt.cluster_inland_) {
    for (auto const s_id : inland) {
      station_id_to_cls_station_id[s_id] = next_cls_station;
      permutation.push_back(s_id);
      ++next_cls_station;
    }
  }

  // The clustering does not have every s_id, since it does not care
  // about footpaths and does not contain stations without routes
  // So here give every station not touched by the clustering a
  // valid cluster station id
  for (auto s_id = 0; s_id < station_id_to_cls_station_id.size(); ++s_id) {
    if (!valid(station_id_to_cls_station_id[s_id])) {
      station_id_to_cls_station_id[s_id] = next_cls_station;
      permutation.push_back(s_id);
      ++next_cls_station;
    }
  }

  // Now we have the new station ids,
  // we have to change them on the whole schedule

  // Start with changing the cluster border / inland stations
  for (auto& border : ttt.cluster_border_) {
    for (auto& s_id : border) {
      s_id = station_id_to_cls_station_id[s_id];
    }
  }

  for (auto& inland : ttt.cluster_inland_) {
    for (auto& s_id : inland) {
      s_id = station_id_to_cls_station_id[s_id];
    }
  }

  decltype(ttt.stations_) new_stations;
  new_stations.reserve(ttt.stations_.size());
  for (auto s_id = 0; s_id < ttt.stations_.size(); ++s_id) {
    new_stations.push_back(ttt.stations_[permutation[s_id]]);
    // new_stations.push_back(std::move(ttt.stations_[permutation[s_id]]));
  }
  ttt.stations_ = new_stations;

  // Replace station ids in footpaths
  for (auto& station : ttt.stations_) {
    for (auto& fp : station.footpaths_) {
      fp.from_ = station_id_to_cls_station_id[fp.from_];
      fp.to_ = station_id_to_cls_station_id[fp.to_];
    }

    for (auto& inc_fp : station.incoming_footpaths_) {
      inc_fp.from_ = station_id_to_cls_station_id[inc_fp.from_];
      inc_fp.to_ = station_id_to_cls_station_id[inc_fp.to_];
    }
  }

  // Replace station ids in routes
  for (auto& route : ttt.routes_) {
    for (auto& rs : route.route_stops_) {
      rs = station_id_to_cls_station_id[rs];
    }

    for (auto& trip : route.trips_) {
      for (auto& lcon : trip.lcons_) {
        lcon.from_ = station_id_to_cls_station_id[lcon.from_];
        lcon.to_ = station_id_to_cls_station_id[lcon.to_];
      }
    }
  }

  // Generate new route ids from the clustering
  std::vector<route_id> route_id_mapping(ttt.routes_.size());
  std::vector<route_id> route_permutation;
  route_permutation.reserve(ttt.routes_.size());

  route_id new_route_id = 0;
  for (auto& partition : ttt.partitioning_.partitions_) {
    for (auto& r_id : partition) {
      route_id_mapping[r_id] = new_route_id;
      route_permutation.push_back(r_id);
      r_id = new_route_id;
      ++new_route_id;
    }
  }

  // Adjust the partitioning for the new route ids
  auto route_to_partition_copy = ttt.partitioning_.route_to_partition_;
  for (auto r_id = 0u; r_id < ttt.partitioning_.route_to_partition_.size();
       ++r_id) {
    auto const new_route_id = route_id_mapping[r_id];
    ttt.partitioning_.route_to_partition_[new_route_id] =
        route_to_partition_copy[r_id];
  }

  decltype(ttt.routes_) new_routes;
  new_routes.reserve(ttt.routes_.size());
  for (auto r_id = 0; r_id < ttt.routes_.size(); ++r_id) {
    // new_routes.push_back(std::move(ttt.routes_[route_permutation[r_id]]));
    new_routes.push_back(ttt.routes_[route_permutation[r_id]]);
  }
  ttt.routes_ = new_routes;

  // Replace route ids in stations
  for (auto& station : ttt.stations_) {
    for (auto& sr : station.stop_routes_) {
      sr = route_id_mapping[sr];
    }
  }
}

auto get_station_departure_events(transformable_timetable const& ttt,
                                  station_id const s_id) {
  std::vector<motis::time> dep_events;

  auto const& station = ttt.stations_[s_id];
  for (auto const r_id : station.stop_routes_) {
    auto const& route = ttt.routes_[r_id];

    for (stop_offset offset = 0; offset < route.route_stops_.size() - 1;
         ++offset) {
      if (route.route_stops_[offset] != s_id) {
        continue;
      }

      for (auto const& trip : route.trips_) {
        if (!trip.lcons_[offset].in_allowed_) {
          continue;
        }
        dep_events.push_back(trip.lcons_[offset].departure_);
      }
    }
  }

  return dep_events;
}

auto get_initialization_footpaths(transformable_timetable const& ttt) {
  std::vector<std::vector<raptor_footpath>> init_footpaths(
      ttt.stations_.size());

  for (auto const& s : ttt.stations_) {
    for (auto const& f : s.footpaths_) {
      init_footpaths[f.from_].emplace_back(f.to_, f.duration_);
    }
  }

  return init_footpaths;
}

std::unique_ptr<raptor_schedule> transformable_to_schedule(
    transformable_timetable& ttt) {
  auto raptor_sched = std::make_unique<raptor_schedule>();

  // generate initialization footpaths BEFORE removing empty stations
  raptor_sched->initialization_footpaths_ = get_initialization_footpaths(ttt);

  raptor_sched->transfer_times_.reserve(ttt.stations_.size());
  raptor_sched->raptor_id_to_eva_.reserve(ttt.stations_.size());
  raptor_sched->station_id_to_index_.reserve(ttt.stations_.size());

  raptor_sched->incoming_footpaths_.resize(ttt.stations_.size());
  raptor_sched->departure_events_.resize(ttt.stations_.size());

  // Loop over the stations
  for (auto s_id = 0; s_id < ttt.stations_.size(); ++s_id) {
    auto const& s = ttt.stations_[s_id];

    raptor_sched->station_id_to_index_.push_back(s.motis_station_index_);
    raptor_sched->transfer_times_.push_back(s.transfer_time_);
    raptor_sched->raptor_id_to_eva_.push_back(s.eva_);
    raptor_sched->eva_to_raptor_id_.emplace(
        s.eva_, raptor_sched->eva_to_raptor_id_.size());

    // set incoming footpaths
    for (auto const& inc_f : s.incoming_footpaths_) {
      auto const tt = ttt.stations_[inc_f.from_].transfer_time_;
      raptor_sched->incoming_footpaths_[s_id].emplace_back(
          inc_f.from_, inc_f.duration_ - tt);
    }

    // set departure events
    raptor_sched->departure_events_[s_id] =
        get_station_departure_events(ttt, s_id);

    // gather all departure events from stations reachable by foot
    for (auto const& f : ttt.stations_[s_id].footpaths_) {
      for_each(
          get_station_departure_events(ttt, f.to_), [&](auto const& dep_event) {
            raptor_sched->departure_events_[s_id].emplace_back(dep_event -
                                                               f.duration_);
          });
    }

    sort_and_unique(raptor_sched->departure_events_[s_id]);
  }

  // Loop over the routes
  for (auto const& r : ttt.routes_) {
    for (auto const& t : r.trips_) {
      raptor_sched->lcon_ptr_.push_back(nullptr);
      for (auto const& rlc : t.lcons_) {
        raptor_sched->lcon_ptr_.push_back(rlc.lcon_);
      }
    }
  }

  auto [raptor_tt, sched_dbg] = create_raptor_timetable(ttt);

  raptor_sched->timetable_ = std::move(raptor_tt);
  raptor_sched->dbg_ = std::move(sched_dbg);

  // preadd the transfer times
  for (auto const& route : raptor_sched->timetable_.routes_) {
    for (auto trip = 0; trip < route.trip_count_; ++trip) {
      for (auto offset = 0; offset < route.stop_count_; ++offset) {
        auto const rsi = route.index_to_route_stops_ + offset;
        auto const sti =
            route.index_to_stop_times_ + (trip * route.stop_count_) + offset;

        auto const s_id = raptor_sched->timetable_.route_stops_[rsi];
        auto const tt = raptor_sched->transfer_times_[s_id];
        auto& arrival = raptor_sched->timetable_.stop_times_[sti].arrival_;
        if (valid(arrival)) {
          arrival += tt;
        }
      }
    }
  }

  // remove transfer times from the footpaths
  for (auto s_id = 0; s_id < raptor_sched->timetable_.stop_count(); ++s_id) {
    auto const& station = raptor_sched->timetable_.stops_[s_id];
    for (auto f_idx = station.index_to_transfers_;
         f_idx < station.index_to_transfers_ + station.footpath_count_;
         ++f_idx) {
      auto& footpath = raptor_sched->timetable_.footpaths_[f_idx];
      auto const tt = raptor_sched->transfer_times_[s_id];
      footpath.duration_ -= tt;
    }
  }

  raptor_sched->cluster_border_ = ttt.cluster_border_;
  raptor_sched->cluster_inland_ = ttt.cluster_inland_;
  raptor_sched->total_border_stations_ = ttt.total_border_stations_;
  raptor_sched->partitioning_ = ttt.partitioning_;

  LOG(info) << "RAPTOR Stations: " << raptor_sched->timetable_.stop_count();
  LOG(info) << "RAPTOR Routes: " << raptor_sched->timetable_.route_count();
  LOG(info) << "RAPTOR Footpaths: "
            << raptor_sched->timetable_.footpath_count();

  return raptor_sched;
}

std::unique_ptr<raptor_schedule> get_raptor_schedule(
    schedule const& sched, bool const write_conflict_graph,
    std::string const& partitioning_path) {
  scoped_timer timer("building RAPTOR timetable");

  transformable_timetable ttt;

  init_stops(sched, ttt);
  init_routes(sched, ttt);
  add_footpaths(sched, ttt);

  // after stops and routes are initialized
  init_stop_routes(ttt);

  //  if (write_conflict_graph) {
  //    auto const& conflict_graph = get_route_conflict_graph(ttt);
  //    write_conflict_graph_to_file(conflict_graph);
  //    write_route_graph_information_to_file(ttt);
  //  }

  //  if (partitioning_path != "") {
  //    introduce_clustering(ttt, partitioning_path);
  //  }

  return transformable_to_schedule(ttt);
}

}  // namespace motis::raptor