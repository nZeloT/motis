#include "motis/core/common/logging.h"
#include "motis/raptor-core/raptor_schedule.h"
#include "motis/raptor-core/raptor_timetable.h"
#include "motis/raptor-core/gpu_timetable.h"
#include "motis/raptor-core/raptor_util.h"

namespace motis {
namespace raptor {

host_gpu_timetable get_host_gpu_timetable(raptor_schedule const& sched) {
  auto const& tt = sched.timetable_;

  host_gpu_timetable h_gtt;

  // Copy the members, which are identical on CPU and GPU
  h_gtt.stops_ = tt.stops_;
  h_gtt.routes_ = tt.routes_;
  // h_gtt.stop_times_ = tt.stop_times_;
  h_gtt.route_stops_ = tt.route_stops_;
  h_gtt.stop_routes_ = tt.stop_routes_;

  // h_gtt.transfer_times_ = sched.transfer_times_;

  for (auto const& station_footpaths : sched.initialization_footpaths_) {
    h_gtt.initialization_footpaths_indices_.push_back(
        h_gtt.initialization_footpaths_.size());
    append_vector(h_gtt.initialization_footpaths_, station_footpaths);
  }
  h_gtt.initialization_footpaths_indices_.push_back(
      h_gtt.initialization_footpaths_.size());

  // Create GPU footpaths, with from and to station
  h_gtt.footpaths_.resize(tt.footpath_count());
  for (station_id s_id = 0; s_id < tt.stop_count(); ++s_id) {
    auto const& stop = tt.stops_[s_id];
    auto const& next_stop = tt.stops_[s_id + 1];

    for (auto foot_idx = stop.index_to_transfers_;
         foot_idx < next_stop.index_to_transfers_; ++foot_idx) {
      auto const& f = tt.footpaths_[foot_idx];
      h_gtt.footpaths_[foot_idx].from_ = s_id;
      h_gtt.footpaths_[foot_idx].to_ = f.to_;
      h_gtt.footpaths_[foot_idx].duration_ = f.duration_;
    }
  }

  // Create split stop times arrays
  h_gtt.stop_arrivals_.reserve(tt.stop_times_.size());
  h_gtt.stop_departures_.reserve(tt.stop_times_.size());
  for (auto const stop_time : tt.stop_times_) {
    h_gtt.stop_arrivals_.push_back(stop_time.arrival_);
    h_gtt.stop_departures_.push_back(stop_time.departure_);
  }

  if (sched.partitioning_.empty()) {
    return h_gtt;
  }

  // Create members for clustered timetable
  h_gtt.clustered_route_stops_.resize(h_gtt.route_stops_.size());

  route_id total_route_count = 0;
  for (auto c_id = 0u; c_id < sched.partitioning_.size(); ++c_id) {
    auto const& partition = sched.partitioning_.partitions_[c_id];
    auto const& border = sched.cluster_border_[c_id];
    auto const& inland = sched.cluster_inland_[c_id];

    cluster c;
    c.border_station_count_ = border.size();
    c.inland_station_count_ = inland.size();
    c.route_count_ = partition.size();
    c.route_start_index_ = total_route_count;
    c.border_mapping_index_ = h_gtt.border_mappings_.size();

    if (c_id == 0) {
      c.arrivals_start_index_ = sched.total_border_stations_;
    } else {
      c.arrivals_start_index_ = h_gtt.clusters_.back().arrivals_start_index_ +
                                h_gtt.clusters_.back().inland_station_count_;
    }

    total_route_count += partition.size();

    h_gtt.clusters_.push_back(c);

    // create border mapping and quick lookup for route stop translation
    std::vector<cls_station_id> station_lookup(tt.stop_count(),
                                               invalid<cls_station_id>);
    for (cls_station_id local = 0; local < border.size(); ++local) {
      auto const s_id = border[local];
      station_lookup[s_id] = local;
      h_gtt.border_mappings_.push_back(s_id);
    }

    cls_station_id local = 0;
    for (station_id s_id = c.arrivals_start_index_;
         s_id < c.arrivals_start_index_ + c.inland_station_count_; ++s_id) {
      station_lookup[s_id] = border.size() + local;
      ++local;
    }

    for (auto const r_id : partition) {
      auto const& route = h_gtt.routes_[r_id];
      for (auto rsi = route.index_to_route_stops_;
           rsi < route.index_to_route_stops_ + route.stop_count_; ++rsi) {
        auto const route_stop = h_gtt.route_stops_[rsi];
        h_gtt.clustered_route_stops_[rsi] = station_lookup[route_stop];
      }
    }
  }

  return h_gtt;
}

}  // namespace raptor
}  // namespace motis