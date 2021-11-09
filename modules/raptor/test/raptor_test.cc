#include "raptor_test.h"

#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <mutex>

#include "motis/loader/loader.h"
#include "motis/raptor-core/config/configs.h"
#include "motis/raptor-core/raptor_query.h"
#include "motis/raptor/cpu/cpu_raptor.h"
#include "motis/raptor/get_raptor_schedule.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/reconstructor.h"

#include "motis/core/access/station_access.h"
#include "motis/protocol/RoutingRequest_generated.h"
#include "motis/routing/search.h"
#include "motis/routing/search_dispatch.h"
#include "motis/core/journey/journey.h"
#include "motis/routing/mem_retriever.h"

// 64MB default start size
constexpr auto LABEL_STORE_START_SIZE = 64 * 1024 * 1024;

namespace motis::raptor {

cpu_raptor_test::cpu_raptor_test(loader::loader_options opts)
    : opts_{std::move(opts)}, rp_sched_{nullptr} {}

void cpu_raptor_test::SetUp() {
  sched_ = loader::load_schedule(opts_);
  manipulate_schedule();
  rp_sched_ = get_raptor_schedule(*sched_, false, "");
  check_mock_on_rp_sched();
}

uint32_t cpu_raptor_test::get_raptor_r_id(const std::string& gtfs_trip_id) {
  auto& trip2route = rp_sched_->dbg_.trip_dbg_to_route_trips_;
  for (auto const& entry : trip2route) {
    if (entry.first.starts_with(gtfs_trip_id)) {
      if (entry.second.size() > 1) {
        throw std::runtime_error{
            "No unique mapping between route ids and gtfs ids!"};
      } else {
        return entry.second.begin()->first;
      }
    }
  }

  throw std::runtime_error{"No entries in trip2route! => GTFS id is unknown"};
}

std::vector<journey> cpu_raptor_test::execute_raptor(
    time dep, std::string const& eva_from, std::string const& eva_to) {
  base_query bq{};
  bq.source_time_begin_ = dep;
  bq.source_time_end_ = bq.source_time_begin_;
  bq.source_ = rp_sched_->eva_to_raptor_id_.at(eva_from);
  bq.target_ = rp_sched_->eva_to_raptor_id_.at(eva_to);
  raptor_query q{bq, rp_sched_->timetable_, OccupancyOnly::trait_size()};
  raptor_statistics st;
  invoke_cpu_raptor<OccupancyOnly>(q, st, *rp_sched_);
  reconstructor<OccupancyOnly> rc{bq, *sched_, *rp_sched_};
  rc.add(dep, *q.result_);
  return rc.get_journeys();
}

std::vector<journey> cpu_raptor_test::execute_routing(
    time dep, std::string const& eva_from, std::string const& eva_to) {

  std::mutex mem_pool_mutex;
  std::vector<std::unique_ptr<routing::memory>> mem_pool;

  routing::mem_retriever mem(mem_pool_mutex, mem_pool, LABEL_STORE_START_SIZE);

  routing::search_query q{};
  q.mem_ = &mem.get();
  q.sched_ = sched_.get();
  q.use_start_footpaths_ = true;
  q.use_dest_metas_ = false;
  q.use_start_metas_ = false;
  q.interval_begin_ = dep;
  q.interval_end_ = INVALID_TIME;
  q.from_ = motis::get_station_node(*sched_, eva_from);
  q.to_ = motis::get_station_node(*sched_, eva_to);
  auto const results = routing::search_dispatch(
      q, routing::Start_OntripStationStart, routing::SearchType_MaxOccupancy,
      routing::SearchDir_Forward);

  auto journeys = results.journeys_;

  //sort for transfers and max occupancy to get the same order as with raptor
  std::sort(journeys.begin(), journeys.end(), [](journey const& lhs, journey const& rhs) {
    if(lhs.trips_.size() > rhs.trips_.size())
      return true;
    if(lhs.trips_.size() == rhs.trips_.size())
      return lhs.occupancy_max_ > rhs.occupancy_max_;
    return false;
  });

  return journeys;
}

}  // namespace motis::raptor