#include "motis/raptor/raptor.h"

#include "boost/program_options.hpp"

#include "motis/module/message.h"

#include "motis/raptor/get_gpu_timetable.h"
#include "motis/raptor/get_raptor_query.h"
#include "motis/raptor/get_raptor_schedule.h"

#include "motis/bootstrap/module_settings.h"

#include "utl/to_vec.h"

#include "motis/raptor-core/cuda_util.h"
#include "motis/raptor-core/raptor_query.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journey_util.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/raptor/get_raptor_schedule.h"
#include "motis/raptor/raptor_search.h"

#include "motis/kernel/copy_timetable.cuh"
#include "motis/kernel/gpu_raptor.cuh"

namespace p = std::placeholders;
using namespace motis::module;
using namespace motis::routing;

namespace motis {
namespace raptor {

raptor::raptor() : module("RAPTOR Options", "raptor") {
  param(partitioning_path_, "partitioning_path",
        "path to a partitioning file, disabled if empty");
  param(write_conflict_graph_, "write_conflict_graph",
        "writes out the route conflict graph");
}

raptor::~raptor() {
  if (initialized_) {
    free_timetable_on_device(d_gtt_);
  }
}

void raptor::init(motis::module::registry& reg) {
  initialized_ = true;

  auto const& sched = get_sched();

  raptor_sched_ =
      *(get_raptor_schedule(sched, write_conflict_graph_, partitioning_path_));

  h_gtt_ = get_host_gpu_timetable(raptor_sched_);
  d_gtt_ = copy_timetable_to_device(h_gtt_);

  auto const& cpu = [&](msg_ptr const& msg) {
    return route_generic<decltype(cpu_raptor), raptor_query>(msg, cpu_raptor);
  };

  auto const& gpu = [&](msg_ptr const& msg) {
    return route_generic<decltype(gpu_raptor), d_query>(msg, gpu_raptor);
  };
  auto const& hy = [&](msg_ptr const& msg) {
    return route_generic<decltype(hybrid_raptor), d_query>(msg, hybrid_raptor);
  };

  auto const& cluster = [&](msg_ptr const& msg) {
    return route_generic<decltype(cluster_raptor), d_query>(msg,
                                                            cluster_raptor);
  };

  reg.register_op("/raptor_cpu", cpu);
  reg.register_op("/raptor_gpu", gpu);
  reg.register_op("/raptor_hybrid", hy);
  reg.register_op("/raptor_cluster", cluster);
  reg.register_op("/raptor", hy);

  std::vector<std::string> const prefs = {"GeForce GTX 1050 Ti",
                                          "GeForce RTX 2080",
                                          "GeForce GTX 1080", "Quadro K600"};
  set_device(prefs);
}

template <typename RaptorFun, typename Query>
msg_ptr raptor::route_generic(msg_ptr const& msg,
                              RaptorFun const& raptor_search) {
  auto const req = motis_content(RoutingRequest, msg);
  auto const& sched = get_sched();

  raptor_statistics stats;
  auto q = get_query<Query>(req, sched, raptor_sched_);

  MOTIS_START_TIMING(total_calculation_time);
  auto const& js = raptor_search(q, stats, sched, raptor_sched_);
  stats.total_calculation_time_ = MOTIS_TIMING_MS(total_calculation_time);

  return make_response(js, req, stats);
}

msg_ptr raptor::make_response(std::vector<journey> const& js,
                              motis::routing::RoutingRequest const* request,
                              raptor_statistics const& stats) {
  auto const& sched = get_sched();

  auto const* start = static_cast<OntripStationStart const*>(request->start());
  auto const interval_start = start->departure_time();
  auto const interval_end = start->departure_time();

  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingResponse,
      CreateRoutingResponse(
          fbb,
          fbb.CreateVector(std::vector<flatbuffers::Offset<Statistics>>{
              to_fbs(fbb, to_stats_category("raptor", stats))}),
          fbb.CreateVector(utl::to_vec(
              js,
              [&](journey const& j) { return motis::to_connection(fbb, j); })),
          motis_to_unixtime(sched, interval_start),
          motis_to_unixtime(sched, interval_end),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<DirectConnection>>()))
          .Union());
  return make_msg(fbb);
}

}  // namespace raptor
}  // namespace motis