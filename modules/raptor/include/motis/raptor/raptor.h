#pragma once

#include "motis/module/module.h"

#include "motis/core/journey/journey.h"

#include "motis/raptor-core/gpu_timetable.h"

#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_schedule.h"
#include "motis/raptor/raptor_implementation_type.h"

namespace motis {
namespace raptor {

struct raptor : public motis::module::module {
  raptor();
  ~raptor() override;

  raptor(raptor const&) = delete;
  raptor& operator=(raptor const&) = delete;

  raptor(raptor&&) = delete;
  raptor& operator=(raptor&&) = delete;

  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr route_generic(motis::module::msg_ptr const&,
                                       implementation_type);

  motis::module::msg_ptr make_response(std::vector<journey> const&,
                                       motis::routing::RoutingRequest const*,
                                       raptor_statistics const&);

  std::string partitioning_path_;
  bool write_conflict_graph_;

  bool initialized_{false};

  raptor_schedule raptor_sched_;
  host_gpu_timetable h_gtt_;
  device_gpu_timetable d_gtt_;
};

}  // namespace raptor
}  // namespace motis