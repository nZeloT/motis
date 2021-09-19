#pragma once

#include "motis/raptor-core/raptor_query.h"
#include "motis/raptor/raptor_schedule.h"

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

namespace motis {
namespace raptor {

using namespace motis::routing;

base_query get_base_query(RoutingRequest const* routing_request,
                          schedule const& sched,
                          raptor_schedule const& raptor_sched) {
  base_query q;

  auto const destination_station = routing_request->destination();
  auto const target_eva = destination_station->id()->str();
  std::string start_eva;

  switch (routing_request->start_type()) {
    case Start::Start_PretripStart: {
      auto const* pretrip_start =
          static_cast<PretripStart const*>(routing_request->start());

      auto const start_station = pretrip_start->station();
      start_eva = start_station->id()->str();

      auto const interval = pretrip_start->interval();
      auto const departure_time_begin = interval->begin();
      auto const departure_time_end = interval->end();

      q.source_time_begin_ =
          unix_to_motistime(sched.schedule_begin_, departure_time_begin);
      q.source_time_end_ =
          unix_to_motistime(sched.schedule_begin_, departure_time_end);
    } break;

    case Start::Start_OntripStationStart: {
      auto const* ontrip_start =
          static_cast<OntripStationStart const*>(routing_request->start());

      auto const start_station = ontrip_start->station();
      start_eva = start_station->id()->str();

      auto const departure_time = ontrip_start->departure_time();
      q.source_time_begin_ =
          unix_to_motistime(sched.schedule_begin_, departure_time);
      q.source_time_end_ = q.source_time_begin_;

    } break;

    default: break;
  }

  q.source_ = raptor_sched.eva_to_raptor_id_.at(start_eva);
  q.target_ = raptor_sched.eva_to_raptor_id_.at(target_eva);

  return q;
}

template <class Query, typename Config>
inline Query get_query(motis::routing::RoutingRequest const* routing_request,
                schedule const& sched, raptor_schedule const& raptor_sched) {
  auto const& tt = raptor_sched.timetable_;
  auto const config_dim_size = Config::trait_size();

  return Query(get_base_query(routing_request, sched, raptor_sched), tt,
               config_dim_size);
}

}  // namespace raptor
}  // namespace motis