#pragma once

#include "utl/verify.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/protocol/RoutingRequest_generated.h"

#include "motis/raptor-core/config/configs.h"
#include "motis/raptor-core/raptor_timetable.h"

#include "motis/raptor/cpu/cpu_raptor.h"
#include "motis/raptor/error.h"
#include "motis/raptor/get_raptor_query.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_implementation_type.h"
#include "motis/raptor/reconstructor.h"

#include "motis/raptor/gpu/cluster_raptor.cuh"
#include "motis/raptor/gpu/gpu_raptor.cuh"
#include "motis/raptor/gpu/hybrid_raptor.cuh"

namespace motis {
namespace raptor {

inline auto get_departure_range(
    motis::time const begin, motis::time const end,
    std::vector<motis::time> const& departure_events) {

  auto const lower = std::lower_bound(std::cbegin(departure_events),
                                      std::cend(departure_events), begin) -
                     1;
  auto const upper = std::upper_bound(std::cbegin(departure_events),
                                      std::cend(departure_events), end) -
                     1;

  return std::pair(lower, upper);
}

template <typename RaptorFun, typename Query>
inline std::vector<journey> raptor_gen(Query& q, raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_schedule const& raptor_sched,
                                       RaptorFun const& raptor_search) {
  reconstructor reconstructor(q, sched, raptor_sched);

  // We have a ontrip query, just a single raptor query is needed
  if (q.source_time_begin_ == q.source_time_end_) {
    stats.raptor_queries_ = 1;

    MOTIS_START_TIMING(raptor_time);
    raptor_search(q);
    stats.raptor_time_ = MOTIS_TIMING_MS(raptor_time);

    MOTIS_START_TIMING(rec_timing);
    reconstructor.add(q.source_time_begin_, *q.result_);
    stats.rec_time_ = MOTIS_TIMING_US(rec_timing);

    return reconstructor.get_journeys();
  }

  // Get departure range before we do the +1 query
  auto const& [lower, upper] =
      get_departure_range(q.source_time_begin_, q.source_time_end_,
                          raptor_sched.departure_events_[q.source_]);

  stats.raptor_queries_ += 1;
  q.source_time_begin_ = q.source_time_end_ + 1;
  MOTIS_START_TIMING(plus_one_time);
  raptor_search(q);
  stats.raptor_time_ += MOTIS_TIMING_US(plus_one_time);

  MOTIS_START_TIMING(plus_one_rec_time);
  reconstructor.add(q.source_time_begin_, *q.result_);
  stats.rec_time_ += MOTIS_TIMING_US(plus_one_rec_time);

  for (auto dep_it = upper; dep_it != lower; --dep_it) {
    stats.raptor_queries_ += 1;
    q.source_time_begin_ = *dep_it;

    MOTIS_START_TIMING(raptor_time);
    raptor_search(q);
    stats.raptor_time_ += MOTIS_TIMING_US(raptor_time);

    MOTIS_START_TIMING(rec_timing);
    reconstructor.add(q.source_time_begin_, *q.result_);
    stats.rec_time_ += MOTIS_TIMING_US(rec_timing);
  }

  return reconstructor.get_journeys(q.source_time_end_);
}

inline std::vector<journey> raptor_dispatch(
    raptor_statistics& stats, schedule const& sched,
    raptor_schedule const& raptor_sched, implementation_type impl_type,
    const motis::routing::RoutingRequest* req) {
  auto const type = req->search_type();

  using ST = motis::routing::SearchType;
  utl::verify_ex(
      type == ST::SearchType_Default || type == ST::SearchType_MaxOccupancy,
      std::system_error{error::search_type_not_supported});

  if (type == ST::SearchType_MaxOccupancy) {
    utl::verify_ex(impl_type == implementation_type::CPU,
                   std::system_error{error::search_type_not_supported});
  }

  switch (impl_type) {
    case implementation_type::CPU: {
      if (type == ST::SearchType_MaxOccupancy) {
        auto q = get_query<raptor_query, OccupancyOnly>(req, sched, raptor_sched);
        auto f = [&stats, &raptor_sched](raptor_query const& q) {
          invoke_cpu_raptor<OccupancyOnly>(q, stats, raptor_sched);
        };
        return raptor_gen(q, stats, sched, raptor_sched, f);
      } else if(type == ST::SearchType_Default){
        auto q = get_query<raptor_query, Default>(req, sched, raptor_sched);
        auto f = [&stats, &raptor_sched](raptor_query const& q) {
          invoke_cpu_raptor<Default>(q, stats, raptor_sched);
        };
        return raptor_gen(q, stats, sched, raptor_sched, f);
      }else{
        throw std::system_error{error::not_implemented};
      }
    }

#ifdef MOTIS_CUDA
    case implementation_type::GPU: {
      auto q = get_query<d_query, Default>(req, sched, raptor_sched);
      return raptor_gen(q, stats, sched, raptor_sched,
                        std::function(invoke_gpu_raptor));
    }

    case implementation_type::HYBRID: {
      auto q = get_query<d_query, Default>(req, sched, raptor_sched);
      return raptor_gen(q, stats, sched, raptor_sched,
                        std::function(invoke_hybrid_raptor));
    }

    case implementation_type::CLUSTER: {
      auto q = get_query<d_query, Default>(req, sched, raptor_sched);
      return raptor_gen(q, stats, sched, raptor_sched,
                        std::function(invoke_cluster_raptor));
    }
#endif
    default: throw std::system_error{error::not_implemented};
  }
}

}  // namespace raptor
}  // namespace motis
