#include "motis/core/schedule/time.h"
#include "motis/core/common/timing.h"
#include "motis/core/journey/journey.h"

#include "motis/raptor-core/raptor_timetable.h"
#include "motis/raptor-core/raptor_util.h"

#include "motis/raptor/reconstructor.h"
#include "motis/raptor/cpu_raptor.h"
#include "motis/raptor/print_raptor.h"

#include "motis/kernel/gpu_raptor.cuh"
#include "motis/kernel/hybrid_raptor.cuh"
#include "motis/kernel/cluster_raptor.cuh"

namespace motis {
namespace raptor {

inline auto get_departure_range(
    motis::time const begin, motis::time const end,
    std::vector<motis::time> const& departure_events) {

  auto const lower = std::lower_bound(std::cbegin(departure_events),
                                      std::cend(departure_events), begin) - 1;
  auto const upper = std::upper_bound(std::cbegin(departure_events),
                                      std::cend(departure_events), end) - 1;

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

inline std::vector<journey> cpu_raptor(raptor_query& q,
                                       raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_schedule const& raptor_sched) {
  auto f = std::bind(&invoke_cpu_raptor, std::placeholders::_1, 
                     std::ref(stats), std::cref(raptor_sched));
  return raptor_gen(q, stats, sched, raptor_sched, f);
}

inline std::vector<journey> hybrid_raptor(d_query& dq, raptor_statistics& stats,
                                          schedule const& sched,
                                          raptor_schedule const& raptor_sched) {
  return raptor_gen(dq, stats, sched, raptor_sched,
                std::function(invoke_hybrid_raptor));
}

inline std::vector<journey> gpu_raptor(d_query& dq, raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_schedule const& raptor_sched) {
  return raptor_gen(dq, stats, sched, raptor_sched,
                std::function(invoke_gpu_raptor));
}

inline std::vector<journey> cluster_raptor(d_query& dq,
                                           raptor_statistics& stats,
                                           schedule const& sched,
                                          raptor_schedule const& raptor_sched) {
  return raptor_gen(dq, stats, sched, raptor_sched,
                std::function(invoke_cluster_raptor));
}

} // namespace raptor
} // namespace motis
