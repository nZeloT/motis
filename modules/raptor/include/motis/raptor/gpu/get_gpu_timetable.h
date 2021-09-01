#pragma once

#include "motis/raptor-core/gpu_timetable.h"

namespace motis {
namespace raptor {

host_gpu_timetable get_host_gpu_timetable(raptor_schedule const& sched); 

}  // namespace raptor
}  // namespace motis