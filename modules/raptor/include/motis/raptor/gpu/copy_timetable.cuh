#pragma once 

#include "motis/raptor-core/gpu_timetable.h"

namespace motis {

// never put this in raptor namespace again

device_gpu_timetable copy_timetable_to_device(host_gpu_timetable const&);
void free_timetable_on_device(device_gpu_timetable const&);

} // namespace motis