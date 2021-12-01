#pragma once

#include "motis/raptor/gpu/gpu_timetable.cuh"

namespace motis::raptor {

__device__ void mark(uint32_t* store, unsigned int const idx);

__device__ bool marked(uint32_t const* const store, unsigned int idx);

__device__ void reset_store(unsigned int* store, int const store_size);

__device__ void convert_station_to_route_marks(uint32_t* station_marks,
                                               uint32_t* route_marks,
                                               bool* any_station_marked,
                                               device_gpu_timetable const& tt);
}