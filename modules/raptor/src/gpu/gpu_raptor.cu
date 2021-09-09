#include <algorithm>
#include <iostream>
#include <cstdio>

#include "cooperative_groups.h"
#include "cuda_profiler_api.h"

#include "motis/core/common/timing.h"

namespace motis {

using namespace cooperative_groups;

using namespace raptor;

// leader type must be unsigned 32bit
// no leader is a zero ballot vote (all 0) minus 1 => with underflow all 1's
constexpr unsigned int FULL_MASK = 0xFFFFffff;
constexpr unsigned int NO_LEADER = FULL_MASK;

__device__ __forceinline__
unsigned int get_block_thread_id() {
  return threadIdx.x + (blockDim.x * threadIdx.y);
}

__device__ __forceinline__
unsigned int get_global_thread_id() {
  return get_block_thread_id() + (blockDim.x * blockDim.y * blockIdx.x);
}

__device__ __forceinline__
unsigned int get_block_stride() {
  return blockDim.x * blockDim.y;
}

__device__ __forceinline__
unsigned int get_global_stride() {
  return get_block_stride() * gridDim.x * gridDim.y;
}

__device__
void mark(unsigned int* store, unsigned int const idx) {
  unsigned int const store_idx = (idx >> 5); // divide by 32
  unsigned int const mask = 1 << (idx % 32); 
  atomicOr(&store[store_idx], mask);
}

__device__
bool marked(unsigned int const * const store, unsigned int idx) {
  unsigned int const store_idx = (idx >> 5); // divide by 32
  unsigned int const val = store[store_idx];
  unsigned int const mask = 1 << (idx % 32);
  return (bool) (val & mask);
}

__device__
void reset_store(unsigned int* store, int const store_size) {
  auto const t_id = get_global_thread_id();
  auto const stride = get_global_stride();
  for (auto idx = t_id; idx < store_size; idx += stride) {
    store[idx] = 0;
  }
}

__device__
void convert_station_to_route_marks(unsigned int* station_marks, 
                                    unsigned int* route_marks) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for (auto idx = global_t_id; idx < GTT.stop_count_; idx += global_stride) {
    if (marked(station_marks, idx)) {
      if (!ANY_STATION_MARKED) { ANY_STATION_MARKED = true; }
      auto const stop =  GTT.stops_[idx];
      for (auto sri = stop.index_to_stop_routes_; 
                sri < stop.index_to_stop_routes_ + stop.route_count_;
              ++sri) {
        mark(route_marks, GTT.stop_routes_[sri]);
      }
    }
  }
}

__device__
motis::time get_stop_arrival(stop_times_index const sti) {
  return GTT.stop_times_[sti].arrival_;
}

__device__
motis::time get_stop_departure(stop_times_index const sti) {
  return GTT.stop_times_[sti].departure_;
}

__device__
motis::time get_stop_arrival_split(stop_times_index const sti) {
  return GTT.stop_arrivals_[sti];
}

__device__
motis::time get_stop_departure_split(stop_times_index const sti) {
  return GTT.stop_departures_[sti];
}

__device__
station_id get_route_stop(route_stops_index const rsi) {
  return GTT.route_stops_[rsi];
}

__device__
station_id get_route_stop_cls(route_stops_index const rsi) {
  return GTT.clustered_route_stops_[rsi];
}

__device__
stop_time get_stop_time(stop_times_index const sti) {
  return GTT.stop_times_[sti];
  // auto const addr = &GTT.stop_times_[idx];
  // uint32_t return_value;

  // asm("ld.global.cg.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  // return *((stop_time*) &return_value);
}


__device__ 
motis::time get_arrival(global_mem_time const * const base, 
                        station_id const s_id) {
  // motis::time return_value;
  // asm("ld.global.ca.s16 %0, [%1];" : "=h"(return_value) : "l"(&base[s_id]));
  // return return_value;
  // return __ldg(&base[s_id]);
  return base[s_id];
}

#if SMALL_TIME /* SMALL TIME -- 16 bit time values in global memory */

__device__
bool update_arrival(global_mem_time * const base, 
                    station_id const s_id, 
                    global_mem_time const val) {

#if __CUDA_ARCH__ >= 700

  auto old_value = base[s_id];
  global_mem_time assumed;

  do {
    if (old_value <= val) { return false; }

    assumed = old_value;

    old_value = atomicCAS(&base[s_id], assumed, val);
  } while (assumed != old_value);
  
  return true;

#else

  // we have a 16-bit time value array, but only 32-bit atomic operations
  // therefore every two 16-bit time values are read as one 32-bit time value
  // then they are the corresponding part is updated and stored if a better
  // time value was found while the remaining 16 bit value part remains
  // unchanged

  global_mem_time * const arr_address = &base[s_id];
  unsigned int* base_address = (unsigned int*)((size_t) arr_address & ~2);
  unsigned int old_value, assumed, new_value, compare_val;

  old_value = *base_address;

  do {
    assumed = old_value;

    if ((size_t) arr_address & 2) {
      compare_val = (0x0000FFFF & assumed) ^ (((unsigned int) val) << 16);
    } else {
      compare_val = (0xFFFF0000 & assumed) ^ (unsigned int) val;
    }

    new_value = __vminu2(old_value, compare_val);

    if (new_value == old_value) { return false; }

    old_value = atomicCAS(base_address, assumed, new_value);
  } while (assumed != old_value);

  return true;

#endif
}

#else /* BIG TIME -- 32bit time values in global memory */

__device__
motis::time get_arrival(global_mem_time const * const base, 
                        station_id const s_id) {
  return (motis::time) (base[s_id]);
}

__device__
void update_arrival(global_mem_time* const base, 
                    station_id const s_id, 
                    motis::time const val) {
  global_mem_time* address = &base[s_id];
  atomicMin(address, (motis::time32) val);
}

#endif /* END IF STATEMENT */

__device__
void reset_arrivals(motis::time* const arrivals) {
  auto const global_stride = get_global_stride();
  auto arr_idx = get_global_thread_id();

  for (; arr_idx < GTT.stop_count_; arr_idx += global_stride) {
    arrivals[arr_idx] = invalid<global_mem_time>;
  }
}

__device__
void copy_marked_arrivals(motis::time* const to, motis::time* const from, unsigned int* station_marks) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  for (; arr_idx < GTT.stop_count_; arr_idx += global_stride) {
    if (marked(station_marks, arr_idx)) {
      to[arr_idx] = from[arr_idx]; 
    } else {
      to[arr_idx] = invalid<motis::time>;
    }
  }
}

__device__
void copy_arrivals(motis::time* const to, motis::time* const from) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  for (; arr_idx < GTT.stop_count_; arr_idx += global_stride) {
    to[arr_idx] = from[arr_idx]; 
  }
}

__device__
void copy_and_min_arrivals(motis::time* const to, motis::time* const from) {
  auto const global_stride = get_global_stride();

  auto arr_idx = get_global_thread_id();
  for (; arr_idx < GTT.stop_count_; arr_idx += global_stride) {
    to[arr_idx] = min(from[arr_idx], to[arr_idx]); 
  }
}

template<GetArrivalFun GetArrival, 
         UpdateArrivalFun UpdateArrival,
         GetRouteStopFun GetRouteStop,
         GetStopArrivalFun GetStopArrival, 
         GetStopDepartureFun GetStopDeparture>
__device__
void update_route_larger32_gen(gpu_route const& route,
                               global_mem_time const * const prev_arrivals,
                               global_mem_time* const arrivals,
                               unsigned int* station_marks) {
  auto const t_id = threadIdx.x;

  station_id station_id_t = invalid<station_id>;
  motis::time prev_arrival = invalid<motis::time>;
  motis::time stop_arrival = invalid<motis::time>;
  motis::time stop_departure = invalid<motis::time>;
  
  int active_stop_count = route.stop_count_;

  // this is ceil(stop_count / 32)
  int const stage_count = (route.stop_count_ + (32 - 1)) >> 5;
  int active_stage_count = stage_count;

  unsigned int leader = NO_LEADER;
  unsigned int any_arrival = 0;

  for (int trip_offset = 0; 
               trip_offset < route.trip_count_; 
             ++trip_offset) {

    for (int current_stage = 0; 
             current_stage < active_stage_count; 
           ++current_stage) {

      int stage_id = (current_stage << 5) + t_id;

      // load the prev arrivals for the current stage
      if (stage_id < active_stop_count) {
        station_id_t = GetRouteStop(route.index_to_route_stops_ + stage_id);
        prev_arrival = GetArrival(prev_arrivals, station_id_t);
      } 

      any_arrival |= __any_sync(FULL_MASK, valid(prev_arrival));
      if (current_stage == active_stage_count - 1 && !any_arrival) { return; }
      if (!any_arrival) { continue; }

      // load the stop times for the current stage
      if (stage_id < active_stop_count) {
        auto const st_idx = route.index_to_stop_times_
                    + (trip_offset * route.stop_count_)
                    + stage_id;
        stop_departure = GetStopDeparture(st_idx);
      } 

      // get the current stage leader
      unsigned int ballot = __ballot_sync(FULL_MASK,
                                    (stage_id < active_stop_count) 
                                  && valid(prev_arrival) 
                                  && valid(stop_departure)
                                  && (prev_arrival <= stop_departure));
      leader = __ffs(ballot) - 1;

      if (leader != NO_LEADER) {
        leader += current_stage << 5;
      } 

      // first update the current stage
      if (leader != NO_LEADER && stage_id < active_stop_count) {

        if (stage_id > leader) {
        auto const st_idx = route.index_to_stop_times_
                    + (trip_offset * route.stop_count_)
                    + stage_id;
          stop_arrival = GetStopArrival(st_idx);
          bool updated = UpdateArrival(arrivals, station_id_t, stop_arrival);
          if (updated) { mark(station_marks, station_id_t); }
        } 
      }

      // then update all upward stages
      if (leader != NO_LEADER) {
        for (int upward_stage = current_stage + 1; 
                 upward_stage < active_stage_count;
               ++upward_stage) {

          int upwards_id = (upward_stage << 5) + t_id;
          if (upwards_id < active_stop_count) {

          auto const st_idx = route.index_to_stop_times_
                          + (trip_offset * route.stop_count_) 
                          + upwards_id;

            stop_arrival = GetStopArrival(st_idx);
            station_id_t = GetRouteStop(route.index_to_route_stops_ 
                                        + upwards_id);
            bool updated = UpdateArrival(arrivals, station_id_t, stop_arrival);
            if (updated) { mark(station_marks, station_id_t); }
          } 
        }

        // for this route we do not need to update any station higher than the leader anymore
        active_stop_count = leader;
        active_stage_count = (active_stop_count + (32 - 1)) >> 5;
        leader = NO_LEADER;
      } 
    }
  }
}

template<GetArrivalFun GetArrival, 
         UpdateArrivalFun UpdateArrival,
         GetRouteStopFun GetRouteStop, 
         GetStopArrivalFun GetStopArrival,
         GetStopDepartureFun GetStopDeparture>
__device__
void update_route_smaller32_fronts_gen(gpu_route const route,
                                   global_mem_time const * const prev_arrivals,
                                   global_mem_time * const arrivals,
                                   unsigned int* station_marks) {
  auto const t_id = threadIdx.x;
  auto const index_to_stop_times = route.index_to_stop_times_;
  auto const stop_count = route.stop_count_;

  station_id station_id_t = invalid<station_id>;
  motis::time prev_arrival = invalid<motis::time>;
  motis::time stop_arrival = invalid<motis::time>;
  motis::time stop_departure = invalid<motis::time>;

  unsigned leader = stop_count;
  unsigned int active_stop_count = stop_count;

  unsigned const front_count = 32 / stop_count;
  unsigned const front_lane = threadIdx.x % stop_count;
  unsigned const excess_threshold = front_count * stop_count;

  unsigned const front_id = threadIdx.x / stop_count;
  stop_times_index st_idx = invalid<stop_times_index>;

  if (front_lane < active_stop_count && t_id < excess_threshold) {
    station_id_t = GTT.route_stops_[route.index_to_route_stops_ + front_lane];
    prev_arrival = GetArrival(prev_arrivals, station_id_t);
  }

  if (!__any_sync(FULL_MASK, valid(prev_arrival))) { return; }

  for (unsigned trip_offset = front_id; 
                trip_offset < route.trip_count_;
                trip_offset += front_count) {

    if (front_lane < active_stop_count && t_id < excess_threshold) {
      st_idx = index_to_stop_times 
             + (trip_offset * stop_count) 
             + front_lane;
      stop_departure = GetStopDeparture(st_idx);
    }

    unsigned ballot = __ballot_sync(FULL_MASK, 
                                    (front_lane < active_stop_count)
                                 && (t_id < excess_threshold)
                                 && valid(prev_arrival)
                                 && valid(stop_departure)
                                 && (prev_arrival <= stop_departure));

    for (unsigned f_id = 0; f_id < front_count; ++f_id) {
      if (f_id == front_id) {

        leader = __ffs(ballot) - 1;

        if (front_lane > leader && front_lane < active_stop_count) {
          stop_arrival = GetStopArrival(st_idx);
          bool updated = UpdateArrival(arrivals, station_id_t, stop_arrival);
          if (updated) { mark(station_marks, station_id_t); }
        } 

        if (leader != NO_LEADER) {
          active_stop_count = leader;
        }
      }
      
      ballot = ballot >> stop_count;
    }
  }
}

template<GetArrivalFun GetArrival, 
         UpdateArrivalFun UpdateArrival,
         GetRouteStopFun GetRouteStop, 
         GetStopArrivalFun GetStopArrival,
         GetStopDepartureFun GetStopDeparture>
__device__
void update_route_smaller32_gen(gpu_route const route,
                                global_mem_time const * const prev_arrivals,
                                global_mem_time * const arrivals,
                                unsigned int* station_marks) {
  auto const t_id = threadIdx.x;

  station_id station_id_t = invalid<station_id>;
  motis::time prev_arrival = invalid<motis::time>;
  motis::time stop_arrival = invalid<motis::time>;
  motis::time stop_departure = invalid<motis::time>;

  unsigned leader = route.stop_count_;
  unsigned int active_stop_count = route.stop_count_;

  if (t_id < active_stop_count) {
    station_id_t = GetRouteStop(route.index_to_route_stops_ + t_id);
    prev_arrival = GetArrival(prev_arrivals, station_id_t);
  }

  if (!__any_sync(FULL_MASK, valid(prev_arrival))) { return; }

  for (int trip_offset = 0; 
           trip_offset < route.trip_count_; 
         ++trip_offset) {

    if (t_id < active_stop_count) {
      auto const st_idx = route.index_to_stop_times_ 
                        + (trip_offset * route.stop_count_) 
                        + t_id;
      stop_departure = GetStopDeparture(st_idx);
    }

    // elect leader
    unsigned ballot = __ballot_sync(FULL_MASK, 
                                    (t_id < active_stop_count)
                                 && valid(prev_arrival)
                                 && valid(stop_departure)
                                 && (prev_arrival <= stop_departure));
    leader = __ffs(ballot) - 1;

    if (t_id > leader && t_id < active_stop_count) {
      auto const st_idx = route.index_to_stop_times_ 
                        + (trip_offset * route.stop_count_) 
                        + t_id;

      stop_arrival = GetStopArrival(st_idx);
      bool updated = UpdateArrival(arrivals, station_id_t, stop_arrival);
      if (updated) { mark(station_marks, station_id_t); }
    } 

    if (leader != NO_LEADER) {
      active_stop_count = leader;
    }
    leader = NO_LEADER;
  }
}

template <GetArrivalFun GetArrival, UpdateArrivalFun UpdateArrival>
__device__ void update_footpaths_dev_scratch_gen(
    global_mem_time const* const read_arrivals,
    global_mem_time* const write_arrivals, unsigned int* station_marks) {

  auto const global_stride = get_global_stride();

  auto foot_idx = get_global_thread_id();
  for (; foot_idx < GTT.footpath_count_; foot_idx += global_stride) {
    auto const footpath = GTT.footpaths_[foot_idx];

    motis::time const from_arrival = GetArrival(read_arrivals, footpath.from_);
    motis::time const new_arrival = from_arrival + footpath.duration_;

    if (valid(from_arrival) && marked(station_marks, footpath.from_)) {
      bool updated = UpdateArrival(write_arrivals, footpath.to_, new_arrival);
      if (updated) { mark(station_marks, footpath.to_); }
    }
  }
}

__device__
auto const update_footpaths_dev_scratch
        =  update_footpaths_dev_scratch_gen<get_arrival, update_arrival>;

template<GetArrivalFun GetArrival, UpdateArrivalFun UpdateArrival>
__device__
void update_footpaths_dev_new_gen(global_mem_time const * const arrivals,
                                  global_mem_time* const next_arrivals,
                                  unsigned int* station_marks) {
  auto const global_stride = get_global_stride();

  auto foot_idx = get_global_thread_id();
  for (; foot_idx < GTT.footpath_count_; foot_idx += global_stride) {
    auto const footpath = GTT.footpaths_[foot_idx];

    motis::time from_arrival = arrivals[footpath.from_];
    motis::time new_arrival = from_arrival + footpath.duration_;

    if (valid(from_arrival) && marked(station_marks, footpath.from_)) {
      bool updated = UpdateArrival(next_arrivals, footpath.to_, new_arrival);
      if (updated) { mark(station_marks, footpath.to_); }
    }
  }
}

__device__
auto const update_footpaths_dev_new
        =  update_footpaths_dev_new_gen<get_arrival, update_arrival>;

__device__
auto const update_route_smaller32
         = update_route_smaller32_gen<get_arrival, 
                                      update_arrival,  
                                      get_route_stop, 
                                      get_stop_arrival, 
                                      get_stop_departure>;
__device__
auto const update_route_larger32
         = update_route_larger32_gen<get_arrival, 
                                      update_arrival,
                                      get_route_stop, 
                                      get_stop_arrival,
                                      get_stop_departure>;

__device__
auto const update_route_smaller32_cls
         = update_route_smaller32_gen<get_arrival, 
                                      update_arrival,
                                      get_route_stop_cls, 
                                      get_stop_arrival,
                                      get_stop_departure>;
__device__
auto const update_route_larger32_cls
         = update_route_larger32_gen<get_arrival, 
                                     update_arrival,
                                     get_route_stop_cls,
                                     get_stop_arrival,
                                     get_stop_departure>;
__device__
auto const update_route_smaller32_cls_split
         = update_route_smaller32_gen<get_arrival, 
                                      update_arrival,
                                      get_route_stop_cls, 
                                      get_stop_arrival_split,
                                      get_stop_departure_split>;
__device__
auto const update_route_larger32_cls_split
         = update_route_larger32_gen<get_arrival, 
                                     update_arrival,
                                     get_route_stop_cls,
                                     get_stop_arrival_split,
                                     get_stop_departure_split>;

__device__
auto const update_route_smaller32_fronts
         = update_route_smaller32_fronts_gen<get_arrival, 
                                             update_arrival,
                                             get_route_stop,
                                             get_stop_arrival,
                                             get_stop_departure>;

__device__
auto const update_route_smaller32_split
         = update_route_smaller32_gen<get_arrival, 
                                      update_arrival,  
                                      get_route_stop, 
                                      get_stop_arrival_split, 
                                      get_stop_departure_split>;

__device__
auto const update_route_larger32_split
         = update_route_larger32_gen<get_arrival, 
                                      update_arrival,
                                      get_route_stop, 
                                      get_stop_arrival_split,
                                      get_stop_departure_split>;

__device__
void update_routes_dev(global_mem_time const * const prev_arrivals,
                       global_mem_time* const arrivals,
                       unsigned int* station_marks,
                       unsigned int* route_marks) {

  if (get_global_thread_id() == 0) { ANY_STATION_MARKED = false; }

  convert_station_to_route_marks(station_marks, route_marks);
  this_grid().sync();

  auto const station_store_size = (GTT.stop_count_ / 32) + 1;
  reset_store(station_marks, station_store_size);
  this_grid().sync();

  if (!ANY_STATION_MARKED) { return; }

  auto const stride = blockDim.y * gridDim.x;
  auto const start_r_id = threadIdx.y + (blockDim.y * blockIdx.x);
  for (auto r_id = start_r_id; r_id < GTT.route_count_; r_id += stride) {
    if (!marked(route_marks, r_id)) { continue; }

    auto const route = GTT.routes_[r_id];
    if (route.stop_count_ <= 32) {
      // update_route_smaller32_split(route, prev_arrivals, arrivals, r_id);
      update_route_smaller32_split(route, 
                                   prev_arrivals, arrivals,
                                   station_marks);
    } else {
      // update_route_larger32_split(route, prev_arrivals, arrivals, r_id);
      update_route_larger32_split(route, 
                                  prev_arrivals, arrivals,
                                  station_marks);
    }
  }  

  this_grid().sync();

  auto const store_size = (GTT.route_count_ / 32) + 1;
  reset_store(route_marks, store_size);
}

__device__
void init_arrivals_dev(d_query const& dq) {
  auto const t_id = get_global_thread_id();

  auto const station_store_size = (GTT.stop_count_ / 32) + 1;
  reset_store(dq.station_marks_, station_store_size);

  auto const route_store_size = (GTT.route_count_ / 32) + 1;
  reset_store(dq.route_marks_, route_store_size);

  if (t_id == 0) { ANY_STATION_MARKED = false; }

  if (t_id == 0) {
    dq.d_arrivals_[0][dq.source_] = dq.source_time_begin_;
    mark(dq.station_marks_, dq.source_);
  }

  auto const footpath_count =
      GTT.initialization_footpaths_indices_[dq.source_ + 1] -
      GTT.initialization_footpaths_indices_[dq.source_];
  if (t_id < footpath_count) {
    auto const index_into_footpaths =
               GTT.initialization_footpaths_indices_[dq.source_];
    auto const f = GTT.initialization_footpaths_[index_into_footpaths + t_id];

    global_mem_time const new_value = dq.source_time_begin_ + f.duration_;
    bool updated = update_arrival(dq.d_arrivals_[0], f.to_, new_value);
    if (updated) { mark(dq.station_marks_, f.to_); }
  }
}

__device__
void update_footpaths_dev(d_query const dq, int round_k) {
  global_mem_time* const arrivals = dq.d_arrivals_[round_k];
  global_mem_time* const next_arrivals = dq.d_arrivals_[round_k + 1];

  // we must only copy the marked arrivals, 
  // since an earlier raptor query might have used a footpath
  // to generate the current arrival, a new optimum from this value 
  // would be generated using a double walk -> not correct!
  copy_marked_arrivals(dq.footpaths_scratchpad_, arrivals, dq.station_marks_);
  this_grid().sync();

  update_footpaths_dev_scratch(dq.footpaths_scratchpad_, arrivals,
                               dq.station_marks_);
  this_grid().sync();

  copy_and_min_arrivals(next_arrivals, arrivals);
  this_grid().sync();
}

__global__
__launch_bounds__((block_dim_x * block_dim_y), min_blocks_per_sm)
void init_arrivals_kernel(d_query const d_query) {
  init_arrivals_dev(d_query);
}

__global__
__launch_bounds__((block_dim_x * block_dim_y), min_blocks_per_sm)
void update_footpaths_kernel(d_query const dq, int round_k) {
  update_footpaths_dev(dq, round_k);
}

__global__
__launch_bounds__((block_dim_x * block_dim_y), min_blocks_per_sm)
void update_routes_kernel(d_query const dq, int round_k) {
  global_mem_time const * const prev_arrivals = dq.d_arrivals_[round_k - 1];
  global_mem_time* const arrivals = dq.d_arrivals_[round_k];

  update_routes_dev(prev_arrivals, arrivals, 
                    dq.station_marks_, dq.route_marks_);
}

__global__
__launch_bounds__((block_dim_x * block_dim_y), min_blocks_per_sm)
void gpu_raptor_kernel(d_query const dq) {
  init_arrivals_dev(dq);
  this_grid().sync();

  for (int8_t round_k = 1; round_k < max_round_k; ++round_k) {
    const global_mem_time* const prev_arrivals = dq.d_arrivals_[round_k-1];
    global_mem_time* const arrivals = dq.d_arrivals_[round_k];
    global_mem_time* const next_arrivals = dq.d_arrivals_[round_k + 1];

    update_routes_dev(prev_arrivals, arrivals,
                      dq.station_marks_, dq.route_marks_);
    this_grid().sync();

    update_footpaths_dev(dq, round_k);
  }
}

void invoke_gpu_raptor(d_query& dq) {
  void* kernel_args[] = {(void*)&dq};
  cudaLaunchCooperativeKernel((void*)gpu_raptor_kernel, grid_dim,
                              threads_per_block, kernel_args);
  cc();
  cudaDeviceSynchronize();
  cc();

  fetch_result_from_device(dq);
}

} // namespace motis