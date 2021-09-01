#include "motis/raptor-core/raptor_util.h"

namespace motis {

size_t copied_bytes = 0;

template <typename T>
inline void copy_vector_to_device(std::vector<T> const& vec, T** ptr) {

  const auto size_in_bytes = vec_size_bytes(vec);
  cudaMalloc(ptr, size_in_bytes);             cc();
  cudaMemcpy(*ptr,
            vec.data(),
            size_in_bytes,
            cudaMemcpyHostToDevice);         cc();
  copied_bytes += size_in_bytes;
}

device_gpu_timetable copy_timetable_to_device(host_gpu_timetable const& h_gtt) {
  device_gpu_timetable d_gtt;

  copy_vector_to_device(h_gtt.stops_, &(d_gtt.stops_));
  copy_vector_to_device(h_gtt.routes_, &(d_gtt.routes_));

  copy_vector_to_device(h_gtt.footpaths_, &(d_gtt.footpaths_));

  copy_vector_to_device(h_gtt.stop_times_, &(d_gtt.stop_times_));
  copy_vector_to_device(h_gtt.stop_arrivals_, &(d_gtt.stop_arrivals_));
  copy_vector_to_device(h_gtt.stop_departures_, &(d_gtt.stop_departures_));

  copy_vector_to_device(h_gtt.route_stops_, &(d_gtt.route_stops_));
  copy_vector_to_device(h_gtt.stop_routes_, &(d_gtt.stop_routes_));

  d_gtt.stop_count_ = h_gtt.stop_count();
  d_gtt.route_count_ = h_gtt.route_count();
  d_gtt.footpath_count_ = h_gtt.footpaths_.size();

  copy_vector_to_device(h_gtt.initialization_footpaths_indices_,
                        &(d_gtt.initialization_footpaths_indices_));
  copy_vector_to_device(h_gtt.initialization_footpaths_, &(d_gtt.initialization_footpaths_));

  copy_vector_to_device(h_gtt.transfer_times_, &(d_gtt.transfer_times_));

  copy_vector_to_device(h_gtt.clusters_, &(d_gtt.clusters_));
  d_gtt.cluster_count_ = h_gtt.clusters_.size();
  copy_vector_to_device(h_gtt.clustered_route_stops_,
                                            &(d_gtt.clustered_route_stops_));
  copy_vector_to_device(h_gtt.border_mappings_, &(d_gtt.border_mappings_));

  cudaMemcpyToSymbol(GTT,
                     &d_gtt,
                     sizeof(device_gpu_timetable),
                     0,
                     cudaMemcpyHostToDevice);             cc();

//  LOG(info) << "Finished copying RAPTOR timetable to device";
//  LOG(info) << "Copied " << ((double) copied_bytes) / (1024 * 1024)
//            << " mibi bytes";

  return d_gtt;
}

void free_timetable_on_device(device_gpu_timetable const& d_gtt) {
  cuda_free(d_gtt.stops_);
  cuda_free(d_gtt.routes_);
  cuda_free(d_gtt.footpaths_);
  cuda_free(d_gtt.transfer_times_);
  cuda_free(d_gtt.stop_times_);
  cuda_free(d_gtt.stop_arrivals_);
  cuda_free(d_gtt.stop_departures_);
  cuda_free(d_gtt.route_stops_);
  cuda_free(d_gtt.stop_routes_);
  cuda_free(d_gtt.initialization_footpaths_indices_);
  cuda_free(d_gtt.initialization_footpaths_);
  cuda_free(d_gtt.clusters_);
  cuda_free(d_gtt.clustered_route_stops_);
  cuda_free(d_gtt.border_mappings_);
}

} // namespace motis