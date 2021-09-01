namespace motis {

__device__
void update_single_cluster_dev(global_mem_time const * const prev_arrivals,
                               global_mem_time* const arrivals,
                               motis::cluster const cluster,
                               unsigned int* station_marks) {

  __shared__ motis::time shared_arrivals[8 * 1024];
  __shared__ motis::time shared_prev_arrivals[8 * 1024];

  __syncthreads();

  auto const block_stride = get_block_stride();

  // Copy arrival values from prev arrival (global mem) to shared arrivals
  auto arr_idx = get_block_thread_id();
  for (; arr_idx < cluster.border_station_count_; arr_idx += block_stride) {
    auto const bm_idx = cluster.border_mapping_index_ + arr_idx;
    shared_arrivals[arr_idx] = prev_arrivals[GTT.border_mappings_[bm_idx]];
    shared_prev_arrivals[arr_idx] = prev_arrivals[GTT.border_mappings_[bm_idx]];
  } 
  
  arr_idx = get_block_thread_id();
  for (; arr_idx < cluster.inland_station_count_; arr_idx += block_stride) {
    auto const sa_idx = cluster.border_station_count_ + arr_idx;
    auto const pa_idx = cluster.arrivals_start_index_ + arr_idx;
    shared_arrivals[sa_idx] = prev_arrivals[pa_idx];
    shared_prev_arrivals[sa_idx] = prev_arrivals[pa_idx];
  } 
  __syncthreads();


  // Update all routes for the cluster
  // auto stride = blockDim.y * gridDim.x;
  auto stride = block_dim_y;
  auto route_offset = threadIdx.y;
  for (route_id r_id = cluster.route_start_index_ + route_offset; 
                r_id < cluster.route_start_index_ + cluster.route_count_;
                r_id += stride) {
    auto const route = GTT.routes_[r_id];
    if (route.stop_count_ <= 32) {
      update_route_smaller32_cls_split(route, shared_prev_arrivals, shared_arrivals, station_marks);
    } else {
      update_route_larger32_cls_split(route, shared_prev_arrivals, shared_arrivals, station_marks);
    }
  } 
  
  __syncthreads();

  // TODO check if update value valid before push to global
  // check if any update value from global mem is valid before updating routes 

  // Copy arrival values from shared arrivals to arrivals (global mem)
  // arr_idx = get_global_thread_id();
  arr_idx = get_block_thread_id();
  for (; arr_idx < cluster.border_station_count_; arr_idx += block_stride) {
    auto const bm_idx = cluster.border_mapping_index_ + arr_idx;
    if (valid(shared_arrivals[arr_idx])) {
      update_arrival(arrivals, GTT.border_mappings_[bm_idx], 
                              shared_arrivals[arr_idx]);
    }
  } 
  
  arr_idx = get_block_thread_id();
  for (; arr_idx < cluster.inland_station_count_; arr_idx += block_stride) {
    auto const sa_idx = cluster.border_station_count_ + arr_idx;
    auto const pa_idx = cluster.arrivals_start_index_ + arr_idx;
    if (valid(shared_arrivals[sa_idx])) {
      arrivals[pa_idx] = shared_arrivals[sa_idx];
    }
  } 
}

__device__
void update_cluster_dev(global_mem_time const * const prev_arrivals,
                        global_mem_time* const arrivals,
                        unsigned int* station_marks,
                        unsigned int* route_marks) {

  auto const cluster_stride = num_blocks;
  for (cls_id c_id = blockIdx.x; 
              c_id < GTT.cluster_count_; 
              c_id += cluster_stride) {
    // auto const cluster = GTT.clusters_[c_id];
    update_single_cluster_dev(prev_arrivals, arrivals, GTT.clusters_[c_id],
                              station_marks);
  } 
}

__global__
__launch_bounds__((block_dim_x * block_dim_y), min_blocks_per_sm)
void update_cluster_kernel(d_query const dq, int round_k) {
  global_mem_time const * const prev_arrivals = dq.d_arrivals_[round_k - 1];
  global_mem_time* const arrivals = dq.d_arrivals_[round_k];

  update_cluster_dev(prev_arrivals, arrivals, 
                     dq.station_marks_, dq.route_marks_);
}

__global__
__launch_bounds__((block_dim_x * block_dim_y), min_blocks_per_sm)
void cluster_raptor_kernel(d_query const dq) {
  init_arrivals_dev(dq);
  this_grid().sync();

  for (int8_t round_k = 1; round_k < max_round_k; ++round_k) {
    global_mem_time const * const prev_arrivals = dq.d_arrivals_[round_k - 1];
    global_mem_time* const arrivals = dq.d_arrivals_[round_k];
    global_mem_time* const next_arrivals = dq.d_arrivals_[round_k + 1];

    update_cluster_dev(prev_arrivals, arrivals, dq.station_marks_, dq.route_marks_);
    this_grid().sync();

    update_footpaths_dev(dq, round_k);
  }
}

void invoke_cluster_raptor(d_query& dq) {
  auto& result = *dq.result_;

  cudaStream_t transfer_stream;
  cudaStream_t proc_stream;

  auto transfer_result = cudaStreamCreate(&transfer_stream);         cc();
  auto processing_result = cudaStreamCreate(&proc_stream);           cc();

  init_arrivals_kernel<<<nb, tpb, 0, proc_stream>>>(dq);             cc();
  cudaStreamSynchronize(proc_stream);                                cc();

  cudaMemcpyAsync(result[0],
                  dq.d_arrivals_[0],
                  dq.stop_count_ * sizeof(global_mem_time),
                  cudaMemcpyDeviceToHost,
                  transfer_stream);                                  cc();

  // init_arrivals_kernel<<<num_blocks, threads_per_block>>>(dq);          cc();
  // cudaDeviceSynchronize();                                              cc();

  for (int k = 1; k < max_round_k; ++k) {
    update_cluster_kernel<<<nb, tpb, 0, proc_stream>>>(dq,k);        cc();
    cudaStreamSynchronize(proc_stream);                                cc();

    // update_cluster_kernel<<<num_blocks, threads_per_block>>>(dq,k);        cc();
    // cudaDeviceSynchronize();                                               cc();

    void* kernel_args[] = { (void*) &dq, (void*) &k };
    cudaLaunchCooperativeKernel((void*) update_footpaths_kernel,
                                grid_dim,
                                threads_per_block,
                                kernel_args,
                                0, 
                                proc_stream);                        cc();
    cudaStreamSynchronize(proc_stream);                              cc();

    cudaMemcpyAsync(result[k], 
                    dq.d_arrivals_[k],
                    dq.stop_count_ * sizeof(global_mem_time),
                    cudaMemcpyDeviceToHost,
                    transfer_stream);                                cc();
    // cudaDeviceSynchronize();                                               cc();
    // return;
  }

  cudaDeviceSynchronize();                                           cc();
  return;

  // #if __CUDA_ARCH__ >= 610
  //   void* kernel_args[] = { (void*) &dq };
  //   cudaLaunchCooperativeKernel((void*) cluster_raptor_kernel,
  //                               grid_dim,
  //                               threads_per_block,
  //                               kernel_args);                       cc();

  // #else
  //   cluster_raptor_kernel<<<num_blocks, threads_per_block>>>(dq);       cc();
  // #endif
   
  // cudaDeviceSynchronize();                                          cc();
  // return fetch_result_from_device(dq.d_arrivals_, dq.stop_count_);

  // fetch_result_from_device(dq);
}

} // namespace motis