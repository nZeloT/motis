namespace motis {

void invoke_hybrid_raptor(d_query& dq) {
  auto const& transfer_stream = dq.transfer_stream_;
  auto const& proc_stream = dq.proc_stream_;

  void* init_args[] = { (void*) &dq };
  launch_coop_kernel(init_arrivals_kernel, init_args, proc_stream);
  cudaStreamSynchronize(proc_stream);                                cc();

  fetch_arrivals_async(dq, 0, transfer_stream);

  for (int k = 1; k < max_round_k; ++k) {
    void* kernel_args[] = { (void*) &dq, (void*) &k };

    launch_coop_kernel(update_routes_kernel, kernel_args, proc_stream);

    cudaStreamSynchronize(proc_stream);                              cc();

    launch_coop_kernel(update_footpaths_kernel, kernel_args, proc_stream);
    cudaStreamSynchronize(proc_stream);                              cc();

    bool any_station_marked = true;
    cudaMemcpyFromSymbol(&any_station_marked, ANY_STATION_MARKED,
                         sizeof(any_station_marked), 0, cudaMemcpyDeviceToHost);
    if (!any_station_marked) { cudaDeviceSynchronize(); return; }

    fetch_arrivals_async(dq, k, transfer_stream);
  }

  cudaDeviceSynchronize();                                           cc();
}

} // namespace motis