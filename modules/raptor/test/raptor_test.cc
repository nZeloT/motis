#include "raptor_test.h"

#include <stdexcept>
#include <unordered_map>

#include "motis/loader/loader.h"
#include "motis/raptor/get_raptor_schedule.h"

namespace motis::raptor {

cpu_raptor_test::cpu_raptor_test(loader::loader_options opts)
    : opts_{std::move(opts)}, rp_sched_{nullptr} {}

void cpu_raptor_test::SetUp() {
  sched_ = loader::load_schedule(opts_);
  manipulate_schedule();
  rp_sched_ = get_raptor_schedule(*sched_, false, "");
}

uint32_t cpu_raptor_test::get_raptor_r_id(const std::string& gtfs_trip_id) {
  auto& trip2route = rp_sched_->dbg_.trip_dbg_to_route_trips_;
  for(auto const& entry : trip2route) {
    if(entry.first.starts_with(gtfs_trip_id)){
      if(entry.second.size() > 1) {
        throw std::runtime_error {
            "No unique mapping between route ids and gtfs ids!"
        };
      }else{
        return entry.second.begin()->first;
      }
    }
  }
}

}  // namespace motis::raptor