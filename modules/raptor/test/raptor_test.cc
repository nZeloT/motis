#include "raptor_test.h"

#include <stdexcept>

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

}  // namespace motis::raptor