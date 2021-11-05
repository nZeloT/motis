#pragma once

#include "gtest/gtest.h"

#include "motis/core/schedule/schedule.h"
#include "motis/loader/loader_options.h"

namespace motis::raptor {
struct raptor_schedule;

class cpu_raptor_test : public ::testing::Test {
protected:
  explicit cpu_raptor_test(loader::loader_options);

  void SetUp() override;
  virtual void manipulate_schedule() = 0;

  uint32_t get_raptor_r_id(std::string const& gtfs_trip_id);

  schedule_ptr sched_;
  loader::loader_options opts_;
  std::unique_ptr<raptor_schedule> rp_sched_;
};

}