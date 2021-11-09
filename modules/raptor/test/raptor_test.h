#pragma once

#include "gtest/gtest.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"
#include "motis/loader/loader_options.h"



namespace motis::raptor {
struct raptor_schedule;
struct base_query;

class cpu_raptor_test : public ::testing::Test {
protected:
  explicit cpu_raptor_test(loader::loader_options);

  void SetUp() override;
  virtual void manipulate_schedule() = 0;
  virtual void check_mock_on_rp_sched() = 0;

  uint32_t get_raptor_r_id(std::string const&);

  std::vector<journey> execute_raptor(time const departure, std::string const& eva_from, std::string const& eva_to);
  std::vector<journey> execute_routing(time const departure, std::string const& eva_from, std::string const& eva_to);

  schedule_ptr sched_;
  loader::loader_options opts_;
  std::unique_ptr<raptor_schedule> rp_sched_;
};

}