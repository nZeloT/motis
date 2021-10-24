#include "gtest/gtest.h"

#include "raptor_test.h"

#include "motis/test/schedule/raptor_moc.h"

#include "motis/raptor-core/config/configs.h"
#include "motis/raptor-core/raptor_query.h"
#include "motis/raptor-core/raptor_timetable.h"
#include "motis/raptor/cpu/cpu_raptor.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_schedule.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/reconstructor.h"

#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"

using namespace motis;

namespace motis::raptor {

class mc_cpu_raptor_tests : public cpu_raptor_test {
public:
  mc_cpu_raptor_tests()
      : cpu_raptor_test(test::schedule::raptor_moc::dataset_opt){};

  void manipulate_schedule() override {
    // Method is called before the raptor schedule is generated
    manip_sched_for_tc1();
    manip_sched_for_tc2();
    manip_sched_for_tc3();
    manip_sched_for_tc4();
  }

  void manip_sched_for_tc1() {
    // Manipulation for test "force_trip_change_but_keep_route"
    auto trip_1 = get_trip(*sched_, "TC1-TR1", 1630281600L);
    auto t1_edges = trip_1->edges_;
    auto t1_lcon_idx = trip_1->lcon_idx_;
    t1_edges->at(0)->m_.route_edge_.conns_[t1_lcon_idx].occupancy_ = 0;  // A->B
    t1_edges->at(1)->m_.route_edge_.conns_[t1_lcon_idx].occupancy_ = 0;  // B->C
    t1_edges->at(2)->m_.route_edge_.conns_[t1_lcon_idx].occupancy_ = 1;  // C->D

    auto trip_2 = get_trip(*sched_, "TC1-TR2", 1630281600L);
    auto t2_edges = trip_2->edges_;
    auto t2_lcon_idx = trip_2->lcon_idx_;
    t2_edges->at(0)->m_.route_edge_.conns_[t2_lcon_idx].occupancy_ = 1;  // A->B
    t2_edges->at(1)->m_.route_edge_.conns_[t2_lcon_idx].occupancy_ = 0;  // B->C
    t2_edges->at(2)->m_.route_edge_.conns_[t2_lcon_idx].occupancy_ = 0;  // C->D
  }

  void manip_sched_for_tc2() {
    //Manipulation for test "find_result_for_all_occ_levels"
    auto trip1 = get_trip(*sched_, "TC2-TR1", 1630368000L);
    auto t1edges = trip1->edges_;
    auto t1lcidx = trip1->lcon_idx_;
    t1edges->at(0)->m_.route_edge_.conns_[t1lcidx].occupancy_ = 0;//A->B
    t1edges->at(1)->m_.route_edge_.conns_[t1lcidx].occupancy_ = 1;//B->C
    t1edges->at(2)->m_.route_edge_.conns_[t1lcidx].occupancy_ = 2;//C->D

    auto trip2 = get_trip(*sched_, "TC2-TR2", 1630368000L);
    auto t2edges = trip2->edges_;
    auto t2lcidx = trip2->lcon_idx_;
    t2edges->at(0)->m_.route_edge_.conns_[t2lcidx].occupancy_ = 0;//A->B
    t2edges->at(1)->m_.route_edge_.conns_[t2lcidx].occupancy_ = 0;//B->C
    t2edges->at(2)->m_.route_edge_.conns_[t2lcidx].occupancy_ = 1;//C->D

    auto trip3 = get_trip(*sched_, "TC2-TR3", 1630368000L);
    auto t3edges = trip3->edges_;
    auto t3lcidx = trip3->lcon_idx_;
    t3edges->at(0)->m_.route_edge_.conns_[t3lcidx].occupancy_ = 0;//A->B
    t3edges->at(1)->m_.route_edge_.conns_[t3lcidx].occupancy_ = 0;//B->C
    t3edges->at(2)->m_.route_edge_.conns_[t3lcidx].occupancy_ = 0;//C->D
  }

  void manip_sched_for_tc3() {
    //Manipulation for test "journey_w_max_occ_two_is_dominated_by_j_w_max_occ_one"
    auto trip1 = get_trip(*sched_, "TC3-1-TR1", 1630454400L);
    auto t1edges = trip1->edges_;
    auto t1lcidx = trip1->lcon_idx_;
    t1edges->at(0)->m_.route_edge_.conns_[t1lcidx].occupancy_ = 0;//A->B

    auto trip2 = get_trip(*sched_, "TC3-2-TR1", 1630454400L);
    auto t2edges = trip2->edges_;
    auto t2lcidx = trip2->lcon_idx_;
    t2edges->at(0)->m_.route_edge_.conns_[t2lcidx].occupancy_ = 2;//A->C

    auto trip3 = get_trip(*sched_, "TC3-3-TR1", 1630454400L);
    auto t3edges = trip3->edges_;
    auto t3lcidx = trip3->lcon_idx_;
    t3edges->at(0)->m_.route_edge_.conns_[t3lcidx].occupancy_ = 0;//B->C
    t3edges->at(1)->m_.route_edge_.conns_[t3lcidx].occupancy_ = 0;//C->D
  }

  void manip_sched_for_tc4() {
    //Manipulation for test " force_later_departure_stop_when_earlier_is_available"
    auto trip1 = get_trip(*sched_, "TC4-1-TR1", 1629936000L);
    auto t1edges = trip1->edges_;
    auto t1lcidx = trip1->lcon_idx_;
    t1edges->at(0)->m_.route_edge_.conns_[t1lcidx].occupancy_ = 0;//A->B

    auto trip2 = get_trip(*sched_, "TC4-2-TR1", 1629936000L);
    auto t2edges = trip2->edges_;
    auto t2lcidx = trip2->lcon_idx_;
    t2edges->at(0)->m_.route_edge_.conns_[t2lcidx].occupancy_ = 0;//A->C

    auto trip3 = get_trip(*sched_, "TC4-3-TR1", 1629936000L);
    auto t3edges = trip3->edges_;
    auto t3lcidx = trip3->lcon_idx_;
    t3edges->at(0)->m_.route_edge_.conns_[t3lcidx].occupancy_ = 1;//B->C
    t3edges->at(1)->m_.route_edge_.conns_[t3lcidx].occupancy_ = 0;//C->D
  }

};

// Test uses the following schedule (occupancy on trip arrows)
// Stops:   A  --->  B  --->  C  --->  D
// Trip 0:  0' -0-> 10' -0-> 20' -1-> 30'
// Trip 1: 10' -1-> 20' -0-> 30' -0-> 40'
//
// Expectation: Two Journeys. One with Max Occupancy = 1 (just using Trip 0)
//              and one with Max Occupancy = 0 (using Trip 0 and 1).
//              Just using Trip 1 for the max occ. = 0 journey is prevented by
//              the first leg occupancy of 1.
TEST_F(mc_cpu_raptor_tests, tc1_force_trip_change_but_keep_route) {
  auto& sched = *this->rp_sched_;
  auto& tt = this->rp_sched_->timetable_;

  // manipulate the occupancy in the timetable so that it fits the test
  auto const route = tt.routes_[10];
  auto const trip_0_first_sti = route.index_to_stop_times_;
  auto const trip_1_first_sti = trip_0_first_sti + route.stop_count_;

  // make sure the schedule manipulation went as expected
  EXPECT_EQ(4, route.stop_count_);  // little safety net
  EXPECT_EQ(1, tt.stop_occupancies_[trip_0_first_sti + 3].inbound_occupancy_);
  EXPECT_EQ(0, tt.stop_occupancies_[trip_0_first_sti + 2].inbound_occupancy_);
  EXPECT_EQ(0, tt.stop_occupancies_[trip_0_first_sti + 1].inbound_occupancy_);

  EXPECT_EQ(0, tt.stop_occupancies_[trip_1_first_sti + 3].inbound_occupancy_);
  EXPECT_EQ(0, tt.stop_occupancies_[trip_1_first_sti + 2].inbound_occupancy_);
  EXPECT_EQ(1, tt.stop_occupancies_[trip_1_first_sti + 1].inbound_occupancy_);
  // end manipulate timetable

  auto const start_sti = tt.stop_times_[trip_0_first_sti];

  base_query bq{};

  bq.source_time_begin_ = start_sti.departure_;
  bq.source_time_end_ = bq.source_time_begin_;
  bq.source_ = sched.eva_to_raptor_id_.at("A1");
  bq.target_ = sched.eva_to_raptor_id_.at("D1");

  raptor_query q{bq, tt, OccupancyOnly::trait_size()};

  raptor_statistics st;

  invoke_cpu_raptor<OccupancyOnly>(q, st, sched);
  reconstructor<OccupancyOnly> rc{bq, *sched_, sched};
  rc.add(bq.source_time_begin_, *q.result_);
  auto const journeys = rc.get_journeys();

  auto const last_s_id = tt.route_stops_[route.index_to_route_stops_ + 3];
  auto const last_stop_tt = sched.transfer_times_[last_s_id];

  // Start checking expectation
  EXPECT_EQ(2, journeys.size());

  // Check first journey with Max Occupancy = 0
  auto const j0 = journeys[0];
  EXPECT_EQ(1, j0.occupancy_max_);
  EXPECT_EQ(1, j0.trips_.size());
  EXPECT_EQ(4, j0.stops_.size());
  auto const stop_time_d1_0 = tt.stop_times_[trip_0_first_sti + 3];
  auto const j0_arrival = motis_to_unixtime(
      sched_->schedule_begin_, stop_time_d1_0.arrival_ - last_stop_tt);
  EXPECT_EQ(j0_arrival, j0.stops_[3].arrival_.timestamp_);

  // Check second journey
  auto const j1 = journeys[1];
  EXPECT_EQ(0, j1.occupancy_max_);
  EXPECT_EQ(2, j1.trips_.size());
  EXPECT_EQ(4, j1.stops_.size());

  // to get arrival time we need to substract the last station transfer time
  auto const stop_time_d1_1 = tt.stop_times_[trip_1_first_sti + 3];
  auto const j1_arrival = motis_to_unixtime(
      sched_->schedule_begin_, stop_time_d1_1.arrival_ - last_stop_tt);
  EXPECT_EQ(j1_arrival, j1.stops_[3].arrival_.timestamp_);
}

TEST_F(mc_cpu_raptor_tests, tc2_find_result_for_all_occ_levels) {

}

TEST_F(mc_cpu_raptor_tests, tc3_journey_w_max_occ_two_is_dominated_by_j_w_max_occ_one) {

}

TEST_F(mc_cpu_raptor_tests, tc4_force_later_departure_stop_when_earlier_is_available) {

}


}  // namespace motis::raptor