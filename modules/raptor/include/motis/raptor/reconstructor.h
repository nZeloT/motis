#pragma once

#include <algorithm>
#include <utility>

#include <cstdio>
#include <iomanip>
#include <iostream>

#include "motis/core/journey/journey_util.h"

#include "motis/raptor-core/raptor_query.h"
#include "motis/raptor-core/raptor_timetable.h"

#include "motis/raptor/raptor_util.h"

#include "motis/routing/output/stop.h"
#include "motis/routing/output/to_journey.h"
#include "motis/routing/output/transport.h"

namespace motis {
namespace raptor {

using namespace motis::routing::output;

template <typename Config>
struct reconstructor {
  using TraitsData = typename Config::TraitsData;

  struct candidate {
    candidate() = delete;
    candidate(motis::time const dep, motis::time const arr, transfers const t,
              int const trait_offset, TraitsData trait_data,
              bool const ends_with_footpath)
        : departure_(dep),
          arrival_(arr),
          transfers_(t),
          trait_offset_{trait_offset},
          trait_data_{std::move(trait_data)},
          ends_with_footpath_(ends_with_footpath) {}
    motis::time departure_;
    motis::time arrival_;
    uint8_t transfers_;
    bool ends_with_footpath_;

    int trait_offset_;
    TraitsData trait_data_;
  };

  std::string to_string(candidate const& c) {
    return "Dep: " + std::to_string(c.departure_) +
           " Arr: " + std::to_string(c.arrival_) +
           " Transfers: " + std::to_string(c.transfers_);
  }

  reconstructor() = delete;
  reconstructor(base_query const& q, schedule const& sched,
                raptor_schedule const& raptor_sched)
      : source_(q.source_),
        target_(q.target_),
        sched_(sched),
        raptor_sched_(raptor_sched),
        timetable_(raptor_sched.timetable_) {}

  bool dominates(journey const& j, candidate const& c) {
    auto const motis_arrival =
        unix_to_motistime(sched_.schedule_begin_, get_arrival(j));

#ifdef _DEBUG
    auto const transfers = get_transfers(j);
    auto dom_time = motis_arrival <= c.arrival_;
    auto dom_tran = transfers <= c.transfers_;
    auto dom_trat = Config::dominates(j, c);

    auto dominates = dom_time && dom_tran && dom_trat;

    if (dominates) {
      std::cout << "Journey: arr: " << motis_arrival << "; tr: " << transfers
                << "; moc: " << j.occupancy_max_
                << "; dominates Candidate arr: " << c.arrival_
                << "; tr: " << +c.transfers_ << "; moc: " << c.trait_offset_
                << std::endl;
    }

    return dominates;
#else
    return (motis_arrival <= c.arrival_ && get_transfers(j) <= c.transfers_) &&
           Config::dominates(j, c);
#endif
  }

  void add(motis::time const departure, raptor_result const& result) {
    auto const trait_size = Config::trait_size();

    for (raptor_round round_k = 1; round_k < max_round_k; ++round_k) {

      // also go through all trait dimensions to check for viable solutions
      for (int t_offset = 0; t_offset < trait_size; ++t_offset) {
        auto const arrival_idx = Config::get_arrival_idx(target_, t_offset);
        if (!valid(result[round_k][arrival_idx])) {
          continue;
        }

        auto const tt = raptor_sched_.transfer_times_[target_];

        auto trait_vals = Config::get_traits_data(t_offset);
        candidate c(departure, result[round_k][arrival_idx], round_k - 1,
                    t_offset, std::move(trait_vals), true);
        for (; c.arrival_ < result[round_k][arrival_idx] + tt; c.arrival_++) {
          c.ends_with_footpath_ = journey_ends_with_footpath(c, result);
          if (!c.ends_with_footpath_) {
            break;
          }
        }

        c.arrival_ -= tt;

        // Candidate has to arrive earlier than any already reconstructed
        // journey
        bool const dominated =
            std::any_of(std::begin(journeys_), std::end(journeys_),
                        [&](auto const& j) -> bool { return dominates(j, c); });

        // Candidate is dominated by already reconstructed journey
        if (dominated) {
          continue;
        }

        if (!c.ends_with_footpath_) {
          c.arrival_ += tt;
        }

        auto const j = reconstruct_journey(c, result);
        if (j.duration_ <= 86400) {
          journeys_.push_back(j);
        }
        // else{
        //   std::cout << "Filtered journey with duration " << +j.duration_ <<
        //   ";\tTrips: " << +j.trips_.size() << ";\tMoc: " << j.occupancy_max_
        //   << std::endl;
        // }
      }
    }

    // TODO check if there is a need to prune dominated journeys
    //      this could happen if an already known journey becomes dominated
    //      by a candidate
    //      Though, this should only be necessary if one of the used criteria
    //      is does not fulfill the criteria that A dominates B iff a_c < b_c
    //      e.g. if a criteria uses inverse ordering or similar
  }

  auto get_journeys() { return journeys_; }

  auto get_journeys(motis::time const end) {
    erase_if(journeys_, [&](auto const& j) -> bool {
      return unix_to_motistime(sched_.schedule_begin_, get_departure(j)) > end;
    });
    return journeys_;
  }

  struct intermediate_journey {
    intermediate_journey(transfers const trs) : transfers_(trs) {}

    void add_footpath(station_id const to, motis::time const a_time,
                      motis::time const d_time, motis::time const duration,
                      raptor_schedule const& raptor_sched) {
      auto const motis_index = raptor_sched.station_id_to_index_[to];
      stops_.emplace_back(stops_.size(), motis_index, 0, 0, a_time, d_time,
                          a_time, d_time, timestamp_reason::SCHEDULE,
                          timestamp_reason::SCHEDULE, false, false);

      transports_.emplace_back(stops_.size() - 1, stops_.size(), duration, 0, 0,
                               0);
    }

    motis::time add_route(station_id const from, route_id const r_id,
                          trip_id const trip, stop_offset const exit_offset,
                          raptor_schedule const& raptor_sched) {
      auto const& tt = raptor_sched.timetable_;
      auto const& route = tt.routes_[r_id];

      auto const sti_base =
          route.index_to_stop_times_ + (trip * route.stop_count_);

      // Add the stops in backwards fashion, reverse the stop vector at the end
      for (int16_t s_offset = static_cast<int16_t>(exit_offset); s_offset >= 0;
           --s_offset) {
        auto const rsi = route.index_to_route_stops_ + s_offset;
        auto const s_id = tt.route_stops_[rsi];

        auto const sti = sti_base + s_offset;
        auto const stop_time = tt.stop_times_[sti];

        auto const d_time = stop_time.departure_;
        auto const tt = raptor_sched.transfer_times_[s_id];
        auto const a_time = stop_time.arrival_ - tt;

        if (s_id == from && d_time != 0) {
          return d_time;
        }

        auto const motis_index = raptor_sched.station_id_to_index_[s_id];

        stops_.emplace_back(stops_.size(), motis_index, 0, 0, a_time, d_time,
                            a_time, d_time, timestamp_reason::SCHEDULE,
                            timestamp_reason::SCHEDULE, false, false);

        auto const lcon = raptor_sched.lcon_ptr_[sti];
        transports_.emplace_back(stops_.size() - 1, stops_.size(), lcon);
      }

      LOG(motis::logging::warn)
          << "Could not correctly reconstruct RAPTOR journey";
      return invalid<motis::time>;
    }

    void add_start_station(station_id const start,
                           raptor_schedule const& raptor_sched,
                           motis::time const d_time) {
      auto const motis_index = raptor_sched.station_id_to_index_[start];
      stops_.emplace_back(stops_.size(), motis_index, 0, 0, INVALID_TIME,
                          d_time, INVALID_TIME, d_time,
                          timestamp_reason::SCHEDULE,
                          timestamp_reason::SCHEDULE, false, false);
    }

    journey to_journey(schedule const& sched) {
      journey j;

      std::reverse(std::begin(transports_), std::end(transports_));
      unsigned idx = 0;
      for (auto& t : transports_) {
        t.from_ = idx;
        t.to_ = ++idx;
      }

      j.transports_ = generate_journey_transports(transports_, sched);
      j.trips_ = generate_journey_trips(transports_, sched);
      j.attributes_ = generate_journey_attributes(transports_);

      std::reverse(std::begin(stops_), std::end(stops_));

      // HACK enter and exit flags TODO(julian)
      for (auto ts = 0; ts < transfers_ + 1; ++ts) {
        stops_[ts].enter_ = true;
        stops_[ts].exit_ = true;
      }

      j.stops_ = generate_journey_stops(stops_, sched);
      if (!j.stops_.empty()) {
        j.duration_ = j.stops_[j.stops_.size() - 1].arrival_.timestamp_ -
                      j.stops_[0].departure_.timestamp_;
      } else {
        j.duration_ = 0;
      }
      j.transfers_ = transfers_;
      j.db_costs_ = 0;
      j.price_ = 0;
      j.night_penalty_ = 0;
      j.occupancy_max_ =
          std::max_element(std::begin(transports_), std::end(transports_),
                           [](auto const& a, auto const& b) -> bool {
                             if (a.con_ == nullptr) return true;
                             if (b.con_ == nullptr) return false;
                             return (a.con_->occupancy_ < b.con_->occupancy_);
                           })
              ->con_->occupancy_;
      return j;
    }

    transfers transfers_;
    std::vector<intermediate::stop> stops_;
    std::vector<intermediate::transport> transports_;
  };

  bool journey_ends_with_footpath(candidate const c,
                                  raptor_result const& result) {
    auto const tuple =
        get_previous_station(target_, c.arrival_, c.transfers_ + 1,
                             c.trait_data_, c.trait_offset_, result);
    return !valid(std::get<0>(tuple));
  }

  journey reconstruct_journey(candidate const c, raptor_result const& result) {
    intermediate_journey ij(c.transfers_);

#ifdef _DEBUG
    std::cout << "Reconstructing new journey with " << +c.transfers_
              << " transfers and trait offset " << +c.trait_offset_
              << std::endl;
    std::cout << "---------------------------------------------------------"
              << std::endl;
#endif

    auto arrival_station = target_;
    auto last_departure = invalid<motis::time>;
    auto station_arrival = c.arrival_;
    for (auto result_idx = c.transfers_ + 1; result_idx > 0; --result_idx) {

      auto [previous_station, used_route, used_trip, stop_offset] =
          get_previous_station(arrival_station, station_arrival, result_idx,
                               c.trait_data_, c.trait_offset_, result);

      if (!valid(previous_station)) {

        for (auto const& inc_f :
             raptor_sched_.incoming_footpaths_[arrival_station]) {

          auto const adjusted_arrival = station_arrival - inc_f.duration_;
          std::tie(previous_station, used_route, used_trip, stop_offset) =
              get_previous_station(inc_f.from_, adjusted_arrival, result_idx,
                                   c.trait_data_, c.trait_offset_, result);

          if (valid(previous_station)) {
#ifdef _DEBUG
            std::cout
                << "res_idx: " << result_idx << ";\tTook Footpath"
                << ";\t\tFrom " << std::setw(6) << inc_f.from_
                << ";\tArriving at " << std::setw(6) << adjusted_arrival << " ("
                << motis_to_unixtime(sched_.schedule_begin_, adjusted_arrival)
                << ")"
                << ";\tTo " << std::setw(6) << arrival_station
                << ";\tArriving at " << std::setw(6) << station_arrival << " ("
                << motis_to_unixtime(sched_.schedule_begin_, station_arrival)
                << ")" << std::endl;
#endif

            ij.add_footpath(arrival_station, station_arrival, last_departure,
                            inc_f.duration_, raptor_sched_);
            last_departure =
                ij.add_route(previous_station, used_route, used_trip,
                             stop_offset, raptor_sched_);
            break;
          }
        }
      } else {
        last_departure = ij.add_route(previous_station, used_route, used_trip,
                                      stop_offset, raptor_sched_);
      }

#ifdef _DEBUG
      std::cout << "res_idx: " << result_idx << ";\tTook route " << std::setw(6)
                << used_route << ";\tFrom " << std::setw(6) << previous_station;
#endif

      auto const to_station = arrival_station;
      arrival_station = previous_station;
      auto const arr_idx =
          Config::get_arrival_idx(arrival_station, c.trait_offset_);
      auto const old_station_arrival = station_arrival;
      station_arrival = result[result_idx - 1][arr_idx];

#ifdef _DEBUG
      std::cout << ";\tArriving at " << std::setw(6) << station_arrival << " ("
                << motis_to_unixtime(sched_.schedule_begin_, station_arrival)
                << ")"
                << ";\tTo " << std::setw(6) << to_station << ";\tArriving at "
                << std::setw(6) << old_station_arrival << " ("
                << motis_to_unixtime(sched_.schedule_begin_,
                                     old_station_arrival)
                << ")"
                << ";\tTrip Id " << std::setw(3) << used_trip << std::endl;
#endif
    }

    // Add last footpath if necessary
    if (arrival_station != source_) {
      for (auto const& inc_f :
           raptor_sched_.incoming_footpaths_[arrival_station]) {
        if (inc_f.from_ != source_) {
          continue;
        }
        ij.add_footpath(arrival_station, last_departure, last_departure,
                        inc_f.duration_, raptor_sched_);

        motis::time const first_footpath_duration =
            inc_f.duration_ + raptor_sched_.transfer_times_[source_];
        ij.add_start_station(source_, raptor_sched_,
                             last_departure - first_footpath_duration);
        break;
      }
    } else {
      ij.add_start_station(source_, raptor_sched_, last_departure);
    }

    return ij.to_journey(sched_);
  }

  std::tuple<station_id, route_id, trip_id, stop_offset> get_previous_station(
      station_id const arrival_station, motis::time const stop_arrival,
      uint8_t const result_idx, TraitsData const& traits_data,
      uint32_t trait_offset, raptor_result const& result) {
    auto const arrival_stop = timetable_.stops_[arrival_station];

    auto const route_count = arrival_stop.route_count_;
    for (auto sri = arrival_stop.index_to_stop_routes_;
         sri < arrival_stop.index_to_stop_routes_ + route_count; ++sri) {

      auto const r_id = timetable_.stop_routes_[sri];
      auto const& route = timetable_.routes_[r_id];

      for (stop_offset offset = 1; offset < route.stop_count_; ++offset) {
        auto const rsi = route.index_to_route_stops_ + offset;
        auto const s_id = timetable_.route_stops_[rsi];
        if (s_id != arrival_station) {
          continue;
        }

        auto const arrival_trip =
            get_arrival_trip_at_station(r_id, stop_arrival, offset);

        if (!valid(arrival_trip)) {
          continue;
        }

        auto const board_station = get_board_station_for_trip(
            r_id, arrival_trip, result, result_idx - 1, trait_offset,
            traits_data, offset);

        if (valid(board_station)) {
          return {board_station, r_id, arrival_trip, offset};
        }
      }
    }

    return {invalid<station_id>, invalid<route_id>, invalid<trip_id>,
            invalid<stop_offset>};
  }

  trip_id get_arrival_trip_at_station(route_id const r_id,
                                      motis::time const arrival,
                                      stop_offset const offset) {
    auto const& route = timetable_.routes_[r_id];

    for (auto trip = 0; trip < route.trip_count_; ++trip) {
      auto const sti =
          route.index_to_stop_times_ + (trip * route.stop_count_) + offset;
      if (timetable_.stop_times_[sti].arrival_ == arrival) {
        return trip;
      }
    }

    return invalid<trip_id>;
  }

  station_id get_board_station_for_trip(route_id const r_id, trip_id const t_id,
                                        raptor_result const& result,
                                        raptor_round const result_idx,
                                        int const trait_offset,
                                        TraitsData const& trait_data,
                                        stop_offset const arrival_offset) {
    auto const& r = timetable_.routes_[r_id];

    auto const first_stop_times_index =
        r.index_to_stop_times_ + (t_id * r.stop_count_);

    // -1, since we cannot board a trip at the last station
    auto const max_offset =
        std::min(static_cast<stop_offset>(r.stop_count_ - 1), arrival_offset);
    for (auto stop_offset = 0; stop_offset < max_offset; ++stop_offset) {
      auto const rsi = r.index_to_route_stops_ + stop_offset;
      auto const station_id = timetable_.route_stops_[rsi];

      auto const sti = first_stop_times_index + stop_offset;
      auto const departure = timetable_.stop_times_[sti].departure_;

      auto const arr_idx = Config::get_arrival_idx(station_id, trait_offset);
      if (result[result_idx][arr_idx] <= departure &&
          Config::trip_matches_traits(trait_data, timetable_, r_id, t_id,
                                      stop_offset, arrival_offset)) {
        return station_id;
      }
    }

    return invalid<station_id>;
  }

  station_id const source_;
  station_id const target_;

  schedule const& sched_;
  raptor_schedule const& raptor_sched_;
  raptor_timetable const& timetable_;

  std::vector<journey> journeys_;
};

}  // namespace raptor
}  // namespace motis