#pragma once

namespace motis::raptor{

template <typename... Filter>
struct filter;

template <typename FirstFilter, typename... RestFilters>
struct filter<FirstFilter, RestFilters...> {

  template <typename Timetable>
  static bool is_filtered(Timetable const& tt, int stop_time_idx) {
    return FirstFilter::is_filtered(tt, stop_time_idx) ||
           filter<RestFilters...>::is_filtered(tt, stop_time_idx);
  }

};

template <>
struct filter<> {
  template <typename Timetable>
  static bool is_filtered(Timetable const&) {
    return false;
  }
};
};