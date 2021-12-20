#pragma once

namespace motis {
namespace routing {

template <int Index, typename... Selectors>
struct selector {

  template <typename Label>
  static bool create_needed(std::tuple<Selectors...>& t,
                            light_connection const* c, Label& l) {
    bool create = std::get<Index>(t).create_needed(c, l);
    return create | selector<Index - 1, Selectors...>::create_needed(t, c, l);
  }

  static bool continue_search(std::tuple<Selectors...>& t) {

    bool cont = std::get<Index>(t).continue_search();
    return cont | selector<Index - 1, Selectors...>::continue_search(t);
  }

  template <typename Label, typename Edge>
  static bool minimum_found(std::tuple<Selectors...>& t,
                            light_connection const* c, Label& l,
                            Edge const& e) {
    return std::get<Index>(t).minimum_found(c, l, e) &&
           selector<Index - 1, Selectors...>::minimum_found(t, c, l, e);
  }
};

template <typename... Selectors>
struct selector<0, Selectors...> {
  template <typename Label>
  static bool create_needed(std::tuple<Selectors...>& t,
                            light_connection const* c, Label& l) {
    return std::get<0>(t).create_needed(c, l);
  }

  static bool continue_search(std::tuple<Selectors...>& t) {
    return std::get<0>(t).continue_search();
  }

  template <typename Label, typename Edge>
  static bool minimum_found(std::tuple<Selectors...>& t,
                            light_connection const* c, Label& l,
                            Edge const& e) {
    return std::get<0>(t).minimum_found(c, l, e);
  }
};

template <typename... Traits>
struct single_criterion_con_selector {
  std::tuple<Traits...> tup_;
  template <typename Label>
  bool create_needed(light_connection const* c, Label& l) {
    const auto size = std::tuple_size<std::tuple<Traits...>>::value;
    return selector<size - 1, Traits...>::create_needed(tup_, c, l);
  }
  template <typename Label, typename Edge>
  bool continue_search(light_connection const*, Label&, Edge const&) {
    const auto size = std::tuple_size<std::tuple<Traits...>>::value;
    return selector<size - 1, Traits...>::continue_search(tup_);
  }
};

template <typename... Traits>
struct multi_criterion_con_selector {
  std::tuple<Traits...> tup_;
  template <typename Label>
  bool create_needed(light_connection const*, Label&) {
    return true;
  }

  template <typename Label, typename Edge>
  bool continue_search(light_connection const* c, Label& l, Edge const& e) {
    const auto size = std::tuple_size<std::tuple<Traits...>>::value;
    return !selector<size - 1, Traits...>::minimum_found(tup_, c, l, e);
  }
};

}  // namespace routing
}  // namespace motis
