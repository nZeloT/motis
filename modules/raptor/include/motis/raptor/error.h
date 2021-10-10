#pragma once

#include <system_error>
#include <type_traits>

namespace motis::raptor {

namespace error {
enum error_code_t {
  ok = 0,
  not_implemented = 1,
  internal_error = 2,
  search_type_not_supported = 3,
  via_not_supported = 4,
  additional_edges_not_supported = 5,
  journey_date_not_in_schedule = 6,
  source_station_not_in_schedule = 7,
  target_station_not_in_schedule = 8
};
}

class error_category_impl : public std::error_category {
public:
  [[nodiscard]] const char* name() const noexcept override {
    return "motis::raptor";
  }

  [[nodiscard]] std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "raptor: no error";
      case error::not_implemented: return "raptor: not implemented";
      case error::internal_error: return "raptor: internal error";
      case error::search_type_not_supported: return "raptor: search type not supported";
      case error::via_not_supported: return "raptor: via not supported";
      case error::additional_edges_not_supported: return "raptor: additional edges not supported";
      case error::journey_date_not_in_schedule: return "raptor: journey date not in schedule";
      case error::source_station_not_in_schedule: return "raptor: source station_id is not in schedule";
      case error::target_station_not_in_schedule: return "raptor: target station_id is not in schedule";
      default: return "raptor: unknown error";
    }
  }
};

inline const std::error_category& error_category() {
  static error_category_impl instance;
  return instance;
}

namespace error {
inline std::error_code make_error_code(error_code_t e) noexcept {
  return std::error_code(static_cast<int>(e), error_category());
}
}
};  // namespace motis::raptor

namespace std {

template<>
struct is_error_code_enum<motis::raptor::error::error_code_t>
: public std::true_type {};

}