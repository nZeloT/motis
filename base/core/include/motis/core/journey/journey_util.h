#pragma once

#include <cstdint>
#include <ctime>

namespace motis {
struct journey;

uint16_t get_duration(journey const&);
uint16_t get_transfers(journey const&);
uint16_t get_accessibility(journey const&);
std::time_t get_arrival(journey const&);
std::time_t get_departure(journey const&);

}  // namespace motis
