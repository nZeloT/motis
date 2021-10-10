#pragma once

#include "motis/raptor-core/config/config.h"
#include "motis/raptor-core/config/filters.h"
#include "motis/raptor-core/config/traits.h"
#include "motis/raptor-core/config/traits/occupancy.h"

namespace motis::raptor {

using Default = config<traits<>, filter<>>;

using OccupancyOnly = config<traits<trait_max_occupancy>, filter<>>;

}  // namespace motis::raptor