#pragma once

#include "motis/raptor-core/config/config.h"
#include "motis/raptor-core/config/traits.h"
#include "motis/raptor-core/config/traits/max_occupancy.h"

namespace motis::raptor {

using Default = config<traits<>>;

using OccupancyOnly = config<traits<trait_max_occupancy>>;

}  // namespace motis::raptor