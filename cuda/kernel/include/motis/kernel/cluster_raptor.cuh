#pragma once 

#include "motis/raptor-core/raptor_query.h"

namespace motis {

// never put this in raptor namespace again

// std::vector<std::vector<motis::time>> 
// raptor_result 
void invoke_cluster_raptor(d_query&);

} // namespace motis