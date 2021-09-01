#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/footpath.h"

#include "motis/raptor/get_raptor_schedule.h"
//#include "motis/raptor-core/raptor_timetable.h"

namespace motis {
namespace raptor {

using vertex_id = route_id;
using vertex_weight = uint32_t;
using edge_weight = uint16_t;
using color = uint16_t;

struct conflict_edge {
  conflict_edge(route_id const t, edge_weight const w) : to_(t), weight_(w) {}

  route_id to_;
  edge_weight weight_;
};

struct conflict_graph {
  conflict_graph(size_t const s) : vertex_weights_(s, 0), edges_(s) {}

  auto size() const { return vertex_weights_.size(); }

  std::vector<vertex_weight> vertex_weights_;
  std::vector<std::vector<conflict_edge>> edges_;
};

void write_conflict_graph_to_file(conflict_graph const&);
conflict_graph get_route_conflict_graph(transformable_timetable const&);
void write_route_graph_information_to_file(transformable_timetable const&);

} // namespace raptor
} // namespace motis