#include "motis/raptor/route_conflict_graph.h"

#include <fstream>

#include "motis/core/common/logging.h"

#include "motis/raptor/raptor_util.h"

#include "utl/parallel_for.h"

namespace motis {
namespace raptor {

using namespace motis::logging;

/* uses the METIS file format, also used by karhipp */
void write_conflict_graph_to_file(conflict_graph const& cg) {
  LOG(info) << "Writing conflict graph to file";
  auto number_of_vertices = cg.size();
  auto number_of_edges = 0;
  for (auto const& v : cg.edges_) { number_of_edges += v.size(); }
  number_of_edges /= 2;

  // 0 no weights | 1 edge weights
  // 10 node weights | 11 edge and node weights
  auto const weights = "11";

  std::ofstream conflict_graph_file("route-conflict-graph.txt");

  conflict_graph_file << number_of_vertices << " "
                      << number_of_edges << " "
                      << weights << '\n';

  for (auto v_id = 0u; v_id < cg.size(); ++v_id) {
    conflict_graph_file << cg.vertex_weights_[v_id] << " ";
    for (auto edge : cg.edges_[v_id]) { 
      conflict_graph_file << (edge.to_ + 1) << " " << edge.weight_ << " "; 
    }
    conflict_graph_file << '\n';
  }

  conflict_graph_file.close();
}
conflict_graph get_route_conflict_graph(transformable_timetable const& ttt) {
  LOG(info) << "Generating route conflict graph";
  conflict_graph cg(ttt.routes_.size());

  auto get_confict_edges = [&](size_t const r_id) {
    auto const& route = ttt.routes_[r_id];

    cg.vertex_weights_[r_id] = route.route_stops_.size() * route.trips_.size();

    std::vector<edge_weight> edge_weights(ttt.routes_.size(), 0);

    auto route_stops_copy = route.route_stops_;
    sort_and_unique(route_stops_copy);

    for (auto s_id : route_stops_copy) {
      auto const& s = ttt.stations_[s_id];
      for (auto conflict_r_id : s.stop_routes_) {
        if (conflict_r_id == r_id) { continue; }
        ++edge_weights[conflict_r_id];
      }
    }

    for (route_id cr_id = 0; cr_id < edge_weights.size(); ++cr_id) {
      if (edge_weights[cr_id] == 0) { continue; }
      cg.edges_[r_id].emplace_back(cr_id, edge_weights[cr_id]);
    }
  };

//  utl::parallel_for(ttt.routes_.size(), get_confict_edges);

  return cg;
}
void write_route_graph_information_to_file(transformable_timetable const& ttt) {
  LOG(info) << "Writing route graph information to file";
  std::ofstream route_graph_file("route-graph-information.txt");

  route_graph_file << ttt.routes_.size() << '\n';
  for (route_id r_id = 0; r_id < ttt.routes_.size(); ++r_id) {
    auto const& route = ttt.routes_[r_id];
    route_graph_file << route.trips_.size() << " ";
    for (auto rs : route.route_stops_) {
      route_graph_file << rs << " ";
    }
    route_graph_file << '\n';
  }
  route_graph_file.close();
}

}  // namespace raptor
}  // namespace motis