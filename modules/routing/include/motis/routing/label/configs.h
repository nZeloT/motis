#pragma once

#include "motis/routing/label/comparator.h"
#include "motis/routing/label/connection_selector.h"
#include "motis/routing/label/criteria/absurdity.h"
#include "motis/routing/label/criteria/accessibility.h"
#include "motis/routing/label/criteria/late_connections.h"
#include "motis/routing/label/criteria/no_intercity.h"
#include "motis/routing/label/criteria/occupancy.h"
#include "motis/routing/label/criteria/transfers.h"
#include "motis/routing/label/criteria/travel_time.h"
#include "motis/routing/label/criteria/weighted.h"
#include "motis/routing/label/criteria/transfer_classes.h"
#include "motis/routing/label/dominance.h"
#include "motis/routing/label/filter.h"
#include "motis/routing/label/initializer.h"
#include "motis/routing/label/label.h"
#include "motis/routing/label/tie_breakers.h"
#include "motis/routing/label/updater.h"

namespace motis::routing {

template <search_dir Dir>
using default_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, absurdity>,
          initializer<travel_time_initializer, transfers_initializer,
                      absurdity_initializer>,
          updater<travel_time_updater, transfers_updater, absurdity_updater>,
          filter<travel_time_filter, transfers_filter>,
          default_edge_cost_function,
          single_criterion_con_selector<default_con_selector>,
          dominance<absurdity_tb, travel_time_dominance, transfers_dominance>,
          dominance<absurdity_post_search_tb, travel_time_alpha_dominance,
                    transfers_dominance>,
          comparator<transfers_dominance>>;

template <search_dir Dir>
using tc_label = label<
    Dir, MAX_TRAVEL_TIME, true, get_travel_time_lb,
    label_data<travel_time, transfers, absurdity, transfer_classes>,
    initializer<travel_time_initializer, transfers_initializer,
                absurdity_initializer, transfer_classes_initializer>,
    updater<travel_time_updater, transfers_updater, absurdity_updater,
            transfer_classes_updater>,
    filter<travel_time_filter, transfers_filter>,
    transfer_class_edge_cost_function,
    single_criterion_con_selector<transfer_classes_con_selector,
                                  default_con_selector>,
    dominance<absurdity_tb, travel_time_dominance, transfers_dominance,
              transfer_classes_max_dominance>,
    dominance<absurdity_post_search_tb, travel_time_alpha_dominance,
              transfers_dominance, transfer_classes_max_dominance>,
    comparator<transfers_dominance>>;

template <search_dir Dir>
using default_simple_label = label<
    Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
    label_data<travel_time, transfers>,
    initializer<travel_time_initializer, transfers_initializer>,
    updater<travel_time_updater, transfers_updater>,
    filter<travel_time_filter, transfers_filter>, default_edge_cost_function,
    single_criterion_con_selector<default_con_selector>,
    dominance<default_tb, travel_time_dominance, transfers_dominance>,
    dominance<post_search_tb, travel_time_alpha_dominance, transfers_dominance>,
    comparator<transfers_dominance>>;

template <search_dir Dir>
using single_criterion_label =
    label<Dir, MAX_WEIGHTED, false, get_weighted_lb, label_data<weighted>,
          initializer<weighted_initializer>, updater<weighted_updater>,
          filter<weighted_filter>, default_edge_cost_function,
          single_criterion_con_selector<default_con_selector>,
          dominance<default_tb, weighted_dominance>, dominance<post_search_tb>,
          comparator<weighted_dominance>>;

template <search_dir Dir>
using single_criterion_no_intercity_label =
    label<Dir, MAX_WEIGHTED, false, get_weighted_lb, label_data<weighted>,
          initializer<weighted_initializer>, updater<weighted_updater>,
          filter<weighted_filter, no_intercity_filter>,
          default_edge_cost_function,
          single_criterion_con_selector<default_con_selector>,
          dominance<default_tb, weighted_dominance>, dominance<post_search_tb>,
          comparator<weighted_dominance>>;

template <search_dir Dir>
using max_occ_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, absurdity, occupancy>,
          initializer<travel_time_initializer, transfers_initializer,
                      absurdity_initializer, occupancy_initializer>,
          updater<travel_time_updater, transfers_updater, absurdity_updater,
                  occupancy_updater<false>>,
          filter<travel_time_filter, transfers_filter>,
          default_edge_cost_function,
          single_criterion_con_selector<default_con_selector>,
          dominance<default_tb, travel_time_dominance, transfers_dominance,
                    occupancy_dominance_max>,
          dominance<post_search_tb, travel_time_alpha_dominance,
                    transfers_dominance, occupancy_dominance_max>,
          comparator<transfers_dominance>>;

template<search_dir Dir>
using occ_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, absurdity, occupancy>,
          initializer<travel_time_initializer, transfers_initializer,
                      absurdity_initializer, occupancy_initializer>,
          updater<travel_time_updater, transfers_updater, absurdity_updater,
                  occupancy_updater<false>>,
          filter<travel_time_filter, transfers_filter>,
          default_edge_cost_function,
          single_criterion_con_selector<default_con_selector>,
          dominance<default_tb, travel_time_dominance, transfers_dominance,
                    occupancy_dominance>,
          dominance<post_search_tb, travel_time_alpha_dominance,
                    transfers_dominance, occupancy_dominance>,
          comparator<transfers_dominance>>;


template <search_dir Dir>
using late_connections_label = label<
    Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
    label_data<travel_time, transfers, late_connections, absurdity>,
    initializer<travel_time_initializer, transfers_initializer,
                late_connections_initializer, absurdity_initializer>,
    updater<travel_time_updater, transfers_updater, late_connections_updater,
            absurdity_updater>,
    filter<travel_time_filter, transfers_filter, late_connections_filter>,
    default_edge_cost_function,
    single_criterion_con_selector<default_con_selector>,
    dominance<absurdity_tb, travel_time_dominance, transfers_dominance,
              late_connections_dominance>,
    dominance<absurdity_post_search_tb, travel_time_alpha_dominance,
              transfers_dominance, late_connections_post_search_dominance>,
    comparator<transfers_dominance>>;

template <search_dir Dir>
using late_connections_label_for_tests = label<
    Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
    label_data<travel_time, transfers, late_connections>,
    initializer<travel_time_initializer, transfers_initializer,
                late_connections_initializer>,
    updater<travel_time_updater, transfers_updater, late_connections_updater>,
    filter<travel_time_filter, transfers_filter, late_connections_filter>,
    default_edge_cost_function,
    single_criterion_con_selector<default_con_selector>,
    dominance<default_tb, travel_time_dominance, transfers_dominance,
              late_connections_dominance>,
    dominance<post_search_tb, travel_time_alpha_dominance, transfers_dominance,
              late_connections_post_search_dominance_for_tests>,
    comparator<transfers_dominance>>;

template <search_dir Dir>
using accessibility_label =
    label<Dir, MAX_TRAVEL_TIME, false, get_travel_time_lb,
          label_data<travel_time, transfers, accessibility, absurdity>,
          initializer<travel_time_initializer, transfers_initializer,
                      accessibility_initializer, absurdity_initializer>,
          updater<travel_time_updater, transfers_updater, accessibility_updater,
                  absurdity_updater>,
          filter<travel_time_filter, transfers_filter>,
          default_edge_cost_function,
          single_criterion_con_selector<default_con_selector>,
          dominance<absurdity_tb, travel_time_dominance, transfers_dominance,
                    accessibility_dominance>,
          dominance<absurdity_post_search_tb, travel_time_alpha_dominance,
                    transfers_dominance, accessibility_dominance>,
          comparator<transfers_dominance, accessibility_dominance>>;

}  // namespace motis::routing
