#pragma once

#include "motis/core/schedule/edges.h"
#include "motis/routing/lower_bounds.h"

namespace motis::routing {

struct default_con_selector {
  bool created_ = false;
  bool continue_search() { return !created_; }
  template <typename Label>
  bool create_needed(light_connection const*, Label&) {
    if (created_) {
      return false;
    }
    created_ = true;
    return true;
  }

  template <typename Label, typename Edge>
  bool minimum_found(light_connection const*, Label&, Edge const&) {
    return true;
  }
};

struct default_edge_cost_function {
  static edge_cost change_ec(edge_cost ec) { return ec; }
};

template <typename... DataClass>
struct label_data : public DataClass... {};

template <search_dir Dir, std::size_t MaxBucket,
          bool PostSearchDominanceEnabled, typename GetBucket, typename Data,
          typename Init, typename Updater, typename Filter,
          typename EdgeCostFunct, typename ConSelector, typename Dominance,
          typename PostSearchDominance, typename Comparator>
struct label : public Data {  // NOLINT
  enum : std::size_t { MAX_BUCKET = MaxBucket };

  label() = default;  // NOLINT

  label(edge const* e, label* pred, time now, lower_bounds& lb,
        light_connection const* lcon = nullptr)
      : pred_(pred),
        edge_(e),
        connection_(lcon),
        start_(pred != nullptr ? pred->start_ : now),
        now_(now),
        dominated_(false) {
    Init::init(*this, lb);
  }

  node const* get_node() const { return edge_->get_destination<Dir>(); }

  template <typename Edge, typename LowerBounds>
  bool create_single_label(label& l, Edge const& e, LowerBounds& lb, bool no_cost,
                    int additional_time_cost = 0) {
    if (pred_ && e.template get_destination<Dir>() == pred_->get_node()) {
      return false;
    }
    if ((e.type() == edge::BWD_EDGE ||
         e.type() == edge::AFTER_TRAIN_BWD_EDGE) &&
        edge_->type() == edge::EXIT_EDGE) {
      return false;
    }

    auto ec = e.template get_edge_cost<Dir>(now_, connection_);
    if (!ec.is_valid()) {
      return false;
    }
    if (no_cost) {
      ec.time_ = 0;
    } else {
      ec.time_ += additional_time_cost;
    }

    l = *this;
    l.pred_ = this;
    l.edge_ = &e;
    l.connection_ = ec.connection_;
    l.now_ += (Dir == search_dir::FWD) ? ec.time_ : -ec.time_;

    Updater::update(l, ec, lb);
    return !l.is_filtered();
  }

  template <typename Edge, typename LowerBounds, typename CreateFunc>
  bool create_label(Edge const& e, LowerBounds& lb, bool no_cost,
                    CreateFunc func, int additional_time_cost = 0) {
    label& old = *this;
    if (pred_ && e.template get_destination<Dir>() == pred_->get_node()) {
      return false;
    }

    if (e.type() == edge::BWD_EDGE && edge_->type() == edge::EXIT_EDGE) {
      return false;
    }
    if (e.template get_destination<Dir>()->is_foot_node() && pred_ &&
        pred_->pred_ == nullptr) {
      return false;
    }

    auto ec = e.template get_edge_cost<Dir>(now_, connection_);
    if (!ec.is_valid()) {
      return false;
    }
    if (no_cost) {
      ec.time_ = 0;
    } else {
      ec.time_ += additional_time_cost;
    }
    ec = EdgeCostFunct::change_ec(ec);
    auto add_wait_time = 0;
    ConSelector sel;
    auto sel_con_nr = 0;
    auto old_ec_con = ec.connection_;
    do {
      sel_con_nr++;
      old_ec_con = ec.connection_;
      if (sel.create_needed(ec.connection_, *this)) {
        label l = *this;
        l.pred_ = this;
        l.edge_ = &e;
        l.selected_con_nr_ = sel_con_nr;

        l.connection_ = ec.connection_;
        ec.time_ += add_wait_time;
        l.now_ += ((Dir == search_dir::FWD) ? ec.time_ : -ec.time_);
        Updater::update(l, ec, lb);
        if (!l.is_filtered()) {
          func(l);
        }
      }
      // Only select multiple connections if the route was just entered
      if (ec.connection_ == nullptr || old.edge_->type() != edge::ENTER_EDGE) {
        break;
      }
      auto const time = (Dir == search_dir::FWD ? ec.connection_->d_time_ + 1
                                                : ec.connection_->a_time_ - 1);
      add_wait_time = ec.connection_->d_time_ + 1 - now_;

      ec = e.template get_edge_cost<Dir>(time, old.connection_);
      ec = EdgeCostFunct::change_ec(ec);
      if (!ec.is_valid()) {
        break;
      }
    } while (sel.continue_search(old_ec_con, *this, e));
    return true;
  }

  inline bool is_filtered() { return Filter::is_filtered(*this); }

  bool dominates(label const& o) const {
    if (incomparable(o)) {
      return false;
    }
    return Dominance::dominates(false, *this, o);
  }

  bool incomparable(label const& o) const {
    return current_begin() < o.current_begin() ||
           current_end() > o.current_end();
  }

  time current_begin() const { return Dir == search_dir::FWD ? start_ : now_; }

  time current_end() const { return Dir == search_dir::FWD ? now_ : start_; }

  bool dominates_post_search(label const& o) const {
    return PostSearchDominance::dominates(false, *this, o);
  }

  bool operator<(label const& o) const {
    return Comparator::lexicographical_compare(*this, o);
  }

  static inline bool is_post_search_dominance_enabled() {
    return PostSearchDominanceEnabled;
  }

  std::size_t get_bucket() const { return GetBucket()(this); }

  label* pred_;
  edge const* edge_;
  light_connection const* connection_;
  time start_, now_;
  bool dominated_;
  uint8_t selected_con_nr_;
};

}  // namespace motis::routing
