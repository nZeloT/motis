#pragma once

#include <functional>

namespace motis {
namespace raptor {

// returns the size of a vectors contents in bytes
// works only if T is trivial
template <typename T>
inline auto vec_size_bytes(std::vector<T> const& vec) { 
  return sizeof(T) * vec.size(); 
}

// append contents of a vector to another one
template<typename T>
inline void append_vector(std::vector<T>& dst, std::vector<T> const& elems) {
  dst.insert(std::end(dst), std::begin(elems), std::end(elems));
}

// application of the erase-remove idiom
// removes all elements from a vector if the predicate is evaluated to true
template<typename T, typename Fn>
void erase_if(std::vector<T>& v, Fn fun) {
  v.erase(std::remove_if(std::begin(v), std::end(v), fun), std::end(v));
}

template<typename Container, typename UnaryOp>
inline void for_each(Container& c, UnaryOp const& unary_op) {
  std::for_each(std::begin(c), std::end(c), unary_op);
}

template<typename Container, typename UnaryOp>
inline void for_each(Container const& c, UnaryOp const& unary_op) {
  std::for_each(std::begin(c), std::end(c), unary_op);
}

// application of map on the passed container and simply returns the container,
// instead of an iterator
template<typename T, typename Result>
auto map(std::vector<T>& v, std::function<Result(T&)> const& fun) {
  std::transform(std::begin(v), std::end(v), std::begin(v), fun);
  return v;
}

// sorts the container and removes all duplicates
template<typename Container, typename SortFun>
inline void sort_and_unique(Container& c, SortFun const& sort_fun) {
  std::sort(std::begin(c), std::end(c), sort_fun);
  c.erase(std::unique(std::begin(c), std::end(c)), std::end(c));
}

// sorts the container and removes all duplicates
template <typename Container>
inline void sort_and_unique(Container& c) {
  std::sort(std::begin(c), std::end(c));
  c.erase(std::unique(std::begin(c), std::end(c)), std::end(c));
}


} // namespace raptor
} // namespace motis
