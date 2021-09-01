project(motis)
cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE raptor-core-files src/*.cc)
add_library(raptor-core STATIC ${raptor-core-files})
target_compile_options(raptor-core PRIVATE ${MOTIS_CXX_FLAGS})
#target_link_libraries(raptor-core
#  utl
#  sparsehash
#  motis-loader
#  motis-module
#  cuda-library
#)
