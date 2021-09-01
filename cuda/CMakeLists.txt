cmake_minimum_required(VERSION 3.8)
project(motis)

include_directories(kernel/include)
include_directories(kernel/src)

if(MOTIS_CUDA)
  file(GLOB_RECURSE cuda-kernel-files kernel/src/everything.cu)

  #set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  #CUDA_ADD_LIBRARY(cuda-library STATIC ${cuda-kernel-files})
  #set_target_properties(cuda-library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_library(cuda-library SHARED kernel/src/everything.cu)
  set_target_properties(cuda-library PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/"
  )
  target_include_directories(cuda-library PUBLIC kernel/include)
  set_property(TARGET cuda-library PROPERTY CUDA_ARCHITECTURES 75 61)
  target_compile_options(cuda-library PRIVATE "--expt-relaxed-constexpr")
  
  target_link_libraries(cuda-library
   raptor-core
 # motis-core
 #  motis-loader
 #  motis-module
 #  motis-bootstrap
  )
endif()
