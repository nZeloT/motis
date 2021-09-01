cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-raptor-files src/*.cc)
add_library(motis-raptor STATIC ${motis-raptor-files})
target_include_directories(motis-raptor PUBLIC include)
target_compile_features(motis-raptor PUBLIC cxx_std_17)
if (MOTIS_CUDA)
  target_link_libraries(motis-raptor 
    motis-core
    ${Boost_PROGRAM_OPTIONS_LIBRARY} 
    ${Boost_SYSTEM_LIBRARY} 
    cuda-library
    motis-module
    motis-routing
    CUDA::cudart_static
  )
else()
  target_link_libraries(motis-raptor
    ${Boost_PROGRAM_OPTIONS_LIBRARY} 
    ${Boost_SYSTEM_LIBRARY} 
    motis-module
    motis-routing
  )
endif()

target_compile_options(motis-raptor PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-raptor PRIVATE ${MOTIS_COMPILE_DEFINITIONS})
