cmake_minimum_required(VERSION 3.17)
project(motis)

include_directories(include)

include_directories(src/gpu)

if(MOTIS_CUDA)
  add_library(cuda-library SHARED src/gpu/everything.cu)
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


file(GLOB_RECURSE motis-validator-files eval/src/validate.cc)
add_executable(motis-validator EXCLUDE_FROM_ALL ${motis-validator-files})
target_compile_features(motis-validator PUBLIC cxx_std_17)
target_link_libraries(motis-validator
        motis-bootstrap
        motis-core
        motis-module
        conf
        ${CMAKE_THREAD_LIBS_INIT}
        ianatzdb-res
)
target_compile_options(motis-validator PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-validator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-print-raptor-files eval/src/print_raptor.cc)
add_executable(motis-print-raptor EXCLUDE_FROM_ALL ${motis-print-raptor-files})
target_compile_features(motis-print-raptor PUBLIC cxx_std_17)
target_link_libraries(motis-print-raptor
        motis-bootstrap
        motis-core
        motis-module
        conf
        ${CMAKE_THREAD_LIBS_INIT}
        ianatzdb-res
        )
target_compile_options(motis-print-raptor PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-print-raptor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-xtract-gtfs-files eval/src/gtfs-xtract.cc)
add_executable(motis-xtract-gtfs EXCLUDE_FROM_ALL ${motis-xtract-gtfs-files})
target_compile_features(motis-xtract-gtfs PUBLIC cxx_std_17)
target_link_libraries(motis-xtract-gtfs
        motis-bootstrap
        motis-core
        motis-module
        conf
        ${CMAKE_THREAD_LIBS_INIT}
        ianatzdb-res
        )
target_compile_options(motis-xtract-gtfs PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-xtract-gtfs PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")