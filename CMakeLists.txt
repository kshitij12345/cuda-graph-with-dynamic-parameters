cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(graph)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_executable(graph graph.cu)
target_sources(graph PRIVATE gpu_graph.cpp)
target_sources(graph PRIVATE gpu_graph.hpp)
target_compile_features(graph PUBLIC cxx_range_for)
target_link_libraries(graph ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
