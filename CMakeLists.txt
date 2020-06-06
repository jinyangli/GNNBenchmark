cmake_minimum_required(VERSION 2.8)
project(GNNBenchmark C CXX)

message(STATUS "Start configuring project ${PROJECT_NAME}")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
endif(OPENMP_FOUND)

# include directories
include_directories("minigun/minigun/")
include_directories("minigun/third_party/dlpack/include")
include_directories("minigun/third_party/dmlc-core/include")
include_directories("minigun/third_party/moderngpu/src")

add_subdirectory(minigun/third_party/dmlc-core)
add_subdirectory(tests)
