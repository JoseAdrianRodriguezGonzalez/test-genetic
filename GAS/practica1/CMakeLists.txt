cmake_minimum_required(VERSION 3.10)
project(SimpleGeneticAlgorithm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(core
    src/ga/sga.cpp
    src/experiments/program.cpp 
    src/io/reader.cpp
)

target_include_directories(core PUBLIC lib)

add_executable(programa app/main.cpp)

target_link_libraries(programa core)
